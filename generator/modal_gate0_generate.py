"""Gate 0 goal generation: K_m final masks x K_a goal images per HF test row.

Reproduces the SGE-Goal CALVIN inference settings exactly
(``inferencers/sge_goal_pipeline/{fer-mask-prediction-code,image-editing-code}``):

* Step 2 (FER mask): Step-2 pipeline variant (no final-mask input), prompt =
  instruction + DEFAULT_INSTRUCTION, guidance 1.0, 10 steps, area capped at
  256^2, sides floored to multiples of 32, output resized back and saved as JPEG.
* Step 3 (goal image): multi-image pipeline, ``build_enhanced_prompt``, default
  guidance, 10 steps, original size, predicted mask read back from the JPEG.

Deviation (deliberate): Step 1 is skipped and the dataset's
``initial_edit_region_mask`` is used, which isolates Steps 2 and 3.

Only the per-sample seeds differ from the original scripts, so samples vary.

Output layout on the ``rm-policy-eval-data`` volume (what ``eval_method.py``
reads, one sub-folder per variant):

    /eval_data/<out_name>/m{j}a{k}/data/<hf_row+1>/{original_image.png,edited_image.jpg,edit_instruction.txt}
    /eval_data/<out_name>/masks/<hf_row+1>/mf_{j}.jpg

Usage:
    modal run modal_gate0_generate.py::prefetch               # CPU only, caches FLUX weights
    modal run modal_gate0_generate.py --limit 5               # pilot: time it, inspect outputs
    modal run --detach modal_gate0_generate.py --num-shards 2 # full run
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import modal

APP_NAME = "sge-gate0-generate"
GPU_TYPE = "L40S"
BASE_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PIPE_DIR = REPO / "inferencers/sge_goal_pipeline"
STEP2_DIR = PIPE_DIR / "fer-mask-prediction-code"
STEP3_DIR = PIPE_DIR / "image-editing-code"

HF_TEST_URL = (
    "https://huggingface.co/datasets/SuyashAjitChavan2002/Calvin-Dataset-Final/"
    "resolve/main/data/test-00000-of-00001-f87345d1f3d9dd3f.parquet"
)
PARQUET = "/hf_cache/calvin_test.parquet"

ckpt_vol = modal.Volume.from_name("rm-policy-ckpts", create_if_missing=True)
eval_data_vol = modal.Volume.from_name("rm-policy-eval-data", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("rm-policy-hf-cache", create_if_missing=True)
VOLUMES = {"/ckpts": ckpt_vol, "/eval_data": eval_data_vol, "/hf_cache": hf_cache_vol}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0", "torchvision",
        "diffusers", "transformers", "accelerate", "peft",
        "sentencepiece", "protobuf", "huggingface_hub", "hf_transfer",
        "pillow", "numpy", "pyarrow", "scikit-image", "scipy", "tqdm",
    )
    .env({
        "HF_HOME": "/hf_cache/hf_home",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    # Step 3 pipeline keeps its original module name (flux_kontext_eval_utils imports it).
    .add_local_file(STEP3_DIR / "pipeline_flux_kontext_multi_imgs.py",
                    "/root/gate0/pipeline_flux_kontext_multi_imgs.py")
    .add_local_file(STEP3_DIR / "flux_kontext_eval_utils.py",
                    "/root/gate0/flux_kontext_eval_utils.py")
    .add_local_file(STEP2_DIR / "pipeline_flux_kontext_multi_imgs.py",
                    "/root/gate0/pipeline_step2_fer.py")
    .add_local_file(STEP2_DIR / "modal_fer_mask_inference.py",
                    "/root/gate0/modal_fer_mask_inference.py")
)

app = modal.App(APP_NAME, image=image)
hf_secret = modal.Secret.from_name("huggingface-secret")


def _read_constant(py_file: str, name: str):
    """Read a literal module-level constant without importing the file."""
    tree = ast.parse(Path(py_file).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {py_file}")


def _ensure_parquet():
    if not Path(PARQUET).exists():
        import urllib.request
        Path(PARQUET).parent.mkdir(parents=True, exist_ok=True)
        print(f"[gate0] downloading {HF_TEST_URL}")
        urllib.request.urlretrieve(HF_TEST_URL, PARQUET)
        hf_cache_vol.commit()


def _resolve_lora(name: str) -> Path:
    for cand in (Path("/ckpts/ckpts") / name, Path("/ckpts") / name):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"LoRA {name} not found under /ckpts or /ckpts/ckpts")


@app.function(volumes=VOLUMES, secrets=[hf_secret], timeout=3600, cpu=2, memory=8192)
def prefetch():
    """CPU-only: cache FLUX Kontext weights + the HF test parquet in the volume."""
    from huggingface_hub import snapshot_download

    _ensure_parquet()
    path = snapshot_download(BASE_MODEL, token=os.environ.get("HF_TOKEN"))
    hf_cache_vol.commit()
    print(f"[gate0] cached {BASE_MODEL} at {path}")


@app.function(gpu=GPU_TYPE, volumes=VOLUMES, secrets=[hf_secret],
              timeout=6 * 3600, cpu=4, memory=32768)
def generate_shard(rows: list[int], km: int, ka: int, out_name: str, seed_base: int):
    import io
    import sys
    import time

    sys.path.insert(0, "/root/gate0")
    import pyarrow.parquet as pq
    import torch
    from PIL import Image

    from pipeline_flux_kontext_multi_imgs import FluxKontextPipeline as Step3Pipeline
    from pipeline_step2_fer import FluxKontextPipeline as Step2Pipeline
    from flux_kontext_eval_utils import build_enhanced_prompt

    fer_src = "/root/gate0/modal_fer_mask_inference.py"
    fer_instruction = _read_constant(fer_src, "DEFAULT_INSTRUCTION").strip()
    fer_cfg = _read_constant(fer_src, "INFERENCE_CONFIG")
    step3_steps = 10  # modal_flux_kontext_inference.py: num_inference_steps = 10

    _ensure_parquet()
    table = pq.read_table(PARQUET, columns=[
        "id", "original_image", "initial_edit_region_mask", "edit_instruction",
    ]).take(rows)

    def _img(cell):
        return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")

    dtype = torch.bfloat16
    t0 = time.perf_counter()
    pipe3 = Step3Pipeline.from_pretrained(BASE_MODEL, torch_dtype=dtype,
                                          token=os.environ.get("HF_TOKEN"))
    fer_lora = _resolve_lora("calvin-fer-mask-prediction.safetensors")
    edit_lora = _resolve_lora("calvin-image-editing.safetensors")
    pipe3.load_lora_weights(str(fer_lora.parent), weight_name=fer_lora.name, adapter_name="fer")
    pipe3.load_lora_weights(str(edit_lora.parent), weight_name=edit_lora.name, adapter_name="edit")
    pipe3 = pipe3.to("cuda")
    pipe2 = Step2Pipeline(**pipe3.components)  # shares transformer / VAE / text encoders
    for p in (pipe2, pipe3):
        p.set_progress_bar_config(disable=True)
    print(f"[gate0] models loaded in {time.perf_counter() - t0:.1f}s; {len(rows)} rows")

    out_root = Path("/eval_data") / out_name
    variants = [f"m{j}a{k}" for j in range(km) for k in range(ka)]
    timings = {"step2": [], "step3": []}

    for i, hf_row in enumerate(rows):
        sample = f"{hf_row + 1}"
        done = all((out_root / v / "data" / sample / "edited_image.jpg").exists() for v in variants)
        if done:
            continue

        orig = _img(table["original_image"][i].as_py())
        init_mask = _img(table["initial_edit_region_mask"][i].as_py())
        instr = table["edit_instruction"][i].as_py().strip()
        orig_w, orig_h = orig.size

        # ---- Step 2: K_m final masks -------------------------------------
        mult = 32
        w2, h2 = (orig_w // mult) * mult, (orig_h // mult) * mult
        cap = fer_cfg["resolution"] ** 2
        if w2 * h2 > cap:
            aspect = w2 / h2
            w2 = int((cap * aspect) ** 0.5) // mult * mult
            h2 = int((cap / aspect) ** 0.5) // mult * mult

        pipe3.set_adapters(["fer"])
        with torch.no_grad():
            pe2, ppe2, _ = pipe3.encode_prompt(instr + "\n\n" + fer_instruction, prompt_2=None)
        mask_dir = out_root / "masks" / sample
        mask_dir.mkdir(parents=True, exist_ok=True)
        masks = []
        for j in range(km):
            gen = torch.Generator(device="cuda").manual_seed(seed_base + 1000 * hf_row + j)
            ts = time.perf_counter()
            with torch.autocast("cuda"):
                out = pipe2(
                    original_image=orig, initial_edit_region_mask=init_mask,
                    prompt_embeds=pe2, pooled_prompt_embeds=ppe2, generator=gen,
                    height=h2, width=w2,
                    guidance_scale=fer_cfg["guidance_scale"],
                    num_inference_steps=fer_cfg["num_inference_steps"],
                    max_sequence_length=fer_cfg["max_sequence_length"],
                    _auto_resize=False, max_area=w2 * h2,
                )
            torch.cuda.synchronize()
            timings["step2"].append(time.perf_counter() - ts)
            m = out.images[0]
            if m.size != (orig_w, orig_h):
                m = m.resize((orig_w, orig_h), resample=Image.BILINEAR)
            mp = mask_dir / f"mf_{j}.jpg"
            m.save(mp, quality=95)
            masks.append(Image.open(mp).convert("RGB"))  # Step 3 consumed the JPEG

        # ---- Step 3: K_a goal images per mask ----------------------------
        pipe3.set_adapters(["edit"])
        with torch.no_grad():
            pe3, ppe3, _ = pipe3.encode_prompt(build_enhanced_prompt(instr), prompt_2=None)
        for j, mf in enumerate(masks):
            for k in range(ka):
                gen = torch.Generator(device="cuda").manual_seed(
                    seed_base + 1000 * hf_row + 100 + 10 * j + k)
                ts = time.perf_counter()
                with torch.autocast("cuda"):
                    out = pipe3(
                        original_image=orig, initial_edit_region_mask=init_mask,
                        final_edit_region_mask=mf,
                        prompt_embeds=pe3, pooled_prompt_embeds=ppe3,
                        num_inference_steps=step3_steps,
                        height=orig_h, width=orig_w,
                        _auto_resize=False, max_area=orig_h * orig_w,
                        generator=gen,
                    )
                torch.cuda.synchronize()
                timings["step3"].append(time.perf_counter() - ts)
                g = out.images[0]
                if g.size != (orig_w, orig_h):
                    g = g.resize((orig_w, orig_h), resample=Image.BILINEAR)
                d = out_root / f"m{j}a{k}" / "data" / sample
                d.mkdir(parents=True, exist_ok=True)
                g.save(d / "edited_image.jpg", "JPEG", quality=95)
                orig.save(d / "original_image.png")
                (d / "edit_instruction.txt").write_text(instr + "\n")

        if (i + 1) % 10 == 0:
            eval_data_vol.commit()
        s2 = sum(timings["step2"]) / max(1, len(timings["step2"]))
        s3 = sum(timings["step3"]) / max(1, len(timings["step3"]))
        print(f"[gate0] row {hf_row} ({i + 1}/{len(rows)})  step2 {s2:.2f}s/pass  "
              f"step3 {s3:.2f}s/pass", flush=True)

    eval_data_vol.commit()
    return {k: (sum(v) / len(v) if v else None) for k, v in timings.items()}


@app.local_entrypoint()
def main(
    rows_json: str = str(REPO / "variant_a_alignment/gate0_rows.json"),
    num_shards: int = 1,
    limit: int = 0,
    km: int = 3,
    ka: int = 2,
    out_name: str = "gate0",
    seed_base: int = 0,
):
    rows = json.loads(Path(rows_json).read_text())["rows"]
    if limit:
        rows = rows[:limit]
    shards = [rows[i::num_shards] for i in range(num_shards)]
    shards = [s for s in shards if s]
    print(f"[gate0] {len(rows)} rows, K_m={km}, K_a={ka}, {len(shards)} x {GPU_TYPE}")
    args = [(s, km, ka, out_name, seed_base) for s in shards]
    for res in generate_shard.starmap(args):
        print(f"[gate0] shard mean latency: {res}")
    n_pass = len(rows) * (km + km * ka)
    print(f"[gate0] done: {n_pass} FLUX passes -> rm-policy-eval-data:/{out_name}/")
