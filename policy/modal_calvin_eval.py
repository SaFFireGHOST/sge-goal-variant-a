# Copyright (c) 2026
"""Modal app that runs the GR-MG final-goal-conditioned policy evaluation
pipeline on a remote GPU.

This file is a thin wrapper around the existing scripts in ``evaluate/``:
``build_hf_calvin_mapping.py``, ``eval_hf.py``, ``eval.py``, ``visualize.py``.
The Python *logic* of those scripts is untouched — Modal just provides the
GPU, the build environment, and the persistent storage.

Modal volumes used (created on first run):

* ``rm-policy-calvin-data`` — CALVIN-D dataset (``task_D_D/`` tree).
* ``rm-policy-ckpts``       — GR-MG / IP2P / MAE checkpoints.
* ``rm-policy-runs``        — outputs (mapping JSON, eval results, mp4s).
* ``rm-policy-hf-cache``    — cached HF parquet downloads.

Entrypoints (``modal run modal_app.py::<name> -- ...``):

* ``setup_check``         — prints CUDA / PyBullet / CALVIN install status.
* ``upload_calvin_split`` — push a local CALVIN split tree into the data volume.
* ``upload_ckpt``         — push a local GR-MG / IP2P ckpt file into the ckpts volume.
* ``download_hf_parquet`` — pre-fetch the HF test parquet into the cache volume.
* ``build_mapping``       — run ``evaluate/build_hf_calvin_mapping.py`` remotely.
* ``eval_hf``             — run ``evaluate/eval_hf.py`` remotely.
* ``eval_calvin``         — run ``evaluate/eval.py`` remotely (the standard
                            LH-MTLC evaluation, with either IP2P or manifest
                            goal-image source).
* ``visualize``           — run ``evaluate/visualize.py`` remotely and copy
                            the mp4 back to the local machine.

See ``USAGE.md`` (Modal section) for end-to-end commands.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME = "rm-policy"

# Where stuff lives *inside* the Modal container.
WORKSPACE = "/workspace/RM-Policy"
CALVIN_REPO = "/opt/calvin"            # mees/calvin clone (provides calvin_agent, calvin_env)
DATA_MOUNT = "/data"                   # rm-policy-calvin-data volume
CKPT_MOUNT = "/ckpts"                  # rm-policy-ckpts volume
RUNS_MOUNT = "/runs"                   # rm-policy-runs volume
HF_CACHE_MOUNT = "/hf_cache"           # rm-policy-hf-cache volume

# Volumes (auto-created if missing).
data_vol = modal.Volume.from_name("rm-policy-calvin-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("rm-policy-ckpts", create_if_missing=True)
runs_vol = modal.Volume.from_name("rm-policy-runs", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("rm-policy-hf-cache", create_if_missing=True)
# CALVIN-EVALS goal-image generation method outputs (per-method dirs of 1087 samples).
eval_data_vol = modal.Volume.from_name(
    "rm-policy-eval-data", create_if_missing=True
)

EVAL_DATA_MOUNT = "/eval_data"

VOLUMES = {
    DATA_MOUNT: data_vol,
    CKPT_MOUNT: ckpt_vol,
    RUNS_MOUNT: runs_vol,
    HF_CACHE_MOUNT: hf_cache_vol,
    EVAL_DATA_MOUNT: eval_data_vol,
}

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
# CALVIN needs PyBullet with EGL, OpenGL drivers, and Python 3.9 (GR-MG was
# tested with 3.9 + CUDA 12.1). We install the CALVIN repo (with its env +
# agent submodules) into ``/opt/calvin`` so that ``calvin_agent`` and
# ``calvin_env`` are importable, and the CALVIN ``conf/`` tree is reachable
# at ``./calvin/calvin_models/conf`` from ``WORKSPACE``.

_apt_packages = [
    "git",
    "wget",
    "build-essential",
    "cmake",
    "libegl1-mesa", "libgl1-mesa-glx", "libgles2-mesa",
    "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev",
    "libosmesa6", "libosmesa6-dev",
    "libglvnd-dev", "pkg-config",
    "ffmpeg",
    "aria2",
    "clang", "llvm",
]

_pip_packages = [
    "numpy<1.24",
    "scipy",
    "pillow",
    "pyarrow",
    "pandas",
    "tqdm",
    "termcolor",
    "matplotlib",
    "imageio[ffmpeg]",
    "opencv-python-headless",
    "omegaconf",
    "hydra-core==1.1.1",
    "pytorch-lightning==1.8.6",
    "transformers==4.36.2",
    "diffusers==0.27.2",
    "accelerate",
    "huggingface_hub",
    "datasets",
    "sentence-transformers",
    "ftfy", "regex",
    "pybullet",
    "open3d",
    "moviepy",
    "imageio-ffmpeg",
    "requests",
    "flamingo-pytorch",
    "einops",
    "timm",
    "git+https://github.com/openai/CLIP.git",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    .env({
        "DEBIAN_FRONTEND": "noninteractive",
        "MESA_GL_VERSION_OVERRIDE": "3.3",
        "PYOPENGL_PLATFORM": "egl",
    })
    .run_commands(
        # Enable universe/multiverse repos (needed for ffmpeg, clang, llvm, libosmesa, etc.)
        "sed -i 's/^# deb http/deb http/g' /etc/apt/sources.list; "
        "grep -q universe /etc/apt/sources.list || "
        "echo 'deb http://archive.ubuntu.com/ubuntu jammy main restricted universe multiverse' >> /etc/apt/sources.list; "
        "echo 'deb http://archive.ubuntu.com/ubuntu jammy-updates main restricted universe multiverse' >> /etc/apt/sources.list; "
        "echo 'deb http://archive.ubuntu.com/ubuntu jammy-security main restricted universe multiverse' >> /etc/apt/sources.list",
    )
    .run_commands(
        # apt-get with retries to handle transient mirror failures.
        "rm -rf /var/lib/apt/lists/* && "
        "ok=0; for i in 1 2 3 4 5; do "
        "apt-get update "
        "&& apt-get install -y --fix-missing "
        + " ".join(_apt_packages)
        + " && ok=1 && break; "
        "echo \"apt retry $i/5\"; "
        "rm -rf /var/lib/apt/lists/*; "
        "sleep 15; done; "
        "[ $ok -eq 1 ] || { echo 'FATAL: apt-get install failed after 5 retries'; exit 1; }",
    )
    # Pin numpy<2 first so torch installs against numpy 1.x ABI.
    .pip_install("numpy==1.23.5")
    .pip_install(
        "torch==2.1.2", "torchvision==0.16.2",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(*_pip_packages)
    # gym==0.21.0 fails with modern setuptools; install it in isolation
    # with pinned build deps and no build isolation.
    .run_commands(
        "pip install 'setuptools==65.5.0' 'wheel==0.38.4'",
        "pip install --no-build-isolation 'gym==0.21.0'",
    )
    # Install CALVIN (env + agent). pyhash needs old setuptools.
    .run_commands(
        "pip install 'setuptools<58' wheel",
        f"git clone --recurse-submodules https://github.com/mees/calvin.git {CALVIN_REPO}",
        # Patch CALVIN's install scripts to skip the egl_check sub-build that
        # requires a working display at install time.
        f"cd {CALVIN_REPO}/calvin_env && pip install -e . || true",
        f"cd {CALVIN_REPO}/calvin_env/tacto && pip install -e . || true",
        f"cd {CALVIN_REPO}/calvin_models && pip install --no-deps -e .",
        # pyhash needs clang + llvm-ar at /tools/llvm/bin (Modal sysconfig path).
        "mkdir -p /tools/llvm/bin && ln -sf $(which clang) /tools/llvm/bin/clang && ln -sf $(which llvm-ar) /tools/llvm/bin/llvm-ar",
        "pip install pyhash",
    )
    # Re-pin numpy LAST — torch 2.1 needs numpy<2 and the _ARRAY_API symbol.
    .run_commands(
        "pip install --force-reinstall 'numpy==1.23.5'",
        "python -c \"import numpy; import torch; print(f'numpy={numpy.__version__} torch.from_numpy OK={torch.from_numpy(numpy.zeros(1)) is not None}')\"",
    )
    # Pre-download CLIP ViT-B/32 so containers never hit transient network errors at runtime.
    .run_commands(
        "python -c \"import clip; clip.load('ViT-B/32', device='cpu'); print('CLIP ViT-B/32 cached')\"",
    )
    # Drop the workspace into the image. ``add_local_dir`` copies at run
    # time, so this stays cheap to rebuild while iterating.
    .add_local_dir(
        Path(__file__).parent.as_posix(),
        WORKSPACE,
        ignore=[
            "__pycache__", "*.pyc",
            ".git", ".venv", "node_modules",
            # Don't ship local artefacts; they belong on volumes.
            "runs", "outputs", "logs", ".runlogs",
            "CLOVER-repo",
        ],
    )
)

app = modal.App(APP_NAME, image=image)


# ---------------------------------------------------------------------------
# Tiny shared helpers (run inside the container).
# ---------------------------------------------------------------------------

def _enter_workspace_with_calvin_conf():
    """Switch to WORKSPACE and ensure ``./calvin/calvin_models/conf`` exists.

    The eval scripts hard-code ``./calvin/calvin_models/conf`` as the relative
    path to the CALVIN config tree. We satisfy that by symlinking the
    installed CALVIN repo into the workspace.
    """
    os.chdir(WORKSPACE)
    link = Path(WORKSPACE) / "calvin"
    if not link.exists():
        os.symlink(CALVIN_REPO, link)


def _run(cmd: list, env=None) -> int:
    import subprocess

    print("[modal] $", " ".join(cmd), flush=True)
    full_env = os.environ.copy()
    # Ensure workspace is on PYTHONPATH so `policy`, `evaluate`, etc. are importable.
    pp = full_env.get("PYTHONPATH", "")
    full_env["PYTHONPATH"] = f"{WORKSPACE}:{pp}" if pp else WORKSPACE
    if env:
        full_env.update(env)
    return subprocess.call(cmd, env=full_env)


# ---------------------------------------------------------------------------
# In-Modal data/ckpt fetchers (avoid going through local disk)
# ---------------------------------------------------------------------------

# CALVIN-D validation-only fetch.  The full task_D_D split is hosted as a
# single ~70 GB zip but we only need validation/ (~4 GB compressed).
# Strategy: read the ZIP central directory via HTTP Range requests, then
# download *only* the validation/ members using concurrent Range GETs.
# This cuts download from ~70 GB / 7 h to ~4 GB / 15-30 min.
_CALVIN_D_URL = "http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip"
_CALVIN_D_SCENE_INFO_FIX = (
    "http://calvin.cs.uni-freiburg.de/scene_info_fix/task_D_D_scene_info.zip"
)


def _partial_zip_extract(url: str, prefix: str, dest: str,
                         workers: int = 16):
    """Download only entries matching *prefix* from a remote ZIP via Range reqs.

    Works by:
      1. HTTP HEAD → total file size
      2. Range GET last 4 MB → locate End-of-Central-Directory
      3. Range GET full central directory → parse all entries
      4. Filter entries whose name starts with *prefix*
      5. Concurrent Range GETs for each entry's local-header + compressed data
      6. Decompress to *dest*

    Falls back to full download + selective extract if the server does not
    support Range requests.
    """
    import io
    import struct
    import zlib
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import requests

    sess = requests.Session()
    sess.headers["User-Agent"] = "Mozilla/5.0 (CALVIN-fetch)"

    # ---- 1. HEAD -----------------------------------------------------------
    head = sess.head(url, allow_redirects=True, timeout=30)
    total_size = int(head.headers.get("Content-Length", 0))
    accept_ranges = head.headers.get("Accept-Ranges", "")
    print(f"[partial-zip] total_size={total_size/1e9:.2f} GB  "
          f"Accept-Ranges={accept_ranges!r}")
    if total_size == 0 or "bytes" not in accept_ranges.lower():
        print("[partial-zip] server does NOT support Range requests — "
              "falling back to full download")
        return None  # caller will use fallback

    # ---- 2. Read EOCD (End of Central Directory) ---------------------------
    eocd_read = min(4 * 1024 * 1024, total_size)  # last 4 MB
    r = sess.get(url, headers={
        "Range": f"bytes={total_size - eocd_read}-{total_size - 1}",
    }, timeout=60)
    r.raise_for_status()
    tail = r.content

    # Find EOCD signature (0x06054b50) scanning from the end
    eocd_sig = b"\x50\x4b\x05\x06"
    eocd_pos = tail.rfind(eocd_sig)
    if eocd_pos == -1:
        print("[partial-zip] EOCD signature not found")
        return None

    # Parse EOCD
    eocd = tail[eocd_pos:eocd_pos + 22]
    (_, _, _, _, _, cd_size, cd_offset, _) = struct.unpack("<IHHHHIIH", eocd)
    print(f"[partial-zip] central_dir offset={cd_offset}  size={cd_size}")

    # Check for ZIP64 EOCD locator (0x07064b50)
    zip64_locator_sig = b"\x50\x4b\x06\x07"
    zip64_loc_pos = tail.rfind(zip64_locator_sig)
    if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or zip64_loc_pos != -1:
        # ZIP64: read the ZIP64 EOCD record
        if zip64_loc_pos != -1:
            loc = tail[zip64_loc_pos:zip64_loc_pos + 20]
            _, _, zip64_eocd_offset, _ = struct.unpack("<IIQI", loc)
            # Read ZIP64 EOCD
            rng = f"bytes={zip64_eocd_offset}-{zip64_eocd_offset + 55}"
            r64 = sess.get(url, headers={"Range": rng}, timeout=60)
            r64.raise_for_status()
            z64 = r64.content
            if z64[:4] == b"\x50\x4b\x06\x06":
                cd_size = struct.unpack("<Q", z64[40:48])[0]
                cd_offset = struct.unpack("<Q", z64[48:56])[0]
                print(f"[partial-zip] ZIP64 central_dir offset={cd_offset}  "
                      f"size={cd_size}")

    # ---- 3. Read full central directory ------------------------------------
    r = sess.get(url, headers={
        "Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}",
    }, timeout=120)
    r.raise_for_status()
    cd_data = r.content
    print(f"[partial-zip] central directory read: {len(cd_data)} bytes")

    # ---- 4. Parse CD entries -----------------------------------------------
    entries = []
    pos = 0
    cd_sig = b"\x50\x4b\x01\x02"
    while pos < len(cd_data) - 46:
        if cd_data[pos:pos + 4] != cd_sig:
            break
        (sig, _ver_made, _ver_needed, _flags, compress_method,
         _mod_time, _mod_date, _crc32,
         compressed_size, uncompressed_size,
         name_len, extra_len, comment_len,
         _disk_start, _int_attr, _ext_attr,
         local_header_offset) = struct.unpack(
            "<4sHHHHHHIIIHHHHHII", cd_data[pos:pos + 46])
        name = cd_data[pos + 46:pos + 46 + name_len].decode("utf-8", "replace")

        # Handle ZIP64 extra field if sizes are 0xFFFFFFFF
        extra_data = cd_data[pos + 46 + name_len:pos + 46 + name_len + extra_len]
        if uncompressed_size == 0xFFFFFFFF or compressed_size == 0xFFFFFFFF or \
           local_header_offset == 0xFFFFFFFF:
            # Parse ZIP64 extended info extra field (tag 0x0001)
            epos = 0
            while epos < len(extra_data) - 4:
                tag, sz = struct.unpack("<HH", extra_data[epos:epos + 4])
                if tag == 0x0001:
                    vals = extra_data[epos + 4:epos + 4 + sz]
                    idx = 0
                    if uncompressed_size == 0xFFFFFFFF and idx + 8 <= len(vals):
                        uncompressed_size = struct.unpack("<Q", vals[idx:idx + 8])[0]
                        idx += 8
                    if compressed_size == 0xFFFFFFFF and idx + 8 <= len(vals):
                        compressed_size = struct.unpack("<Q", vals[idx:idx + 8])[0]
                        idx += 8
                    if local_header_offset == 0xFFFFFFFF and idx + 8 <= len(vals):
                        local_header_offset = struct.unpack("<Q", vals[idx:idx + 8])[0]
                        idx += 8
                    break
                epos += 4 + sz

        entries.append({
            "name": name,
            "compress_method": compress_method,
            "compressed_size": compressed_size,
            "uncompressed_size": uncompressed_size,
            "local_header_offset": local_header_offset,
        })
        pos += 46 + name_len + extra_len + comment_len

    print(f"[partial-zip] total entries in archive: {len(entries)}")

    # Filter for validation/ entries
    wanted = [e for e in entries
              if f"/{prefix}" in e["name"] or e["name"].startswith(prefix)]
    total_dl = sum(e["compressed_size"] for e in wanted)
    print(f"[partial-zip] entries matching '{prefix}': {len(wanted)}  "
          f"compressed_total={total_dl/1e9:.2f} GB")

    if not wanted:
        print("[partial-zip] no matching entries — falling back")
        return None

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    # ---- 5 + 6. Download and extract each entry concurrently ---------------
    done_count = [0]  # mutable counter for progress
    errors = []

    def fetch_and_extract(entry):
        """Download one ZIP entry via Range request and decompress it."""
        name = entry["name"]
        if name.endswith("/"):
            # Directory entry — just create it
            (dest_path / name).mkdir(parents=True, exist_ok=True)
            return
        offset = entry["local_header_offset"]
        # Local file header is 30 bytes + name_len + extra_len; read a bit
        # extra to cover the header, then the compressed data follows.
        # First read 30 bytes of local header to learn actual name_len/extra_len.
        hdr_r = sess.get(url, headers={
            "Range": f"bytes={offset}-{offset + 29}",
        }, timeout=30)
        hdr_r.raise_for_status()
        hdr = hdr_r.content
        loc_name_len = struct.unpack("<H", hdr[26:28])[0]
        loc_extra_len = struct.unpack("<H", hdr[28:30])[0]
        data_offset = offset + 30 + loc_name_len + loc_extra_len
        comp_size = entry["compressed_size"]
        if comp_size == 0:
            # Empty file
            out = dest_path / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"")
            return

        data_r = sess.get(url, headers={
            "Range": f"bytes={data_offset}-{data_offset + comp_size - 1}",
        }, timeout=120)
        data_r.raise_for_status()
        raw = data_r.content

        # Decompress
        method = entry["compress_method"]
        if method == 0:
            content = raw
        elif method == 8:
            content = zlib.decompress(raw, -15)
        else:
            raise ValueError(f"unsupported compression method {method} for {name}")

        out = dest_path / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(content)

        done_count[0] += 1
        if done_count[0] % 2000 == 0:
            print(f"[partial-zip]   extracted {done_count[0]}/{len(wanted)}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_and_extract, e): e for e in wanted}
        for f in as_completed(futs):
            exc = f.exception()
            if exc:
                errors.append((futs[f]["name"], str(exc)))
                if len(errors) > 20:
                    print(f"[partial-zip] too many errors, aborting")
                    break

    if errors:
        print(f"[partial-zip] {len(errors)} errors:")
        for name, msg in errors[:10]:
            print(f"  {name}: {msg}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    print(f"[partial-zip] extracted {done_count[0]}/{len(wanted)} files")
    return dest_path


@app.function(
    volumes={DATA_MOUNT: data_vol},
    timeout=60 * 60 * 4,   # partial-zip should be ~30 min; full fallback ~4 h
    cpu=4,
    memory=16384,
)
def fetch_calvin_d(validation_only: bool = True, force: bool = False):
    """Download CALVIN-D validation into the rm-policy-calvin-data volume.

    Uses HTTP Range requests to download *only* the validation/ entries from
    the remote ZIP (~4 GB instead of 70 GB). Falls back to full download if
    the server rejects Range requests.
    """
    import shutil
    import subprocess
    import zipfile

    target_split = Path(DATA_MOUNT) / "task_D_D"
    val_dir = target_split / "validation"
    if val_dir.exists() and not force:
        n = sum(1 for _ in val_dir.iterdir())
        print(f"[fetch] {val_dir} already exists with {n} children; skipping. "
              "Pass force=True to redownload.")
        return str(val_dir)

    target_split.mkdir(parents=True, exist_ok=True)
    scratch = Path("/tmp/calvin_dl")
    scratch.mkdir(parents=True, exist_ok=True)

    # ---- Try partial-ZIP (fast path) ---- 
    result = _partial_zip_extract(
        _CALVIN_D_URL, "validation/", str(scratch), workers=16,
    )

    if result is not None:
        # Find the validation dir in scratch. Archive root is task_D_D/.
        src_val = None
        for candidate in [
            scratch / "task_D_D" / "validation",
            scratch / "validation",
        ]:
            if candidate.exists():
                src_val = candidate
                break
        if src_val is None:
            # Search recursively
            for p in scratch.rglob("validation"):
                if p.is_dir():
                    src_val = p
                    break
        if src_val is None:
            print("[fetch] partial-zip succeeded but validation/ not found "
                  "— falling back to full download")
            result = None

    if result is None:
        # ---- Fallback: full download with aria2c / wget ----
        zip_path = scratch / "task_D_D.zip"
        print(f"[fetch] FALLBACK: downloading full zip {_CALVIN_D_URL}")

        # Try aria2c (multi-connection) first
        rc = subprocess.call([
            "aria2c", "-x", "16", "-s", "16", "-k", "1M",
            "--file-allocation=none", "--continue=true",
            "-d", str(scratch), "-o", "task_D_D.zip", _CALVIN_D_URL,
        ])
        if rc != 0:
            rc = subprocess.call([
                "wget", "-q", "--show-progress", "-c", _CALVIN_D_URL,
                "-O", str(zip_path),
            ])
        if rc != 0:
            raise SystemExit(f"download failed (rc={rc})")

        print(f"[fetch] extracting validation/ from full zip...")
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            wanted = [n for n in names
                      if "/validation/" in n or n.startswith("validation/")]
            print(f"[fetch]   {len(wanted)}/{len(names)} members")
            for i, m in enumerate(wanted):
                zf.extract(m, scratch)
                if (i + 1) % 5000 == 0:
                    print(f"[fetch]     {i+1}/{len(wanted)}")

        src_val = scratch / "task_D_D" / "validation"
        if not src_val.exists():
            for child in scratch.iterdir():
                if child.is_dir() and (child / "validation").exists():
                    src_val = child / "validation"
                    break

    if src_val is None or not src_val.exists():
        raise SystemExit("validation/ dir not found after extraction")

    if val_dir.exists():
        shutil.rmtree(val_dir)
    print(f"[fetch] moving {src_val} -> {val_dir}")
    shutil.move(str(src_val), str(val_dir))

    # Apply the scene_info fix.
    print("[fetch] applying scene_info fix...")
    fix_zip = scratch / "scene_info_fix.zip"
    rc = subprocess.call([
        "wget", "-q", _CALVIN_D_SCENE_INFO_FIX, "-O", str(fix_zip),
    ])
    if rc == 0:
        with zipfile.ZipFile(fix_zip) as zf:
            zf.extractall(target_split)
        print("[fetch]   scene_info fix applied")
    else:
        print("[fetch]   scene_info fix download failed (non-fatal)")

    # Cleanup scratch.
    print("[fetch] cleaning scratch dir...")
    shutil.rmtree(scratch, ignore_errors=True)

    # Sanity: count episode_*.npz files.
    n_npz = sum(1 for _ in val_dir.glob("episode_*.npz"))
    has_lang = (val_dir / "lang_annotations" / "auto_lang_ann.npy").exists()
    print(f"[fetch] DONE  episode_npz_count={n_npz}  lang_ann={has_lang}")
    data_vol.commit()
    return str(val_dir)


_GR_MG_POLICY_URL = (
    "https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/gr_mg_release/"
    "epoch=47-step=83712.ckpt"
)
_GR_MG_GOAL_GEN_URL = (
    "https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/gr_mg_release/"
    "goal_gen.ckpt"
)
_MAE_URL = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"


@app.function(
    volumes={CKPT_MOUNT: ckpt_vol},
    timeout=60 * 60 * 4,
    cpu=2,
    memory=8192,
)
def fetch_grmg_ckpts(
    policy: bool = True,
    goal_gen: bool = False,
    mae: bool = True,
    force: bool = False,
):
    """Download GR-MG checkpoints directly into the ckpts volume.

    Defaults: policy + MAE. ``goal_gen`` is off because we use user-supplied
    final goal images, not the IP2P generator.
    """
    import subprocess

    plan = []
    if policy:
        plan.append((_GR_MG_POLICY_URL, "gr_mg/epoch=47-step=83712.ckpt"))
    if goal_gen:
        plan.append((_GR_MG_GOAL_GEN_URL, "goal_gen.ckpt"))
    if mae:
        plan.append((_MAE_URL, "MAE/mae_pretrain_vit_base.pth"))

    for url, rel in plan:
        out = Path(CKPT_MOUNT) / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not force:
            print(f"[fetch] {out} already present ({out.stat().st_size/1e6:.1f} MB), skip")
            continue
        print(f"[fetch] downloading {url} -> {out}")
        rc = subprocess.call(["wget", "-q", "--show-progress", "-c", url, "-O", str(out)])
        if rc != 0:
            rc = subprocess.call(["curl", "-fL", "-o", str(out), url])
            if rc != 0:
                raise SystemExit(f"download failed for {url}")
        print(f"[fetch]   ok ({out.stat().st_size/1e6:.1f} MB)")

    ckpt_vol.commit()
    print("[fetch] all done")


# ---------------------------------------------------------------------------
# Setup / diagnostics
# ---------------------------------------------------------------------------

@app.function(volumes=VOLUMES, gpu="T4", timeout=60 * 30)
def setup_check():
    """Print versions and check that GPU + CALVIN imports work."""
    import importlib
    import platform

    print("python:", platform.python_version())
    try:
        import torch
        print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
              "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
    except Exception as e:
        print("torch import failed:", e)

    for mod in ("calvin_agent", "calvin_env", "pybullet", "hydra", "clip",
                "diffusers", "pyarrow"):
        try:
            m = importlib.import_module(mod)
            print(f"{mod}: ok ({getattr(m, '__version__', '?')})")
        except Exception as e:
            print(f"{mod}: FAILED ({e})")

    print("data dir:", os.listdir(DATA_MOUNT) if os.path.isdir(DATA_MOUNT) else "(empty)")
    print("ckpt dir:", os.listdir(CKPT_MOUNT) if os.path.isdir(CKPT_MOUNT) else "(empty)")
    print("runs dir:", os.listdir(RUNS_MOUNT) if os.path.isdir(RUNS_MOUNT) else "(empty)")


# ---------------------------------------------------------------------------
# Upload helpers (data + ckpts).
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def upload_calvin_split(
    local_path: str,
    split_name: str = "task_D_D",
):
    """Upload a local CALVIN split tree into the calvin-data volume.

    Usage::

        modal run modal_app.py::upload_calvin_split \\
            --local-path /home/me/calvin/dataset/task_D_D \\
            --split-name task_D_D

    After upload, the split is reachable inside container functions at
    ``/data/<split_name>``.
    """
    import subprocess
    src = Path(local_path).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"local_path is not a directory: {src}")
    remote = f"{split_name}"
    print(f"[modal] uploading {src} -> rm-policy-calvin-data:/{remote} ...")
    # `modal volume put` recurses into directories.
    cmd = ["modal", "volume", "put", "rm-policy-calvin-data",
           str(src), remote, "--force"]
    rc = subprocess.call(cmd)
    raise SystemExit(rc)


@app.local_entrypoint()
def upload_ckpt(local_path: str, remote_name: str = ""):
    """Upload a single ckpt file to the ckpts volume.

    Usage::

        modal run modal_app.py::upload_ckpt \\
            --local-path ./epoch=47-step=83712.ckpt \\
            --remote-name gr_mg/epoch=47-step=83712.ckpt
    """
    import subprocess
    src = Path(local_path).expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"local_path is not a file: {src}")
    if not remote_name:
        remote_name = src.name
    print(f"[modal] uploading {src} -> rm-policy-ckpts:/{remote_name} ...")
    cmd = ["modal", "volume", "put", "rm-policy-ckpts",
           str(src), remote_name, "--force"]
    rc = subprocess.call(cmd)
    raise SystemExit(rc)


# ---------------------------------------------------------------------------
# HF parquet pre-fetch
# ---------------------------------------------------------------------------

_HF_TEST_URL = (
    "https://huggingface.co/datasets/SuyashAjitChavan2002/Calvin-Dataset-Final/"
    "resolve/main/data/test-00000-of-00001-f87345d1f3d9dd3f.parquet"
)
_HF_CACHE_FILE = f"{HF_CACHE_MOUNT}/calvin_test.parquet"


@app.function(volumes={HF_CACHE_MOUNT: hf_cache_vol}, timeout=60 * 30)
def download_hf_parquet(force: bool = False):
    """Download the HF test parquet into the cache volume."""
    import urllib.request

    out = Path(_HF_CACHE_FILE)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        print(f"[modal] HF parquet already cached: {out} ({out.stat().st_size} B)")
        return _HF_CACHE_FILE
    print(f"[modal] downloading {_HF_TEST_URL} -> {out}")
    urllib.request.urlretrieve(_HF_TEST_URL, out)
    hf_cache_vol.commit()
    print(f"[modal] cached: {out} ({out.stat().st_size} B)")
    return _HF_CACHE_FILE


# ---------------------------------------------------------------------------
# Mapping builder
# ---------------------------------------------------------------------------

@app.function(volumes=VOLUMES, gpu=None, timeout=60 * 60 * 6,
              cpu=4, memory=16384)
def build_mapping(
    split_name: str = "task_D_D",
    output_remote: str = "hf_calvin_mapping.json",
    nn_max_dist: float = 20.0,
):
    """Run ``evaluate/build_hf_calvin_mapping.py`` on Modal.

    No GPU needed (pure md5 / numpy work). Outputs go to
    ``rm-policy-runs:/<output_remote>``.
    """
    _enter_workspace_with_calvin_conf()
    # Make sure the parquet is cached.
    parquet_path = Path(_HF_CACHE_FILE)
    if not parquet_path.exists():
        import urllib.request
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[modal] HF parquet missing; downloading...")
        urllib.request.urlretrieve(_HF_TEST_URL, parquet_path)
        hf_cache_vol.commit()

    val_dir = f"{DATA_MOUNT}/{split_name}/validation"
    out_path = f"{RUNS_MOUNT}/{output_remote}"
    rc = _run([
        "python", "evaluate/build_hf_calvin_mapping.py",
        "--calvin_val_dir", val_dir,
        "--hf_parquet", str(parquet_path),
        "--output", out_path,
        "--nn_max_dist", str(nn_max_dist),
    ])
    runs_vol.commit()
    if rc != 0:
        raise SystemExit(rc)
    print(f"[modal] mapping written to rm-policy-runs:/{output_remote}")
    return out_path


# ---------------------------------------------------------------------------
# eval_hf — evaluate on the 1087 HF test rows
# ---------------------------------------------------------------------------

@app.function(volumes=VOLUMES, gpu="T4", timeout=60 * 60 * 12,
              cpu=4, memory=32768)
def eval_hf(
    split_name: str = "task_D_D",
    ckpt_remote: str = "gr_mg/epoch=47-step=83712.ckpt",
    config_path: str = "policy/config/train.json",
    mapping_remote: str = "hf_calvin_mapping.json",
    run_tag: str = "gt_final",
    epoch: int = 47,
    instruction_source: str = "hf",
    limit: int = -1,
    ep_len: int = 360,
    custom_parquet_remote: str = "",
):
    """Run ``evaluate/eval_hf.py`` on Modal.

    ``ckpt_remote`` is a path inside the ckpts volume. The script expects a
    *directory* of ckpts; we pass the parent of ``ckpt_remote``.

    ``custom_parquet_remote`` (optional) lets you point at an alternative
    parquet living inside the runs volume — typically one whose
    ``edited_image`` column has been replaced with the output of an
    image-editing model. Path is interpreted relative to ``/runs``.
    """
    _enter_workspace_with_calvin_conf()

    ckpt_full = Path(CKPT_MOUNT) / ckpt_remote
    if not ckpt_full.exists():
        raise SystemExit(f"checkpoint not found in volume: {ckpt_full}")
    ckpt_dir = str(ckpt_full.parent)

    if custom_parquet_remote:
        parquet_path = str(Path(RUNS_MOUNT) / custom_parquet_remote)
    else:
        parquet_path = _HF_CACHE_FILE
        if not Path(parquet_path).exists():
            import urllib.request
            Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_HF_TEST_URL, parquet_path)
            hf_cache_vol.commit()

    out_dir = f"{RUNS_MOUNT}/eval_hf_{split_name}_{run_tag}_epoch{epoch}"

    cmd = [
        "python", "evaluate/eval_hf.py",
        "--dataset_path", f"{DATA_MOUNT}/{split_name}",
        "--ckpt_dir", ckpt_dir,
        "--epoch", str(epoch),
        "--config_path", config_path,
        "--mapping_json", f"{RUNS_MOUNT}/{mapping_remote}",
        "--hf_parquet", parquet_path,
        "--output_dir", out_dir,
        "--instruction_source", instruction_source,
        "--ep_len", str(ep_len),
    ]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    rc = _run(cmd, env={"MESA_GL_VERSION_OVERRIDE": "3.3"})
    runs_vol.commit()
    if rc != 0:
        raise SystemExit(rc)
    print(f"[modal] eval_hf finished. results in rm-policy-runs:/{Path(out_dir).name}")


# ---------------------------------------------------------------------------
# eval_calvin — standard LH-MTLC evaluation (manifest or IP2P goal source)
# ---------------------------------------------------------------------------

@app.function(volumes=VOLUMES, gpu="T4", timeout=60 * 60 * 24,
              cpu=4, memory=32768)
def eval_calvin(
    split_name: str = "task_D_D",
    ckpt_remote: str = "gr_mg/epoch=47-step=83712.ckpt",
    config_path: str = "policy/config/train.json",
    epoch: int = 47,
    goal_image_source: str = "manifest",
    goal_images_manifest_remote: str = "manifest.json",
    ip2p_ckpt_remote: str = "goal_gen.ckpt",
    num_sequences: int = 1000,
    run_tag: str = "lh_mtlc",
):
    """Run ``evaluate/eval.py`` on Modal — the official 1000-sequence
    long-horizon evaluation, with either ``manifest`` or ``ip2p`` goal-image
    source.
    """
    _enter_workspace_with_calvin_conf()

    ckpt_full = Path(CKPT_MOUNT) / ckpt_remote
    if not ckpt_full.exists():
        raise SystemExit(f"checkpoint not found in volume: {ckpt_full}")
    ckpt_dir = str(ckpt_full.parent)

    cmd = [
        "python", "evaluate/eval.py",
        "--dataset_path", f"{DATA_MOUNT}/{split_name}",
        "--ckpt_dir", ckpt_dir,
        "--epoch", str(epoch),
        "--config_path", config_path,
        "--goal_image_source", goal_image_source,
        "--num_sequences", str(num_sequences),
        "--run_tag", run_tag,
    ]
    if goal_image_source == "manifest":
        cmd += ["--goal_images_manifest",
                str(Path(RUNS_MOUNT) / goal_images_manifest_remote)]
    elif goal_image_source == "ip2p":
        cmd += ["--ip2p_ckpt_path", str(Path(CKPT_MOUNT) / ip2p_ckpt_remote)]
    else:
        raise SystemExit(f"unknown goal_image_source: {goal_image_source}")

    rc = _run(cmd, env={"MESA_GL_VERSION_OVERRIDE": "3.3"})
    # eval.py writes into ckpt_dir; copy artefacts onto the runs volume.
    import shutil
    sub_dir = f"{split_name}_{goal_image_source}_{epoch}_epoch_{run_tag}"
    src = Path(ckpt_dir) / sub_dir
    if src.exists():
        dst = Path(RUNS_MOUNT) / sub_dir
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"[modal] copied results to rm-policy-runs:/{sub_dir}")
    ckpt_vol.commit()
    runs_vol.commit()
    if rc != 0:
        raise SystemExit(rc)


# ---------------------------------------------------------------------------
# visualize — render a single test sample's mp4 and stream it back
# ---------------------------------------------------------------------------

@app.function(volumes=VOLUMES, gpu="T4", timeout=60 * 60,
              cpu=2, memory=16384)
def _visualize_remote(
    split_name: str,
    ckpt_remote: str,
    epoch: int,
    config_path: str,
    output_remote: str,
    fps: int,
    ep_len: int,
    # HF-row mode (preferred)
    hf_row: int = -1,
    mapping_remote: str = "hf_calvin_mapping.json",
    # Legacy sequence mode
    sequence_idx: int = -1,
    subtask_idx: int = 0,
    num_sequences: int = 1000,
    goal_image_remote: str = "",
) -> bytes:
    _enter_workspace_with_calvin_conf()

    ckpt_full = Path(CKPT_MOUNT) / ckpt_remote
    if not ckpt_full.exists():
        raise SystemExit(f"checkpoint not found in volume: {ckpt_full}")
    ckpt_dir = str(ckpt_full.parent)

    output_path = str(Path(RUNS_MOUNT) / output_remote)

    cmd = [
        "python", "evaluate/visualize.py",
        "--dataset_path", f"{DATA_MOUNT}/{split_name}",
        "--ckpt_dir", ckpt_dir,
        "--epoch", str(epoch),
        "--config_path", config_path,
        "--output", output_path,
        "--fps", str(fps),
        "--ep_len", str(ep_len),
    ]

    if hf_row >= 0:
        # Ensure HF parquet is available
        parquet_path = _HF_CACHE_FILE
        if not Path(parquet_path).exists():
            import urllib.request
            Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_HF_TEST_URL, parquet_path)
            hf_cache_vol.commit()
        cmd += [
            "--hf_row", str(hf_row),
            "--mapping_json", str(Path(RUNS_MOUNT) / mapping_remote),
            "--hf_parquet", parquet_path,
        ]
    else:
        goal_image_path = str(Path(RUNS_MOUNT) / goal_image_remote)
        cmd += [
            "--sequence_idx", str(sequence_idx),
            "--subtask_idx", str(subtask_idx),
            "--num_sequences", str(num_sequences),
            "--goal_image", goal_image_path,
        ]

    rc = _run(cmd, env={"MESA_GL_VERSION_OVERRIDE": "3.3"})
    runs_vol.commit()
    if rc != 0:
        raise SystemExit(rc)

    with open(output_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def visualize(
    output: str = "./visualization.mp4",
    hf_row: int = -1,
    mapping_remote: str = "hf_calvin_mapping.json",
    goal_image: str = "",
    sequence_idx: int = -1,
    subtask_idx: int = 0,
    num_sequences: int = 1000,
    split_name: str = "task_D_D",
    ckpt_remote: str = "gr_mg/epoch=47-step=83712.ckpt",
    epoch: int = 47,
    config_path: str = "policy/config/train.json",
    fps: int = 30,
    ep_len: int = 360,
):
    """Local entrypoint: run visualisation on Modal and stream the mp4 back.

    Preferred usage (HF-row mode — correct env reset + real goal image)::

        modal run modal_app.py::visualize --hf-row 121

    Legacy usage (user-supplied goal image + CALVIN sequence index)::

        modal run modal_app.py::visualize \\
            --goal-image ./my_goal.png \\
            --sequence-idx 3 --subtask-idx 0
    """
    import uuid

    out_local = Path(output).expanduser().resolve()
    out_local.parent.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:8]
    output_remote = f"viz/{job_id}/out.mp4"

    if hf_row >= 0:
        print(f"[modal] visualising HF row {hf_row} (mapping={mapping_remote})")
        mp4_bytes = _visualize_remote.remote(
            split_name=split_name,
            ckpt_remote=ckpt_remote,
            epoch=epoch,
            config_path=config_path,
            output_remote=output_remote,
            fps=fps,
            ep_len=ep_len,
            hf_row=hf_row,
            mapping_remote=mapping_remote,
        )
    else:
        import subprocess

        if not goal_image:
            raise SystemExit("Must specify either --hf-row or --goal-image")
        src = Path(goal_image).expanduser().resolve()
        if not src.is_file():
            raise SystemExit(f"goal image not found: {src}")
        goal_remote = f"viz/{job_id}/goal{src.suffix}"
        print(f"[modal] uploading goal image -> rm-policy-runs:/{goal_remote}")
        rc = subprocess.call([
            "modal", "volume", "put", "rm-policy-runs",
            str(src), goal_remote, "--force",
        ])
        if rc != 0:
            raise SystemExit(rc)

        print(f"[modal] running remote visualisation...")
        mp4_bytes = _visualize_remote.remote(
            split_name=split_name,
            ckpt_remote=ckpt_remote,
            epoch=epoch,
            config_path=config_path,
            output_remote=output_remote,
            fps=fps,
            ep_len=ep_len,
            sequence_idx=sequence_idx,
            subtask_idx=subtask_idx,
            num_sequences=num_sequences,
            goal_image_remote=goal_remote,
        )

    out_local.write_bytes(mp4_bytes)
    print(f"[modal] wrote {out_local} ({len(mp4_bytes)} B)")


# ===========================================================================
# Multi-method sharded evaluation
# ===========================================================================
# Each goal-image generation method has its outputs stored under
# ``/eval_data/<method_name>/<subdir>/<row_1>/{original_image.*, edited_image.*,
# edit_instruction.txt, ...}``. ``upload_eval_method`` uploads one method's
# directory to ``rm-policy-eval-data`` so that GPU shards can read it.

@app.local_entrypoint()
def upload_eval_method(local_path: str, method_name: str = ""):
    """Upload one CALVIN-EVALS/<method>/ directory to rm-policy-eval-data."""
    src = Path(local_path).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"local CALVIN-EVALS method dir not found: {src}")
    if not method_name:
        method_name = src.name
    remote_dir = f"{method_name}"
    print(f"[modal] uploading {src} -> rm-policy-eval-data:/{remote_dir}/")
    with eval_data_vol.batch_upload(force=True) as up:
        up.put_directory(str(src), remote_dir)
    print(f"[modal] upload_eval_method done: {method_name}")


@app.function(volumes=VOLUMES, gpu="T4", timeout=60 * 60 * 12,
              cpu=4, memory=32768)
def eval_method_shard(
    method_name: str,
    row_start: int,
    row_end: int,
    run_tag: str,
    ckpt_remote: str = "gr_mg/epoch=47-step=83712.ckpt",
    config_path: str = "policy/config/train.json",
    mapping_remote: str = "hf_calvin_mapping.json",
    split_name: str = "task_D_D",
    epoch: int = 47,
    ep_len: int = 360,
    use_hf_parquet_goal: bool = False,
    conditioning: str = "both",
):
    """One GPU worker: rolls out HF rows [row_start, row_end) for ``method_name``.

    Writes ``per_sample_<row_start>_<row_end>.jsonl`` and
    ``shard_done_<row_start>_<row_end>.json`` under
    ``/runs/eval_method/<run_tag>/``.

    If ``use_hf_parquet_goal`` is True, ignores ``method_name`` lookup in
    /eval_data and uses HF parquet's ``edited_image`` as the goal source.
    """
    _enter_workspace_with_calvin_conf()

    ckpt_full = Path(CKPT_MOUNT) / ckpt_remote
    if not ckpt_full.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_full}")
    ckpt_dir = str(ckpt_full.parent)

    out_dir = Path(RUNS_MOUNT) / "eval_method" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "evaluate/eval_method.py",
        "--dataset_path", f"{DATA_MOUNT}/{split_name}",
        "--ckpt_dir", ckpt_dir,
        "--epoch", str(epoch),
        "--config_path", config_path,
        "--mapping_json", f"{RUNS_MOUNT}/{mapping_remote}",
        "--output_dir", str(out_dir),
        "--row_start", str(row_start),
        "--row_end", str(row_end),
        "--ep_len", str(ep_len),
        "--method_name", method_name,
        "--conditioning", conditioning,
    ]
    if use_hf_parquet_goal:
        # Ensure parquet exists in hf_cache.
        parquet_path = "/hf_cache/calvin_test.parquet"
        if not Path(parquet_path).exists():
            import urllib.request
            Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_HF_TEST_URL, parquet_path)
            hf_cache_vol.commit()
        cmd += ["--hf_parquet_goal", parquet_path]
    else:
        cmd += ["--method_dir", f"{EVAL_DATA_MOUNT}/{method_name}"]

    rc = _run(cmd, env={"MESA_GL_VERSION_OVERRIDE": "3.3"})
    runs_vol.commit()
    if rc != 0:
        raise SystemExit(rc)
    print(f"[modal] eval_method_shard done: rows [{row_start}, {row_end})")


@app.local_entrypoint()
def run_method(
    method_name: str,
    num_shards: int = 5,
    use_hf_parquet_goal: bool = False,
    ep_len: int = 360,
    run_tag: str = "",
    conditioning: str = "both",
):
    """Spawn ``num_shards`` GPU workers in parallel to evaluate ``method_name``
    over the full 1087 HF test rows."""
    if not run_tag:
        run_tag = method_name
    rows = list(range(0, 1087))
    # Even split.
    chunks = []
    for i in range(num_shards):
        s = (1087 * i) // num_shards
        e = (1087 * (i + 1)) // num_shards
        chunks.append((s, e))
    print(f"[modal] run_method method={method_name} shards={chunks}")
    args_list = [
        (method_name, s, e, run_tag,
         "gr_mg/epoch=47-step=83712.ckpt",
         "policy/config/train.json",
         "hf_calvin_mapping.json",
         "task_D_D",
         47,
         ep_len,
         use_hf_parquet_goal,
         conditioning)
        for (s, e) in chunks
    ]
    list(eval_method_shard.starmap(args_list))
    print(f"[modal] run_method finished for {method_name}")


@app.local_entrypoint()
def run_shard(
    method_name: str,
    row_start: int,
    row_end: int,
    use_hf_parquet_goal: bool = False,
    ep_len: int = 360,
    run_tag: str = "",
    conditioning: str = "both",
):
    """Re-run a single shard for ``method_name`` rows [row_start, row_end)."""
    if not run_tag:
        run_tag = method_name
    print(f"[modal] run_shard method={method_name} rows=[{row_start}, {row_end})")
    eval_method_shard.remote(
        method_name, row_start, row_end, run_tag,
        "gr_mg/epoch=47-step=83712.ckpt",
        "policy/config/train.json",
        "hf_calvin_mapping.json",
        "task_D_D",
        47,
        ep_len,
        use_hf_parquet_goal,
        conditioning,
    )
    print(f"[modal] run_shard done: rows [{row_start}, {row_end})")


# ---------------------------------------------------------------------------
# CPU-only one-shot bootstrap helpers (run once per fresh account).
# ---------------------------------------------------------------------------

@app.function(volumes={RUNS_MOUNT: runs_vol}, timeout=60 * 5)
def upload_mapping_bytes(content: bytes, remote_name: str = "hf_calvin_mapping.json"):
    p = Path(RUNS_MOUNT) / remote_name
    p.write_bytes(content)
    runs_vol.commit()
    print(f"[modal] wrote {p} ({len(content)} B)")


@app.local_entrypoint()
def upload_mapping(local_path: str = ".runlogs/hf_calvin_mapping.json",
                   remote_name: str = "hf_calvin_mapping.json"):
    src = Path(local_path).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"local mapping not found: {src}")
    upload_mapping_bytes.remote(src.read_bytes(), remote_name)
    print(f"[modal] uploaded mapping -> rm-policy-runs:/{remote_name}")


# ---------------------------------------------------------------------------
# Chain-aware visualization.
# ---------------------------------------------------------------------------

@app.function(volumes=VOLUMES, gpu="T4", timeout=60 * 60 * 2,
              cpu=2, memory=16384)
def viz_chain_remote(
    method_name: str,
    skeleton_idx: int,
    start_in_chain: int,
    chain_length: int,
    use_hf_parquet_goal: bool = False,
    ep_len_per_step: int = 360,
    fps: int = 30,
    continue_on_failure: bool = True,
    ckpt_remote: str = "gr_mg/epoch=47-step=83712.ckpt",
    config_path: str = "policy/config/train.json",
    mapping_remote: str = "hf_calvin_mapping.json",
    chain_skeletons_remote: str = "chain_skeletons.json",
    split_name: str = "task_D_D",
    epoch: int = 47,
):
    """Render one chain-instance video on a Modal GPU; return the mp4 bytes."""
    _enter_workspace_with_calvin_conf()
    ckpt_full = Path(CKPT_MOUNT) / ckpt_remote
    ckpt_dir = str(ckpt_full.parent)

    out_path = "/tmp/viz_chain.mp4"
    if Path(out_path).exists():
        Path(out_path).unlink()

    cmd = [
        "python", "evaluate/visualize_chain.py",
        "--dataset_path", f"{DATA_MOUNT}/{split_name}",
        "--ckpt_dir", ckpt_dir,
        "--epoch", str(epoch),
        "--config_path", config_path,
        "--mapping_json", f"{RUNS_MOUNT}/{mapping_remote}",
        "--chain_skeletons", f"{RUNS_MOUNT}/{chain_skeletons_remote}",
        "--skeleton_idx", str(skeleton_idx),
        "--start_in_chain", str(start_in_chain),
        "--chain_length", str(chain_length),
        "--output", out_path,
        "--ep_len_per_step", str(ep_len_per_step),
        "--fps", str(fps),
    ]
    if continue_on_failure:
        cmd += ["--continue_on_failure"]
    if use_hf_parquet_goal:
        parquet_path = "/hf_cache/calvin_test.parquet"
        if not Path(parquet_path).exists():
            import urllib.request
            Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_HF_TEST_URL, parquet_path)
            hf_cache_vol.commit()
        cmd += ["--hf_parquet_goal", parquet_path]
    else:
        cmd += ["--method_dir", f"{EVAL_DATA_MOUNT}/{method_name}"]

    rc = _run(cmd, env={"MESA_GL_VERSION_OVERRIDE": "3.3"})
    if rc != 0 or not Path(out_path).exists():
        raise SystemExit(f"viz_chain failed rc={rc}")
    return Path(out_path).read_bytes()


@app.local_entrypoint()
def upload_chain_skeletons(
    local_path: str = ".runlogs/chain_skeletons.json",
    remote_name: str = "chain_skeletons.json",
):
    src = Path(local_path).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"local chain skeletons not found: {src}")
    upload_mapping_bytes.remote(src.read_bytes(), remote_name)
    print(f"[modal] uploaded chain skeletons -> rm-policy-runs:/{remote_name}")


@app.local_entrypoint()
def viz_chain(
    method_name: str,
    skeleton_idx: int,
    start_in_chain: int,
    chain_length: int,
    output: str,
    use_hf_parquet_goal: bool = False,
    ep_len_per_step: int = 360,
    fps: int = 30,
):
    """Local-entrypoint wrapper: render chain video remotely and save to ``output``."""
    out_local = Path(output).expanduser().resolve()
    out_local.parent.mkdir(parents=True, exist_ok=True)
    print(f"[modal] running chain visualisation: skel={skeleton_idx} "
          f"start={start_in_chain} length={chain_length} method={method_name}")
    mp4_bytes = viz_chain_remote.remote(
        method_name=method_name,
        skeleton_idx=skeleton_idx,
        start_in_chain=start_in_chain,
        chain_length=chain_length,
        use_hf_parquet_goal=use_hf_parquet_goal,
        ep_len_per_step=ep_len_per_step,
        fps=fps,
    )
    out_local.write_bytes(mp4_bytes)
    print(f"[modal] wrote {out_local} ({len(mp4_bytes)} B)")
