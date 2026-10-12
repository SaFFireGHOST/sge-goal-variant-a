# Goal Images Are Control Targets: Spatially-Grounded Goal-Image Synthesis for Robotic Manipulation

**Anonymous WACV Applications Track Submission**  
**Paper ID:** 1893  
**Target Venue:** IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2027)

---

## Abstract

Goal-conditioned manipulation policies are steered by a goal image, a picture of the desired post-manipulation scene synthesized from the current observation and a language instruction. The prevailing assumption is that better image-editing quality yields better goals, so goal generators are built and compared using general image-editing metrics. We show this assumption is false. The reality is much more nuanced. Across two independent goal-conditioned policies on CALVIN, the image editors that top aggregate editing-quality leaderboards (Gemini 3 Pro Image and GPT Image 1.5) produce worse control targets than our proposed far smaller finetuned open models, and the gap widens sharply for the multi-step (long-horizon) tasks robots actually face.

A per-dimension analysis explains why: downstream policy success is governed almost entirely by manipulation adherence (selecting the right object and the right final state; Spearman $\rho \approx 0.9$ with success), while photorealism and aesthetic fidelity, precisely where large proprietary editors excel, are uncorrelated with success ($\rho \approx 0$). This dissociation holds on both CALVIN and the real-world-aligned SIMPLER-Bridge benchmark.

We then present **SGE-Goal**, a spatially-grounded decomposed editing framework that predicts initial and final 2D edit-region masks before goal-image synthesis. SGE-Goal turns open base models (FLUX.1 Kontext, FLUX.2 klein) into the strongest goal generators we evaluate on CALVIN, and second only to Gemini 3 Pro Image on SIMPLER-Bridge. Base models fine-tuned with SGE-Goal also hold up under chaining far better than the best proprietary editor, sustaining roughly seven times the success rate at the longest (five-step) chains. We will release all code and model checkpoints, our automated data-generation pipeline, the processed datasets, and a 15-dimension human-aligned diagnostic evaluator. Our results argue that goal generators must be selected by downstream utility, not by editing-quality leaderboards.

---

> **Figure 1: Goal images are control targets, not artworks.**  
> General-purpose editors try optimizing for editing quality in general, but for robotic manipulation a goal image must primarily:  
> 1. Manipulate the correct object to the correct final state, and  
> 2. Avoid spurious changes that mislead the policy.  
> SGE-Goal achieves this through explicit spatial grounding, while strong general editors introduce unmentioned changes, miss the instruction, or hallucinate objects.

---

## 1. Introduction

Goal-conditioned policy learning is a powerful paradigm for robotic manipulation [3, 8, 11, 12, 33]. Instead of acting from language alone, the robot is given a **goal image**: a picture of the desired post-manipulation scene that turns an ambiguous instruction into an explicit, geometric target. Conditioning a policy on a visual goal together with language has been shown to yield substantially higher manipulation success than language alone [29]. Because the policy cannot observe the future, this goal image must be synthesized from the current observation and the instruction—an image-editing task. Goal-image quality is therefore decisive: a faithful goal gives the policy a clear target, while an unfaithful one misleads it from the first step.

A natural and now-standard methodology has emerged: generate goal images with a strong language-guided image editor, and compare candidate generators using image-editing quality metrics and benchmarks [1, 26, 34]. The implicit assumption is that a better quality edit is a better goal. Modern proprietary editors such as Gemini 3 Pro Image and GPT Image 1.5 produce strikingly photorealistic edits and dominate such benchmarks, suggesting that the goal-image synthesis problem is essentially solved by scale.

**This assumption is false.** The reality is more nuanced, and that is our starting point. We evaluate goal images not by how they look but by what they are for: the success of a downstream manipulation policy conditioned on them. Across two independent, pre-trained goal-conditioned policies (GR-MG [19] and GHIL-Glue [13]) on CALVIN [25], we find that the editors topping aggregate image-editing quality (Gemini 3 Pro Image and GPT Image 1.5) yield worse downstream policy success rates than our proposed far smaller finetuned open models, and the gap widens for the multi-step tasks robots actually perform (Figure 2). The two policies agree on the method ordering almost perfectly (Spearman $\rho = 0.96$), so this is not a quirk of one policy.

### Why does editing quality mislead?
A fine-grained, 15-dimension analysis of every goal image reveals a clean dissociation (Section 4.3.2). **Manipulation adherence**, selecting the correct object and reaching the correct final state, is the dominant predictor of policy success ($\rho \approx 0.9$), whereas **photorealism and aesthetic fidelity**, the dimensions that most impress human viewers and on which proprietary editors win, are essentially uncorrelated with it ($\rho \approx 0$). A goal image is a control signal, not an artwork; a policy rewards a target that points at the right object in the right place, free of spurious changes, not visual polish.

Guided by this finding, we present **SGE-Goal**, a spatially-grounded, decomposed editing framework designed to maximize adherence and suppress the spurious changes that mislead policies. Rather than mapping instruction-plus-image directly to an edited image, SGE-Goal first predicts where the manipulated objects are and where they will be—an initial and a final 2D edit-region mask—and then synthesizes the goal conditioned on both (Figure 1). This decomposition disentangles spatial reasoning from appearance synthesis and prevents the "shortcuts" (background drift, hallucination) that entangled end-to-end editors take. Crucially, SGE-Goal is a lightweight fine-tuning recipe: it turns ordinary open base models (FLUX.1 Kontext [2], FLUX.2 klein [4]) into the strongest goal generators we evaluate on CALVIN, while running fully on-premise.

To train SGE-Goal without manual labels, we introduce a fully automated data-generation pipeline that extracts spatially-grounded training tuples from raw robot videos using a chain of vision foundation models. To diagnose why methods differ, we build a 15-dimension, human-aligned evaluation framework (91.7% agreement with human judgment) whose granularity is what enables the dissociation analysis of Section 4.3.2. We stress that we do not propose this, or any, image-editing metric as a substitute for downstream evaluation; it is a diagnostic that explains downstream behavior.

---

> **Figure 2: Downstream policy success vs. chain length for two pre-trained policies on CALVIN.**  
> SGE-Goal goal images yield the best success rates at every chain length and retain performance as chain length increases, while the strongest general editors collapse: at chain length 5, SGE-Goal (FLUX.1 Kontext [Dev]) reaches 30.9% vs. Gemini 3 Pro’s 4.4% under GR-MG. Long-horizon manipulation is where goal-image utility matters most. (Full values tabulated in Table 2; base = untuned editor).

---

### Contributions
- **SGE-Goal Framework**: We propose SGE-Goal, a spatially-grounded decomposed editing framework that turns small open base models into the strongest goal generators we evaluate on CALVIN and second only to Gemini 3 Pro Image on the real-world-aligned SIMPLER-Bridge benchmark. Base models fine-tuned with SGE-Goal also hold up under chaining far better than the best proprietary editor.
- **Open Artifacts Release**: We release code and model checkpoints, an automated data-generation pipeline, the processed datasets, and a 15-dimension human-aligned diagnostic evaluator.
- **Visual Goals Beat Text Goals Long-Horizon**: Using our own closed-loop setup, we confirm that visual-goal conditioning beats text-only conditioning at every chain length (1–5), extending the single-step finding of MUTEX [29] to the long-horizon regime and motivating the pursuit of high-quality visual goals.
- **Image-Editing Quality is Misaligned with Utility**: We show empirically that image-editing quality is a misaligned proxy for goal-image utility in robotic manipulation. On both CALVIN and SIMPLER-Bridge, the editing-quality ranking of generators fails to match their downstream success-rate ranking, so goal generators must be selected by downstream success rather than by editing-quality leaderboards.
- **The Adherence/Photorealism Dissociation**: We identify and validate the underlying mechanism: an adherence/photorealism dissociation in which adherence predicts policy success and photorealism does not, and we show it holds consistently on both CALVIN and SIMPLER-Bridge.

---

## 2. Related Work

### Goal-image synthesis for manipulation
Visual goals provide an information-rich interface between language and low-level control [5, 6, 16, 32]. SuSIE [3] pioneered subgoal synthesis by fine-tuning InstructPix2Pix [7] on robot video, but predicts only near-future subgoals whose quality degrades rapidly when chained toward an end goal. UniPi [11] and Clover [8] generate goal/subgoal frames via text-conditioned video generation and closed-loop replanning, while Seer [33] uses predictive inverse-dynamics models; all are prone to hallucinated objects, confusion between similar objects, and incoherent or drifting predictions. LfVoid [12] applies pre-trained text-to-image models without task-specific fine-tuning and frequently departs from the original scene. A limitation common to these methods is the lack of explicit spatial control over where objects move and explicit consistency control over what should not change. SGE-Goal addresses both via mask-based spatial grounding.

### Goal-conditioned policies
Our evaluation conditions pre-trained policies on synthesized goals. GR-MG [19] is a multi-modal goal-conditioned policy leveraging partially-annotated data; GHIL-Glue [13] hierarchically filters subgoal images to suppress off-task drift; MUTEX [29] learns unified policies from multimodal task specifications and reports that visual goals improve success over text-only conditioning. We use GR-MG and GHIL-Glue as fixed downstream evaluators of goal-image utility and corroborate the visual-vs-text trend reported by MUTEX.

### Language-guided image editing
Instruction-based editing has advanced rapidly with diffusion models [15, 28]. InstructPix2Pix [7] and HIVE [39] follow natural-language edits; inversion methods such as Prompt-to-Prompt [14] manipulate cross-attention; localized methods such as LIME [30] restrict edits to a region of interest. These methods model the object’s initial location but not where it ends up after manipulation, making them ill-suited to displacement. Recent open editors (Qwen Image Edit [35], FLUX.1 Kontext [2], FLUX.2 klein [4], BAGEL [10]) and proprietary systems (Gemini 3 Pro Image, GPT Image 1.5) offer strong general-purpose editing; we use them both as base models for fine-tuning and as strong baselines, and show that their general strength does not translate into manipulation utility.

### Evaluating edits and the missing link to control
Classical metrics (PSNR, LPIPS [38], CLIP score [27]) give limited semantic insight, while recent benchmarks add fine-grained editing dimensions (I2I-Bench [34]), operation detection (DICE [1]), physical realism (PICABench [26]), or LLM judges (RISEBench [41]). Two gaps matter for robotics: none is curated for manipulation (they weight photorealism heavily—dimensions we show are uncorrelated with policy success), and none is validated against downstream control. We provide a manipulation-specific, human-aligned diagnostic and, more importantly, measure how each editing dimension relates to actual policy success, exposing the proxy gap.

---

> **Figure 3: Automated data-extraction pipeline.**  
> Each robot video is processed by a chain of vision foundation models to produce spatially-grounded training tuples $(I_o, I_e, M_i, M_f, T)$ without manual annotation.

---

## 3. Method

We describe three components:
1. A fully automated pipeline that extracts spatially-grounded training data from robot videos (Section 3.1).
2. SGE-Goal, a three-step decomposed editing framework trained on this data (Section 3.2).
3. A 15-dimension diagnostic evaluator (Section 3.3).

---

### 3.1. Automated Data Extraction

Each training sample is a tuple $(I_o, I_e, M_i, M_f, T)$:
- $I_o$: Original image (before manipulation).
- $I_e$: Edited image (after manipulation).
- $M_i$: Initial edit-region mask localizing the dynamic (to-be-manipulated) objects in $I_o$.
- $M_f$: Final edit-region mask marking their post-manipulation locations in $I_e$.
- $T$: Natural language instruction.

The masks encode the before/after spatial state of the manipulated objects. We extract these tuples from raw robot videos with no manual annotation, chaining six vision foundation models (Figure 3):
1. **Robotic-arm masking**: EVF-SAM [40] proposes per-frame arm masks, verified by BLIP-2 [18] VQA; a multi-pass strategy selects the first frame with consistent arm detection as $I_o$.
2. **Object segmentation & tracking**: SAM [17] produces class-agnostic masks on $I_o$, filtered and merged, then propagated through the video with XMem [9].
3. **Dynamic-object identification**: Per-object motion is quantified by LoFTR [31] feature matching (centroid distance as fallback); $K=2$ KMeans on these distances separates dynamic from stationary objects.
4. **Mask and edited-frame extraction**: $M_i$ is the union of arm and dynamic-object masks on $I_o$; the edited frame is detected when arm and dynamic-object masks cease to overlap (release), giving $I_e$ and the corresponding $M_f$.
5. **Instruction generation**: For datasets lacking text, $T$ is generated by a video VLM [21].

#### Composite vs. atomic manipulations
We distinguish composite instructions, which decompose into multiple necessary steps (e.g., *"open the drawer and put the apple in it"*), from atomic ones, which cannot be decomposed without introducing unnecessary (hallucinated) steps. SGE-Goal operates at the atomic level: composite instructions are decomposed into their necessary-and-sufficient atomic constituents, each processed in sequence with the generated goal $\hat{I}_e$ becoming the next step’s $I_o$. Extraction is performed at atomic granularity.

---

> **Figure 4: SGE-Goal synthesizes the goal image in three steps with explicit 2D spatial grounding via initial mask $M_i$ and final mask $M_f$.**

---

### 3.2. Decomposed Goal-Image Synthesis

#### What we mean by "spatial grounding"
SGE-Goal grounds editing in the 2D image plane via binary edit-region masks; it does not estimate 3D geometry, depth, or camera pose. The masks act as an explicit spatial prior that is absent in end-to-end editors, specifying where objects are and where they should go.

Given $(I_o, T)$, the pipeline sequentially predicts $M_i$, then $M_f$, then the goal image $\hat{I}_e$ (Figure 4):

#### Step 1: Initial mask $M_i$ (Training-Free)
We localize dynamic target objects in $I_o$ with an ensemble of pre-trained models without fine-tuning. The instruction is parsed for target-object names; candidate bounding boxes come from Florence-2 [36] and Grounding DINO [23] with progressively relaxed thresholds and per-object NMS. Each candidate is verified by BLIP-2 [18] VQA, and surviving boxes prompt SAM [17] to produce pixel-level $M_i$.

#### Step 2: Final mask $M_f$
We fine-tune a base image-to-image model with LoRA to predict where dynamic objects reside after manipulation, conditioned on $(I_o, T, M_i)$. Inputs are VAE-encoded; the original-image and $M_i$ latents are clean conditioning, while the target $M_f$ latent is noised by the rectified-flow schedule:
$$M_f^t = (1 - \sigma)M_f^0 + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

The model predicts velocity via the rectified-flow loss [24]:
$$\mathcal{L}_{\text{flow}} = \mathbb{E}_{\sigma, \epsilon} \left\| v_\theta(M_f^t, \sigma) - (\epsilon - M_f^0) \right\|^2 \tag{1}$$

From velocity we recover the one-step clean latent estimate: $\hat{M}_f^0 = \epsilon - v_\theta(M_f^t, \sigma)$. Since $\hat{M}_f$ is binary (identical RGB channels), an auxiliary channel-consistency loss is applied when $\sigma < \tau_{\text{rec}}$:
$$\mathcal{L}_{\text{chan}} = \frac{1}{HW} \sum_{h,w} \sum_{c < c'} \left( \hat{M}_{f,c} - \hat{M}_{f,c'} \right)^2 \tag{2}$$
summing over channel pairs $c < c'$ from $\{R, G, B\}$. The total mask objective is:
$$\mathcal{L}_{\text{mask}} = \mathcal{L}_{\text{flow}} + \lambda_{\text{chan}} \mathbf{1}[\sigma < \tau_{\text{rec}}] \mathcal{L}_{\text{chan}}$$
with $\lambda_{\text{chan}} = 25$, $\tau_{\text{rec}} = 0.35$.

#### Step 3: Goal image $\hat{I}_e$
A second LoRA-fine-tuned model synthesizes the goal from $(I_o, T, M_i, M_f)$. The original, $M_i$, and $M_f$ latents are clean conditioning; the target $I_e$ latent is noised and trained with $\mathcal{L}_{\text{flow}}$ (Eq. 1). To preserve static regions, an auxiliary **static-scene-consistency loss (SSCL)** penalizes deviations outside the union mask $M_\cup = \max(M_i, M_f)$ (gated by $\sigma < \tau_{\text{rec}}$):
$$\mathcal{L}_{\text{static}} = \frac{1}{CHW} \left\| (\hat{I}_e^{\text{img}} - I_o^{\text{img}}) \odot (1 - \bar{M}_\cup) \right\|^2 \tag{3}$$
where superscript $\text{img}$ denotes pixel-space VAE-decoded images, and $\bar{M}_\cup$ is the channel-mean of decoded $M_\cup$. The edit objective is:
$$\mathcal{L}_{\text{edit}} = \mathcal{L}_{\text{flow}} + \lambda_{\text{static}} \mathbf{1}[\sigma < \tau_{\text{rec}}] \mathcal{L}_{\text{static}}$$
with $\lambda_{\text{static}} = 7$, $\tau_{\text{rec}} = 0.35$.

#### Why decomposition helps
Decomposition breaks one entangled task (generating $I_e$ directly from $I_o, T$) into disentangled sub-tasks: foundation-model grounding (Step 1), pure spatial reasoning (Step 2), and appearance synthesis given spatial guidance (Step 3). End-to-end editors must learn spatial reasoning and visual synthesis jointly; lacking explicit guidance, they take shortcuts—editing in place or drifting the background. SGE-Goal concentrates quality on manipulation-relevant axes: adherence and the absence of spurious changes.

---

### 3.3. A 15-Dimension Diagnostic Evaluator

To understand why goal generators differ, we use an automated evaluator scoring each triplet $(I_o, I_e, T)$ across 15 manipulation-specific dimensions:
- **Adherence**: R14 (Source Adherence), R15 (Destination Adherence).
- **Object Preservation**: R1–R4 (Stationary, moved, removed, hallucinated objects).
- **Photorealism**: R13.
- **Scene Artifacts**: R5–R12 (Color/texture/shape consistency, lighting, camera shift).

Evaluations use open models: Video VLM [21], SAM, XMem, BLIP-2, LoFTR, Grounding DINO, and Qwen2.5 [37]. We aggregate dimensions via a relevance-tiered weighted average of per-method ranks:
- **Adherence**: 60% (30% each for R14 and R15).
- **Object Preservation**: 24%.
- **Photorealism**: 8%.
- **Scene Artifacts**: 8%.

#### Table 1: Agreement with Human Judgment (%, $\uparrow$ higher is better)
| Metric | Adherence (R14–15) | Object Pres. (R1–4) | Scene Art. (R5–12) | Realism (R13) | Overall Avg. (R1–15) | I2I-Bench Baseline [34] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Alignment (%)** | **85.5%** | **90.3%** | **92.8%** | **100.0%** | **91.7%** | 71.2% |

Across 278 human-annotated triplets, our evaluator attains **91.7% agreement with human judgment**, well above I2I-Bench (71.2%).

---

## 4. Experiments

### 4.1. Setup

#### Datasets
- **BridgeData V2 (teleoperated)**: ~41k train, 2k test tuples.
- **CALVIN-ABCD [25]**: ~22k train, 1.1k test tuples.
- **LIBERO-10/90 [22]**: ~6.5k train, 376 test tuples.

#### Base Models & Baselines
- **Proposed**: SGE-Goal fine-tuned on FLUX.1 Kontext [Dev] [2] and FLUX.2 klein [4].
- **Untuned base models**: FLUX.1 Kontext (base), FLUX.2 klein (base).
- **Open editing baseline**: Qwen Image Edit [35].
- **Robotics-specific baselines**: SuSIE [3], Clover [8].
- **Proprietary editors**: GPT Image 1.5, Gemini 3 Pro Image (Nano Banana Pro).

#### Downstream Protocol (Primary Ground Truth)
We condition two pre-trained frozen policies, **GR-MG** [19] and **GHIL-Glue** [13], on each generator's goal images on CALVIN and record success rate (SR%) vs. chain length (CL 1–5). Holding policies fixed isolates goal-image utility. An oracle policy is evaluated with ground-truth goal images ($I_e^{\text{GT}}$ at CL=1).

#### Real-World-Aligned Benchmark
We evaluate on **SIMPLER-Bridge** [20], a WidowX/BridgeData-V2 simulator correlating strongly with real-robot performance across >1,500 real/sim rollouts. We test on 4 atomic WidowX tasks (24 episodes each) driving frozen GHIL-Glue.

---

> **Figure 5: Training data extraction quality.**  
> Human quality ratings (1–5 scale) for $I_o$, $I_e$, $M_i$, $M_f$ across the three datasets (50 annotators, 500 tuples/dataset). Ratings concentrate at the top (lowest mean: 4.44/5).

---

### 4.2. Downstream Policy Success

#### Table 2: Downstream Policy Success (SR%, $\uparrow$) on CALVIN Across Chain Lengths (1–5)
Evaluated with two frozen policies: GR-MG [19] and GHIL-Glue [13]. Bold indicates best non-oracle performance.

| Method | GR-MG CL1 | GR-MG CL2 | GR-MG CL3 | GR-MG CL4 | GR-MG CL5 | GHIL-Glue CL1 | GHIL-Glue CL2 | GHIL-Glue CL3 | GHIL-Glue CL4 | GHIL-Glue CL5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *GT goal (oracle)* | *73.6* | *—* | *—* | *—* | *—* | *39.0* | *—* | *—* | *—* | *—* |
| **SGE-Goal (FLUX.1 Kontext)** | **73.1** | **59.7** | **47.3** | **36.8** | **30.9** | **29.2** | **12.0** | **3.7** | **0.8** | 0.0 |
| **SGE-Goal (FLUX.2 klein)** | 71.8 | 53.8 | 34.8 | 21.4 | 17.3 | 22.8 | 8.1 | 1.5 | 0.1 | 0.0 |
| Gemini 3 Pro Image | 63.3 | 36.9 | 18.1 | 8.1 | 4.4 | 13.2 | 2.4 | 0.0 | 0.0 | 0.0 |
| Clover | 63.2 | 34.2 | 12.8 | 3.6 | 1.9 | 17.3 | 3.2 | 0.2 | 0.0 | 0.0 |
| GPT Image 1.5 | 60.2 | 36.7 | 17.1 | 6.2 | 3.4 | 8.0 | 0.7 | 0.0 | 0.0 | 0.0 |
| Qwen Image Edit | 59.4 | 33.7 | 13.4 | 4.6 | 2.2 | 7.2 | 1.1 | 0.0 | 0.0 | 0.0 |
| SuSIE | 55.8 | 28.4 | 9.9 | 2.6 | 1.0 | 6.1 | 0.1 | 0.0 | 0.0 | 0.0 |
| FLUX.2 klein (base) | 52.0 | 29.6 | 13.4 | 5.2 | 2.9 | 2.9 | 0.0 | 0.0 | 0.0 | 0.0 |
| FLUX.1 Kontext (base) | 50.3 | 28.6 | 12.8 | 4.8 | 2.6 | 2.6 | 0.0 | 0.0 | 0.0 | 0.0 |

**Key Findings:**
1. **Oracle Parity**: At CL=1 under GR-MG, SGE-Goal (73.1%) nearly matches the Ground-Truth Oracle (73.6%) and outperforms its untuned base by +22.8 points.
2. **Superiority over Proprietary APIs**: SGE-Goal outperforms Gemini 3 Pro Image (63.3%) by +9.8% and GPT Image 1.5 (60.2%) by +12.9%.
3. **Cross-Policy Consensus**: GR-MG and GHIL-Glue rank the models near-identically (Spearman rank correlation $\rho = 0.96$).
4. **7× Advantage at Long Horizons**: At chain length 5 under GR-MG, SGE-Goal maintains **30.9%** success, while Gemini 3 Pro collapses to **4.4%** and SuSIE falls to **1.0%** (a **7× advantage**).

---

#### Table 3: Downstream Policy Success on SIMPLER-Bridge (SR%, $\uparrow$)
Four atomic WidowX manipulation tasks under the frozen GHIL-Glue policy (24 episodes each).

| Method | Spoon | Carrot | Stack | Eggplant | Mean SR% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Gemini 3 Pro Image | 16.7 | 29.2 | 0.0 | 4.2 | **12.5%** |
| **SGE-Goal (FLUX.2 klein)** | 16.7 | 16.7 | 0.0 | 8.3 | **10.4%** |
| **SGE-Goal (FLUX.1 Kontext)** | 8.3 | 20.8 | 0.0 | 4.2 | **8.3%** |
| SuSIE | 16.7 | 8.3 | 4.2 | 0.0 | 7.3% |
| Qwen Image Edit | 8.3 | 4.2 | 0.0 | 0.0 | 3.1% |
| GPT Image 1.5 | 4.2 | 0.0 | 0.0 | 4.2 | 2.1% |
| FLUX.2 klein (base) | 4.2 | 4.2 | 0.0 | 0.0 | 2.1% |
| FLUX.1 Kontext (base) | 4.2 | 4.2 | 0.0 | 0.0 | 2.1% |
| Clover | 4.2 | 0.0 | 0.0 | 0.0 | 1.0% |

*Note*: SGE-Goal is the strongest open model and outperforms SuSIE (7.3%) despite GHIL-Glue being the low-level policy natively trained for SuSIE subgoals.

---

#### Table 4: Visual Goals Beat Text Goals at Every Chain Length
GR-MG policy success rate (SR%, $\uparrow$) comparing Image+Text (I+T) goal conditioning vs. Text-Only (T) conditioning.

| Conditioning Mode | CL 1 | CL 2 | CL 3 | CL 4 | CL 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Image + Text (I+T)** | **73.1%** | **59.7%** | **47.3%** | **36.8%** | **30.9%** |
| Text Only (T) | 59.0% | 37.8% | 13.1% | 2.0% | 0.2% |

Visual conditioning extends single-step MUTEX findings [29] across multi-step horizon rollouts.

---

> **Figure 6: Chained multi-step prediction by SGE-Goal.**  
> The generated goal at each step becomes the input for the next. Scene structure is preserved across all five steps and only the instructed objects move, preventing error drift.

---

### 4.3. Image Editing Evaluation Results

#### Table 5: Runtime Profile (Seconds, $\downarrow$ lower is better)
Measured on a single NVIDIA A10G GPU for inference and NVIDIA RTX A6000 for offline extraction.

| Inference Component (A10G) | Time (s) | Offline Data Generation Stage (A6000) | Time (s) |
| :--- | :---: | :--- | :---: |
| Initial Mask ($M_i$) | 2.47 s | EVF-SAM + BLIP-2 | 4.46 s |
| Final Mask ($M_f$) | 5.16 s | SAM (first frame) | 1.07 s |
| Goal Image ($\hat{I}_e$) | 2.92 s | XMem Tracking | 7.76 s |
| **Total Inference** | **10.55 s** | LoFTR + KMeans Clustering | 8.74 s |

---

#### Table 6: Image-Editing Quality on BridgeData V2 (Br), CALVIN (Ca), and LIBERO (Li)
Comparing our human-aligned evaluator (weighted rank, $\downarrow$ lower is better) and I2I-Bench (weighted score, $\uparrow$ higher is better).

| Method | Our Eval Br ($\downarrow$) | Our Eval Ca ($\downarrow$) | Our Eval Li ($\downarrow$) | I2I-Bench Br ($\uparrow$) | I2I-Bench Ca ($\uparrow$) | I2I-Bench Li ($\uparrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Gemini 3 Pro Image | **2.07** | **2.65** | **1.81** | 0.785 | 0.707 | 0.675 |
| GPT Image 1.5 | 5.15 | 5.09 | 3.04 | **0.801** | **0.716** | **0.774** |
| Qwen Image Edit | 4.14 | 4.17 | 2.06 | 0.646 | 0.642 | 0.680 |
| **SGE-Goal (FLUX.2 klein)** | 2.99 | 3.41 | 4.98 | 0.655 | 0.626 | 0.484 |
| **SGE-Goal (FLUX.1 Kontext)** | 3.81 | 3.18 | 5.54 | 0.649 | 0.616 | 0.479 |
| Clover | — | 5.48 | — | — | 0.606 | — |
| SuSIE | 6.16 | 7.86 | 5.39 | 0.486 | 0.407 | 0.377 |
| FLUX.1 Kontext (base) | 5.48 | 7.33 | 6.51 | 0.533 | 0.449 | 0.378 |
| FLUX.2 klein (base) | 6.20 | 5.83 | 6.66 | 0.568 | 0.470 | 0.324 |

---

### 4.3.2. Why Editing Quality Misleads: The Adherence/Photorealism Dissociation

#### Table 7: Strength of Correlation with Policy Success (Absolute Spearman $|\rho|$)
Correlation between editing dimensions and downstream policy success across models.

| Evaluation Dimension | CALVIN: GR-MG $|\rho|$ | CALVIN: GHIL-Glue $|\rho|$ | SIMPLER-Bridge $|\rho|$ |
| :--- | :---: | :---: | :---: |
| **Source Adherence (R14)** | **0.92** | **0.89** | **0.85** |
| **Destination Adherence (R15)** | **0.90** | **0.79** | **0.66** |
| **Photorealism (R13)** | 0.25 | 0.07 | 0.02 |

- **Adherence Governs Control**: Source and destination adherence correlate strongly ($|\rho| \approx 0.79 - 0.92$) with policy success. Permutation testing confirms high statistical significance: $p < 10^{-3}$ for source adherence and $p = 0.001$ for destination adherence.
- **Photorealism is Irrelevant**: Photorealism correlation is indistinguishable from chance ($\rho = 0.03, p = 0.95$ on CALVIN; $|\rho| = 0.02$ on SIMPLER-Bridge). Proprietary editors excel at photorealism, which inflates their leaderboard scores without providing control utility.

---

> **Figure 7: Qualitative goal-image comparison.**  
> SGE-Goal moves the target object to the correct destination while strictly preserving unmentioned background objects. General-purpose proprietary editors (Gemini 3 Pro, GPT Image 1.5) render photorealistic surfaces but frequently move the wrong objects or alter unrelated scene regions.

---

### 4.4. Ablations

#### Table 8: Ablation by Downstream Success Rate (SR%, $\uparrow$)
Ablation of spatial mask conditioning and static scene consistency loss $\mathcal{L}_{\text{static}}$ across CALVIN (chain lengths 1–5) and SIMPLER-Bridge.

| Model Variant | GR-MG CL1 | GR-MG CL2 | GR-MG CL3 | GR-MG CL4 | GR-MG CL5 | GHIL CL1 | GHIL CL2 | GHIL CL3 | GHIL CL4 | GHIL CL5 | SIMPLER-Bridge |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FLUX.1 Kontext** | | | | | | | | | | | |
| **SGE-Goal (full)** | **73.1** | **59.7** | **47.3** | **36.8** | **30.9** | **29.2** | **12.0** | **3.7** | **0.8** | **0.0** | **8.3%** |
| w/o Final Mask ($M_f$) | 60.1 | 28.4 | 12.9 | 5.7 | 3.5 | 13.8 | 0.3 | 0.0 | 0.0 | 0.0 | 5.2% |
| w/o Masks ($M_i, M_f$) | 50.7 | 21.3 | 7.5 | 2.8 | 1.1 | 7.3 | 0.0 | 0.0 | 0.0 | 0.0 | 5.2% |
| Base Editor (untuned) | 50.3 | 28.6 | 12.8 | 4.8 | 2.6 | 2.6 | 0.0 | 0.0 | 0.0 | 0.0 | 2.1% |
| **FLUX.2 klein** | | | | | | | | | | | |
| **SGE-Goal (full)** | **71.8** | **53.8** | **34.8** | **21.4** | **17.3** | **22.8** | **8.1** | **1.5** | **0.1** | **0.0** | **10.4%** |
| w/o Final Mask ($M_f$) | 63.5 | 35.0 | 15.2 | 5.7 | 2.9 | 13.0 | 1.5 | 0.0 | 0.0 | 0.0 | 7.3% |
| w/o Masks ($M_i, M_f$) | 60.0 | 32.1 | 12.5 | 4.1 | 1.9 | 12.6 | 1.2 | 0.0 | 0.0 | 0.0 | 2.1% |
| Base Editor (untuned) | 52.0 | 29.6 | 13.4 | 5.2 | 2.9 | 2.9 | 0.0 | 0.0 | 0.0 | 0.0 | 2.1% |

**Ablation Takeaways:**
1. **Dominance of Final Mask $M_f$**: Supplying the destination prior $M_f$ is the single largest performance booster, lifting CL=1 success by +13.0 points on GR-MG (60.1% $\rightarrow$ 73.1%) and +15.4 points on GHIL-Glue (13.8% $\rightarrow$ 29.2%).
2. **Unguided Fine-Tuning Degrades Base**: Fine-tuning without masks (`w/o Masks`) often falls below the untuned base at longer chain lengths (e.g. 1.1% vs 2.6% at CL=5), indicating catastrophic forgetting when spatial constraints are omitted.
3. **Decomposition Prevents Long-Horizon Collapse**: At CL=5, only the full model preserves strong performance (30.9% and 17.3%), whereas all ablated variants collapse below 4%.

---

## 5. Discussion and Conclusion

### Practical Recommendation
For deploying goal-conditioned policies, practitioners should not default to proprietary visual editing APIs. A small open base model fine-tuned with SGE-Goal (run locally on an A10G in ~10.6 s) produces vastly superior control targets, with an immense advantage on multi-step chained manipulations. Goal generators must be evaluated and selected by downstream manipulation success.

### Limitations
1. **2D Grounding Scope**: SGE-Goal operates strictly in the 2D image plane without explicit 3D geometry or camera pose estimation; large viewpoint changes or heavy self-occlusions remain challenging.
2. **Local Consistency Tradeoffs**: By executing genuine, large displacements, SGE-Goal can slightly perturb adjacent pixels; its goal is task adherence and suppressing policy-critical errors, not minimal change.
3. **Downstream Protocol**: Evaluated on CALVIN and SIMPLER-Bridge with frozen policies; physical robot hardware rollouts and co-training policy networks with SGE-Goal are future milestones.

### Conclusion
Goal images are control targets, not artworks. Aggregate image-editing benchmarks misrepresent goal-image utility because policies care about manipulation adherence and stationary object preservation, not photorealism. SGE-Goal introduces explicit 2D spatial grounding via initial and final edit-region masks and static consistency regularization, transforming standard open diffusion editors into state-of-the-art robotic goal generators.

---

## References

- **[1]** L. Baraldi, D. Bucciarelli, F. Betti, M. Cornia, L. Baraldi, N. Sebe, and R. Cucchiara. *What changed? Detecting and evaluating instruction-guided image edits with multimodal large language models.* arXiv:2505.20405, 2025.
- **[2]** S. Batifol, A. Blattmann, F. Boesel, S. Consul, C. Diagne, T. Dockhorn, J. English, Z. English, P. Esser, S. Kulal, et al. *FLUX.1 Kontext: Flow matching for in-context image generation and editing in latent space.* arXiv:2506.xxxxx, 2025.
- **[3]** K. Black, M. Nakamoto, P. Atreya, H. Walke, C. Finn, A. Kumar, and S. Levine. *Zero-shot robotic manipulation with pretrained image-editing diffusion models.* arXiv:2310.10639, 2023.
- **[4]** Black Forest Labs. *FLUX.2: Frontier visual intelligence.* https://bfl.ai, 2025.
- **[5]** A. Brohan, N. Brown, J. Carbajal, et al. *RT-1: Robotics Transformer for real-world control at scale.* arXiv:2212.06817, 2022.
- **[6]** A. Brohan, N. Brown, J. Carbajal, et al. *RT-2: Vision-language-action models transfer web knowledge to robotic control.* arXiv:2307.15818, 2023.
- **[7]** T. Brooks, A. Holynski, and A. A. Efros. *InstructPix2Pix: Learning to follow image editing instructions.* In CVPR, pages 18392–18402, 2023.
- **[8]** Q. Bu, J. Zeng, L. Chen, Y. Yang, G. Zhou, J. Yan, P. Luo, H. Cui, Y. Ma, and H. Li. *Closed-loop visuomotor control with generative expectation for robotic manipulation.* In NeurIPS, 37:139002–139029, 2024.
- **[9]** H. K. Cheng and A. G. Schwing. *XMem: Long-term video object segmentation with an Atkinson-Shiffrin memory model.* In ECCV, pages 640–658. Springer, 2022.
- **[10]** C. Deng, D. Zhu, K. Li, C. Gou, F. Li, Z. Wang, S. Zhong, W. Yu, X. Nie, Z. Song, et al. *Emerging properties in unified multimodal pretraining.* arXiv:2505.14683, 2025.
- **[11]** Y. Du, M. Yang, B. Dai, H. Dai, O. Nachum, J. B. Tenenbaum, D. Schuurmans, and P. Abbeel. *Learning universal policies via text-guided video generation.* In NeurIPS, 36, 2023.
- **[12]** J. Gao, K. Hu, G. Xu, and H. Xu. *Can pre-trained text-to-image models generate visual goals for reinforcement learning?* arXiv:2307.07837, 2023.
- **[13]** K. B. Hatch, A. Balakrishna, O. Mees, S. Nair, S. Park, B. Wulfe, M. Itkina, B. Eysenbach, S. Levine, T. Kollar, and B. Burchfiel. *GHIL-Glue: Hierarchical control with filtered subgoal images.* In ICRA, 2025.
- **[14]** A. Hertz, R. Mokady, J. Tenenbaum, K. Aberman, Y. Pritch, and D. Cohen-Or. *Prompt-to-Prompt image editing with cross attention control.* arXiv:2208.01626, 2022.
- **[15]** J. Ho, A. Jain, and P. Abbeel. *Denoising diffusion probabilistic models.* In NeurIPS, 33:6840–6851, 2020.
- **[16]** M. J. Kim, et al. *OpenVLA: An open-source vision-language-action model.* arXiv:2406.09246, 2024.
- **[17]** A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, and R. Girshick. *Segment Anything.* In ICCV, pages 4015–4026, 2023.
- **[18]** J. Li, D. Li, S. Savarese, and S. Hoi. *BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models.* In ICML, pages 19730–19742. PMLR, 2023.
- **[19]** P. Li, H. Wu, Y. Huang, C. Cheang, L. Wang, and T. Kong. *GR-MG: Leveraging partially-annotated data via multi-modal goal-conditioned policy.* IEEE Robotics and Automation Letters (RA-L), 2025.
- **[20]** X. Li, K. Hsu, J. Gu, K. Pertsch, O. Mees, H. R. Walke, C. Fu, I. Lunawat, I. Sieh, S. Kirmani, S. Levine, J. Wu, C. Finn, H. Su, Q. Vuong, and T. Xiao. *Evaluating real-world robot manipulation policies in simulation.* In CoRL, 2024.
- **[21]** X. Li, Y. Wang, J. Yu, X. Zeng, Y. Zhu, H. Huang, J. Gao, K. Li, Y. He, C. Wang, et al. *VideoChat-Flash: Hierarchical compression for long-context video modeling.* arXiv:2501.00574, 2024.
- **[22]** B. Liu, et al. *LIBERO: Benchmarking knowledge transfer for lifelong robot learning.* In NeurIPS, 36, 2024.
- **[23]** S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su, et al. *Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection.* In ECCV, pages 38–55. Springer, 2024.
- **[24]** X. Liu, C. Gong, and Q. Liu. *Flow straight and fast: Learning to generate and transfer data with rectified flow.* arXiv:2209.03003, 2022.
- **[25]** O. Mees, L. Hermann, E. Rosete-Beas, and W. Burgard. *CALVIN: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks.* IEEE Robotics and Automation Letters (RA-L), 7(3):7327–7334, 2022.
- **[26]** Y. Pu, L. Zhuo, S. Han, J. Xing, K. Zhu, S. Cao, B. Fu, S. Liu, H. Li, Y. Qiao, et al. *PicaBench: How far are we from physically realistic image editing?* arXiv:2510.17681, 2025.
- **[27]** A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. *Learning transferable visual models from natural language supervision.* In ICML, pages 8748–8763. PMLR, 2021.
- **[28]** R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. *High-resolution image synthesis with latent diffusion models.* In CVPR, pages 10684–10695, 2022.
- **[29]** R. Shah, R. Martín-Martín, and Y. Zhu. *MUTEX: Learning unified policies from multimodal task specifications.* In CoRL, 2023.
- **[30]** E. Simsar, A. Tonioni, Y. Xian, T. Hofmann, and F. Tombari. *LIME: Localized image editing via attention regularization in diffusion models.* In WACV, pages 222–231. IEEE, 2025.
- **[31]** J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou. *LoFTR: Detector-free local feature matching with transformers.* In CVPR, pages 8922–8931, 2021.
- **[32]** Octo Model Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, et al. *Octo: An open-source generalist robot policy.* arXiv:2405.12213, 2024.
- **[33]** Y. Tian, S. Yang, J. Zeng, P. Wang, D. Lin, H. Dong, and J. Pang. *Predictive inverse dynamics models are scalable learners for robotic manipulation.* arXiv:2412.15109, 2024.
- **[34]** J. Wang, J. Wang, H. Duan, J. Kang, G. Zhai, and X. Min. *I2I-Bench: A comprehensive benchmark suite for image-to-image editing models.* arXiv:2512.04660, 2025.
- **[35]** C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S.-M. Yin, S. Bai, X. Xu, Y. Chen, et al. *Qwen-Image technical report.* arXiv:2508.02324, 2025.
- **[36]** B. Xiao, H. Wu, W. Xu, X. Dai, H. Hu, Y. Lu, M. Zeng, C. Liu, and L. Yuan. *Florence-2: Advancing a unified representation for a variety of vision tasks.* In CVPR, pages 4818–4829, 2024.
- **[37]** A. Yang, B. Yang, B. Zhang, B. Hui, B. Wang, B. Zheng, C. Yu, D. Lv, F. Du, H. Huang, et al. *Qwen2.5 technical report.* arXiv:2412.15115, 2024.
- **[38]** R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. *The unreasonable effectiveness of deep features as a perceptual metric.* In CVPR, pages 586–595, 2018.
- **[39]** S. Zhang, X. Yang, Y. Feng, C. Qin, C.-C. Chen, N. Yu, Z. Chen, H. Wang, S. Savarese, S. Ermon, et al. *HIVE: Harnessing human feedback for instructional visual editing.* In CVPR, pages 9026–9036, 2024.
- **[40]** Y. Zhang, T. Cheng, L. Zhu, R. Hu, L. Liu, H. Liu, L. Ran, X. Chen, W. Liu, and X. Wang. *EVF-SAM: Early vision-language fusion for text-prompted segment anything model.* arXiv:2406.20076, 2024.
- **[41]** X. Zhao, P. Zhang, K. Tang, X. Zhu, H. Li, W. Chai, Z. Zhang, et al. *Envisioning beyond the pixels: Benchmarking reasoning-informed visual editing.* arXiv:2504.02826, 2025.
