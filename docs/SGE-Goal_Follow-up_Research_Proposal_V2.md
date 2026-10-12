# Policy-Aligned Training of Spatially Grounded Goal-Image Synthesis for Robotic Manipulation

**Research Proposal • Follow-up to SGE-Goal (WACV 2027)**  
**Institution:** IIIT Bangalore  
**Date:** October 2026  
**Target Venue:** ICCV 2027 (ICORE2026 A\*) • *Fallback:* NeurIPS 2027 (A\*)

---

## At a Glance

- **Objective:** Train SGE-Goal’s goal-image generator with feedback from the manipulation policy that executes its goals, and establish how generator and policy should adapt to each other, keeping SGE-Goal’s principle of one final goal per atomic manipulation step.
- **Design:** A $2 \times 2$ study over which module is trained (§1.3):
  - **Variant A:** Trains the generator with the policy frozen.
  - **Variant C:** Trains the policy with the generator frozen.
  - **Variant B:** Trains both in alternation.  
  Task success reaches the generator only through which of its samples it is trained on, how strongly, and which of two goals it prefers, so SGE-Goal’s rectified-flow and consistency losses stay unchanged. A dense, differentiable policy signal reuses the one-step image estimate that SGE-Goal already computes, and credit is assigned per stage: where the object goes (the final mask) separately from how the goal looks. Policies learn by imitation and self-imitation; online reinforcement learning is optional.
- **Scope:** The work does not propose a new state-of-the-art manipulation system. It studies how an explicit, inspectable goal image and the policy that executes it should adapt to each other, in matched comparisons that hold policy architectures and demonstration data fixed, and it measures what explicit goals provide beyond raw success (§1.4). It also tests SGE-Goal’s principle itself: one final goal per atomic step against subgoals within the step, with everything else matched (§5.6).
- **Competitors:** 
  1. The untrained SGE-Goal (baseline to beat).
  2. Prior methods of learning from feedback (success filtering, judged-quality rewards, test-time selection, gradients through the plan, policy-only adaptation, online reinforcement learning, variance-routed credit) re-implemented in the same setting.
  3. Intermediate subgoals within each step, both as published and in a matched comparison.  
  *State-of-the-art systems are reported for context only (§5.7).*
- **Expected Contributions:**
  1. Goal-image synthesis trained directly for the success of the policy that executes the goals (frozen in Variant A or co-trained in Variant B).
  2. Stage-wise credit assignment enabled by SGE-Goal’s decomposition, crediting success separately to where the object goes ($M_f$) versus how the scene looks ($\hat{I}_e$).
  3. A measured faithfulness–utility trade-off.
  4. A controlled comparison of generator-side, policy-side, and joint adaptation, and of reward sources (task success, edit adherence, photorealism).
  5. Measured benefits of explicit goals beyond raw success: reuse across unadapted policies and pre-execution human/VLM inspection and correction.
  6. A matched test of one final goal per atomic step against intermediate subgoals.
- **Novelty:** Screening all SGE-Goal references, follow-up works, and literature up to September 2026 (§2) reveals no method that trains an explicit, spatially grounded goal image from the task success of the executing policy while keeping the goal a faithful visual end-state and measuring the impact on faithfulness.
- **Plan and Resources:** Four phases over ~20 weeks with three decision gates (§6). Target venue: ICCV 2027 (March 2027 deadline). Compute: shared lab 4× RTX A6000 (48 GB) and 1× RTX Pro 6000 (96 GB), estimated at 1.0–1.4k A6000-hours for Phases 0–2 (§8).

---

## 1. Introduction

### 1.1. SGE-Goal Recap
SGE-Goal [1] synthesizes goal images for goal-conditioned manipulation policies. It works on atomic steps—single instructions such as *"open the drawer"* that cannot be split further without adding unnecessary steps—and produces **one final goal per atomic step**: an image of the scene at the end of that step. 

Given a source image $I_o$ and step instruction $T$:
- **Step 1:** Localizes manipulated objects with pre-trained grounding and segmentation models without training, producing the initial edit-region mask $M_i$.
- **Step 2:** A rectified-flow image editor fine-tuned with Low-Rank Adaptation (LoRA) predicts the final edit-region mask $M_f$ (where dynamic objects end up).
- **Step 3:** A second LoRA-tuned editor synthesizes the goal image $\hat{I}_e$ conditioned on $(I_o, T, M_i, M_f)$.

The base editors are FLUX.1 Kontext (main) and FLUX.2 klein. Steps 2 and 3 are trained on automatically extracted tuples $(I_o, I_e, M_i, M_f, T)$, where $I_e$ is the ground-truth (GT) final frame. Step 2 uses a channel-consistency loss, and Step 3 uses a static-scene-consistency loss outside $M_\cup = \max(M_i, M_f)$.

Downstream, two pre-trained frozen goal-conditioned policies—**GR-MG** [2] and **GHIL-Glue** [3]—execute goals on CALVIN [4]. With both policies frozen, SGE-Goal achieved the highest success among all evaluated generators (including Gemini 3 Pro Image and GPT Image 1.5) across all chain lengths (CL 1–5). Policy success tracked adherence (Spearman $\rho = 0.98$ for source, $0.92$ for destination) but was uncorrelated with photorealism ($\rho = 0.03$).

---

### 1.2. Motivation
SGE-Goal showed that goal generators should be selected by downstream policy success; this proposal **trains them for it**. Five key observations define the problem:
1. **Long Horizons Hold the Headroom:** Under GR-MG, SGE-Goal success drops from 73.1% at CL 1 to 30.9% at CL 5; under GHIL-Glue, it reaches at most 3.7% from CL 3 onward [1].
2. **Policies Were Not Trained for One Goal per Step:** Both evaluated policies were trained for multi-subgoal conditioning (following intermediate frames within a step). Driven by SGE-Goal's single end-goal, they reached average lengths of ~2.5, whereas with native subgoal loops they report 4.04 (GR-MG) and 3.69 (GHIL-Glue).
3. **A Frozen Policy Bounds Goal Gains:** With a frozen GR-MG, SGE-Goal at CL 1 (73.1%) already approaches the GT oracle (73.6%). Further gains require co-adapting the policy.
4. **Reward-Driven Training Can Destroy Faithfulness:** Optimizing goals purely for policy actions risks reward hacking—producing uninterpretable policy-specific adversarial codes rather than faithful visual scenes.
5. **Foresight Tokens vs. Inspectable Goals:** Black-box end-to-end models cannot be inspected by humans before execution, cannot be corrected, and cannot transfer across different robot embodiments.

---

### 1.3. Proposed Study ($2 \times 2$ Design)

The study crosses two factors: whether the generator is trained, and whether the policy is trained:

#### Table 1: Training configurations of the $2 \times 2$ study
| | **Policy Frozen** | **Policy Trained** |
| :--- | :--- | :--- |
| **Generator Frozen** | **Ref-Pub / Ref-IM:** Baselines using frozen SGE-Goal [1]. | **Variant C (Policy Adaptation):** Policy adapts to SGE-Goal via imitation & self-imitation. |
| **Generator Trained** | **Variant A (Generator Alignment):** SGE-Goal trained via dense behavior matching and task success. | **Variant B (Co-Training):** Alternating updates of generator (Variant A) and policy (Variant C). |

- **$\pi_{\text{IM}}$ (Interface-Matched Policy):** A policy checkpoint fine-tuned on demonstration data paired with single GT end goals rather than multi-subgoals. $\pi_{\text{IM}}$ serves as the fair reference baseline across all conditions.

---

### 1.4. Scope of the Claims
The study does not claim to present an undisputed new state-of-the-art robot system. The claims rest on:
1. **Matched Comparisons:** Holding policy architecture, base editors, and training data fixed across all conditions.
2. **Within-Family References:** Comparing against native multi-subgoal loops of the same policies on identical test chains.
3. **Benefits Beyond Raw Success:** Measuring goal inspectability, human/VLM error correction before rollout, and cross-policy transfer.
4. **Granularity Test:** Formally evaluating one final goal per atomic step against intermediate subgoals under identical compute budgets.

---

## 2. Related Work & Positioning

#### Table 2: Positioning against the closest related literature
| Approach | Visual Target | Training Setup | Use of Task Success | Faithfulness of Target |
| :--- | :--- | :--- | :--- | :--- |
| **Seer [5], PAD [11], GR-1 [12]** | Foresight tokens or future frames | Joint imitation / action-generation | None | None (internal representations) |
| **ImageWAM [13] (2026)** | Editor features; goal never decoded | Editor branch + action expert | None | No image produced |
| **SuSIE [8], GR-MG [2], SWEET [9] (2026)** | (Sub)goal images or keyframes | Separate or joint imitation | None | Implicit (grounded in real frames) |
| **GHIL-Glue [3], TaKSIE [10], CLOVER [6]** | Subgoals or video plans | Separate | Progress/distance estimates select subgoals or trigger replanning | Implicit |
| **VideoAgent [7]** | Video plans | Generator refined | VLM feedback selects plans; successes fine-tune generator | Implicit + VLM checks |
| **RoboTALES [14] (2026)** | Video futures | Generator refined | VLM-critic rewards for generator | Implicit (reconstruction) |
| **WAM-RL [15] (2026)** | World-model predictions | Actor-critic RL on world model | Environment interaction; reconstruction rewards | Implicit (reconstruction) |
| **ORCHID [16]** | Subgoals from small planner from scratch | Co-adaptation | Success-filtered rollouts train both | Implicit |
| **DARS [20] (2026)** | Edited image from text plan | RL on vision-language planner + diffusion renderer | Judged editing rewards; variance-routed learning (no robot) | Rewarded as judged edit quality |
| **This Proposal** | **End-of-step goal image with 2D masks from pre-trained editor** | **Generator alignment (A), policy adaptation (C), or joint co-training (B)** | **Dense policy-based loss; stage-wise preferences & weights; self-imitation** | **Enforced (masks, static loss) & verified (F1–F6)** |

---

## 3. Objectives, Research Questions, and Hypotheses

#### Table 3: Research questions and hypotheses
| Question | Description | Formal Hypothesis |
| :--- | :--- | :--- |
| **RQ1 (Variant A)** | Can training only the generator raise a frozen policy's success? | **H1:** Yes, where goals for the same context differ in success. Gains grow with chain length and policy dependence on goals (higher on GHIL-Glue than GR-MG). Dense loss refines appearance; success feedback trains destination masks ($M_f$). |
| **RQ2 (Variant C)** | How much does adapting only the policy recover? | **H2:** Variant C exceeds Ref-IM at all chain lengths as the policy learns SGE-Goal's goal distribution and residual artifacts. CL-1 success exceeds public checkpoints and narrows the gap to native subgoals. |
| **RQ3 (Variant B)** | Does joint adaptation beat one-sided variants? | **H3:** Variant B exceeds the better of A and C, particularly at long horizons (CL 3–5). |
| **RQ4 (Faithfulness)** | Do goals stay faithful under success-driven training? | **H4:** Unconstrained end-to-end backprop causes goals to drift into policy-specific adversarial codes. SGE-Goal's masks and static loss preserve visual fidelity while capturing most success gains. |
| **RQ5 (Credit)** | Does SGE-Goal's decomposition improve sample efficiency? | **H5:** Crediting final masks ($M_f$) and appearance ($\hat{I}_e$) separately aligns the generator in fewer rollouts than whole-goal credit or variance-routed credit. |
| **RQ6 (Reward Source)** | Which reward aligns the generator best? | **H6:** Success and adherence rewards both improve policy performance; photorealism rewards (R13) improve aesthetic scores but produce zero downstream gain. |
| **RQ7 (Explicit Goals)** | What do explicit goals provide beyond raw success? | **H7:** Trained goals transfer to unseen policies; pre-execution automated VLM/human inspection catches failing goals prior to execution. |
| **RQ8 (Granularity)** | Is one final goal per step non-inferior to subgoals? | **H8:** Matched for compute and training, one final goal per step is non-inferior (within 0.15 average length) at significantly lower generation cost. |

---

## 4. Proposed Method

### 4.1. Mathematical Formulation & SGE-Goal Objectives
Let context $c = (I_o, T)$ define an atomic step. SGE-Goal generator $G_\theta$ maps $c \to (M_i, M_f, \hat{I}_e)$ with trainable LoRA parameters $\theta = (\theta_2, \theta_3)$. 

A goal-conditioned policy $\pi_\psi$ receives observation $o_t$, goal $g$, and $T$, outputting action $a_t \sim \pi_\psi(\cdot \mid o_t, g, T)$. A rollout $\tau = (o_1, a_1, \dots, o_H, a_H)$ produces binary task success $R(\tau) \in \{0, 1\}$.

The expected success objective is:
$$J(\theta, \psi) = \mathbb{E}_{c} \, \mathbb{E}_{\hat{I}_e \sim G_\theta(c)} \, \mathbb{E}_{\tau \sim \pi_\psi(\cdot \mid \hat{I}_e, T)} \left[ R(\tau) \right] \tag{1}$$

For rectified-flow training with latent target $x$, noise $\epsilon \sim \mathcal{N}(0, I)$, and noise scale $\sigma \in [0, 1]$, noisy latent $x_\sigma = (1 - \sigma)x + \sigma \epsilon$, the per-sample flow matching loss is:
$$\ell_\theta(x; c) = \left\| v_\theta(x_\sigma, \sigma; c) - (\epsilon - x) \right\|^2 \tag{2}$$

SGE-Goal maintains anchoring losses on dataset tuples:
- **Step 2:** $\mathcal{L}_{\text{mask}} = \mathcal{L}_{\text{flow}} + \lambda_{\text{chan}} \mathbf{1}[\sigma < \tau_{\text{rec}}] \mathcal{L}_{\text{chan}}$ ($\lambda_{\text{chan}} = 25$).
- **Step 3:** $\mathcal{L}_{\text{edit}} = \mathcal{L}_{\text{flow}} + \lambda_{\text{static}} \mathbf{1}[\sigma < \tau_{\text{rec}}] \mathcal{L}_{\text{static}}$ ($\lambda_{\text{static}} = 7$).

---

### 4.2. Dense Policy Signal: Behavior Matching
Reusing the clean one-step decoded image estimate $\hat{I}_e^{\text{img}}$ from $\mathcal{L}_{\text{static}}$, we feed it directly into policy $\pi_\psi$:
$$\mathcal{L}_{\text{beh}} = \mathbb{E}_{t \in \mathcal{S}, \, \xi \sim \Xi} \left[ D_\pi\left( o_t, \xi(\hat{I}_e^{\text{img}}), \xi(I_e) \right) \right] \tag{3}$$
- $\mathcal{S}$: Demonstration time steps.
- $\xi \sim \Xi$: Differentiable augmentations (color jitter, mild blur, affine translation) preventing adversarial texture exploits.
- $D_\pi$: Action discrepancy (MSE for regression action heads; noise-prediction error for diffusion policies).

---

### 4.3. Task Success as a Training Signal
Since binary success $R(\tau)$ is non-differentiable, success guides flow-matching updates through three channels:

1. **Filtering:** Samples with success rate $\hat{p} \ge \eta$ become supervised targets.
2. **Reward-Weighted Regression (Flow-RWR):**
   $$\mathcal{L}_{\text{RWR}} = \mathbb{E}_{(c, x) \sim \mathcal{B}} \left[ w(c, x) \, \ell_\theta(x; c) \right], \quad w = \min\left( \exp(A / \beta_w), w_{\max} \right) \tag{4}$$
3. **Flow-DPO (Direct Preference Optimization):**
   On pairs of goals $(x_w, x_l)$ where $\hat{p}(x_w) - \hat{p}(x_l) \ge \delta$:
   $$\mathcal{L}_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma_{\text{sigmoid}}\left( -\beta \left( \Delta_\theta(x_w) - \Delta_\theta(x_l) \right) \right) \right] \tag{5}$$
   $$\Delta_\theta(x) = \ell_\theta(x; c) - \ell_{\text{ref}}(x; c)$$

#### Combined Stage Objectives
$$\mathcal{L}_2 = \mathcal{L}_{\text{mask}} + \lambda_{\text{succ}} \mathcal{L}_{\text{succ}} \tag{6}$$
$$\mathcal{L}_3 = \mathcal{L}_{\text{edit}} + \lambda_{\text{beh}} \mathbf{1}[\sigma < \tau_{\text{rec}}] \mathcal{L}_{\text{beh}} + \lambda_{\text{succ}} \mathcal{L}_{\text{succ}} \tag{7}$$

---

### 4.4. Stage-Wise Credit Assignment
Goals are generated in two stages: Step 2 decides **where objects go** ($M_f$), and Step 3 decides **how the scene looks** ($\hat{I}_e$). When one goal succeeds and another fails, we isolate responsibility using **factorial sampling**:
- Sample $K_m$ destination masks $M_f$.
- For each mask, sample $K_a$ appearances $\hat{I}_e$.
- Execute each with $K_s$ policy seeds ($K_m \times K_a \times K_s$ total rollouts per context).

Let $U(M_f)$ be the average success of mask $M_f$, and $b(c)$ the context baseline. The advantage decomposes exactly:
$$A(x; c) = \underbrace{U(M_f) - b(c)}_{A_2(M_f; c) \text{ [Step 2 Mask Advantage]}} + \underbrace{\hat{p}(x; c) - U(M_f)}_{A_3(x; c) \text{ [Step 3 Appearance Advantage]}} \tag{8}$$

---

### 4.5. Faithfulness Safeguards & Diagnostics (F1–F6)
To verify that goals do not degenerate into uninterpretable policy hacks, goals are tracked every round:
- **F1 (Adherence):** R14/R15 evaluated by held-out judges (DICE [35] or independent VLM).
- **F2 (Spurious Changes):** Preservation of non-target objects (R1–R12).
- **F3 (Oracle Transfer):** Success rate when fed into a policy trained only on GT frames.
- **F4 (Cross-Policy Transfer):** Success when evaluated on an independent, unadapted policy.
- **F5 (Perturbation Fragility):** Retention of success under mild JPEG compression, Gaussian blur, and noise.
- **F6 (Residual Action Probe):** Verifying that demonstration actions cannot be linearly decoded from unedited background pixels.

---

### 4.6. Training Procedures
- **Variant A (Generator Alignment):**
  1. *Stage 1 (Dense):* Optimize Step 3 with $\mathcal{L}_{\text{beh}}$ on dataset tuples (cheap, no rollouts) $\to$ **A-dense**.
  2. *Stage 2 (Success):* 3 offline rounds collecting factorially sampled rollouts, computing stage-wise advantages, and updating Steps 2 and 3 $\to$ **A-full**.
- **Variant C (Policy Adaptation):**  
  Generator frozen. Policy $\pi_\psi$ fine-tunes on demonstration tuples paired with a mixture $q$ of GT frames, augmented frames, and SGE-Goal outputs, plus self-imitation on successful rollouts:
  $$\mathcal{L}_\pi = \mathbb{E}_{\mathcal{D}_{\text{demo}}, g \sim q} \left[ \ell_{\text{BC}}(\pi_\psi(o_t, g, T), a_t) \right] + \lambda_{\text{self}} \mathbb{E}_{\mathcal{D}_{\text{succ}}} \left[ \ell_{\text{BC}}(\pi_\psi(o_t, g, T), a_t) \right] \tag{9}$$
- **Variant B (Co-Training):**  
  Alternating rounds:
  1. *Collect:* Factorial rollouts with $(G_r, \pi_r)$.
  2. *Generator Phase:* Update generator against an exponential moving average (EMA) copy $\bar{\pi}_r$.
  3. *Policy Phase:* Update policy with $\mathcal{L}_\pi$ using fresh samples from $G_{r+1}$.

---

## 5. Experimental Design

### 5.1. Experimental Conditions

#### Table 4: Complete experimental conditions
| Condition | Generator | Policy | Training Signals |
| :--- | :--- | :--- | :--- |
| **Ref-Pub$^\star$** | SGE-Goal (frozen) | Public checkpoint | None (reproduces WACV baseline) |
| **Ref-IM$^\star$** | SGE-Goal (frozen) | $\pi_{\text{IM}}$ (frozen) | None (fair baseline for matched interface) |
| **Native$^\star$** | Policy's native subgoal generator | Public checkpoint | Native multi-subgoal loop |
| **A-dense** | SGE-Goal trained with $\mathcal{L}_{\text{beh}}$ | $\pi_{\text{IM}}$ (frozen) | Dense behavior matching only |
| **A-full$^\star$** | A-dense + 3 success rounds | $\pi_{\text{IM}}$ (frozen) | Dense + stage-wise success (DPO/RWR) |
| **A-adh** | A-dense + 3 reward rounds | $\pi_{\text{IM}}$ (frozen) | Dense + adherence reward (R14/R15) |
| **A-photo** | A-dense + 3 reward rounds | $\pi_{\text{IM}}$ (frozen) | Dense + photorealism reward (R13) |
| **A-select** | SGE-Goal (frozen), best of $N$ | $\pi_{\text{IM}}$ (frozen) | None; test-time utility model selection |
| **C-IL** | SGE-Goal (frozen) | $\pi_{\text{IM}}$ trained on mixture $q$ | Imitation learning on goal mixture |
| **C-full$^\star$** | SGE-Goal (frozen) | C-IL + self-imitation | Imitation + self-imitation on successes |
| **B-full$^\star$** | Co-trained | Co-trained | Alternating generator & policy updates |
| **B-filter** | Co-trained | Co-trained | Success-filtered rollouts only |
| **E2E-naive$^\star$** | Mask-free editor, no $\mathcal{L}_{\text{static}}$ | Co-trained | End-to-end backprop through policy |
| **E2E-SGE** | SGE-Goal | Co-trained | End-to-end backprop with SGE-Goal losses |

---

### 5.2. Granularity Study: One Final Goal vs. Intermediate Subgoals

#### Table 5: Control variables in the granularity study
| Dimension | Native Comparison Flaw | Granularity Study Control |
| :--- | :--- | :--- |
| **Policy** | Different checkpoints trained differently | Same policy backbone ($\pi_{\text{IM}}$ vs. $\pi_{\text{SG}}$) |
| **Generator** | Different base architectures | Same base editor (FLUX.1 Kontext) |
| **Anchoring** | Chained from previous goals vs. re-observed | Both tested with execution-time re-anchoring |
| **Compute** | 1 generation/step vs. multiple subgoals/step | Test-time compute matched using best-of-$N$ selection (**Gr-select**) |

#### Table 6: Granularity outcomes and paper framing
| Empirical Outcome | Meaning | Framing of Paper |
| :--- | :--- | :--- |
| **Gr-final $\ge$ Gr-sub** | One final goal per atomic step suffices at lower compute | One final goal recommended as default interface |
| **Gr-SGE $\ge$ Gr-sub, Gr-final $<$ Gr-sub** | Final goals succeed when spatially grounded via masks | SGE-Goal's spatial grounding validated |
| **Both inferior; GT-final $\approx$ GT-sub** | Gap is due to generation quality, not interface limits | Goal training (H1–H3) directly targets this gap |
| **Both inferior; GT-final $\ll$ GT-sub** | Subgoals provide motion trajectory cues absent in end-states | Trade-off quantified: success vs. generation cost & transfer |

---

### 5.3. Competitor Analysis & Win Criteria

#### Table 7: Competitors and definition of winning
| Competitor Category | Target Baseline | Definition of Winning |
| :--- | :--- | :--- |
| **Untrained SGE-Goal** | Ref-Pub and Ref-IM | $\ge +3$ points at CL 1 or $+0.15$ average length over Ref-IM with 95% CI $> 0$ and F1–F5 within tolerance. |
| **Success Filtering (ORCHID [16], VideoAgent [7])** | B-filter | B-full outperforms B-filter (H3). |
| **Judged-Quality Rewards (RoboTALES [14])** | A-adh, A-photo | A-full achieves higher task success; photorealism adds no gain (H6). |
| **Test-Time Selection (CLOVER [6], GHIL-Glue [3])** | A-select (best of $N$) | A-full matches or exceeds A-select while generating 1 goal instead of $N$. |
| **Gradient Through Plan (Seer [5], ImageWAM [13])** | E2E-naive | E2E-SGE and B-full retain performance while preserving faithfulness F1–F5 (H4). |
| **Variance-Routed Credit (DARS [20])** | Variance-routed baseline | Stage-wise credit reaches equal success with fewer rollouts (H5). |
| **Subgoals within Step (Native Loops [2, 3])** | Native published loops | Matched comparison shows non-inferiority at lower latency (H8). |

---

## 6. Work Plan & Decision Gates

#### Table 8: Timeline and decision gates (20 Weeks Total)
| Phase | Weeks | Scope & Key Milestones | Decision Gate |
| :--- | :---: | :--- | :--- |
| **P0: Foundations** | 1–3 | Evaluation harness, common random seeds, $\pi_{\text{IM}}$ checkpoints, Ref-Pub & Ref-IM baselines. | **Gate 0 (Headroom):** Cross-fitted gain of best goal $\ge 5$ points and $\ge 30\%$ contexts have mixed outcomes. |
| **P1: One-Sided Adaptation** | 4–8 | Variant A (A-dense, A-full, A-adh, A-photo), Variant C (C-IL, C-full), Native subgoal baselines. | **Gate 1:** A-full or C-full reaches Ref-IM $+3$ points at CL 1 or $+0.15$ avg length with F1–F5 preserved. |
| **P2: Co-Training** | 9–15 | Variant B (B-full, B-filter), E2E ablations, granularity comparison (RQ8), transfer & cost analysis. | **Gate 2:** B-full exceeds better of A-full and C-full by $\ge 0.15$ avg length with non-overlapping CIs. |
| **P3: Consolidation** | 16–20 | SIMPLER-Bridge rollouts, optional real robot, paper writing, code/model artifact cleanup. | **Submission:** ICCV 2027 (Weeks 21–22). |

#### Table 9: Work packages and person-week allocation (~43–56 Person-Weeks)
| Work Package | Deliverable | Phase | Effort |
| :--- | :--- | :---: | :---: |
| **WP1: Infrastructure** | Evaluation harness, parallel rollout pipeline, experiment registry | P0–P1 | 5–6 pw |
| **WP2: Policy Adaptation** | $\pi_{\text{IM}}$ checkpoints, C-IL, C-full | P0–P1 | 6–8 pw |
| **WP3: Generator Alignment** | $\mathcal{L}_{\text{beh}}$, DPO/RWR stage-wise credit, A-dense, A-full | P1 | 8–10 pw |
| **WP4: Co-Training** | B-full, B-filter, E2E-naive, E2E-SGE | P2 | 6–8 pw |
| **WP5: Faithfulness & Analysis** | F1–F6 metrics, transfer matrix, variance decomposition | P1–P3 | 5–7 pw |
| **WP6: External Evaluation** | SIMPLER-Bridge, real-robot verification | P3 | 4–6 pw |
| **WP7: Writing & Release** | Manuscript preparation, code and checkpoint releases | P3 | 4 pw |
| **WP8: Granularity Study** | Horizon editor, Gr-final, Gr-sub, Gr-SGE, Gr-select | P1–P2 | 5–7 pw |

---

## 7. Risks and Mitigations

#### Table 10: Risk management matrix
| # | Risk Description | Root Cause | Mitigation Strategy | Early Warning Signal |
| :---: | :--- | :--- | :--- | :--- |
| **1** | Binary non-differentiable success | Cannot backprop through simulator | Advantage weighting (DPO/RWR), $K_s$ seeds, dense $\mathcal{L}_{\text{beh}}$ | Pair labels disagree across seeds in $>30\%$ pairs |
| **2** | No intra-context variance | All sampled goals succeed or fail | Skip uninformative contexts, chain-length curriculum | $>60\%$ contexts all-success or all-failure |
| **3** | Misattributed failures | Policy execution noise | Within-context comparisons, common random seeds | Advantage signs flip across seed halves |
| **4** | Frozen policy bound | Variant A bounded by policy capacity | $\pi_{\text{IM}}$ baseline; Variants B & C co-adapt policy | Ref-IM already near GT ceiling |
| **5** | Zero signal at CL $\ge 3$ | Policy collapse on long chains | Interface matching, per-step reward labels, CL curriculum | CL 3 success $< 5\%$ at Phase 0 |
| **6** | Optimistic headroom | Overfitting noisy rollout gains | Cross-fitting at Gate 0 with split seeds | Selection gain $\gg$ held-out gain |
| **7** | VAE/denoising gradient break | 20 sampling steps in computation graph | One-step clean estimate $\hat{I}_e^{\text{img}}$ used in $\mathcal{L}_{\text{beh}}$ | $\mathcal{L}_{\text{beh}}$ decreases but rollout success stalls |
| **8** | Co-training instability | Moving target dynamics | Alternating updates, EMA policy copy $\bar{\pi}$, lower policy LR | Round-to-round swings exceed CI |
| **9** | Loss term imbalance | Multiple loss terms on different scales | Per-term gradient norm balancing on validation split | One gradient norm $>10\times$ others |
| **10** | Catastrophic forgetting | Fine-tuning degrades pre-trained prior | LoRA tuning, DPO reference penalty, replay dataset tuples | Validation editing scores drop below Round 0 |
| **11** | VRAM exhaustion on $\mathcal{L}_{\text{beh}}$ | Joint editor + policy graph | Run on 96 GB RTX Pro 6000, gradient checkpointing, bf16 | OOM or step latency $>2\times$ baseline |
| **12** | Policy-specific adversarial codes | Policy exploits visual generator artifacts | Enforce $M_i, M_f$, static loss $\mathcal{L}_{\text{static}}$, augmentations $\Xi$ | F5 fragility spikes or F4 cross-policy drop |
| **13** | Policy lock-in | Policy loses ability to follow GT goals | Include GT frames in training goal mixture $q$ | Policy fails on GT goals ($F_3$ drops) |
| **14** | Goal-action mismatch | Generated goal contradicts demonstrated action | Filter generated goals with mask IoU threshold $\kappa$ | Imitation loss on generated goals $\gg$ on GT |
| **15** | Goal diversity collapse | DPO collapses to single conservative goal | Regularize DPO $\beta$, track per-context mask diversity | Distance between sampled masks $M_f$ drops |
| **16** | Interface confound | Gains solely reflect learning 1-goal interface | Compare against $\pi_{\text{IM}}$ (trained on 1 goal) | B-full $\approx$ C-full |
| **17** | Reward hacking | Overfitting to diagnostic evaluator | Use held-out judges (F1) and human evaluations | Evaluator diverges from physical success |
| **18** | Test-set contamination | Leakage of evaluation contexts | Strictly isolate train/val/test splits | Gate and test results diverge |
| **19** | Low statistical power | Gains obscured by seed variance | Paired bootstrap tests, 3 seeds on all test chains | CI wider than effect size |
| **20** | Gap to SOTA systems | Comparing against 4+ step SOTA | Frame as controlled scientific study, not SOTA claim | Matched gains small relative to native gap |
| **21** | Weak re-implementations | Baselines underperforming originals | Grant equal tuning budget, frame as principles | Gain far smaller than published baseline |
| **22** | Granularity loss | Subgoals fundamentally beat final goals | Decoupled framing (Table 6), highlight cost & transfer | GT-sub $\gg$ GT-final at Gate 1 |
| **23** | Concurrent literature | Rapidly evolving research field | Continuous arXiv monitoring, precise contrasts | New paper training goals on task success |
| **24** | Evaluation compute blowout | Factorial sampling latency | Cache Step 1 and chained goals, reduced denoising steps | GPU-hours per round exceed budget |
| **25** | Lab compute contention | Shared GPU cluster availability | Agreed quotas, job checkpointing, run gate tests first | Queue delays $>2$ days |
| **26** | Framework incompatibility | PyTorch vs. JAX policy codebases | Bridge PyTorch policy tensors in Phase 0 | Unresolved framework bugs in Week 2 |
| **27** | Sim-to-Real gap | Simulation alignment fails on real robot | SIMPLER visual matching, optional real-world DPO | Real-robot success fails to track sim |

---

## 8. Resources & Compute Budget

The laboratory provides:
- **4× NVIDIA RTX A6000 (48 GB VRAM)**: Dedicated to parallel rollouts, policy fine-tuning, DPO/RWR updates, and FLUX.2 klein experiments.
- **1× NVIDIA RTX Pro 6000 (96 GB VRAM)**: Dedicated to combined computational graphs ($\mathcal{L}_{\text{beh}}$ and end-to-end backprop).

#### Table 11: Compute budget breakdown (A6000-Equivalent Hours)
| Item | Workload Basis | Compute Estimate |
| :--- | :--- | :---: |
| **Unit Cost: Goal Generation** | ~4.5 s/goal with Step 1 cached (3 masks $\times$ 2 appearances) | — |
| **Unit Cost: CALVIN Rollout** | 5–10 s per step rollout including policy inference | — |
| **Phase 0: Baseline & Gate 0** | Policy fine-tuning; 300 contexts $\times$ 8 goals $\times$ 4 seeds | 30–70 h |
| **Variant A** | Dense stage + 3 rounds (1k contexts $\times$ 6 goals $\times$ 2 seeds) | 75–120 h |
| **Reward Variants** | A-adh and A-photo (3 rounds each with evaluator scoring) | 90–135 h |
| **Variant C** | Goal synthesis for demonstrations + 2 self-imitation rounds | 70–100 h |
| **Variant B & B-filter** | 3 rounds each of alternating generator and policy updates | 160–235 h |
| **End-to-End Ablations** | Joint training with shared gradients on RTX Pro 6000 | 60–100 h |
| **Evaluation Runs** | 13 conditions across all test chains + re-anchoring ablation | 280–330 h |
| **Native Subgoal Baselines** | Public checkpoints with native subgoal generators | 40–60 h |
| **Explicit Goal Analyses** | Inspection and regeneration for 3 conditions | 25–40 h |
| **Granularity Study** | $\pi_{\text{SG}}$, horizon editor, Gr-final, Gr-sub, Gr-SGE, Gr-select | 150–240 h |
| **Total (Phases 0–2)** | | **0.98k – 1.43k h** |
| **Cluster Capacity** | 6.5 A6000-equivalents at 40% share & 60% utilization | **~260 h / week** (~4–6 weeks total) |

---

## 9. Dissemination Strategy

1. **Target Conference: ICCV 2027 (ICORE2026 A\*)**
   - *Estimated Deadlines:* Abstract: March 1, 2027; Full Paper: March 8, 2027.
   - *Rationale:* Focuses on visual goal synthesis, spatial grounding, and faithfulness; matches 20-week schedule.
   - *Fallback:* **NeurIPS 2027 (A\*)** (May 2027 deadline) or **IROS 2027 (A)**.
2. **Follow-Up Journal Article:**
   - Consolidates SGE-Goal (WACV 2027) and this policy-aligned training study into an extended IEEE TPAMI or IJCV submission following conference review.
3. **Open-Source Release:**
   - Model checkpoints, training code, rollout harness, and benchmark logs released upon publication.

---

## 10. Approval & Phase-0 Checklist

### Items for Immediate Approval
1. **Scope:** $2 \times 2$ factorial study (Variants A, B, C), granularity study (§5.6), and explicit-goal benefits (RQ7).
2. **Schedule:** 20-week roadmap with decision Gates 0, 1, and 2.
3. **Target Venue:** ICCV 2027 with NeurIPS 2027 fallback.
4. **Compute Allocation:** Lab cluster commitment of ~1.0–1.4k A6000-hours across Phases 0–2.

### Phase-0 Resolution Checklist (Weeks 1–3)
1. **Policy Codebase Integration:** Confirm GR-MG and GHIL-Glue fine-tuning pipelines and differentiable PyTorch action heads.
2. **Harness & Common Random Seeds:** Validate parallel CALVIN rollout engine with deterministic seed pairing.
3. **Gate 0 Execution:** Measure cross-fitted headroom on 300 validation contexts to confirm mixed-outcome density before launching success rounds.

---

## References

- **[1]** Anonymous. *Goal images are control targets: Spatially-grounded goal-image synthesis for robotic manipulation.* WACV 2027 Submission (Paper ID 1893).
- **[2]** P. Li, H. Wu, Y. Huang, C. Cheang, L. Wang, and T. Kong. *GR-MG: Leveraging partially-annotated data via multi-modal goal-conditioned policy.* IEEE RA-L, 2025.
- **[3]** K. B. Hatch, A. Balakrishna, O. Mees, S. Nair, S. Park, B. Wulfe, M. Itkina, B. Eysenbach, S. Levine, T. Kollar, and B. Burchfiel. *GHIL-Glue: Hierarchical control with filtered subgoal images.* In ICRA, 2025.
- **[4]** O. Mees, L. Hermann, E. Rosete-Beas, and W. Burgard. *CALVIN: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks.* IEEE RA-L, 2022.
- **[5]** Y. Tian, S. Yang, J. Zeng, P. Wang, D. Lin, H. Dong, and J. Pang. *Predictive inverse dynamics models are scalable learners for robotic manipulation.* In ICLR, 2025.
- **[6]** Q. Bu, J. Zeng, L. Chen, Y. Yang, G. Zhou, J. Yan, P. Luo, H. Cui, Y. Ma, and H. Li. *Closed-loop visuomotor control with generative expectation for robotic manipulation.* In NeurIPS, 2024.
- **[7]** A. Soni, et al. *VideoAgent: Self-improving video generation for embodied planning.* In NeurIPS, 2025.
- **[8]** K. Black, M. Nakamoto, P. Atreya, H. Walke, C. Finn, A. Kumar, and S. Levine. *Zero-shot robotic manipulation with pretrained image-editing diffusion models.* In ICLR, 2024.
- **[9]** Y. Song, et al. *SWEET: Sparse world modeling with image editing for embodied task execution.* arXiv:2605.19319, 2026.
- **[10]** X. Kang and Y.-L. Kuo. *Incorporating task progress knowledge for subgoal generation in robotic manipulation through image edits.* In WACV, 2025.
- **[11]** Y. Guo, et al. *Prediction with action: Visual policy learning via joint denoising process.* In NeurIPS, 2024.
- **[12]** H. Wu, et al. *Unleashing large-scale video generative pre-training for visual robot manipulation.* In ICLR, 2024.
- **[13]** Y. Zhang, et al. *ImageWAM: Do world action models really need video generation, or just image editing?* arXiv:2606.19531, 2026.
- **[14]** H. Gani, et al. *RoboTALES: Learning reasoning-guided robot policies via task-aligned simulated futures.* arXiv:2607.06018, 2026.
- **[15]** Z. Qian, et al. *WAM-RL: World-action model reinforcement learning with reconstruction rewards and online video SFT.* arXiv:2606.17906, 2026.
- **[16]** C. Grislain, et al. *Online self-training for co-adaptation in hierarchical diffusion policies.* arXiv:2603.05291, 2026.
- **[17]** Y. Qiao, et al. *Achieve what you imagined: Learning to align actions with visual plans.* arXiv:2609.33832, 2026.
- **[18]** S. Zhang, et al. *HIVE: Harnessing human feedback for instructional visual editing.* In CVPR, 2024.
- **[19]** H. Guo, et al. *Leveraging verifier-based reinforcement learning in image editing.* arXiv:2604.27505, 2026.
- **[20]** H. Cao, et al. *DARS: Dual-level credit assignment RL with structured reasoning for instruction-based image editing.* arXiv:2608.20161, 2026.
- **[21]** K. Frans, L. B. Soros, and O. Witkowski. *CLIPDraw: Exploring text-to-drawing synthesis through language-image encoders.* In NeurIPS, 2022.
- **[22]** A. Athalye, et al. *Synthesizing robust adversarial examples.* In ICML, 2018.
- **[23]** J. Choi, et al. *Perception prioritized training of diffusion models.* In CVPR, 2022.
- **[24]** K. Clark, et al. *Directly fine-tuning diffusion models on differentiable rewards.* In ICLR, 2024.
- **[25]** J. Liu, et al. *Flow-GRPO: Training flow matching models via online RL.* In NeurIPS, 2025.
- **[26]** H. Dong, et al. *RAFT: Reward ranked finetuning for generative foundation model alignment.* TMLR, 2023.
- **[27]** C. Gulcehre, et al. *Reinforced self-training (ReST) for language modeling.* arXiv, 2023.
- **[28]** J. Peters and S. Schaal. *Reinforcement learning by reward-weighted regression for operational space control.* In ICML, 2007.
- **[29]** J. Liu, et al. *Improving video generation with human feedback.* In NeurIPS, 2025 (Introduces Flow-DPO & Flow-RWR).
- **[30]** R. Rafailov, et al. *Direct preference optimization: Your language model is secretly a reward model.* In NeurIPS, 2023.
- **[31]** B. Wallace, et al. *Diffusion model alignment using direct preference optimization.* In CVPR, 2024.
- **[32]** A. Z. Ren, et al. *Diffusion policy policy optimization.* In ICLR, 2025.
- **[33]** Q. Yu, et al. *DAPO: An open-source LLM reinforcement learning system at scale.* arXiv, 2025.
- **[34]** X. He, et al. *TempFlow-GRPO: When timing matters for GRPO in flow models.* In ICLR, 2026.
- **[35]** L. Baraldi, et al. *What changed? Detecting and evaluating instruction-guided image edits with multimodal large language models.* arXiv:2505.20405, 2025.
- **[36]** X. Li, et al. *Evaluating real-world robot manipulation policies in simulation.* In CoRL, 2024.
- **[37]** B. Liu, et al. *LIBERO: Benchmarking knowledge transfer for lifelong robot learning.* In NeurIPS, 2023.
- **[38]** Octo Model Team. *Octo: An open-source generalist robot policy.* In RSS, 2024.
- **[39]** J. Wang, et al. *I2I-Bench: A comprehensive benchmark suite for image-to-image editing models.* arXiv:2512.04660, 2025.
- **[40]** Y. Zhang, et al. *EVF-SAM: Early vision-language fusion for text-prompted segment anything model.* arXiv:2406.20076, 2024.
- **[41]** X. Zhao, et al. *Envisioning beyond the pixels: Benchmarking reasoning-informed visual editing.* arXiv:2504.02826, 2025.
