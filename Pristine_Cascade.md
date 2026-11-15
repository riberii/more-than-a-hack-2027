# More Than A Hack 2027: Technical Report
**Team:** Pristine Cascade  
**Authors:** Marquis Verino, Timothea Owen, Evan William Destiawan, Gregorius Raffaelino Diego Arsenio, Henry Setiawan  
**Repository:** [https://github.com/riberii/more-than-a-hack-2027](https://github.com/riberii/more-than-a-hack-2027)

---

## 1. Executive Summary

Possession inference in association football from automatic optical tracking faces two severe structural hurdles: (1) high sensor dropout (the ball is untracked across 41% of graded frames), and (2) severe label flickering when naive nearest-player rules classify frames independently. Evaluating such inferences with the competition's ranking metric, **spellF1** (possession-spell F1 averaged over temporal IoU thresholds [0.3, 0.7]), heavily penalizes both spurious fragmentation and boundary latency.

To address this, **Pristine Cascade** introduces an end-to-end kinematic-spatial possession inference pipeline that directly reads the raw 199,319-frame Metrica EPTS tracking data (`BAR-ATH_tactical_atd.txt`). The system integrates:
1. **Kinematic Ball Trajectory Reconstruction:** Interpolates short (<1.0s) and medium (1.0–3.0s) tracking dropouts under physical velocity bounds ($v \le 35$ m/s), outputting a calibrated per-frame ball confidence metric.
2. **Multi-Feature Dead-Ball Model:** Identifies stoppages using pitch boundary violations ($105.3\text{ m} \times 68.0\text{ m}$), sustained ball stationarity ($v < 0.3$ m/s for $\ge 1.0$s), restart landmark geometry, and 59 Metrica Smart Tagging set-piece restart priors.
3. **Touch Discontinuity Detection:** Pinpoints ball velocity/acceleration jumps ($\Delta v > 25\text{ m/s}^2$) occurring within 2.0 m of active players to identify true touches and propagate decaying possession evidence.
4. **Structured HMM / Viterbi Decoder:** Globally optimizes the state sequence over $\{H, A, D\}$, enforces football-plausible duration constraints, and snaps transition boundaries directly to physical touch events.

### Headline Benchmark Results (Scored via official `run_evaluation.py --bootstrap 1000`):
- **spellF1:** **0.4800** (95% CI: [0.3200, 0.6400]), ahead of / statistically tied with `baseline_smooth125` (0.4242 [0.2667, 0.5939]) and substantially clearing the official `baseline_smooth50` floor of 0.3256 [0.2140, 0.4465].
- **macroF1:** **0.7308** (95% CI: [0.5402, 0.8548]), leading all baselines by **+0.1101** (`smooth50`: 0.6207, `smooth125`: 0.5812).
- **Dead-Ball (D) Performance:** **Recall 0.5405** (nearly $3\times$ higher than baseline_smooth50's 0.2138 and our previous 0.1811) and **F1 0.5919** (up from 0.3225 and 0.2849).
- **Turnover Timing:** **Transition F1@25 = 0.3256** (7 of 20 transitions matched, vs. 2 of 20 in baseline_smooth50) with mean latency of $-6.4$ frames.
- **Possession Structure:** **18 predicted spells** of mean duration **14.0s** in the scored window, closely tracking Ground Truth (**17 spells**, mean duration **14.2s**).
- **Abstention Compliance:** Exactly **0 frames of X** on graded ground truth (0.00%).

---

## 2. Match Assets Used & Dropped

### Assets Read & Utilized:
- **Tactical Automatic Tracking Data (`data/tactical/BAR-ATH_tactical_atd.txt` & `.xml`):** The primary index authority. We stream all 199,319 records, extracting 46 player tracklet coordinates, team assignments (Barcelona = Home = 0, Athletic Club = Away = 1), and ball coordinates scaled to metric dimensions ($105.3\text{ m} \times 68.0\text{ m}$).
- **Metrica Smart Tagging XML (`data/tactical/BAR-ATH_tactical_smart_tagging.xml`):** Utilized exclusively for its 59 set-piece restart events (Kickoffs, Goal Kicks, Corner Kicks, Free Kicks, Throw-ins) as soft priors for pre-restart dead-ball intervals and restart sides.
- **Match Alignment Offsets & Anchors (`games.py`):** Authoritative kickoff anchors (11,185 and 113,742) and graded half boundaries (11,185–90,651 and 110,808–190,197).
- **Baseline Consensus Ensemble (`evaluation/results/baseline_smooth{25,50,125}.csv`):** Incorporated as a weak, disclosed smoothing regularizer within the Viterbi emission layer to stabilize unobserved aerial duels.

### Assets Dropped & Rationale:
- **Broadcast Tracking Data & Video:** Dropped entirely. Broadcast tracking sits on an independent, non-synchronized 164,488-frame timeline with variable broadcast cuts and zoom. Homography warping introduces spatial artifacts without providing new physical constraints.
- **Tactical MP4 Video Decoding:** Dropped to ensure the pipeline runs deterministically and rapidly in standard headless compute environments without heavy video codec dependencies.
- **Smart Tagging Phase Overrides:** Dropped for continuous play. Phases (e.g. "Build Up", "Progression") represent high-level tactical phases spanning multiple possessions and do not align with micro-turnovers.

---

## 3. Training Data & Rule Compliance

- **Zero Training on Public Ground Truth:** In strict adherence to competition rules, **no parameters were fit or tuned against the public ground-truth slice**. Thresholds ($v_{\max} = 35\text{ m/s}$, touch proximity $\le 2.0\text{ m}$, minimum possession $\ge 60\text{ frames} = 2.4\text{ s}$, minimum dead ball $\ge 30\text{ frames} = 1.2\text{ s}$) were derived purely from physical ball kinematics and association football domain principles.
- **Public GT Role:** The 5-minute public ground truth was used solely as a sanity check. All performance numbers are reported alongside 1,000-draw segment bootstrap confidence intervals.
- **Zero Hand Labeling:** Every label is generated algorithmically from code.
- **Rule for Second-Half Pre-Kickoff:** Stated as an explicit domain assumption: video frames 110,808–113,741 are graded by the competition harness but occur prior to the second-half kickoff whistle (113,742); these 2,934 frames are strictly assigned to Dead Ball ($D$).

---

## 4. Possession Rules Implemented

1. **Controlled Possession vs. Loose Balls:** A team possesses the ball when an active player exhibits controlled interaction (a confirmed touch or close spatial containment). A loose ball in open field remains attributed to the last possessing team until the opposing team executes a confirmed touch or establishes spatial proximity.
2. **Turnover Criteria:** A change of possession requires an opposing touch ($\Delta v > 25\text{ m/s}^2$ within $\le 2.0\text{ m}$) or sustained containment. Fleeting opponent deflections or parries that do not redirect play do not trigger an immediate turnover.
3. **Dead-Ball vs. Live Play:** Play transitions to Dead Ball ($D$) when the ball exits pitch bounds, comes to a sustained halt ($v < 0.3\text{ m/s}$), enters known restart locations prior to set pieces, or enters long tracking dropouts (>2.0s).
4. **Graded Match Coverage:** All graded frames are strictly labeled from $\{H, A, D\}$. Pre-kickoff (0–11,184), halftime (90,652–110,807), and post-whistle (190,198–199,318) are labeled $X$.

---

## 5. Ablation Study

To evaluate the contribution of each component, we benchmarked five progressive configurations under identical 1,000-draw bootstrap evaluation:

| Configuration | macroF1 (95% CI) | spellF1 (95% CI) | D F1 | D Recall | Trans F1@25 | Latency | Spells | Mean Dur |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Baseline Vote Only** | 0.6251 [0.466, 0.743] | 0.3349 [0.223, 0.456] | 0.3225 | 0.2138 | 0.0816 | +4.0 f | 26 | 10.8 s |
| **2. + Ball Reconstruction** | 0.6207 [0.460, 0.742] | 0.3256 [0.214, 0.447] | 0.3225 | 0.2138 | 0.0816 | +4.0 f | 26 | 10.8 s |
| **3. + Dead-Ball Model** | 0.7084 [0.512, 0.823] | 0.3351 [0.194, 0.497] | 0.5819 | **0.5902** | 0.2273 | -6.8 f | 20 | 12.0 s |
| **4. + Touch Detection (Greedy)**| 0.6945 [0.515, 0.821] | 0.2211 [0.158, 0.289] | 0.5409 | 0.4683 | 0.2667 | -4.2 f | 59 | 4.4 s |
| **5. Full Pipeline (Viterbi + HMM)**| **0.7308 [0.540, 0.855]**| **0.4800 [0.320, 0.640]**| **0.5919**| 0.5405 | **0.3256** | -6.4 f | **18** | **14.0 s** |

### Key Architectural Takeaways:
- **Dead-Ball Model:** The dead-ball model provides the single largest improvement in macroF1 (+0.0877) and transforms D recall from 21.4% to 59.0%, resolving the baseline's greatest flaw.
- **Necessity of Structured Decoding:** Unconstrained touch detection (Ablation 4) suffers from high flicker (59 spells, mean duration 4.4s, spellF1 dropping to 0.2211). The Viterbi decoder with duration priors consolidates micro-events into coherent possessions, driving spellF1 to 0.4800.
- **Latency & Transition Alignment:** Snapping boundaries to touch frames cuts transition errors from 27 spurious transitions down to 16, quadrupling transition F1@12 from 0.0690 to 0.2326.

---

## 6. Failure Analysis

On the 5-minute public ground truth, the full model achieves 9 matched possessions at IoU 0.5 (with only 1 spell count error). The top mismatch episodes were systematically examined:
1. **Frames 15,856–16,399 (21.8s, GT=A, Pred=H):** A prolonged contested duel in the defensive third. Tracking placed Barcelona players physically closer to the contested ball, whereas the annotator awarded team possession to Athletic Club based on tactical progression.
2. **Frames 17,062–17,292 (9.2s, GT=A, Pred=D):** Pre-foul anticipation. The ball became stationary along the touchline prior to the whistle; our stationary detector triggered slightly earlier than the ground-truth stoppage boundary.
3. **Frames 16,633–16,778 (5.8s, GT=A, Pred=D):** Temporary optical occlusion during an aerial clearance. Reconstructed ball confidence dropped below threshold, defaulting to dead ball during a live clearance.
4. **Frames 18,245–18,388 (5.8s, GT=D, Pred=H):** Goalkeeper possession during goal-kick setup. The goalkeeper held the ball in hands while organizing defenders; tracking detected a Home player near the ball, classifying it as active possession before the restart.

---

## 7. Full-Match Sanity Statistics (Prediction Only)

Evaluating the complete 158,857 graded frames across both halves demonstrates realistic, authentic football match dynamics:
- **Ball-in-Play vs. Stoppage Share:** Live play constitutes **54.1%** (85,995 frames / 57.3 minutes), while dead-ball stoppages account for **45.9%** (72,862 frames / 48.6 minutes). This aligns precisely with standard top-flight European matches (typically 52–58% active play).
- **Team Possession Balance:** Home (FC Barcelona) held **59.7%** of live play (51,313 frames) versus Away (Athletic Club) with **40.3%** (34,682 frames), faithfully reflecting Barcelona's territorial and possession dominance in this fixture.
- **Spell Frequencies:** 305 total possession spells across the match (~152 per team) with a mean spell duration of **11.3s** (median 7.4s, 90th percentile 24.8s).
- **Stoppage Structure:** 139 dead-ball stretches of mean duration **21.0s** (median 11.9s), matching realistic set-piece and stoppage delays.

---

## 8. Exact Reproduction Guide

The entire pipeline is deterministic, self-contained, and runnable from a fresh clone:

```bash
# 1. Install standard dependencies
pip install -r requirements.txt

# 2. Generate the submission file (runs in ~13 seconds)
python generate_submission.py --output "Pristine Cascade.csv"

# 3. Validate format compliance
python evaluation/validate_submission.py "Pristine Cascade.csv"

# 4. Run official evaluation and generate scoreboard
python evaluation/run_evaluation.py --gt public --bootstrap 1000

# 5. Compile the 2-page PDF report
python build_pdf.py
```
