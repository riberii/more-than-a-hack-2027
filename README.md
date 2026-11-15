# More Than A Hack 2027: Initial Challenge Submission
**Team Name:** Pristine Cascade  
**Team Members:** Marquis Verino, Timothea Owen, Evan William Destiawan, Gregorius Raffaelino Diego Arsenio, Henry Setiawan  
**Official Challenge:** FC Barcelona vs Athletic Club — Automatic Possession Inference  
**Repository:** [https://github.com/riberii/more-than-a-hack-2027](https://github.com/riberii/more-than-a-hack-2027)

---

## Overview

This repository contains the complete algorithmic pipeline, source code, technical documentation, and reproduction scripts for team **Pristine Cascade**'s submission to **More Than A Hack 2027 - Initial Challenge**.

Our system infers frame-accurate match possession labels (`H` = FC Barcelona, `A` = Athletic Club, `D` = Dead Ball, `X` = Ungraded Abstention) for all 199,319 frames (25 fps) of the tactical video recording by directly reading and modeling the raw Metrica EPTS optical tracking data.

### Key Performance Highlights (Scored on Public Ground Truth with 1,000 Bootstrap Draws)
- **spellF1:** `0.4800` (95% CI: `[0.3200, 0.6400]`, ahead of / statistically tied with `baseline_smooth125` at `0.4242` on the 17-possession sample, and clearing the `baseline_smooth50` floor of `0.3256`).
- **macroF1:** `0.7308` (95% CI: `[0.5402, 0.8548]`, leading `baseline_smooth50` by **+0.1101**).
- **Dead-Ball (D) Recall & F1:** **Recall 0.5405** (nearly $3\times$ higher than `baseline_smooth50`'s `0.2138`) and **F1 0.5919** (vs. `0.3225`).
- **Turnover Timing (Transition F1@25):** `0.3256` (matched 7 of 20 transitions, vs. 2 of 20 for `baseline_smooth50`).
- **Possession Scale:** 18 predicted spells of mean duration **14.0s** in the public scored window (Ground Truth has 17 spells of mean **14.2s**).
- **Abstention Compliance:** Exactly 0 frames of `X` on graded ground truth (0.00%).
- **Runtime:** Full-match streaming, reconstruction, touch detection, and Viterbi DP executes in **~13 seconds**.

---

## Algorithmic Architecture

Our pipeline processes the raw tracking data through 5 coordinated modules:

```text
[Raw EPTS Tracking: 199,319 frames]
              │
              ├──> [Ball Kinematics & Reconstruction] ──> Ball Trajectory & Confidence
              │
              ├──> [Touch & Velocity Discontinuities] ──> Touch Evidence & Team Attribution
              │
              ├──> [Multi-Feature Dead-Ball Model]   ──> Pitch Bounds, Stationarity, Set Pieces
              │
              └──> [Structured HMM / Viterbi DP]    ──> Global Sequence Optimization
                                                              │
                                                              ▼
                                               [Touch-Aligned Transitions]
                                                              │
                                                              ▼
                                                 Pristine Cascade.csv (199,319 rows)
```

1. **EPTS Tracking Streamer:** Fast single-pass streaming of `BAR-ATH_tactical_atd.txt` and `.xml`, converting 46 player tracklet coordinates and ball coordinates to metric units ($105.3\text{ m} \times 68.0\text{ m}$).
2. **Kinematic Ball Reconstruction:** Interpolates short dropouts ($\le 1.0\text{s}$) and medium dropouts ($1.0–3.0\text{s}$) under physical velocity bounds ($v \le 35\text{ m/s}$), producing a continuous trajectory and a calibrated per-frame confidence score.
3. **Dead-Ball & Stoppage Model:** Combines pitch boundary checks ($105.3\text{ m} \times 68.0\text{ m}$), sustained ball stationarity ($v < 0.3\text{ m/s}$ for $\ge 1.0\text{s}$), restart landmark geometry, and 59 Metrica Smart Tagging set-piece restart priors (Goal Kicks, Corners, Free Kicks, Throw-ins, Kickoffs).
4. **Touch & Discontinuity Detection:** Detects ball acceleration jumps ($\Delta v > 25\text{ m/s}^2$) within 2.0m of active players, attributing touches to that team and propagating decaying possession evidence ($t_{1/2} \approx 1.2\text{s}$) forward.
5. **Structured HMM / Viterbi Decoder:** Global dynamic programming sequence decoding over $\{H, A, D\}$ balancing touch evidence, nearest-player logistic probabilities, dead-ball likelihood, and baseline consensus regularizer. Transition boundaries are snapped to physical touch frames to eliminate smoothing lag.

---

## Repository Structure

```text
.
├── generate_submission.py       # Main deterministic submission generator (reads tracking data)
├── build_pdf.py                 # Generates official 2-page PDF submission report
├── games.py                     # Challenge alignment constants & asset path resolution
├── requirements.txt             # Pinned dependencies (numpy, pandas, reportlab, matplotlib)
├── Pristine Cascade.csv         # Generated final submission CSV (199,319 rows)
├── Pristine Cascade.pdf         # Generated official 2-page Technical Report
├── Pristine_Cascade.md          # Markdown source of the Technical Report
├── timeline_public_window.png   # Failure analysis timeline strip (GT vs. Prediction)
├── data/                        # Match dataset junction / symlink (or via MTAH_DATA_DIR)
│   └── tactical/
│       ├── BAR-ATH_tactical_atd.txt
│       ├── BAR-ATH_tactical_atd.xml
│       └── BAR-ATH_tactical_smart_tagging.xml
├── evaluation/
│   ├── gt/                      # Public 5-minute ground truth (7,500 graded frames)
│   ├── results/                 # Pre-computed baselines & official submission copy
│   ├── possession_metrics.py    # Official metric specification & scoring functions
│   ├── run_evaluation.py        # Official leaderboard scoring tool (with bootstrap)
│   └── validate_submission.py   # Submission format validator
└── README.md                    # Project documentation & reproduction instructions
```

---

## Reproduction Instructions

### 1. Environment Setup
Python 3.8+ or Python 3.11+:
```bash
git clone https://github.com/riberii/more-than-a-hack-2027.git
cd more-than-a-hack-2027
pip install -r requirements.txt
```

### 2. Match Data Setup
The gated challenge match bundle should be unpacked into `./data` (or pointed to via `$env:MTAH_DATA_DIR` / `export MTAH_DATA_DIR`):
```text
data/
└── tactical/
    ├── BAR-ATH_tactical_atd.txt
    ├── BAR-ATH_tactical_atd.xml
    └── BAR-ATH_tactical_smart_tagging.xml
```

### 3. Generate Submission CSV
Run the end-to-end pipeline:
```bash
python generate_submission.py --output "Pristine Cascade.csv"
```
Execution takes approximately **13 seconds** and outputs `Pristine Cascade.csv` with exactly 199,319 rows.

### 4. Format Validation
Run the official validator:
```bash
python evaluation/validate_submission.py "Pristine Cascade.csv"
```
Output:
```text
OK    Pristine Cascade.csv
      199319 rows, frame 0..199318
      label counts:
        H    51313   25.7%
        A    34682   17.4%
        D    72862   36.6%
        X    40462   20.3%
      reports 305 possessions of mean 11.3 s
Valid.
```

### 5. Evaluate Benchmark Score
Run the official competition scorer with 1,000 bootstrap draws:
```bash
python evaluation/run_evaluation.py --gt public --bootstrap 1000
```
Output summary:
```text
rank  submission          macroF1  spellF1   poss     dur  transF1  lat_f  shareE  frag
   1  Pristine Cascade     0.7308   0.4800     18    14.0   0.3256   -6.4   0.047  1.14
   2  baseline_smooth125   0.5812   0.4242     16    18.5   0.1667   -7.7   0.052  0.81
   3  baseline_smooth50    0.6207   0.3256     26    10.8   0.0816   +4.0   0.048  1.43
   4  baseline_smooth25    0.6200   0.2868     36     7.6   0.2034   +8.2   0.056  1.90
   5  baseline_raw         0.5972   0.0378    131     2.0   0.1410   -3.1   0.046  6.52
```

### 6. Build PDF Report
Compile the official 2-page PDF document:
```bash
python build_pdf.py
```
Outputs `Pristine Cascade.pdf` (exactly 2 pages, letter size).
