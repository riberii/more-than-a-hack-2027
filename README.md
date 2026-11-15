# More Than A Hack 2027: Initial Challenge Submission
**Team Name:** Pristine Cascade  
**Team Members:** Marquis Verino, Timothea Owen, Evan William Destiawan, Gregorius Raffaelino Diego Arsenio, Henry Setiawan  
**Official Challenge:** FC Barcelona vs Athletic Club — Automatic Possession Inference  

---

## Overview

This repository contains the full source code, algorithmic pipeline, and reproduction instructions for team **Pristine Cascade**'s submission to **More Than A Hack 2027 - Initial Challenge**.

Our method infers frame-by-frame match possession labels (`H` = Barcelona, `A` = Athletic Club, `D` = Dead Ball, `X` = Ungraded Abstention) for all 199,319 frames of the tactical video recording.

### Key Performance Highlights (Public Ground Truth Benchmark)
- **spellF1:** `0.4774` (**Rank 1**, +47% relative gain over the `baseline_smooth50` floor of `0.3256`)
- **macroF1:** `0.6274` (Rank 1 frame-level unweighted F1)
- **Accuracy:** `0.7324`
- **Possession Count:** 14 spells in public window (GT = 17) | 293 spells full-match (mean duration: 14.2 s, matching GT 14.19 s)
- **Zero Leakage:** Strictly zero training or tuning on the ground truth.

---

## Algorithmic Architecture

1. **Multi-Scale Temporal Consensus:** Integrates nearest-player tracking signals smoothed at multiple temporal horizons ($\pm 1.0\text{ s}$, $\pm 2.0\text{ s}$, and $\pm 5.0\text{ s}$) with weights `[2, 2, 1]` to balance sharp handover edges with noise immunity.
2. **Match Schedule & Stoppage Anchoring:**
   - Ungraded pre-match, halftime, and post-match frames are cleanly assigned `X`.
   - The 2,934 graded pre-kickoff frames of the second half (frames 110,808 to 113,741) are strictly anchored to Dead Ball (`D`).
3. **Controlled Possession State Persistence:** Enforces the association football rule that a turnover only occurs when an opponent establishes controlled possession ($\ge 75\text{ frames} \approx 3.0\text{ s}$). Short touches, tackles, and deflections are re-absorbed.
4. **Occlusion Bridging:** Gaps of untracked ball ($< 50\text{ frames} = 2.0\text{ s}$) between identical team possessions are bridged, preventing artificial possession cuts.

---

## Repository Structure

```text
more-than-a-hack-2027/
├── generate_submission.py       # Main deterministic submission generator
├── games.py                     # Official challenge constants and offset metadata
├── requirements.txt             # Pinned minimal dependencies
├── Pristine Cascade.pdf         # Short technical document (PDF submission)
├── Pristine_Cascade.md          # Markdown version of technical document
├── evaluation/
│   ├── gt/                      # Public 5-minute ground truth
│   ├── results/                 # Official pre-computed baselines
│   ├── possession_metrics.py    # Metric specification & scorer
│   ├── run_evaluation.py        # Official leaderboard scoring tool
│   └── validate_submission.py   # Official submission format validator
└── README.md                    # Documentation & reproduction guide
```

---

## Reproduction Instructions

### 1. Installation
Clone the repository and install dependencies in Python 3.8+ or Python 3.11+:
```bash
git clone https://github.com/riberii/more-than-a-hack-2027.git
cd more-than-a-hack-2027
pip install -r requirements.txt
```

### 2. Generate the Submission File
Run the end-to-end generation script:
```bash
python generate_submission.py --output "Pristine Cascade.csv"
```
This produces `Pristine Cascade.csv` with exactly 199,319 rows.

### 3. Validate Format
Run the official competition validator:
```bash
python evaluation/validate_submission.py "Pristine Cascade.csv"
```
Output:
```text
OK    Pristine Cascade.csv
      199319 rows, frame 0..199318
      label counts:
        H    59561   29.9%
        A    44770   22.5%
        D    54526   27.4%
        X    40462   20.3%
      reports 293 possessions of mean 14.2 s
Valid.
```

### 4. Evaluate Benchmark Score
Copy the generated file to `evaluation/results/` and run the evaluation script:
```bash
cp "Pristine Cascade.csv" evaluation/results/Pristine_Cascade.csv
python evaluation/run_evaluation.py
```
Output:
```text
rank  submission          macroF1  spellF1   poss     dur  transF1  lat_f
   1  Pristine_Cascade     0.6274   0.4774     14    20.3   0.1143   +4.0
```
