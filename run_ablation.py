"""Run full ablation study with 1000 bootstrap draws and collect all metrics."""
import os
import sys
import time
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation"))
import possession_metrics as pm
import games

GAME = "BAR-ATH"
pitch_len, pitch_wid = games.GAMES[GAME]["pitch"]
n_frames = games.source_frames("tactical", GAME)
fps = games.GAMES[GAME]["fps"]

HALF1_START = 11185
HALF1_END = 90651
HALF2_START = 110808
HALF2_END = 190197
KICKOFF_2ND = 113742

meta_path, raw_path, homography_path, st_path = games.require(
    ["atd_meta", "atd_raw", "homography", "smart_tagging"]
)

# 1. Parse metadata
root = ET.parse(meta_path).getroot()
teams = [t.get("id") for t in root.iter("Team")]
home_id = teams[0]
slots = [p.get("teamId") for p in root.iter("Player")]
team_of_slot = np.array([0 if tid == home_id else 1 for tid in slots], dtype=np.int8)
n_slots = len(team_of_slot)
home_slots = np.flatnonzero(team_of_slot == 0)
away_slots = np.flatnonzero(team_of_slot == 1)

print("1. Streaming tracking data...")
ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
ball_present = np.zeros(n_frames, dtype=bool)
player_xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)

min_dist_h = np.full(n_frames, 999.0, dtype=np.float32)
min_dist_a = np.full(n_frames, 999.0, dtype=np.float32)
closest_slot_h = np.full(n_frames, -1, dtype=np.int8)
closest_slot_a = np.full(n_frames, -1, dtype=np.int8)

with open(raw_path, 'r') as f:
    for idx, line in enumerate(f):
        parts = line.strip().split(':')
        if len(parts) != 3:
            continue
        ball_str = parts[2]
        has_ball = ball_str[0] != 'N'
        if has_ball:
            b_vals = ball_str.split(',')
            bx = float(b_vals[0]) * pitch_len
            by = float(b_vals[1]) * pitch_wid
            ball_xy[idx, 0] = bx
            ball_xy[idx, 1] = by
            ball_present[idx] = True

        p_groups = parts[1].split(';')
        best_dh, best_da = 999.0, 999.0
        best_sh, best_sa = -1, -1
        for s_idx, p_str in enumerate(p_groups):
            if p_str[0] == 'N':
                continue
            c1 = p_str.find(',')
            c2 = p_str.find(',', c1 + 1)
            px = float(p_str[:c1]) * pitch_len
            py = float(p_str[c1+1:c2]) * pitch_wid
            player_xy[idx, s_idx, 0] = px
            player_xy[idx, s_idx, 1] = py
            if has_ball:
                d = ((px - bx)**2 + (py - by)**2)**0.5
                if team_of_slot[s_idx] == 0:
                    if d < best_dh:
                        best_dh = d
                        best_sh = s_idx
                else:
                    if d < best_da:
                        best_da = d
                        best_sa = s_idx

        if has_ball:
            min_dist_h[idx] = best_dh
            min_dist_a[idx] = best_da
            closest_slot_h[idx] = best_sh
            closest_slot_a[idx] = best_sa

# Baseline signals
def load_base(name):
    path = os.path.join("evaluation", "results", name + ".csv")
    df = pd.read_csv(path, dtype={"label": "object"})
    cmap = {"H": 0, "A": 1, "D": 2, "X": 3}
    return np.array([cmap.get(x, 2) for x in df["label"].str.strip().str.upper()], dtype=np.int8)

b25 = load_base("baseline_smooth25")
b50 = load_base("baseline_smooth50")
b125 = load_base("baseline_smooth125")

# Ground truth
gt_codes = pm.label_codes(pm.read_label_csv("evaluation/gt/BAR-ATH_possession_gt_public.csv"))

results = {}

# Ablation 1: Baseline-vote-only (current submission style)
print("Evaluating Ablation 1: Baseline-vote-only...")
w_h = (b25 == 0)*2 + (b50 == 0)*2 + (b125 == 0)*1
w_a = (b25 == 1)*2 + (b50 == 1)*2 + (b125 == 1)*1
w_d = (b25 == 2)*2 + (b50 == 2)*2 + (b125 == 2)*1
abl1 = np.argmax(np.vstack([w_h, w_a, w_d]), axis=0).astype(np.int8)
abl1[:HALF1_START] = 3
abl1[HALF1_END+1:HALF2_START] = 3
abl1[HALF2_START:KICKOFF_2ND] = 2
abl1[HALF2_END+1:] = 3
results["baseline_vote_only"] = pm.evaluate(gt_codes, abl1, taus=(12, 25), bootstrap_n=1000)

# Ablation 2: + Ball Reconstruction
print("Evaluating Ablation 2: + Ball Reconstruction...")
recon_xy = ball_xy.copy()
ball_conf = np.zeros(n_frames, dtype=np.float32)
ball_conf[ball_present] = 1.0

diff = np.diff(ball_present.astype(int))
gap_starts = np.flatnonzero(diff == -1) + 1
gap_ends = np.flatnonzero(diff == 1) + 1
if not ball_present[0]:
    gap_starts = np.concatenate(([0], gap_starts))
if not ball_present[-1]:
    gap_ends = np.concatenate((gap_ends, [n_frames]))

for s, e in zip(gap_starts, gap_ends):
    gap_len = e - s
    if s > 0 and e < n_frames:
        p_prev = ball_xy[s - 1]
        p_next = ball_xy[e]
        dist = np.linalg.norm(p_next - p_prev)
        time_s = (gap_len + 1) / 25.0
        speed = dist / max(time_s, 0.04)
        if gap_len <= 25 and speed <= 35.0:
            alphas = np.linspace(0, 1, gap_len + 2)[1:-1, None]
            recon_xy[s:e] = p_prev + alphas * (p_next - p_prev)
            dist_edge = np.minimum(np.arange(gap_len), np.arange(gap_len)[::-1]) + 1
            ball_conf[s:e] = 0.6 + 0.35 * (1.0 - (dist_edge / (gap_len / 2 + 1)))
        elif gap_len <= 75 and speed <= 30.0:
            alphas = np.linspace(0, 1, gap_len + 2)[1:-1, None]
            recon_xy[s:e] = p_prev + alphas * (p_next - p_prev)
            ball_conf[s:e] = 0.35 * (1.0 - gap_len / 75.0)

# Recompute distances
interpolated = (~ball_present) & (~np.isnan(recon_xy[:, 0]))
for idx in np.flatnonzero(interpolated):
    bx, by = recon_xy[idx]
    p_h = player_xy[idx, home_slots]
    valid_h = ~np.isnan(p_h[:, 0])
    if np.any(valid_h):
        min_dist_h[idx] = np.min(np.linalg.norm(p_h[valid_h] - [bx, by], axis=1))
    p_a = player_xy[idx, away_slots]
    valid_a = ~np.isnan(p_a[:, 0])
    if np.any(valid_a):
        min_dist_a[idx] = np.min(np.linalg.norm(p_a[valid_a] - [bx, by], axis=1))

# Nearest player on reconstructed tracking
abl2 = np.full(n_frames, 2, dtype=np.int8)
has_recon_ball = ~np.isnan(recon_xy[:, 0])
abl2[has_recon_ball & (min_dist_h < min_dist_a)] = 0
abl2[has_recon_ball & (min_dist_a < min_dist_h)] = 1
# Apply 50-frame majority vote
h_cum = np.concatenate(([0], np.cumsum(abl2 == 0)))
a_cum = np.concatenate(([0], np.cumsum(abl2 == 1)))
idx_arr = np.arange(n_frames)
low = np.maximum(0, idx_arr - 50)
high = np.minimum(n_frames, idx_arr + 51)
h_cnt = h_cum[high] - h_cum[low]
a_cnt = a_cum[high] - a_cum[low]
abl2_smooth = np.full(n_frames, 2, dtype=np.int8)
seen = (h_cnt + a_cnt) > 0
abl2_smooth[seen] = np.where(h_cnt[seen] >= a_cnt[seen], 0, 1)
abl2_smooth[:HALF1_START] = 3
abl2_smooth[HALF1_END+1:HALF2_START] = 3
abl2_smooth[HALF2_START:KICKOFF_2ND] = 2
abl2_smooth[HALF2_END+1:] = 3
results["ball_reconstruction"] = pm.evaluate(gt_codes, abl2_smooth, taus=(12, 25), bootstrap_n=1000)

# Ablation 3: + Dead-Ball Model
print("Evaluating Ablation 3: + Dead-Ball Model...")
valid_ball = ~np.isnan(recon_xy[:, 0])
ball_vx = np.zeros(n_frames, dtype=np.float32)
ball_vy = np.zeros(n_frames, dtype=np.float32)
valid_both = valid_ball[:-2] & valid_ball[2:]
indices = np.flatnonzero(valid_both) + 1
ball_vx[indices] = (recon_xy[indices + 1, 0] - recon_xy[indices - 1, 0]) * (25.0 / 2.0)
ball_vy[indices] = (recon_xy[indices + 1, 1] - recon_xy[indices - 1, 1]) * (25.0 / 2.0)
ball_speed = np.sqrt(ball_vx**2 + ball_vy**2)

p_dead = np.zeros(n_frames, dtype=np.float32)
bx, by = recon_xy[:, 0], recon_xy[:, 1]
out_bounds = valid_ball & ((bx < -0.5) | (bx > pitch_len + 0.5) | (by < -0.5) | (by > pitch_wid + 0.5))
p_dead[out_bounds] = 1.0

for s, e in zip(gap_starts, gap_ends):
    if e - s > 50:
        p_dead[s:e] = np.maximum(p_dead[s:e], 0.85)

stationary = valid_ball & (ball_speed < 0.3)
diff_stat = np.diff(stationary.astype(int))
stat_starts = np.flatnonzero(diff_stat == 1) + 1
stat_ends = np.flatnonzero(diff_stat == -1) + 1
if stationary[0]:
    stat_starts = np.concatenate(([0], stat_starts))
if stationary[-1]:
    stat_ends = np.concatenate((stat_ends, [n_frames]))
for s, e in zip(stat_starts, stat_ends):
    if e - s >= 25:
        p_dead[s:e] = np.maximum(p_dead[s:e], 0.75)

st_events = []
for inst in ET.parse(st_path).iter("instance"):
    start_s = float(inst.findtext("start"))
    end_s = float(inst.findtext("end"))
    code = inst.findtext("code")
    tags = {l.findtext("group"): l.findtext("text") for l in inst.findall("label")}
    st_events.append({"code": code, "start_frame": int(round(start_s * 25)), "end_frame": int(round(end_s * 25)), "tags": tags})

sp_windows = {
    "Goal Kick": (450, 0.85), "Corner Kick": (350, 0.80), "Free Kick": (300, 0.75),
    "Throw-in": (180, 0.70), "Kick Off (Start)": (200, 0.95), "Kick Off (After Goal)": (300, 0.90)
}
for e in st_events:
    if e["code"] and "SET PIECE" in e["code"].upper():
        tp = e["tags"].get("Type", "")
        if tp in sp_windows:
            win, prior = sp_windows[tp]
            sf = e["start_frame"]
            ds = max(0, sf - win)
            p_dead[ds:sf] = np.maximum(p_dead[ds:sf], prior)

p_dead[HALF2_START:KICKOFF_2ND] = 1.0

abl3 = abl2_smooth.copy()
abl3[p_dead >= 0.7] = 2
abl3[:HALF1_START] = 3
abl3[HALF1_END+1:HALF2_START] = 3
abl3[HALF2_START:KICKOFF_2ND] = 2
abl3[HALF2_END+1:] = 3
results["dead_ball_model"] = pm.evaluate(gt_codes, abl3, taus=(12, 25), bootstrap_n=1000)

# Ablation 4: + Touch-based Event Detection
print("Evaluating Ablation 4: + Touch Event Detection...")
ball_ax = np.zeros(n_frames, dtype=np.float32)
ball_ay = np.zeros(n_frames, dtype=np.float32)
acc_indices = indices[1:-1]
ball_ax[acc_indices] = (ball_vx[acc_indices + 1] - ball_vx[acc_indices - 1]) * (25.0 / 2.0)
ball_ay[acc_indices] = (ball_vy[acc_indices + 1] - ball_vy[acc_indices - 1]) * (25.0 / 2.0)
ball_acc = np.sqrt(ball_ax**2 + ball_ay**2)

touch_h = np.zeros(n_frames, dtype=np.float32)
touch_a = np.zeros(n_frames, dtype=np.float32)
for idx in range(1, n_frames - 1):
    if not valid_ball[idx]:
        continue
    acc = ball_acc[idx]
    dh, da = min_dist_h[idx], min_dist_a[idx]
    if (acc > 25.0 and min(dh, da) < 2.0) or (min(dh, da) < 0.8):
        strength = min(1.0, acc / 50.0 + 0.3) / (1.0 + min(dh, da))
        if dh < da:
            touch_h[idx] = max(touch_h[idx], strength)
        else:
            touch_a[idx] = max(touch_a[idx], strength)

touch_ev_h = np.zeros(n_frames, dtype=np.float32)
touch_ev_a = np.zeros(n_frames, dtype=np.float32)
for t in range(1, n_frames):
    touch_ev_h[t] = touch_ev_h[t-1] * 0.94 + touch_h[t]
    touch_ev_a[t] = touch_ev_a[t-1] * 0.94 + touch_a[t]

soft_h = 1.0 / (1.0 + np.exp(-np.clip((min_dist_a - min_dist_h) / 1.5, -10, 10)))
soft_a = 1.0 - soft_h

base_cons_h = ((b25 == 0)*2 + (b50 == 0)*2 + (b125 == 0)*1) / 5.0
base_cons_a = ((b25 == 1)*2 + (b50 == 1)*2 + (b125 == 1)*1) / 5.0
base_cons_d = ((b25 == 2)*2 + (b50 == 2)*2 + (b125 == 2)*1) / 5.0

# Without HMM (greedy frame-by-frame on touch + soft + dead)
score_h = 1.5 * touch_ev_h + 1.2 * soft_h + 1.2 * base_cons_h - 2.5 * p_dead
score_a = 1.5 * touch_ev_a + 1.2 * soft_a + 1.2 * base_cons_a - 2.5 * p_dead
score_d = 3.0 * (p_dead - 0.35) + 1.5 * base_cons_d - 1.0 * ball_conf
abl4 = np.argmax(np.vstack([score_h, score_a, score_d]), axis=0).astype(np.int8)
abl4[:HALF1_START] = 3
abl4[HALF1_END+1:HALF2_START] = 3
abl4[HALF2_START:KICKOFF_2ND] = 2
abl4[HALF2_END+1:] = 3
results["touch_based_detection"] = pm.evaluate(gt_codes, abl4, taus=(12, 25), bootstrap_n=1000)

# Ablation 5: Full HMM/Viterbi Decoder + Structured Durations + Touch Alignment
print("Evaluating Ablation 5: Full Pipeline (Viterbi + Durations + Touch Alignment)...")

def decode_viterbi(start_f, end_f):
    L = end_f - start_f + 1
    log_emit = np.zeros((L, 3), dtype=np.float32)
    log_emit[:, 0] = score_h[start_f:end_f+1]
    log_emit[:, 1] = score_a[start_f:end_f+1]
    log_emit[:, 2] = score_d[start_f:end_f+1]

    log_trans = np.array([
        [ 0.0, -4.5, -2.8],
        [-4.5,  0.0, -2.8],
        [-2.0, -2.0,  0.0]
    ], dtype=np.float32)

    viterbi = np.zeros((L, 3), dtype=np.float32)
    backpointer = np.zeros((L, 3), dtype=np.int32)
    viterbi[0] = log_emit[0]
    for t in range(1, L):
        scores = viterbi[t-1, :, None] + log_trans
        backpointer[t] = np.argmax(scores, axis=0)
        viterbi[t] = np.max(scores, axis=0) + log_emit[t]
    path = np.zeros(L, dtype=np.int8)
    path[-1] = np.argmax(viterbi[-1])
    for t in range(L - 2, -1, -1):
        path[t] = backpointer[t + 1, path[t + 1]]
    return path

abl5 = np.full(n_frames, 3, dtype=np.int8)
abl5[HALF1_START:HALF1_END+1] = decode_viterbi(HALF1_START, HALF1_END)
abl5[HALF2_START:HALF2_END+1] = decode_viterbi(HALF2_START, HALF2_END)

# Apply structured duration filtering (p=60, d=30)
def apply_structured(arr, min_poss=60, min_dead=30, min_micro=25, max_iter=15):
    out = arr.copy()
    for start, stop in [(HALF1_START, HALF1_END + 1), (HALF2_START, HALF2_END + 1)]:
        sub = out[start:stop]
        for _ in range(max_iter):
            changed = False
            breaks = np.flatnonzero(sub[1:] != sub[:-1]) + 1
            bounds = np.concatenate(([0], breaks, [len(sub)]))
            starts = bounds[:-1]
            stops = bounds[1:]
            labels = sub[starts]
            durs = stops - starts

            # Micro flickers
            for i in range(len(labels)):
                if labels[i] in (0, 1) and durs[i] < min_micro:
                    prev_l = labels[i-1] if i > 0 else 2
                    next_l = labels[i+1] if i+1 < len(labels) else 2
                    cand = prev_l if (durs[i-1] if i > 0 else 0) >= (durs[i+1] if i+1 < len(labels) else 0) else next_l
                    if cand == 3: cand = 2
                    sub[starts[i]:stops[i]] = cand
                    changed = True

            breaks = np.flatnonzero(sub[1:] != sub[:-1]) + 1
            bounds = np.concatenate(([0], breaks, [len(sub)]))
            starts = bounds[:-1]
            stops = bounds[1:]
            labels = sub[starts]
            durs = stops - starts

            # Dead bridging
            for i in range(1, len(labels) - 1):
                if labels[i] == 2 and durs[i] < min_dead:
                    if labels[i-1] == labels[i+1] and labels[i-1] in (0, 1):
                        sub[starts[i]:stops[i]] = labels[i-1]
                        changed = True

            breaks = np.flatnonzero(sub[1:] != sub[:-1]) + 1
            bounds = np.concatenate(([0], breaks, [len(sub)]))
            starts = bounds[:-1]
            stops = bounds[1:]
            labels = sub[starts]
            durs = stops - starts

            # Min possession
            for i in range(len(labels)):
                if labels[i] in (0, 1) and durs[i] < min_poss:
                    prev_l = labels[i-1] if i > 0 else 2
                    next_l = labels[i+1] if i+1 < len(labels) else 2
                    cand = prev_l if (durs[i-1] if i > 0 else 0) >= (durs[i+1] if i+1 < len(labels) else 0) else next_l
                    if cand == 3: cand = 2
                    sub[starts[i]:stops[i]] = cand
                    changed = True

            if not changed:
                break
        out[start:stop] = sub
    return out

abl5_struct = apply_structured(abl5, min_poss=60, min_dead=30, min_micro=25)

# Touch snapping & transition refinement
for start, stop in [(HALF1_START, HALF1_END + 1), (HALF2_START, HALF2_END + 1)]:
    sub = abl5_struct[start:stop]
    breaks = np.flatnonzero(sub[1:] != sub[:-1]) + 1
    new_sub = sub.copy()
    for brk in breaks:
        from_l = sub[brk - 1]
        to_l = sub[brk]
        if from_l == 3 or to_l == 3:
            continue
        abs_brk = start + brk
        lo = max(0, abs_brk - 10)
        hi = min(n_frames - 1, abs_brk + 15)
        if from_l in (0, 1) and to_l in (0, 1):
            tgt_touch = touch_a if to_l == 1 else touch_h
            win_t = tgt_touch[lo:hi]
            if np.any(win_t > 0):
                shift = (lo + np.argmax(win_t)) - abs_brk
                new_brk = brk + shift
                if 0 < new_brk < len(sub):
                    if shift > 0: new_sub[brk:new_brk] = from_l
                    elif shift < 0: new_sub[new_brk:brk] = to_l
            else:
                shift = 5
                new_brk = brk + shift
                if 0 < new_brk < len(sub): new_sub[brk:new_brk] = from_l
        elif from_l == 2 and to_l in (0, 1):
            tgt_touch = touch_h if to_l == 0 else touch_a
            win_t = tgt_touch[lo:hi]
            if np.any(win_t > 0):
                shift = (lo + np.argmax(win_t)) - abs_brk
                new_brk = brk + shift
                if 0 < new_brk < len(sub):
                    if shift > 0: new_sub[brk:new_brk] = from_l
                    elif shift < 0: new_sub[new_brk:brk] = to_l
    abl5_struct[start:stop] = new_sub

abl5_struct[:HALF1_START] = 3
abl5_struct[HALF1_END+1:HALF2_START] = 3
abl5_struct[HALF2_START:KICKOFF_2ND] = 2
abl5_struct[HALF2_END+1:] = 3
results["full_pipeline"] = pm.evaluate(gt_codes, abl5_struct, taus=(12, 25), bootstrap_n=1000)

# Save ablation results
out_data = {}
for k, v in results.items():
    t12 = v["transitions"][12]
    t25 = v["transitions"][25]
    out_data[k] = {
        "macro_f1": v["macro_f1"],
        "macro_bootstrap": (v["bootstrap_low"], v["bootstrap_high"]),
        "spell_f1_mean": v["spell_f1_mean"],
        "spell_bootstrap": (v["spell_bootstrap_low"], v["spell_bootstrap_high"]),
        "f1_H": v["per_class"]["H"]["f1"],
        "f1_A": v["per_class"]["A"]["f1"],
        "f1_D": v["per_class"]["D"]["f1"],
        "recall_D": v["per_class"]["D"]["recall"],
        "precision_D": v["per_class"]["D"]["precision"],
        "trans_f1_12": t12["f1"],
        "trans_f1_25": t25["f1"],
        "latency_mean": t25["mean_latency"],
        "n_matched_25": t25["n_matched"],
        "n_spurious_25": t25["n_spurious"],
        "n_poss": v["n_pred_spells"],
        "mean_dur_s": v["mean_spell_duration_s"]
    }

with open("ablation_results.json", "w") as f:
    json.dump(out_data, f, indent=2)

print("\n=== ABLATION STUDY RESULTS (1000 Bootstrap Draws) ===")
print("{:<22} {:>8} {:>17} {:>8} {:>17} {:>6} {:>6} {:>8} {:>6} {:>5} {:>5}".format(
    "Configuration", "macroF1", "macroCI", "spellF1", "spellCI", "D_F1", "D_rec", "trans25", "lat_f", "poss", "dur"
))
print("-" * 115)
for k, d in out_data.items():
    m_ci = f"{d['macro_bootstrap'][0]:.3f}-{d['macro_bootstrap'][1]:.3f}"
    s_ci = f"{d['spell_bootstrap'][0]:.3f}-{d['spell_bootstrap'][1]:.3f}"
    lat = f"{d['latency_mean']:+.1f}" if d['latency_mean'] is not None else "--"
    print(f"{k:<22} {d['macro_f1']:>8.4f} {m_ci:>17} {d['spell_f1_mean']:>8.4f} {s_ci:>17} {d['f1_D']:>6.4f} {d['recall_D']:>6.4f} {d['trans_f1_25']:>8.4f} {lat:>6} {d['n_poss']:>5d} {d['mean_dur_s']:>5.1f}s")
