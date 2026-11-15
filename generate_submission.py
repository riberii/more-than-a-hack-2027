"""Generate the final possession submission file for More Than A Hack 2027.

Team Name: Pristine Cascade
Team Members: Marquis Verino, Timothea Owen, Evan William Destiawan,
              Gregorius Raffaelino Diego Arsenio, Henry Setiawan

Architecture:
1. EPTS Automatic Tracking Data (ATD) streaming parser: reads 199,319 frames of
   tactical tracking (Metrica EPTS format) containing 46 player track slots and
   the ball, scaled to pitch dimensions (105.3m x 68.0m).
2. Ball Trajectory Reconstruction: interpolates short (<1.0s) and medium (1-3s)
   occlusions using physical ball kinematics (<35 m/s maximum speed) and outputs
   a per-frame ball confidence metric in [0, 1].
3. Dead-Ball & Restart Model: models dead-ball state from pitch boundary violations,
   sustained ball stationarity (<0.3 m/s for >=1.0s), restart landmarks (corner arcs,
   goal lines, touchlines, centre spot), and 59 Metrica Smart Tagging set-piece
   restart priors (goal kicks, corners, free kicks, throw-ins, kickoffs).
   Includes documented pre-second-half kickoff dead-ball window (frames 110808-113741).
4. Touch & Event Discontinuity Detection: detects ball acceleration and direction
   discontinuities (d_v > 25 m/s^2) with a player within <=2.0m radius, assigning
   the touch to that player's team and propagating possession evidence forward with
   exponential temporal decay.
5. Structured HMM / Viterbi Decoder: global state sequence optimization over
   {H, A, D} using emissions from touch evidence, nearest-player soft distances,
   dead-ball likelihood, and baseline consensus prior, followed by minimum duration
   filtering and touch-aligned transition boundary placement.

Usage:
    python generate_submission.py --output "Pristine Cascade.csv"
"""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import games

# Match Constants (from games.py)
GAME = "BAR-ATH"
pitch_len, pitch_wid = games.GAMES[GAME]["pitch"]
N_FRAMES = games.source_frames("tactical", GAME)
FPS = games.GAMES[GAME]["fps"]

HALF1_START = 11185
HALF1_END = 90651
HALF2_START = 110808
HALF2_END = 190197
KICKOFF_2ND = 113742

H, A, D, X = 0, 1, 2, 3
LABELS = ("H", "A", "D", "X")


def stream_tracking_data(meta_path, raw_path):
    """Stream and parse raw EPTS tracking data into compact numpy arrays.

    Returns:
        ball_xy: float32 (N_FRAMES, 2) in pitch meters [NaN where missing]
        ball_present: bool (N_FRAMES)
        player_xy: float32 (N_FRAMES, 46, 2) in pitch meters
        min_dist_h: float32 (N_FRAMES) distance to nearest Home player
        min_dist_a: float32 (N_FRAMES) distance to nearest Away player
        team_of_slot: int8 (46) 0 for Home (Barcelona), 1 for Away (Athletic)
    """
    # Parse metadata XML for team assignments
    root = ET.parse(meta_path).getroot()
    teams = [t.get("id") for t in root.iter("Team")]
    home_id = teams[0]
    slots = [p.get("teamId") for p in root.iter("Player")]
    team_of_slot = np.array([0 if tid == home_id else 1 for tid in slots], dtype=np.int8)
    n_slots = len(team_of_slot)

    # Pre-allocate
    ball_xy = np.full((N_FRAMES, 2), np.nan, dtype=np.float32)
    ball_present = np.zeros(N_FRAMES, dtype=bool)
    player_xy = np.full((N_FRAMES, n_slots, 2), np.nan, dtype=np.float32)

    min_dist_h = np.full(N_FRAMES, 999.0, dtype=np.float32)
    min_dist_a = np.full(N_FRAMES, 999.0, dtype=np.float32)

    with open(raw_path, "r") as handle:
        for idx, line in enumerate(handle):
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue

            # Ball parsing
            ball_str = parts[2]
            has_ball = ball_str[0] != "N"
            if has_ball:
                b_vals = ball_str.split(",")
                bx = float(b_vals[0]) * pitch_len
                by = float(b_vals[1]) * pitch_wid
                ball_xy[idx, 0] = bx
                ball_xy[idx, 1] = by
                ball_present[idx] = True

            # Player parsing
            p_groups = parts[1].split(";")
            best_dh, best_da = 999.0, 999.0

            for s_idx, p_str in enumerate(p_groups):
                if p_str[0] == "N":
                    continue
                c1 = p_str.find(",")
                c2 = p_str.find(",", c1 + 1)
                px = float(p_str[:c1]) * pitch_len
                py = float(p_str[c1 + 1 : c2]) * pitch_wid
                player_xy[idx, s_idx, 0] = px
                player_xy[idx, s_idx, 1] = py

                if has_ball:
                    d = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
                    if team_of_slot[s_idx] == 0:
                        if d < best_dh:
                            best_dh = d
                    else:
                        if d < best_da:
                            best_da = d

            if has_ball:
                min_dist_h[idx] = best_dh
                min_dist_a[idx] = best_da

    return ball_xy, ball_present, player_xy, min_dist_h, min_dist_a, team_of_slot


def reconstruct_ball_trajectory(ball_xy, ball_present, player_xy, team_of_slot):
    """Interpolate short and medium tracking occlusions using ball kinematics."""
    recon_xy = ball_xy.copy()
    ball_conf = np.zeros(N_FRAMES, dtype=np.float32)
    ball_conf[ball_present] = 1.0

    diff = np.diff(ball_present.astype(int))
    gap_starts = np.flatnonzero(diff == -1) + 1
    gap_ends = np.flatnonzero(diff == 1) + 1
    if not ball_present[0]:
        gap_starts = np.concatenate(([0], gap_starts))
    if not ball_present[-1]:
        gap_ends = np.concatenate((gap_ends, [N_FRAMES]))

    for s, e in zip(gap_starts, gap_ends):
        gap_len = e - s
        if s > 0 and e < N_FRAMES:
            p_prev = ball_xy[s - 1]
            p_next = ball_xy[e]
            dist = np.linalg.norm(p_next - p_prev)
            time_s = (gap_len + 1) / float(FPS)
            speed = dist / max(time_s, 0.04)

            # Short gaps <= 1s
            if gap_len <= 25 and speed <= 35.0:
                alphas = np.linspace(0, 1, gap_len + 2)[1:-1, None]
                recon_xy[s:e] = p_prev + alphas * (p_next - p_prev)
                dist_edge = np.minimum(np.arange(gap_len), np.arange(gap_len)[::-1]) + 1
                ball_conf[s:e] = 0.6 + 0.35 * (1.0 - (dist_edge / (gap_len / 2 + 1)))
            # Medium gaps 1-3s
            elif gap_len <= 75 and speed <= 30.0:
                alphas = np.linspace(0, 1, gap_len + 2)[1:-1, None]
                recon_xy[s:e] = p_prev + alphas * (p_next - p_prev)
                ball_conf[s:e] = 0.35 * (1.0 - gap_len / 75.0)

    # Recompute distances for interpolated frames
    home_slots = np.flatnonzero(team_of_slot == 0)
    away_slots = np.flatnonzero(team_of_slot == 1)
    interpolated = (~ball_present) & (~np.isnan(recon_xy[:, 0]))
    min_dist_h = np.full(N_FRAMES, 999.0, dtype=np.float32)
    min_dist_a = np.full(N_FRAMES, 999.0, dtype=np.float32)

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

    return recon_xy, ball_conf, min_dist_h, min_dist_a, gap_starts, gap_ends


def detect_touches_and_kinematics(recon_xy, min_dist_h, min_dist_a):
    """Detect ball velocity discontinuities and player touch events."""
    valid_ball = ~np.isnan(recon_xy[:, 0])
    ball_vx = np.zeros(N_FRAMES, dtype=np.float32)
    ball_vy = np.zeros(N_FRAMES, dtype=np.float32)

    valid_both = valid_ball[:-2] & valid_ball[2:]
    indices = np.flatnonzero(valid_both) + 1
    ball_vx[indices] = (recon_xy[indices + 1, 0] - recon_xy[indices - 1, 0]) * (FPS / 2.0)
    ball_vy[indices] = (recon_xy[indices + 1, 1] - recon_xy[indices - 1, 1]) * (FPS / 2.0)
    ball_speed = np.sqrt(ball_vx**2 + ball_vy**2)

    # Acceleration / Delta-v
    ball_ax = np.zeros(N_FRAMES, dtype=np.float32)
    ball_ay = np.zeros(N_FRAMES, dtype=np.float32)
    acc_indices = indices[1:-1]
    ball_ax[acc_indices] = (ball_vx[acc_indices + 1] - ball_vx[acc_indices - 1]) * (FPS / 2.0)
    ball_ay[acc_indices] = (ball_vy[acc_indices + 1] - ball_vy[acc_indices - 1]) * (FPS / 2.0)
    ball_acc = np.sqrt(ball_ax**2 + ball_ay**2)

    # Touch detection
    touch_h = np.zeros(N_FRAMES, dtype=np.float32)
    touch_a = np.zeros(N_FRAMES, dtype=np.float32)

    for idx in range(1, N_FRAMES - 1):
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

    # Exponential decay propagation
    touch_ev_h = np.zeros(N_FRAMES, dtype=np.float32)
    touch_ev_a = np.zeros(N_FRAMES, dtype=np.float32)
    decay = 0.94
    for t in range(1, N_FRAMES):
        touch_ev_h[t] = touch_ev_h[t - 1] * decay + touch_h[t]
        touch_ev_a[t] = touch_ev_a[t - 1] * decay + touch_a[t]

    return touch_h, touch_a, touch_ev_h, touch_ev_a, ball_speed, valid_ball


def build_dead_ball_model(recon_xy, valid_ball, ball_speed, gap_starts, gap_ends, st_path):
    """Build dead-ball probability surface combining physical and XML priors."""
    p_dead = np.zeros(N_FRAMES, dtype=np.float32)

    # 1. Pitch boundary violations
    bx, by = recon_xy[:, 0], recon_xy[:, 1]
    out_bounds = valid_ball & ((bx < -0.5) | (bx > pitch_len + 0.5) | (by < -0.5) | (by > pitch_wid + 0.5))
    p_dead[out_bounds] = 1.0

    # 2. Long tracking dropouts (>2.0s = 50 frames)
    for s, e in zip(gap_starts, gap_ends):
        if e - s > 50:
            p_dead[s:e] = np.maximum(p_dead[s:e], 0.85)

    # 3. Sustained stationary ball (<0.3 m/s for >=1.0s)
    stationary = valid_ball & (ball_speed < 0.3)
    diff_stat = np.diff(stationary.astype(int))
    stat_starts = np.flatnonzero(diff_stat == 1) + 1
    stat_ends = np.flatnonzero(diff_stat == -1) + 1
    if stationary[0]:
        stat_starts = np.concatenate(([0], stat_starts))
    if stationary[-1]:
        stat_ends = np.concatenate((stat_ends, [N_FRAMES]))
    for s, e in zip(stat_starts, stat_ends):
        if e - s >= 25:
            p_dead[s:e] = np.maximum(p_dead[s:e], 0.75)

    # 4. Smart Tagging XML set-piece restart priors
    sp_windows = {
        "Goal Kick": (450, 0.85),
        "Corner Kick": (350, 0.80),
        "Free Kick": (300, 0.75),
        "Throw-in": (180, 0.70),
        "Kick Off (Start)": (200, 0.95),
        "Kick Off (After Goal)": (300, 0.90),
    }

    if os.path.isfile(st_path):
        for inst in ET.parse(st_path).iter("instance"):
            code = inst.findtext("code")
            if code and "SET PIECE" in code.upper():
                tags = {l.findtext("group"): l.findtext("text") for l in inst.findall("label")}
                tp = tags.get("Type", "")
                if tp in sp_windows:
                    win, prior = sp_windows[tp]
                    start_s = float(inst.findtext("start"))
                    sf = int(round(start_s * FPS))
                    ds = max(0, sf - win)
                    p_dead[ds:sf] = np.maximum(p_dead[ds:sf], prior)

    # 5. Pre-2nd-half kickoff guaranteed dead ball
    p_dead[HALF2_START:KICKOFF_2ND] = 1.0

    return p_dead


def load_baseline_ensemble(results_dir):
    """Load pre-computed multi-scale baselines as an ensemble prior."""
    code_map = {"H": 0, "A": 1, "D": 2, "X": 3}

    def read_col(name):
        path = os.path.join(results_dir, name + ".csv")
        if not os.path.isfile(path):
            return np.full(N_FRAMES, 2, dtype=np.int8)
        df = pd.read_csv(path, dtype={"label": "object"})
        return np.array([code_map.get(x, 2) for x in df["label"].str.strip().str.upper()], dtype=np.int8)

    b25 = read_col("baseline_smooth25")
    b50 = read_col("baseline_smooth50")
    b125 = read_col("baseline_smooth125")

    base_cons_h = ((b25 == 0) * 2 + (b50 == 0) * 2 + (b125 == 0) * 1) / 5.0
    base_cons_a = ((b25 == 1) * 2 + (b50 == 1) * 2 + (b125 == 1) * 1) / 5.0
    base_cons_d = ((b25 == 2) * 2 + (b50 == 2) * 2 + (b125 == 2) * 1) / 5.0

    return base_cons_h, base_cons_a, base_cons_d


def viterbi_decode_half(start_f, end_f, score_h, score_a, score_d):
    """Run Viterbi Dynamic Programming over {H, A, D} for one half."""
    length = end_f - start_f + 1
    log_emit = np.zeros((length, 3), dtype=np.float32)
    log_emit[:, 0] = score_h[start_f : end_f + 1]
    log_emit[:, 1] = score_a[start_f : end_f + 1]
    log_emit[:, 2] = score_d[start_f : end_f + 1]

    # State transition costs: 0=H, 1=A, 2=D
    log_trans = np.array(
        [[0.0, -4.5, -2.8],
         [-4.5, 0.0, -2.8],
         [-2.0, -2.0, 0.0]],
        dtype=np.float32,
    )

    viterbi = np.zeros((length, 3), dtype=np.float32)
    backpointer = np.zeros((length, 3), dtype=np.int32)
    viterbi[0] = log_emit[0]

    for t in range(1, length):
        scores = viterbi[t - 1, :, None] + log_trans
        backpointer[t] = np.argmax(scores, axis=0)
        viterbi[t] = np.max(scores, axis=0) + log_emit[t]

    path = np.zeros(length, dtype=np.int8)
    path[-1] = np.argmax(viterbi[-1])
    for t in range(length - 2, -1, -1):
        path[t] = backpointer[t + 1, path[t + 1]]

    return path


def structured_duration_filter(arr, min_poss=60, min_dead=30, min_micro=25, max_iter=15):
    """Prune micro-possessions and bridge dead-ball dropouts within plausible bounds."""
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

            # 1. Micro flickers (< min_micro)
            for i in range(len(labels)):
                if labels[i] in (H, A) and durs[i] < min_micro:
                    prev_l = labels[i - 1] if i > 0 else D
                    next_l = labels[i + 1] if i + 1 < len(labels) else D
                    cand = (
                        prev_l
                        if (durs[i - 1] if i > 0 else 0) >= (durs[i + 1] if i + 1 < len(labels) else 0)
                        else next_l
                    )
                    if cand == X:
                        cand = D
                    sub[starts[i] : stops[i]] = cand
                    changed = True

            breaks = np.flatnonzero(sub[1:] != sub[:-1]) + 1
            bounds = np.concatenate(([0], breaks, [len(sub)]))
            starts = bounds[:-1]
            stops = bounds[1:]
            labels = sub[starts]
            durs = stops - starts

            # 2. Bridge brief dead balls between same team (< min_dead)
            for i in range(1, len(labels) - 1):
                if labels[i] == D and durs[i] < min_dead:
                    if labels[i - 1] == labels[i + 1] and labels[i - 1] in (H, A):
                        sub[starts[i] : stops[i]] = labels[i - 1]
                        changed = True

            breaks = np.flatnonzero(sub[1:] != sub[:-1]) + 1
            bounds = np.concatenate(([0], breaks, [len(sub)]))
            starts = bounds[:-1]
            stops = bounds[1:]
            labels = sub[starts]
            durs = stops - starts

            # 3. Prune sub-threshold possession spells (< min_poss)
            for i in range(len(labels)):
                if labels[i] in (H, A) and durs[i] < min_poss:
                    prev_l = labels[i - 1] if i > 0 else D
                    next_l = labels[i + 1] if i + 1 < len(labels) else D
                    cand = (
                        prev_l
                        if (durs[i - 1] if i > 0 else 0) >= (durs[i + 1] if i + 1 < len(labels) else 0)
                        else next_l
                    )
                    if cand == X:
                        cand = D
                    sub[starts[i] : stops[i]] = cand
                    changed = True

            if not changed:
                break
        out[start:stop] = sub

    return out


def snap_transitions_to_touches(arr, touch_h, touch_a):
    """Align transition boundaries to physical touch and restart events."""
    out = arr.copy()
    for start, stop in [(HALF1_START, HALF1_END + 1), (HALF2_START, HALF2_END + 1)]:
        sub = out[start:stop]
        breaks = np.flatnonzero(sub[1:] != sub[:-1]) + 1
        new_sub = sub.copy()

        for brk in breaks:
            from_l = sub[brk - 1]
            to_l = sub[brk]
            if from_l == X or to_l == X:
                continue

            abs_brk = start + brk
            lo = max(0, abs_brk - 10)
            hi = min(N_FRAMES - 1, abs_brk + 15)

            # Possession turnover H <-> A
            if from_l in (H, A) and to_l in (H, A):
                tgt_touch = touch_a if to_l == A else touch_h
                win_t = tgt_touch[lo:hi]
                if np.any(win_t > 0):
                    shift = (lo + np.argmax(win_t)) - abs_brk
                    new_brk = brk + shift
                    if 0 < new_brk < len(sub):
                        if shift > 0:
                            new_sub[brk:new_brk] = from_l
                        elif shift < 0:
                            new_sub[new_brk:brk] = to_l
                else:
                    shift = 5
                    new_brk = brk + shift
                    if 0 < new_brk < len(sub):
                        new_sub[brk:new_brk] = from_l

            # Dead-ball restart D -> H/A
            elif from_l == D and to_l in (H, A):
                tgt_touch = touch_h if to_l == H else touch_a
                win_t = tgt_touch[lo:hi]
                if np.any(win_t > 0):
                    shift = (lo + np.argmax(win_t)) - abs_brk
                    new_brk = brk + shift
                    if 0 < new_brk < len(sub):
                        if shift > 0:
                            new_sub[brk:new_brk] = from_l
                        elif shift < 0:
                            new_sub[new_brk:brk] = to_l

        out[start:stop] = new_sub

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="Pristine Cascade.csv", help="Output path for submission CSV")
    parser.add_argument("--data-dir", default=None, help="Root path of MTAH data bundle")
    parser.add_argument("--results-dir", default=None, help="Directory containing baseline smoothed files")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 70)
    print("Pristine Cascade: More Than A Hack 2027 Possession Inference")
    print("=" * 70)

    # 1. Resolve asset paths via games.py
    meta_path, raw_path, homography_path, st_path = games.require(
        ["atd_meta", "atd_raw", "homography", "smart_tagging"], data_dir=args.data_dir
    )
    print(f"[Data] Found tracking metadata: {meta_path}")
    print(f"[Data] Found raw tracking data: {raw_path}")
    print(f"[Data] Found Smart Tagging XML: {st_path}")

    # Results directory for baseline ensemble
    results_dir = args.results_dir or os.path.join(HERE, "evaluation", "results")

    # 2. Stream ATD tracking data
    print("\n[Step 1] Streaming raw EPTS tracking data...")
    t0 = time.time()
    ball_xy, ball_present, player_xy, min_dist_h, min_dist_a, team_of_slot = stream_tracking_data(
        meta_path, raw_path
    )
    print(f"  -> Streamed {N_FRAMES} frames in {time.time()-t0:.2f}s (ball tracked in {np.sum(ball_present)} frames)")

    # 3. Ball reconstruction
    print("\n[Step 2] Reconstructing ball trajectory and computing confidence...")
    t0 = time.time()
    recon_xy, ball_conf, min_dist_h_rec, min_dist_a_rec, gap_starts, gap_ends = reconstruct_ball_trajectory(
        ball_xy, ball_present, player_xy, team_of_slot
    )
    # Merge distance observations
    use_rec = (~ball_present) & (~np.isnan(recon_xy[:, 0]))
    min_dist_h[use_rec] = min_dist_h_rec[use_rec]
    min_dist_a[use_rec] = min_dist_a_rec[use_rec]
    print(f"  -> Reconstructed ball in {np.sum(use_rec)} missing frames in {time.time()-t0:.2f}s")

    # 4. Touch detection & kinematics
    print("\n[Step 3] Detecting touches and ball kinematics...")
    t0 = time.time()
    touch_h, touch_a, touch_ev_h, touch_ev_a, ball_speed, valid_ball = detect_touches_and_kinematics(
        recon_xy, min_dist_h, min_dist_a
    )
    print(f"  -> Detected {np.sum(touch_h > 0)} Home touches and {np.sum(touch_a > 0)} Away touches in {time.time()-t0:.2f}s")

    # 5. Dead-ball model
    print("\n[Step 4] Building dead-ball probability model...")
    t0 = time.time()
    p_dead = build_dead_ball_model(recon_xy, valid_ball, ball_speed, gap_starts, gap_ends, st_path)
    print(f"  -> Dead-ball model constructed in {time.time()-t0:.2f}s")

    # 6. Nearest-player soft evidence & baseline ensemble
    soft_h = 1.0 / (1.0 + np.exp(-np.clip((min_dist_a - min_dist_h) / 1.5, -10, 10)))
    soft_a = 1.0 - soft_h
    base_cons_h, base_cons_a, base_cons_d = load_baseline_ensemble(results_dir)

    # Emission scores
    score_h = 1.5 * touch_ev_h + 1.2 * soft_h + 1.2 * base_cons_h - 2.5 * p_dead
    score_a = 1.5 * touch_ev_a + 1.2 * soft_a + 1.2 * base_cons_a - 2.5 * p_dead
    score_d = 3.0 * (p_dead - 0.35) + 1.5 * base_cons_d - 1.0 * ball_conf

    # 7. Viterbi decoder
    print("\n[Step 5] Running HMM / Viterbi decoder...")
    t0 = time.time()
    pred_codes = np.full(N_FRAMES, X, dtype=np.int8)
    pred_codes[HALF1_START : HALF1_END + 1] = viterbi_decode_half(HALF1_START, HALF1_END, score_h, score_a, score_d)
    pred_codes[HALF2_START : HALF2_END + 1] = viterbi_decode_half(HALF2_START, HALF2_END, score_h, score_a, score_d)
    print(f"  -> Viterbi DP completed in {time.time()-t0:.2f}s")

    # 8. Structured duration filtering & touch alignment
    print("\n[Step 6] Applying structured duration filter and touch alignment...")
    t0 = time.time()
    pred_codes = structured_duration_filter(pred_codes, min_poss=60, min_dead=30, min_micro=25)
    pred_codes = snap_transitions_to_touches(pred_codes, touch_h, touch_a)

    # 9. Match schedule enforcement
    pred_codes[:HALF1_START] = X
    pred_codes[HALF1_END + 1 : HALF2_START] = X
    pred_codes[HALF2_START:KICKOFF_2ND] = D  # Guaranteed pre-2nd-half kickoff D
    pred_codes[HALF2_END + 1 :] = X
    print(f"  -> Post-processing completed in {time.time()-t0:.2f}s")

    # Write output
    out_df = pd.DataFrame({"frame": np.arange(N_FRAMES), "label": [LABELS[c] for c in pred_codes]})
    out_df.to_csv(args.output, index=False)
    print(f"\n[Output] Wrote {len(out_df)} rows to {args.output}")

    # Also keep Pristine_Cascade.csv in sync
    if args.output == "Pristine Cascade.csv":
        out_df.to_csv("Pristine_Cascade.csv", index=False)
        out_df.to_csv(os.path.join(results_dir, "Pristine_Cascade.csv"), index=False)
        out_df.to_csv(os.path.join(results_dir, "Pristine Cascade.csv"), index=False)

    print(f"[Done] Total execution time: {time.time()-t_start:.2f}s")


if __name__ == "__main__":
    main()
