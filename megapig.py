#!/usr/bin/env python3

import os
import math
import random
from datetime import datetime
import concurrent.futures
from collections import defaultdict
from functools import lru_cache
from tqdm import tqdm

# --- ANSI colors ---
RESET  = "\033[0m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"

TARGET = 100

# =========================
# RNG utility
# =========================
def d6():
    return random.randint(1, 6)

# =========================
# EV helpers (memoized)
# =========================
@lru_cache(maxsize=None)
def ev_threshold(remaining: int, target: int, subtotal: int) -> float:
    """
    Expected points for 'roll until subtotal >= min(target, remaining)' from a given subtotal.
    Bust on a 1 yields 0 for the turn.
    """
    cap = min(target, remaining)
    if subtotal >= cap:
        return float(subtotal)
    # E = (1/6)*0 + sum_{r=2..6} (1/6) * E(subtotal + r)
    s = 0.0
    for r in range(2, 7):
        s += ev_threshold(remaining, target, subtotal + r)
    return s / 6.0

@lru_cache(maxsize=None)
def ev_fixed_n(remaining: int, subtotal: int, rolls_left: int) -> float:
    """
    Expected points for 'roll exactly rolls_left more times' (unless bust or you already hit remaining).
    """
    if subtotal >= remaining:
        return float(subtotal)
    if rolls_left == 0:
        return float(subtotal)
    s = 0.0
    for r in range(2, 7):
        s += ev_fixed_n(remaining, subtotal + r, rolls_left - 1)
    return s / 6.0

@lru_cache(maxsize=None)
def ev_roll3_then_20(remaining: int, subtotal: int, phase: int, rolls_left_phase0: int) -> float:
    """
    Expected points for policy:
      Phase 0: up to 3 front-loaded rolls; stop early if subtotal >= remaining or subtotal >= 20.
      Phase 1: then behave like hold-at-20 (capped by remaining).
    """
    if subtotal >= remaining:
        return float(subtotal)

    if phase == 0:
        if subtotal >= 20:
            # go to phase 1 with no need to roll further; but we already hit target, so we would stop
            return float(subtotal)
        if rolls_left_phase0 == 0:
            # switch to phase 1 (threshold 20)
            return ev_threshold(remaining, 20, subtotal)
        # still in phase 0, take a roll
        s = 0.0
        for r in range(2, 7):
            s += ev_roll3_then_20(remaining, subtotal + r, 0, rolls_left_phase0 - 1)
        # r==1 bust => 0
        return s / 6.0
    else:
        # phase 1 == simple threshold 20 (capped)
        return ev_threshold(remaining, 20, subtotal)

# =========================
# Turn simulators returning dict(points, rolls, ev)
# =========================
def do_threshold_turn(state, base_target: int):
    """
    Simulate a single turn for 'roll until >= min(base_target, remaining)'.
    Returns dict with actual result and EV for this (state, target).
    """
    remaining = TARGET - state["me"]
    target = min(base_target, remaining)
    # Pre-compute EV from subtotal 0 for reporting/offset
    ev = ev_threshold(remaining, target, 0)

    total, rolls = 0, 0
    while True:
        r = d6()
        rolls += 1
        if r == 1:
            return {"points": 0, "rolls": rolls, "ev": ev}
        total += r
        if total >= target:
            return {"points": total, "rolls": rolls, "ev": ev}

def do_fixed_n_turn(state, n: int):
    """
    Simulate 'roll exactly n times' unless bust or you already reach remaining.
    """
    remaining = TARGET - state["me"]
    ev = ev_fixed_n(remaining, 0, n)

    total, rolls = 0, 0
    for _ in range(n):
        r = d6()
        rolls += 1
        if r == 1:
            return {"points": 0, "rolls": rolls, "ev": ev}
        total += r
        if total >= remaining:
            return {"points": total, "rolls": rolls, "ev": ev}
    return {"points": total, "rolls": rolls, "ev": ev}

def do_roll3_then_20_turn(state):
    """
    Simulate the 'R3_then_20' policy.
    """
    remaining = TARGET - state["me"]
    ev = ev_roll3_then_20(remaining, 0, 0, 3)

    total, rolls = 0, 0
    # Phase 0: up to 3 rolls
    for _ in range(3):
        r = d6()
        rolls += 1
        if r == 1:
            return {"points": 0, "rolls": rolls, "ev": ev}
        total += r
        if total >= remaining or total >= 20:
            return {"points": total, "rolls": rolls, "ev": ev}
    # Phase 1: threshold 20
    target = min(20, remaining)
    while total < target:
        r = d6()
        rolls += 1
        if r == 1:
            return {"points": 0, "rolls": rolls, "ev": ev}
        total += r
    return {"points": total, "rolls": rolls, "ev": ev}

# =========================
# Strategies (all return dict)
# =========================
def hold_at_15(state): return do_threshold_turn(state, 15)
def hold_at_20(state): return do_threshold_turn(state, 20)
def hold_at_25(state): return do_threshold_turn(state, 25)

def hold25_with_pressure(state):
    base = 30 if state["opp"] >= 80 else 25
    return do_threshold_turn(state, base)

def random_hold(state):
    base = random.choice([15, 20, 25, 30])
    return do_threshold_turn(state, base)

def adaptive_hold_at_15_20_25(state):
    me, opp = state["me"], state["opp"]
    diff = me - opp
    if diff > 20:
        return do_threshold_turn(state, 15)
    elif diff >= -20:
        return do_threshold_turn(state, 20)
    else:
        return do_threshold_turn(state, 25)

def cautious_endgame(state):
    remaining = TARGET - state["me"]
    if remaining <= 15:
        return do_threshold_turn(state, 10)
    elif remaining <= 30:
        return do_threshold_turn(state, 15)
    else:
        return do_threshold_turn(state, 25)

def pressure_play(state):
    return do_threshold_turn(state, 30 if state["opp"] >= 80 else 20)

def roll4(state): return do_fixed_n_turn(state, 4)
def roll5(state): return do_fixed_n_turn(state, 5)
def roll6(state): return do_fixed_n_turn(state, 6)

def phase_20_20_15(state):
    remaining = TARGET - state["me"]
    if remaining > 60:  return do_threshold_turn(state, 20)
    if remaining > 30:  return do_threshold_turn(state, 20)
    return do_threshold_turn(state, 15)

def phase_25_20_15(state):
    remaining = TARGET - state["me"]
    if remaining > 60:  return do_threshold_turn(state, 25)
    if remaining > 30:  return do_threshold_turn(state, 20)
    return do_threshold_turn(state, 15)

def gap_phase_hybrid(state):
    me, opp = state["me"], state["opp"]
    remaining = TARGET - me
    gap = me - opp
    if remaining > 60:
        base = 25
    elif remaining > 30:
        base = 22
    else:
        base = 15
    if gap >= 20:
        base -= 3
    elif gap <= -20:
        base += 5
    base = max(10, min(35, base))
    return do_threshold_turn(state, base)

def pressure_tiers(state):
    opp = state["opp"]
    if opp >= 90: return do_threshold_turn(state, 32)
    if opp >= 80: return do_threshold_turn(state, 28)
    if opp >= 70: return do_threshold_turn(state, 25)
    return do_threshold_turn(state, 20)

def capped_endgame_25_15(state):
    remaining = TARGET - state["me"]
    base = 25 if remaining > 30 else 15
    return do_threshold_turn(state, base)

def roll3_then_20(state):
    return do_roll3_then_20_turn(state)

def linear_gap_policy(state):
    me, opp = state["me"], state["opp"]
    gap = me - opp
    base = 22
    a, b = 0.10, 0.20
    target = base - a * max(gap, 0) + b * max(-gap, 0)
    target = int(max(10, min(32, target)))
    return do_threshold_turn(state, target)

def surpass_opp_plus10(state):
    me, opp = state["me"], state["opp"]
    goal = max(15, min(30, (opp - me) + 10))
    return do_threshold_turn(state, goal)

def variance_cap(state):
    me, opp = state["me"], state["opp"]
    lead = me - opp
    if lead >= 30:  return do_threshold_turn(state, 15)
    if lead >= 15:  return do_threshold_turn(state, 18)
    if lead <= -30: return do_threshold_turn(state, 30)
    if lead <= -15: return do_threshold_turn(state, 25)
    return do_threshold_turn(state, 22)

def phase_plus_pressure(state):
    me, opp = state["me"], state["opp"]
    remaining = TARGET - me
    base = 25 if remaining > 40 else 18
    if opp >= 85: base += 4
    elif opp >= 75: base += 2
    return do_threshold_turn(state, min(base, 32))

# =========================
# Active engines
# =========================
PLAYERS = [
    {"name": "H@15",         "algorithm": hold_at_15},
    {"name": "H@20",         "algorithm": hold_at_20},
    {"name": "H@25",         "algorithm": hold_at_25},
    {"name": "H@25P",        "algorithm": hold25_with_pressure},
    {"name": "ADP",          "algorithm": adaptive_hold_at_15_20_25},
    {"name": "RAND",         "algorithm": random_hold},
    {"name": "CEG",          "algorithm": cautious_endgame},
    {"name": "PP",           "algorithm": pressure_play},
    {"name": "R4",           "algorithm": roll4},
    {"name": "R5",           "algorithm": roll5},
    {"name": "R6",           "algorithm": roll6},
    {"name": "P20_20_15",    "algorithm": phase_20_20_15},
    {"name": "P25_20_15",    "algorithm": phase_25_20_15},
    {"name": "GAP",          "algorithm": gap_phase_hybrid},
    {"name": "PT",           "algorithm": pressure_tiers},
    {"name": "CEG25_15",     "algorithm": capped_endgame_25_15},
    {"name": "R3_then_20",   "algorithm": roll3_then_20},
    {"name": "LIN_GAP",      "algorithm": linear_gap_policy},
    {"name": "SURP+10",      "algorithm": surpass_opp_plus10},
    {"name": "VAR_CAP",      "algorithm": variance_cap},
    {"name": "PH+PR",        "algorithm": phase_plus_pressure},
]

# =========================
# Core mechanics (now also returns per-game turn stats)
# =========================
def pig_game(player1, player2):
    score1, score2 = 0, 0
    name1, name2 = player1["name"], player2["name"]

    # accumulate per-turn aggregates for this game
    agg = {
        name1: {"points": 0.0, "rolls": 0.0, "ev": 0.0, "turns": 0},
        name2: {"points": 0.0, "rolls": 0.0, "ev": 0.0, "turns": 0},
    }

    current = 1
    while score1 < TARGET and score2 < TARGET:
        if current == 1:
            state = {"me": score1, "opp": score2}
            res = player1["algorithm"](state)  # dict
            pts, rolls, ev = res["points"], res["rolls"], res["ev"]
            score1 += pts
            agg[name1]["points"] += pts
            agg[name1]["rolls"]  += rolls
            agg[name1]["ev"]     += ev
            agg[name1]["turns"]  += 1
            current = 2
        else:
            state = {"me": score2, "opp": score1}
            res = player2["algorithm"](state)
            pts, rolls, ev = res["points"], res["rolls"], res["ev"]
            score2 += pts
            agg[name2]["points"] += pts
            agg[name2]["rolls"]  += rolls
            agg[name2]["ev"]     += ev
            agg[name2]["turns"]  += 1
            current = 1

    winner = 1 if score1 >= TARGET else 2
    return winner, agg

# =========================
# Parallel worker
# =========================
def play_matchup(args):
    player1, player2, num_games = args
    n1, n2 = player1["name"], player2["name"]

    wins = {n1: 0, n2: 0}

    # sums over ALL games (for averaging later)
    sums = {
        n1: {"points": 0.0, "rolls": 0.0, "ev": 0.0, "turns": 0},
        n2: {"points": 0.0, "rolls": 0.0, "ev": 0.0, "turns": 0},
    }

    # per-process random seed
    random.seed(datetime.now().timestamp() + os.getpid())

    for _ in range(num_games):
        winner, agg = pig_game(player1, player2)
        wins[n1] += 1 if winner == 1 else 0
        wins[n2] += 1 if winner == 2 else 0

        # accumulate per-game aggregates
        for nm in (n1, n2):
            sums[nm]["points"] += agg[nm]["points"]
            sums[nm]["rolls"]  += agg[nm]["rolls"]
            sums[nm]["ev"]     += agg[nm]["ev"]
            sums[nm]["turns"]  += agg[nm]["turns"]

    # compute per-directed averages (points/turn, rolls/turn, offset/turn)
    avgs = {
        n1: {
            "points_per_turn": (sums[n1]["points"] / sums[n1]["turns"]) if sums[n1]["turns"] else 0.0,
            "rolls_per_turn":  (sums[n1]["rolls"]  / sums[n1]["turns"]) if sums[n1]["turns"] else 0.0,
            "offset_per_turn": ((sums[n1]["points"] - sums[n1]["ev"]) / sums[n1]["turns"]) if sums[n1]["turns"] else 0.0,
        },
        n2: {
            "points_per_turn": (sums[n2]["points"] / sums[n2]["turns"]) if sums[n2]["turns"] else 0.0,
            "rolls_per_turn":  (sums[n2]["rolls"]  / sums[n2]["turns"]) if sums[n2]["turns"] else 0.0,
            "offset_per_turn": ((sums[n2]["points"] - sums[n2]["ev"]) / sums[n2]["turns"]) if sums[n2]["turns"] else 0.0,
        },
    }

    payload = {"wins": wins, "avgs": avgs}
    return (n1, n2, payload)

# =========================
# Main
# =========================
if __name__ == "__main__":
    num_games   = 100000
    num_cpus    = os.cpu_count() or 1
    max_workers = max(1, int(num_cpus * 0.9))
    print(f"Detected {num_cpus} CPUs, using {max_workers} workers...")

    tasks = []
    for i in range(len(PLAYERS)):
        for j in range(len(PLAYERS)):
            if i != j:
                tasks.append((PLAYERS[i], PLAYERS[j], num_games))

    # overall tallies
    results = {p["name"]: 0 for p in PLAYERS}

    # head-to-head collections (we average across the two directions we run)
    win_h2h      = defaultdict(lambda: defaultdict(list))  # percentages
    points_h2h   = defaultdict(lambda: defaultdict(list))  # avg points per turn
    rolls_h2h    = defaultdict(lambda: defaultdict(list))  # avg rolls per turn
    offset_h2h   = defaultdict(lambda: defaultdict(list))  # avg EV offset per turn

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for p1, p2, payload in tqdm(
            executor.map(play_matchup, tasks),
            total=len(tasks),
            desc="Simulating matchups",
        ):
            wins = payload["wins"]
            avgs = payload["avgs"]

            # overall wins per player
            results[p1] += wins[p1]
            results[p2] += wins[p2]

            # head-to-head win%
            total = wins[p1] + wins[p2]
            win_h2h[p1][p2].append((wins[p1] / total) * 100.0)
            win_h2h[p2][p1].append((wins[p2] / total) * 100.0)

            # head-to-head averages per turn (directed)
            points_h2h[p1][p2].append(avgs[p1]["points_per_turn"])
            points_h2h[p2][p1].append(avgs[p2]["points_per_turn"])

            rolls_h2h[p1][p2].append(avgs[p1]["rolls_per_turn"])
            rolls_h2h[p2][p1].append(avgs[p2]["rolls_per_turn"])

            offset_h2h[p1][p2].append(avgs[p1]["offset_per_turn"])
            offset_h2h[p2][p1].append(avgs[p2]["offset_per_turn"])

    # average across the two directions (and any chunking)
    def avg2(dct):
        return {
            p: {q: (sum(vs) / len(vs) if vs else 0.0) for q, vs in row.items()}
            for p, row in dct.items()
        }

    win_h2h_avg    = avg2(win_h2h)
    points_h2h_avg = avg2(points_h2h)
    rolls_h2h_avg  = avg2(rolls_h2h)
    offset_h2h_avg = avg2(offset_h2h)

    games_per_player = (len(PLAYERS) - 1) * num_games * 2
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    # rank map
    rank_map = {player: idx + 1 for idx, (player, _) in enumerate(sorted_results)}

    # -------- Summary Table --------
    print("\nFinal Results (with 95% CI)")
    print(f"{'Rank':<5} {'Player':<12} {'Wins':>10} {'Win %':>10} {'±CI':>8}")
    print("-" * 50)
    for rank, (player, wins) in enumerate(sorted_results, start=1):
        p = wins / games_per_player
        margin = 1.96 * math.sqrt(p * (1 - p) / games_per_player)
        print(f"#{rank:<4} {player:<12} {wins:>10} {p*100:>9.2f}% {margin*100:>7.2f}%")

    # -------- Matrix Printers --------
    def print_matrix(title, matrix, fmt="{:>.1f}%", colorize_win=False, is_percentage=False):
        print(f"\n{title}")

        names = [p for p, _ in sorted_results]
        labels = [f"#{rank_map[n]} {n}" for n in names]

        MIN_COL = 12
        col_width = max(MIN_COL, max(len(s) for s in labels) + 2)
        row_label_width = col_width

        header = f"{'':<{row_label_width}}" + "".join(f"{s:>{col_width}}" for s in labels)
        print(header)

        for p in names:
            left = f"{f'#{rank_map[p]} {p}':<{row_label_width}}"
            row = [left]
            for q in names:
                if p == q:
                    cell = f"{'--':>{col_width}}"
                else:
                    val = matrix[p].get(q, 0.0)
                    # formatting
                    if is_percentage:
                        base = f"{val:>{col_width-1}.1f}%"
                    else:
                        base = f"{val:>{col_width-2}.2f} "

                    if not colorize_win:
                        cell = base
                    else:
                        # color by advantage around 50% with ±1% yellow band
                        diff = val - 50.0
                        if abs(diff) < 1.0:
                            cell = YELLOW + base + RESET
                        elif diff > 0:
                            cell = GREEN + base + RESET
                        else:
                            cell = RED + base + RESET
                row.append(cell)
            print("".join(row))

    # 1) Win % matrix (same as before, with coloring)
    print_matrix("Head-to-Head Win % Matrix", win_h2h_avg, colorize_win=True, is_percentage=True)

    # 2) Average Score per Turn
    print_matrix("Head-to-Head Avg Points per Turn", points_h2h_avg, fmt="{:>.2f}", colorize_win=False, is_percentage=False)

    # 3) Average Number of Dice Rolled per Turn
    print_matrix("Head-to-Head Avg Rolls per Turn", rolls_h2h_avg, fmt="{:>.2f}", colorize_win=False, is_percentage=False)

    # 4) Average Offset from EV per Turn (points - EV)
    print_matrix("Head-to-Head Avg EV Offset per Turn (points - EV)", offset_h2h_avg, fmt="{:>+.3f}", colorize_win=False, is_percentage=False)
