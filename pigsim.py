#!/usr/bin/env python3

import random
from datetime import datetime
import os
import concurrent.futures
from tqdm import tqdm
import math
from collections import defaultdict

# --- ANSI colors ---
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"

# =========================
# Utilities
# =========================
def d6():
    return random.randint(1, 6)

def roll_until_capped(state, base_target):
    """
    Roll until reaching the smaller of:
      - base_target (engine’s intended threshold)
      - remaining points needed to win (100 - me)
    Stop immediately once the cap is reached; bust on 1.
    """
    remaining = 100 - state["me"]
    target = min(base_target, remaining)
    total = 0
    while True:
        r = d6()
        if r == 1:
            return 0
        total += r
        if total >= target:
            return total

# Fixed-roll, but stop early if we’ve already reached the win.
def _roll_fixed_safe(n, state):
    remaining = 100 - state["me"]
    total = 0
    for _ in range(n):
        r = d6()
        if r == 1:
            return 0
        total += r
        if total >= remaining:
            return total
    return total

# =========================
# Strategies (all capped)
# =========================
def hold_at_15(state): return roll_until_capped(state, 15)
def hold_at_20(state): return roll_until_capped(state, 20)
def hold_at_25(state): return roll_until_capped(state, 25)

def hold25_with_pressure(state):
    """
    Human-easy: hold at 25, but if the opponent is 80+,
    push a bit harder (hold at 30). Always endgame-capped.
    """
    base = 30 if state["opp"] >= 80 else 25
    return roll_until_capped(state, base)

def random_hold(state):
    return roll_until_capped(state, random.choice([15, 20, 25, 30]))

def adaptive_hold_at_15_20_25(state):
    me, opp = state["me"], state["opp"]
    diff = me - opp
    if diff > 20:              # leading big → safer
        return roll_until_capped(state, 15)
    elif diff >= -20:          # close game
        return roll_until_capped(state, 20)
    else:                      # trailing big → push
        return roll_until_capped(state, 25)

def cautious_endgame(state):
    remaining = 100 - state["me"]
    if remaining <= 15:
        return roll_until_capped(state, 10)
    elif remaining <= 30:
        return roll_until_capped(state, 15)
    else:
        return roll_until_capped(state, 25)

def pressure_play(state):
    return roll_until_capped(state, 30 if state["opp"] >= 80 else 20)

def roll4(state): return _roll_fixed_safe(4, state)
def roll5(state): return _roll_fixed_safe(5, state)
def roll6(state): return _roll_fixed_safe(6, state)

def phase_20_20_15(state):
    remaining = 100 - state["me"]
    if remaining > 60:   return roll_until_capped(state, 20)
    if remaining > 30:   return roll_until_capped(state, 20)
    return roll_until_capped(state, 15)

def phase_25_20_15(state):
    remaining = 100 - state["me"]
    if remaining > 60:   return roll_until_capped(state, 25)
    if remaining > 30:   return roll_until_capped(state, 20)
    return roll_until_capped(state, 15)

def gap_phase_hybrid(state):
    me, opp = state["me"], state["opp"]
    remaining = 100 - me
    gap = me - opp
    if remaining > 60: base = 25
    elif remaining > 30: base = 22
    else: base = 15
    if gap >= 20:   base -= 3
    elif gap <= -20: base += 5
    base = max(10, min(35, base))
    return roll_until_capped(state, base)

def pressure_tiers(state):
    opp = state["opp"]
    if opp >= 90:   return roll_until_capped(state, 32)
    if opp >= 80:   return roll_until_capped(state, 28)
    if opp >= 70:   return roll_until_capped(state, 25)
    return roll_until_capped(state, 20)

def capped_endgame_25_15(state):
    remaining = 100 - state["me"]
    base = 25 if remaining > 30 else 15
    return roll_until_capped(state, base)

def roll3_then_20(state):
    # three front-loaded rolls, stopping if we’ve reached remaining/20; then continue to min(20, remaining)
    total = 0
    remaining = 100 - state["me"]
    for _ in range(3):
        r = d6()
        if r == 1: return 0
        total += r
        if total >= remaining or total >= 20:
            return total
    target = min(20, remaining)
    while total < target:
        r = d6()
        if r == 1: return 0
        total += r
    return total

def linear_gap_policy(state):
    me, opp = state["me"], state["opp"]
    gap = me - opp
    base = 22
    a, b = 0.10, 0.20  # down 0.10 per lead point, up 0.20 per deficit point
    target = base - a*max(gap, 0) + b*max(-gap, 0)
    target = int(max(10, min(32, target)))
    return roll_until_capped(state, target)

def surpass_opp_plus10(state):
    me, opp = state["me"], state["opp"]
    goal = max(15, min(30, (opp - me) + 10))
    return roll_until_capped(state, goal)

def variance_cap(state):
    me, opp = state["me"], state["opp"]
    lead = me - opp
    if lead >= 30:  return roll_until_capped(state, 15)
    if lead >= 15:  return roll_until_capped(state, 18)
    if lead <= -30: return roll_until_capped(state, 30)
    if lead <= -15: return roll_until_capped(state, 25)
    return roll_until_capped(state, 22)

def phase_plus_pressure(state):
    me, opp = state["me"], state["opp"]
    remaining = 100 - me
    base = 25 if remaining > 40 else 18
    if opp >= 85: base += 4
    elif opp >= 75: base += 2
    return roll_until_capped(state, min(base, 32))

# =========================
# Active engines
# (Removed redundants: CEG20 is identical to capped H@20.)
# =========================
PLAYERS = [
    {"name": "H@15", "algorithm": hold_at_15},
    {"name": "H@20", "algorithm": hold_at_20},
    {"name": "H@25", "algorithm": hold_at_25},
    {"name": "ADP", "algorithm": adaptive_hold_at_15_20_25},
    {"name": "RAND", "algorithm": random_hold},
    {"name": "CEG", "algorithm": cautious_endgame},
    {"name": "PP", "algorithm": pressure_play},
    {"name": "R4", "algorithm": roll4},
    {"name": "R5", "algorithm": roll5},
    {"name": "R6", "algorithm": roll6},
    {"name": "P20_20_15", "algorithm": phase_20_20_15},
    {"name": "P25_20_15", "algorithm": phase_25_20_15},
    {"name": "GAP", "algorithm": gap_phase_hybrid},
    {"name": "PT", "algorithm": pressure_tiers},
    {"name": "CEG25_15", "algorithm": capped_endgame_25_15},
    {"name": "R3_then_20", "algorithm": roll3_then_20},
    {"name": "LIN_GAP", "algorithm": linear_gap_policy},
    {"name": "SURP+10", "algorithm": surpass_opp_plus10},
    {"name": "VAR_CAP", "algorithm": variance_cap},
    {"name": "PH+PR", "algorithm": phase_plus_pressure},
    {"name": "H@25P", "algorithm": hold25_with_pressure},
]

# =========================
# Core mechanics
# =========================
def pig_game(player1, player2):
    score1, score2 = 0, 0
    current = 1
    while score1 < 100 and score2 < 100:
        if current == 1:
            state = {"me": score1, "opp": score2}
            score1 += player1["algorithm"](state)
            current = 2
        else:
            state = {"me": score2, "opp": score1}
            score2 += player2["algorithm"](state)
            current = 1
    return 1 if score1 >= 100 else 2

# =========================
# Parallel worker
# =========================
def play_matchup(args):
    player1, player2, num_games = args
    wins = {player1["name"]: 0, player2["name"]: 0}
    random.seed(datetime.now().timestamp() + os.getpid())
    for _ in range(num_games):
        winner = pig_game(player1, player2)
        if winner == 1:
            wins[player1["name"]] += 1
        else:
            wins[player2["name"]] += 1
    return (player1["name"], player2["name"], wins)

# =========================
# Main
# =========================
if __name__ == "__main__":
    num_games = 100000
    num_cpus = os.cpu_count() or 1
    max_workers = max(1, int(num_cpus * 0.9))
    print(f"Detected {num_cpus} CPUs, using {max_workers} workers...")

    tasks = []
    for i in range(len(PLAYERS)):
        for j in range(len(PLAYERS)):
            if i != j:
                tasks.append((PLAYERS[i], PLAYERS[j], num_games))

    results = {player["name"]: 0 for player in PLAYERS}
    head_to_head = defaultdict(lambda: defaultdict(list))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for p1, p2, wins in tqdm(
            executor.map(play_matchup, tasks),
            total=len(tasks),
            desc="Simulating matchups",
        ):
            results[p1] += wins[p1]
            results[p2] += wins[p2]
            total = wins[p1] + wins[p2]
            head_to_head[p1][p2].append(wins[p1] / total * 100)
            head_to_head[p2][p1].append(wins[p2] / total * 100)

    # average into final percentages
    head_to_head_avg = {
        p: {q: (sum(vals) / len(vals) if vals else 0) for q, vals in row.items()}
        for p, row in head_to_head.items()
    }

    games_per_player = (len(PLAYERS) - 1) * num_games * 2
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    # rank map
    rank_map = {player: idx + 1 for idx, (player, _) in enumerate(sorted_results)}

    print("\nFinal Results (with 95% CI)")
    print(f"{'Rank':<5} {'Player':<12} {'Wins':>10} {'Win %':>10} {'±CI':>8}")
    print("-" * 50)
    for rank, (player, wins) in enumerate(sorted_results, start=1):
        p = wins / games_per_player
        margin = 1.96 * math.sqrt(p * (1 - p) / games_per_player)
        print(f"#{rank:<4} {player:<12} {wins:>10} {p*100:>9.2f}% {margin*100:>7.2f}%")

    # --- head-to-head matrix (ranked, auto-sized columns, ±1% yellow band) ---
    print("\nHead-to-Head Win % Matrix")

    names = [p for p, _ in sorted_results]
    col_labels = [f"#{rank_map[n]} {n}" for n in names]
    row_labels = col_labels[:]

    MIN_COL = 12
    col_width = max(MIN_COL, max(len(s) for s in col_labels) + 2)
    row_label_width = max(MIN_COL, max(len(s) for s in row_labels) + 2)

    header = f"{'':<{row_label_width}}" + "".join(f"{s:>{col_width}}" for s in col_labels)
    print(header)

    for p in names:
        left = f"{f'#{rank_map[p]} {p}':<{row_label_width}}"
        row = [left]
        for q in names:
            if p == q:
                cell = f"{'--':>{col_width}}"
            else:
                val = head_to_head_avg[p][q]
                diff = val - 50
                base = f"{val:>{col_width-1}.1f}%"
                if abs(diff) < 1.0:
                    cell = YELLOW + base + RESET
                elif diff > 0:
                    cell = GREEN + base + RESET
                else:
                    cell = RED + base + RESET
            row.append(cell)
        print("".join(row))
