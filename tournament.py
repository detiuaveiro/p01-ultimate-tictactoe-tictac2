"""Headless tournament runner using direct Bitboard simulation (no WebSocket overhead).

Supports multiple baseline strategies and computes Elo ratings.
Uses ThreadPoolExecutor for both intra-game search parallelism and inter-game concurrency.
"""

import os
import random
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import IntEnum

from agents.lib.state import (
    BitboardState,
    evaluate_state,
    run_search,
)

# Shared executor for intra-game search parallelism
_SEARCH_EXECUTOR = ThreadPoolExecutor()


# --- Strategy definitions ---

class Strategy(IntEnum):
    """Available agent strategies for tournament play."""

    RANDOM = 0
    GREEDY = 1
    MINIMAX_D2 = 2
    MINIMAX_D4 = 3
    ALPHABETA_1S = 4
    ALPHABETA_4S = 5
    RUST_SOLVER = 6


STRATEGY_LABELS = {
    Strategy.RANDOM: "Random",
    Strategy.GREEDY: "Greedy",
    Strategy.MINIMAX_D2: "Minimax-D2",
    Strategy.MINIMAX_D4: "Minimax-D4",
    Strategy.ALPHABETA_1S: "AlphaBeta-1s",
    Strategy.ALPHABETA_4S: "AlphaBeta-4s",
    Strategy.RUST_SOLVER: "RustSolver",
}

# Time budgets for search-based strategies
STRATEGY_TIME_LIMITS = {
    Strategy.MINIMAX_D2: 0.05,
    Strategy.MINIMAX_D4: 0.2,
    Strategy.ALPHABETA_1S: 1.0,
    Strategy.ALPHABETA_4S: 4.0,
    Strategy.RUST_SOLVER: 1.0,
}


def pick_move(state: BitboardState, player_id: int, strategy: Strategy) -> tuple[int, int]:
    """Select a move according to the given strategy."""
    legal_moves = state.get_legal_moves()
    if not legal_moves:
        return (0, 0)

    if strategy == Strategy.RANDOM:
        return random.choice(legal_moves)

    if strategy == Strategy.GREEDY:
        best_score = float("-inf")
        best = legal_moves[0]
        for move in legal_moves:
            child = state.clone()
            child.apply_move(player_id, move[0], move[1])
            score = evaluate_state(child, player_id)
            if score > best_score:
                best_score = score
                best = move
        return best

    # Time-limited iterative deepening with parallel root search
    if strategy == Strategy.RUST_SOLVER:
        return rust_solver_pick_move(state, player_id, STRATEGY_TIME_LIMITS[strategy])

    return run_search(
        state, STRATEGY_TIME_LIMITS[strategy], player_id,
        executor=_SEARCH_EXECUTOR,
    )


# --- Rust Solver Integration ---

RUST_SOLVER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "external/ultimattt/target/release/ultimattt",
)


def state_to_rust_notation(state: BitboardState, player_id: int) -> str:
    """Convert BitboardState to the notation used by Nelson Elhage's solver."""
    player = "X" if player_id == 1 else "O"

    # Global board state
    global_board = ""
    for b in range(9):
        if state.active_macro == b:
            global_board += "@"
        else:
            bit = 1 << (8 - b)
            if state.macro_p1 & bit:
                global_board += "X"
            elif state.macro_p2 & bit:
                global_board += "O"
            elif state.macro_draw & bit:
                global_board += "#"
            else:
                global_board += "."

    # Local board states
    local_boards = []
    for b in range(9):
        board_str = ""
        for s in range(9):
            bit = 1 << (8 - s)
            if state.boards_p1[b] & bit:
                board_str += "X"
            elif state.boards_p2[b] & bit:
                board_str += "O"
            else:
                board_str += "."
        local_boards.append(board_str)

    return f"{player};{global_board};{'/'.join(local_boards)}"


def rust_solver_pick_move(
    state: BitboardState,
    player_id: int,
    time_limit: float = 1.0,
) -> tuple[int, int]:
    """Call the Nelson Elhage's Rust solver via subprocess."""
    if not os.path.exists(RUST_SOLVER_PATH):
        # Fallback to random if binary not found
        return random.choice(state.get_legal_moves())

    # Format time limit for the Rust solver (it expects integer durations like '1s' or '500ms')
    ms = int(time_limit * 1000)
    limit_str = f"{ms}ms"

    notation = state_to_rust_notation(state, player_id)
    cmd = [
        RUST_SOLVER_PATH,
        "analyze",
        "--engine=minimax",
        f"--limit={limit_str}",
        notation,
    ]

    try:
        # Run the solver, ensuring it doesn't print to stderr unless there's a real issue
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Look for the last "move=..." line in stdout
        move_line = None
        for line in reversed(result.stdout.splitlines()):
            if line.startswith("move="):
                move_line = line
                break

        if move_line:
            # Format: "move=ae"
            move_str = move_line.split("=")[1]
            macro_idx = ord(move_str[0]) - ord('a')
            micro_idx = ord(move_str[1]) - ord('a')
            return (macro_idx, micro_idx)
    except subprocess.CalledProcessError as e:
        # Don't spam errors if the process was interrupted by the user (SIGINT/SIGTERM)
        if e.returncode < 0:
            return random.choice(state.get_legal_moves())
        print(f"Rust solver error: {e}")
    except Exception as e:
        if not isinstance(e, KeyboardInterrupt):
            print(f"Rust solver error: {e}")

    return random.choice(state.get_legal_moves())


# --- Game simulation ---

DRAW = 3
MAX_GAME_MOVES = 81
BOARD_SIDE = 3


def simulate_game(strategy_p1: Strategy, strategy_p2: Strategy) -> int:
    """Play a full game and return the winner (1, 2, or 3 for draw)."""
    state = BitboardState()
    strategies = {1: strategy_p1, 2: strategy_p2}
    current_player = 1

    for _ in range(MAX_GAME_MOVES):
        is_over, winner = state.is_terminal()
        if is_over:
            return winner

        move = pick_move(state, current_player, strategies[current_player])
        state.apply_move(current_player, move[0], move[1])
        current_player = BOARD_SIDE - current_player

    _, winner = state.is_terminal()
    return winner if winner != 0 else DRAW


# --- Elo rating ---

ELO_INITIAL = 1000
ELO_K = 32


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


@dataclass
class TournamentResults:
    """Accumulated results from a round-robin tournament."""

    ratings: dict[Strategy, float] = field(default_factory=lambda: {s: ELO_INITIAL for s in Strategy})
    wins: dict[Strategy, int] = field(default_factory=lambda: defaultdict(int))
    losses: dict[Strategy, int] = field(default_factory=lambda: defaultdict(int))
    draws: dict[Strategy, int] = field(default_factory=lambda: defaultdict(int))
    matchup_wins: dict[tuple[Strategy, Strategy], int] = field(default_factory=lambda: defaultdict(int))
    matchup_total: dict[tuple[Strategy, Strategy], int] = field(default_factory=lambda: defaultdict(int))
    lock: threading.Lock = field(default_factory=threading.Lock)


def update_elo(
    results: TournamentResults,
    player_a: Strategy,
    player_b: Strategy,
    winner: int,
) -> None:
    """Update Elo ratings based on a single game result. Thread-safe."""
    with results.lock:
        ra = results.ratings[player_a]
        rb = results.ratings[player_b]
        ea = expected_score(ra, rb)
        eb = expected_score(rb, ra)

        if winner == 1:
            sa, sb = 1.0, 0.0
            results.wins[player_a] += 1
            results.losses[player_b] += 1
            results.matchup_wins[(player_a, player_b)] += 1
        elif winner == 2:
            sa, sb = 0.0, 1.0
            results.wins[player_b] += 1
            results.losses[player_a] += 1
            results.matchup_wins[(player_b, player_a)] += 1
        else:
            sa, sb = 0.5, 0.5
            results.draws[player_a] += 1
            results.draws[player_b] += 1

        results.ratings[player_a] = ra + ELO_K * (sa - ea)
        results.ratings[player_b] = rb + ELO_K * (sb - eb)
        results.matchup_total[(player_a, player_b)] += 1


# --- Tournament runner ---

def _run_single_game(
    p1_strat: Strategy,
    p2_strat: Strategy,
    results: TournamentResults,
) -> None:
    """Simulate one game and update results. Designed to run in a thread."""
    winner = simulate_game(p1_strat, p2_strat)
    update_elo(results, p1_strat, p2_strat, winner)


def run_tournament(
    strategies: list[Strategy],
    games_per_matchup: int = 100,
    max_workers: int | None = None,
) -> TournamentResults:
    """Run a round-robin tournament between all strategy pairs."""
    import sys
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    if gil_enabled:
        print("\n" + "!" * 60)
        print("WARNING: Global Interpreter Lock (GIL) is ENABLED.")
        print("Parallel search will be throttled to a single core.")
        print("To fix this, use the free-threaded Python build (e.g., python3.14t).")
        print("!" * 60 + "\n")

    if max_workers is None:
        max_workers = os.cpu_count() or 4

    results = TournamentResults()
    total_matchups = len(strategies) * (len(strategies) - 1) // 2
    total_games = total_matchups * games_per_matchup * 2  # Both sides

    print(f"Tournament: {len(strategies)} strategies, {games_per_matchup} games/matchup/side")
    print(f"Total games: {total_games}, workers: {max_workers}")
    print("-" * 60)

    start = time.time()
    game_count = 0
    game_count_lock = threading.Lock()

    # Build the full list of games to play
    games: list[tuple[Strategy, Strategy]] = []
    for i, s1 in enumerate(strategies):
        for s2 in strategies[i + 1:]:
            for side in range(2):
                p1_strat = s1 if side == 0 else s2
                p2_strat = s2 if side == 0 else s1
                for _ in range(games_per_matchup):
                    games.append((p1_strat, p2_strat))

    # Use all available cores for inter-game parallelism
    effective_workers = max_workers

    try:
        with ThreadPoolExecutor(max_workers=effective_workers) as game_executor:
            futures = {
                game_executor.submit(_run_single_game, p1, p2, results): (p1, p2)
                for p1, p2 in games
            }

            for future in as_completed(futures):
                future.result()  # Propagate exceptions
                with game_count_lock:
                    game_count += 1
                    if game_count % 50 == 0:
                        elapsed = time.time() - start
                        rate = game_count / elapsed if elapsed > 0 else 0
                        p1, p2 = futures[future]
                        print(
                            f"  [{game_count}/{total_games}] "
                            f"{STRATEGY_LABELS[p1]} vs {STRATEGY_LABELS[p2]} "
                            f"({rate:.1f} games/s)"
                        )
    except KeyboardInterrupt:
        print("\nTournament interrupted by user. Shutting down...")
        return results

    elapsed = time.time() - start
    print(f"\nCompleted {game_count} games in {elapsed:.1f}s ({game_count / elapsed:.1f} games/s)")
    return results


def print_results(results: TournamentResults) -> None:
    """Print a formatted summary of tournament results."""
    print("\n" + "=" * 60)
    print("FINAL ELO RATINGS")
    print("=" * 60)

    played = [s for s in results.ratings if results.wins[s] + results.losses[s] + results.draws[s] > 0]
    sorted_strats = sorted(played, key=results.ratings.get, reverse=True)
    for rank, strat in enumerate(sorted_strats, 1):
        w = results.wins[strat]
        ls = results.losses[strat]
        d = results.draws[strat]
        total = w + ls + d
        win_rate = w / total * 100 if total > 0 else 0
        print(
            f"  {rank}. {STRATEGY_LABELS[strat]:<16} "
            f"Elo: {results.ratings[strat]:7.1f}  "
            f"W/L/D: {w}/{ls}/{d}  "
            f"Win%: {win_rate:.1f}%"
        )

    print("\n" + "-" * 60)
    print("HEAD-TO-HEAD MATCHUPS")
    print("-" * 60)
    seen: set[frozenset[Strategy]] = set()
    for (a, b), _ in sorted(results.matchup_total.items()):
        key = frozenset({a, b})
        if key in seen:
            continue
        seen.add(key)
        total_ab = results.matchup_total.get((a, b), 0) + results.matchup_total.get((b, a), 0)
        a_total_wins = results.matchup_wins.get((a, b), 0)
        b_total_wins = results.matchup_wins.get((b, a), 0)
        total_draws = total_ab - a_total_wins - b_total_wins
        print(
            f"  {STRATEGY_LABELS[a]:<16} vs {STRATEGY_LABELS[b]:<16}: "
            f"{a_total_wins}W / {total_draws}D / {b_total_wins}L  ({total_ab} games)"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UTTT Tournament Runner")
    parser.add_argument(
        "-n", "--games", type=int, default=50,
        help="Games per matchup per side (default: 50)",
    )
    parser.add_argument(
        "-s", "--strategies", nargs="+",
        choices=[s.name.lower() for s in Strategy],
        default=["random", "greedy", "alphabeta_1s"],
        help="Strategies to include",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=None,
        help="Max worker threads for inter-game parallelism (default: cpu_count)",
    )
    args = parser.parse_args()

    selected = [Strategy[s.upper()] for s in args.strategies]
    results = run_tournament(selected, games_per_matchup=args.games, max_workers=args.workers)
    print_results(results)
