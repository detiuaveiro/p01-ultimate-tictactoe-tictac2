import json
import math
import concurrent.futures
from agents.lib.state import BitboardState, DEFAULT_WEIGHTS, run_search
from tournament import rust_solver_pick_move, DRAW

VALIDATE_GAMES = 1000
TIME_LIMIT = 0.2


def pick_move_weights(
    state: BitboardState, player_id: int, weights: tuple[float, ...], time_limit: float
) -> tuple[int, int]:
    return run_search(state, time_limit, player_id, weights=weights)


def simulate_game(weights: tuple[float, ...], we_are_p1: bool) -> int:
    state = BitboardState()
    current_player = 1

    for _ in range(81):
        is_over, winner = state.is_terminal()
        if is_over:
            return winner

        if (current_player == 1 and we_are_p1) or (
            current_player == 2 and not we_are_p1
        ):
            move = pick_move_weights(
                state, current_player, weights, time_limit=TIME_LIMIT
            )
        else:
            move = rust_solver_pick_move(state, current_player, time_limit=TIME_LIMIT)

        state.apply_move(current_player, move[0], move[1])
        current_player = 3 - current_player

    _, winner = state.is_terminal()
    return winner if winner != 0 else DRAW


def main():
    print("Loading optimized weights from cmaes_weights.json...")
    try:
        with open("cmaes_weights.json", "r") as f:
            data = json.load(f)
            weights = tuple(data["best_weights"])
        print(f"Loaded weights from generation {data.get('generation', 'unknown')}")
    except FileNotFoundError:
        print("cmaes_weights.json not found, using DEFAULT_WEIGHTS")
        weights = DEFAULT_WEIGHTS

    print(f"Playing {VALIDATE_GAMES} games against RUST_SOLVER...")

    matchups = []
    # Half games as P1, half as P2
    for i in range(VALIDATE_GAMES):
        matchups.append(i % 2 == 0)

    wins = 0
    losses = 0
    draws = 0

    def run_matchup(we_are_p1):
        return we_are_p1, simulate_game(weights, we_are_p1)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for we_are_p1, winner in executor.map(run_matchup, matchups):
            if winner == 3:
                draws += 1
            elif (winner == 1 and we_are_p1) or (winner == 2 and not we_are_p1):
                wins += 1
            else:
                losses += 1

            completed = wins + losses + draws
            if completed % 50 == 0:
                print(
                    f"Completed {completed}/{VALIDATE_GAMES} - W: {wins}, L: {losses}, D: {draws}"
                )

    total = wins + losses + draws
    win_rate = wins / total

    # 95% Confidence Interval for proportion: p +/- 1.96 * sqrt(p(1-p)/n)
    margin_of_error = 1.96 * math.sqrt(win_rate * (1 - win_rate) / total)

    print("\\n--- Validation Results ---")
    print(f"Total Games: {total}")
    print(f"Wins: {wins} ({win_rate * 100:.1f}%)")
    print(f"Losses: {losses} ({losses / total * 100:.1f}%)")
    print(f"Draws: {draws} ({draws / total * 100:.1f}%)")
    print(
        f"95% Confidence Interval for Win Rate: {win_rate * 100:.1f}% +/- {margin_of_error * 100:.1f}%"
    )
    print(
        f"CI Range: [{(win_rate - margin_of_error) * 100:.1f}%, {(win_rate + margin_of_error) * 100:.1f}%]"
    )


if __name__ == "__main__":
    main()
