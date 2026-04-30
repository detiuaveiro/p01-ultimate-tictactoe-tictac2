import json
import concurrent.futures
import numpy as np
import cma
from agents.lib.state import BitboardState, DEFAULT_WEIGHTS, run_search, TranspositionTable
from tournament import rust_solver_pick_move, DRAW

POPULATION_SIZE = 20
GAMES_PER_BASELINE = 10  # 10 games as P1, 10 as P2 against the baseline
TIER_1_TIME_LIMIT = 4.0
TIER_2_TIME_LIMIT = 4.0
RUST_GAMES_PER_ELITE = 10  # 10 games per side
STAGNATION_LIMIT = 6


def pick_move_weights(
    state: BitboardState, player_id: int, weights: tuple[float, ...], time_limit: float, tt: TranspositionTable
) -> tuple[int, int]:
    return run_search(state, time_limit, player_id, weights=weights, tt=tt)


def simulate_game_weights(
    weights_1: tuple[float, ...], weights_2: tuple[float, ...]
) -> int:
    state = BitboardState()
    current_player = 1
    w_map = {1: weights_1, 2: weights_2}
    tt_map = {1: TranspositionTable(), 2: TranspositionTable()}

    for _ in range(81):
        is_over, winner = state.is_terminal()
        if is_over:
            return winner

        move = pick_move_weights(
            state, current_player, w_map[current_player], TIER_1_TIME_LIMIT, tt_map[current_player]
        )
        state.apply_move(current_player, move[0], move[1])
        current_player = 3 - current_player

    _, winner = state.is_terminal()
    return winner if winner != 0 else DRAW


def evaluate_tier_1(population):
    n = len(population)
    fitness = np.zeros(n)
    matchups = []

    # Each agent plays against DEFAULT_WEIGHTS
    for i in range(n):
        for _ in range(GAMES_PER_BASELINE):
            matchups.append((i, True))  # i is P1
            matchups.append((i, False)) # i is P2

    def run_matchup(m):
        i, i_is_p1 = m
        w1 = tuple(population[i])
        w_base = DEFAULT_WEIGHTS
        if i_is_p1:
            winner = simulate_game_weights(w1, w_base)
            return (i, i_is_p1, winner)
        else:
            winner = simulate_game_weights(w_base, w1)
            return (i, i_is_p1, winner)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(run_matchup, matchups))

    for i, i_is_p1, winner in results:
        if winner == 3:
            fitness[i] += 2.0
        elif (winner == 1 and i_is_p1) or (winner == 2 and not i_is_p1):
            fitness[i] += 5.0

    return fitness


def simulate_game_vs_rust(weights: tuple[float, ...], we_are_p1: bool) -> int:
    state = BitboardState()
    current_player = 1
    tt = TranspositionTable()

    for _ in range(81):
        is_over, winner = state.is_terminal()
        if is_over:
            return winner

        if (current_player == 1 and we_are_p1) or (
            current_player == 2 and not we_are_p1
        ):
            move = pick_move_weights(
                state, current_player, weights, time_limit=TIER_2_TIME_LIMIT, tt=tt
            )
        else:
            move = rust_solver_pick_move(
                state, current_player, time_limit=TIER_2_TIME_LIMIT
            )

        state.apply_move(current_player, move[0], move[1])
        current_player = 3 - current_player

    _, winner = state.is_terminal()
    return winner if winner != 0 else DRAW


def evaluate_tier_2(population, tier_1_fitness):
    n = len(population)
    elite_count = max(1, n // 5)  # Top 20%
    print(f"Tier 2: evaluating {elite_count} elite agents...")
    elite_indices = np.argsort(tier_1_fitness)[-elite_count:]

    fitness = np.copy(tier_1_fitness)

    matchups = []
    for idx in elite_indices:
        for _ in range(RUST_GAMES_PER_ELITE):
            matchups.append((idx, True))
            matchups.append((idx, False))

    def run_matchup(m):
        idx, we_are_p1 = m
        w = tuple(population[idx])
        winner = simulate_game_vs_rust(w, we_are_p1)
        return (idx, we_are_p1, winner)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(run_matchup, matchups))
    print(f"Tier 2: games finished, processing {len(results)} results...")

    for idx, we_are_p1, winner in results:
        if winner == 3:
            fitness[idx] += 3.0
        elif (winner == 1 and we_are_p1) or (winner == 2 and not we_are_p1):
            fitness[idx] += 10.0

    return fitness


def main():
    print("Initializing Co-CMA-ES...")

    x0 = list(DEFAULT_WEIGHTS)
    sigma0 = 1.0
    stds = [max(1.0, w * 0.3) for w in x0]

    es = cma.CMAEvolutionStrategy(
        x0, sigma0, {
            "popsize": POPULATION_SIZE,
            "CMA_stds": stds,
        }
    )

    best_fitness_history = []
    stagnation_counter = 0
    generation = 0
    best_overall_weights = x0
    best_overall_fitness = -float("inf")

    while not es.stop() and stagnation_counter < STAGNATION_LIMIT:
        generation += 1
        print(f"\n--- Generation {generation} ---")

        population = es.ask()

        print("Running Tier 1 (vs baseline)...")
        tier_1_fitness = evaluate_tier_1(population)

        print("Running Tier 2 (vs Rust solver)...")
        final_fitness = evaluate_tier_2(population, tier_1_fitness)

        costs = [-f for f in final_fitness]

        try:
            es.tell(population, costs)

            current_best_idx = np.argmax(final_fitness)
            current_best_fitness = final_fitness[current_best_idx]
            current_best_weights = population[current_best_idx]

            print(f"Best fitness this gen: {current_best_fitness}")

            if current_best_fitness > best_overall_fitness:
                best_overall_fitness = current_best_fitness
                best_overall_weights = current_best_weights
                stagnation_counter = 0
                print("New overall best found!")
            else:
                stagnation_counter += 1
                print(
                    f"No improvement. Stagnation: {stagnation_counter}/{STAGNATION_LIMIT}"
                )
        except Exception as e:
            print(f"Error during CMA-ES update: {e}")
            import traceback
            traceback.print_exc()
            break

        best_fitness_history.append(current_best_fitness)

        # Save checkpoints
        with open("cmaes_weights.json", "w") as f:
            json.dump(
                {
                    "generation": generation,
                    "best_fitness": best_overall_fitness,
                    "best_weights": list(best_overall_weights),
                },
                f,
                indent=4,
            )
        print(f"Checkpoint saved for generation {generation}.")

    print("\nEvolution finished!")
    print(f"Best fitness achieved: {best_overall_fitness}")
    print("Weights saved to cmaes_weights.json")


if __name__ == "__main__":
    main()
