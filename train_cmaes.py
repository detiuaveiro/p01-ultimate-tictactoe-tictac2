import json
import concurrent.futures
import numpy as np
import cma
from agents.lib.state import BitboardState, DEFAULT_WEIGHTS, run_search
from tournament import rust_solver_pick_move, DRAW

POPULATION_SIZE = 40
GAMES_PER_MATCHUP = 2  # 2 games per side for Tier 1 to eliminate first-mover advantage
TIER_1_TIME_LIMIT = 0.1
TIER_2_TIME_LIMIT = 0.2
RUST_GAMES_PER_ELITE = 4  # 4 games per side
STAGNATION_LIMIT = 3


def pick_move_weights(
    state: BitboardState, player_id: int, weights: tuple[float, ...], time_limit: float
) -> tuple[int, int]:
    return run_search(state, time_limit, player_id, weights=weights)


def simulate_game_weights(
    weights_1: tuple[float, ...], weights_2: tuple[float, ...]
) -> int:
    state = BitboardState()
    current_player = 1
    w_map = {1: weights_1, 2: weights_2}

    for _ in range(81):
        is_over, winner = state.is_terminal()
        if is_over:
            return winner

        move = pick_move_weights(
            state, current_player, w_map[current_player], TIER_1_TIME_LIMIT
        )
        state.apply_move(current_player, move[0], move[1])
        current_player = 3 - current_player

    _, winner = state.is_terminal()
    return winner if winner != 0 else DRAW


def evaluate_tier_1(population):
    n = len(population)
    fitness = np.zeros(n)
    matchups = []

    for i in range(n):
        for j in range(i + 1, n):
            for _ in range(GAMES_PER_MATCHUP):
                matchups.append((i, j, True))
                matchups.append((i, j, False))

    def run_matchup(m):
        i, j, i_is_p1 = m
        w1 = tuple(population[i])
        w2 = tuple(population[j])
        if i_is_p1:
            winner = simulate_game_weights(w1, w2)
            return (i, j, i_is_p1, winner)
        else:
            winner = simulate_game_weights(w2, w1)
            return (i, j, i_is_p1, winner)

    # Use max workers based on cpu count, but since it's heavy python, process pool might be better?
    # Actually, we use ThreadPoolExecutor which runs run_search that uses TT.
    # ThreadPoolExecutor is fine for I/O and free-threaded python.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(run_matchup, matchups))

    for i, j, i_is_p1, winner in results:
        if winner == 3:  # Draw
            fitness[i] += 0.5
            fitness[j] += 0.5
        elif (winner == 1 and i_is_p1) or (winner == 2 and not i_is_p1):
            fitness[i] += 1.0
        else:
            fitness[j] += 1.0

    return fitness


def simulate_game_vs_rust(weights: tuple[float, ...], we_are_p1: bool) -> int:
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
                state, current_player, weights, time_limit=TIER_2_TIME_LIMIT
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

    for idx, we_are_p1, winner in results:
        if winner == 3:
            fitness[idx] += 5.0
        elif (winner == 1 and we_are_p1) or (winner == 2 and not we_are_p1):
            fitness[idx] += 15.0

    return fitness


def main():
    print("Initializing Co-CMA-ES...")

    # Initial weights and std deviation
    x0 = list(DEFAULT_WEIGHTS)
    sigma0 = 1.0
    
    # Scale initial standard deviations proportionally to the default weights
    # ensuring a minimum std dev of 1.0 for small weights
    stds = [max(1.0, w * 0.3) for w in x0]

    es = cma.CMAEvolutionStrategy(x0, sigma0, {"popsize": POPULATION_SIZE, "CMA_stds": stds})

    best_fitness_history = []
    stagnation_counter = 0
    generation = 0
    best_overall_weights = x0
    best_overall_fitness = -float("inf")

    while not es.stop() and stagnation_counter < STAGNATION_LIMIT:
        generation += 1
        print(f"\\n--- Generation {generation} ---")

        # 1. Sample population
        population = es.ask()

        # 2. Evaluate Tier 1 (Round-Robin)
        print("Running Tier 1 (Round-Robin)...")
        tier_1_fitness = evaluate_tier_1(population)

        # 3. Evaluate Tier 2 (vs Rust Solver)
        print("Running Tier 2 (vs Rust Solver)...")
        final_fitness = evaluate_tier_2(population, tier_1_fitness)

        # CMA-ES minimizes, so we negate the fitness
        costs = [-f for f in final_fitness]

        # 4. Update distribution
        es.tell(population, costs)

        # 5. Track best
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

    print("\\nEvolution finished!")
    print(f"Best fitness achieved: {best_overall_fitness}")
    print(f"Weights saved to cmaes_weights.json")


if __name__ == "__main__":
    main()
