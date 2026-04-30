[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/vaeSCq55)
# <img src="frontend/favicon.svg" alt="logo" width="128" height="128" align="middle"> SI2 - Ultimate-TicTacToe

Ultimate Tic-Tac-Toe is played on a 9x9 grid composed of nine 3x3 micro-boards. Players win by capturing three micro-boards in a row on the macro-board. The twist: the cell you play in determines which micro-board your opponent must play next. If that board is already resolved, the opponent gets a free move anywhere.

## Our agent

We built an alpha-beta search agent with iterative deepening, backed by a bitboard state representation and Zobrist-hashed transposition tables. The heuristic evaluates macro-board control (positional weights per board), unblocked/blocked two-in-a-row sequences, micro-board tactics (center control, two-in-a-row), and a penalty for granting the opponent free moves.

Design decisions:

- Bitboard representation. Each micro-board is a 9-bit integer. Win detection, move generation, and state cloning are all bitwise operations. Precomputed lookup tables handle micro-board feature extraction and macro-board sequence counting.
- Incremental evaluation. Micro-board features (center control, two-in-a-row counts) are tracked incrementally on `apply_move` rather than recomputed from scratch, keeping the evaluator fast at depth.
- Root parallelism. The search fans out root moves across threads using `ThreadPoolExecutor`. This is effective on the free-threaded Python 3.14 build (`python3.14t`), where threads actually run in parallel.
- CMA-ES weight tuning. We implemented CMA-ES to optimize the 15 heuristic weights. Due to a lack of time and computing power we could not get it to converge to a result that reliably beats our hand-tuned weights. The infrastructure is there and with more compute time we believe it can do better.

### Zobrist hashing and transposition tables

During alpha-beta search, many move sequences lead to the same board position (transpositions). Re-evaluating these from scratch wastes time. We use Zobrist hashing to identify positions and a transposition table (TT) to cache results.

At startup we generate a table of random 64-bit integers: one per (cell, player) pair (81 cells x 2 players = 162 values) and 10 more for the active constraint (9 boards + 1 for free move). The hash of any position is the XOR of all relevant entries. When a move is made, the hash updates incrementally in O(1):

```
H' = H ^ Z[cell][player] ^ Z_constraint[old] ^ Z_constraint[new]
```

XOR is associative and its own inverse, so applying the same value twice cancels it out and the order does not matter. This lets us maintain a running hash as pieces are placed and constraints change, without scanning the full board. UTTT is unusual in that the constraint (which board you must play in) is part of the game state, so two positions with identical piece placement but different constraints hash to different values.

The TT is a dict keyed by Zobrist hash. Each entry stores the search depth, the evaluated score, a bound flag (exact, lower, or upper), and the best move found. On lookup, if the stored depth >= the current search depth, the stored bounds tighten the alpha-beta window or return immediately. Entries are replaced only when the new depth >= the stored depth (depth-preferred replacement). When the table exceeds 100k entries it is cleared.

Beyond avoiding redundant work, the TT also drives move ordering: at each node, we try the TT move first. In a well-ordered tree, alpha-beta reduces the effective branching factor from b to roughly sqrt(b), so move ordering has an outsized impact on search depth within the same time budget.

### Evaluation engine

The evaluation function runs at every leaf node, so it needs to be fast. We split it into three layers.

Macro-board control: each of the 9 positions has a positional weight (center = 100, corners = 30, sides = 10). The score is the weighted sum of boards held by the maximizing player minus those held by the opponent.

Macro-board sequences: we precompute lookup tables indexed by `(macro_p1 << 9) | macro_p2` that count unblocked pairs (two boards in a winning line with the third empty), blocked pairs (two boards in a line but the third held by the opponent), and adjacent pairs (positional synergy). Each is a single array lookup. Drawn boards need a correction pass over the 8 win masks, but only when draws exist.

Micro-board tactics: each micro-board's features (center control, two-in-a-row counts) are packed into a precomputed LUT indexed by the p1 and p2 bitboards. Instead of scanning the LUT at evaluation time, we track running totals incrementally: when `apply_move` modifies a micro-board, it looks up the old and new packed features and applies the delta. When a board gets resolved, its contribution is subtracted. The evaluation function reads these in O(1) from the state object.

Free-move penalty: granting the opponent a free move is usually bad since they can target whatever board is weakest. The penalty scales with game phase (higher in the late game) and the sign flips depending on whose turn it is.

### Heuristic weights

The evaluation function uses 15 weights:

| Weight | Default | Description |
|:---|---:|:---|
| Macro board (x9) | 30/10/100 | Corner/side/center board control |
| Unblocked pair | 40 | Two boards in a winning line, third empty |
| Blocked pair | 15 | Two boards in a line, third held by opponent |
| Micro center | 3 | Holding the center cell of a micro-board |
| Micro two-in-a-row | 5 | Two in a row within a micro-board |
| Adjacent macro | 15 | Two adjacent macro-boards held |
| Free move | 20 | Penalty/bonus for free-move situations |

## Tournament results

Round-robin tournament, 50 games per matchup per side (2,100 games total), on an 8-core ARM server using free-threaded Python 3.14:

| Rank | Strategy | Elo | W/L/D | Win% |
|:---|:---|---:|:---|---:|
| 1 | AlphaBeta-4s | 1217.4 | 260/3/337 | 43.3% |
| 2 | AlphaBeta-1s | 1122.2 | 216/19/365 | 36.0% |
| 3 | RustSolver | 1102.9 | 83/108/409 | 13.8% |
| 4 | Minimax-D4 | 964.2 | 182/51/367 | 30.3% |
| 5 | Greedy | 920.4 | 108/420/72 | 18.0% |
| 6 | Random | 857.0 | 12/286/302 | 2.0% |
| 7 | Minimax-D2 | 816.0 | 132/106/362 | 22.0% |

### Head-to-head matchups

100 games per pair (50 per side). W/D/L from the perspective of the row player.

| Matchup | W | D | L |
|:---|---:|---:|---:|
| AlphaBeta-4s vs RustSolver | 20 | 80 | 0 |
| AlphaBeta-1s vs RustSolver | 16 | 84 | 0 |
| AlphaBeta-4s vs AlphaBeta-1s | 19 | 79 | 2 |
| AlphaBeta-4s vs Minimax-D4 | 30 | 69 | 1 |
| AlphaBeta-1s vs Minimax-D4 | 15 | 85 | 0 |
| Minimax-D4 vs RustSolver | 17 | 79 | 4 |
| AlphaBeta-4s vs Minimax-D2 | 37 | 63 | 0 |
| AlphaBeta-1s vs Minimax-D2 | 47 | 53 | 0 |
| Minimax-D4 vs Minimax-D2 | 21 | 77 | 2 |
| Minimax-D2 vs RustSolver | 15 | 84 | 1 |
| AlphaBeta-4s vs Greedy | 100 | 0 | 0 |
| AlphaBeta-1s vs Greedy | 87 | 13 | 0 |
| Minimax-D4 vs Greedy | 97 | 3 | 0 |
| Greedy vs Random | 68 | 20 | 12 |
| Greedy vs RustSolver | 50 | 10 | 40 |

### Comparison with the Rust solver

We used Nelson Elhage's [ultimattt](https://github.com/nelhage/ultimattt) solver as an external baseline. It is a minimax engine written in Rust with its own transposition tables and iterative deepening.

On the tournament hardware, the solver processes around 3.7 million nodes per second. Our Python agent does roughly 100-150 thousand. That is a 25-30x throughput difference. Despite this, our agent does not lose a single game to the solver: AlphaBeta-4s goes 20-0-80 and AlphaBeta-1s goes 16-0-84 (W-L-D). Even at the same 1-second time budget, where the solver searches 2-4 ply deeper, our agent comes out ahead.

The explanation is heuristic quality. The solver uses a generic evaluation function. Ours is specialized for UTTT: positional macro-board weights, precomputed LUTs for macro threat counting, incrementally tracked micro-board features, and a phase-aware free-move penalty. A more accurate evaluation means better decisions per node, which compensates for searching fewer of them.

#### Experimental conditions

| Parameter | Our Agent | Rust Solver |
|:---|:---|:---|
| Time per move | 1s or 4s | 1s |
| Threads | 8 (root parallelism) | 1 (default) |
| TT memory | ~100k entries (Python dict) | 128MB |
| Node throughput | ~100-150K nodes/s | ~3.7M nodes/s |
| Typical search depth | 8-12 ply | 12-13 ply |

The solver was run with `--table-mem=128M` instead of the default 1GB. The 16GB server was running both agents and the tournament infrastructure concurrently, so we had to keep memory usage in check. The solver used its default single thread (`--threads=1`). Giving it more threads or memory would likely improve its play, but would also starve our processes of resources, so we kept it at the defaults for a fair comparison on shared hardware.

The `--limit` flag controls thinking time per move. We set it to `1s` to match our AlphaBeta-1s agent. Our AlphaBeta-4s agent gets 4x more time, which partly explains its stronger results, though the 1s-vs-1s comparison already favors our agent.

### Server contributions

We found and fixed two issues in the base simulation code:

1. Race condition: fast agents could submit moves before the server finished processing the previous turn. Added `asyncio.Lock` to `server.py` to serialize move processing.
2. Constraint resolution order: when a move simultaneously wins a micro-board and directs the opponent into that same board, the opponent should get a free move. The original code checked the constraint before updating the board state, which could deny the free move.

## Setup

1.  Launch the simulation:
    ```bash
    docker compose up
    ```
    Frontend at [http://localhost:8080](http://localhost:8080).

2.  Run agents:
    ```bash
    uv run python -m agents.advanced_agent
    # in another terminal
    uv run python -m agents.dummy_agent
    ```

3.  Set up the Rust solver (needed for training and some tournament strategies):
    ```bash
    # Requires git and a Rust toolchain (cargo)
    ./scripts/setup_solver.sh
    ```

4.  Run a tournament:
    For full parallelism, use the free-threaded Python 3.14 build:
    ```bash
    uv run --python python3.14t tournament.py -n 50 -s random greedy alphabeta_1s
    ```
    Without `python3.14t`, the GIL limits threads to a single core. The tournament still works, just slower.

    Available strategies: `random`, `greedy`, `minimax_d2`, `minimax_d4`, `alphabeta_1s`, `alphabeta_4s`, `rust_solver`, `alphabeta_opt_1s`, `alphabeta_opt_4s`.

5.  Train weights with CMA-ES:
    ```bash
    uv run --python python3.14t train_cmaes.py
    ```
    Checkpoints are saved to `cmaes_weights.json` after each generation.

6.  Validate trained weights:
    ```bash
    uv run --python python3.14t validate.py
    ```
    Plays 1000 games against the Rust solver and reports win rate with 95% confidence intervals.

## Project structure

```
agents/
  base_agent.py         Abstract base class (provided)
  dummy_agent.py        Random move agent (provided)
  manual_agent.py       CLI agent for manual play (provided)
  advanced_agent.py     Our alpha-beta agent
  lib/
    state.py            Bitboard state, search engine, evaluation
backend/
  server.py             WebSocket game server
frontend/               HTML5 canvas visualization
scripts/
  setup_solver.sh       Download and compile the Rust solver
tournament.py           Headless round-robin tournament runner
train_cmaes.py          CMA-ES weight optimization
validate.py             Weight validation vs Rust solver
cmaes_weights.json      Best weights from training (gen 4)
```

## Authors

* Mario Antunes - [mariolpantunes](https://github.com/mariolpantunes) (base simulation)
* Andre Cardoso - [mycsina](https://github.com/mycsina) (alpha-beta agent, CMA-ES)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
