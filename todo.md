# Ultimate Tic-Tac-Toe - Project Roadmap & To-Do

## 1. Foundational Architecture and Memory Models
- [x] Establish basic `AdvancedAlphaBetaAgent` class inheriting from `BaseUTTTAgent`.
- [x] Implement robust asynchronous WebSocket loop management (`ProcessPoolExecutor`).
- [x] Implement highly optimized 16-bit Bitboard representation.
- [x] Create efficient bitwise operations for move generation, application, and win detection.
- [x] **Refactor:** Separate game logic and state representation into `agents/lib/state.py`.

## 2. The Deliberative Core and Evaluation
- [x] Implement standard recursive Minimax algorithm.
- [x] Develop the static heuristic evaluation function:
  - [x] Macro-board weights (Center=100, Corner=30, Side=10).
  - [x] Micro-board tactical weights (Center=3, Two-in-a-row=5).
  - [x] Unblocked Sequence scoring (+40 bonus).
- [x] Integrate an "Opening Book" (Center domination).

## 3. Advanced Optimization and Pruning
- [x] Integrate Alpha-Beta Pruning.
- [x] Implement Zobrist Hashing for state indexing.
- [x] Integrate Transposition Tables.
- [x] Implement Iterative Deepening (4.0s search window).
- [x] Implement Move Ordering heuristics (TT-best move first).

## 4. Benchmarking, Auditing, and Documentation
- [x] Script an automated tournament runner (Bitboard-direct simulations).
- [x] Implement Elo rating calculations over 10,000+ matches.
- [x] Draft final `README.md` and report visualizations.

## 5. Deferred: Server Contributions (backend/server.py)
- [x] **Synchronization:** Add `asyncio.Lock()` to prevent WebSocket race conditions.
- [x] **Logic Anomaly:** Fix "Just Won" vs "Full Board" constraint resolution order.
