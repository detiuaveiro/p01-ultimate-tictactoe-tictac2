"""Bitboard state representation and Minimax search engine for Ultimate Tic-Tac-Toe."""

import random
import time
from concurrent.futures import ThreadPoolExecutor
from enum import IntEnum

# Board geometry
BOARD_SIDE = 3
BOARD_CELLS = 9
TOTAL_CELLS = 81
FULL_BOARD = 0b111111111
CENTER_MASK = 0b000010000

# Zobrist hashing
ZOBRIST_SEED = 42
NUM_PIECES = 2
NUM_CONSTRAINTS = 10  # 9 boards + wildcard
WILDCARD_CONSTRAINT = 9

# Active macro sentinel
FREE_MOVE = -1

# Search limits
WIN_SCORE = 1_000_000
MAX_SEARCH_DEPTH = 64

# Heuristic weights
MACRO_WEIGHTS = (30.0, 10.0, 30.0, 10.0, 100.0, 10.0, 30.0, 10.0, 30.0)
UNBLOCKED_PAIR_BONUS = 40.0
BLOCKED_PAIR_PENALTY = 15.0
MICRO_CENTER_BONUS = 3.0
MICRO_TWO_IN_ROW_BONUS = 5.0
ADJACENT_MACRO_BONUS = 15.0
FREE_MOVE_PENALTY = 20.0

DEFAULT_WEIGHTS = MACRO_WEIGHTS + (
    UNBLOCKED_PAIR_BONUS,
    BLOCKED_PAIR_PENALTY,
    MICRO_CENTER_BONUS,
    MICRO_TWO_IN_ROW_BONUS,
    ADJACENT_MACRO_BONUS,
    FREE_MOVE_PENALTY,
)

# Pre-calculated winning bitmasks for a 3x3 board.
# Bit 8 = top-left (0,0), bit 0 = bottom-right (2,2).
WIN_MASKS = (
    0b111000000,  # Top row
    0b000111000,  # Middle row
    0b000000111,  # Bottom row
    0b100100100,  # Left column
    0b010010010,  # Center column
    0b001001001,  # Right column
    0b100010001,  # Main diagonal
    0b001010100,  # Anti-diagonal
)

ADJACENT_MASKS = (
    0b110000000, 0b011000000, 0b000110000, 0b000011000, 0b000000110, 0b000000011, # horizontal
    0b100100000, 0b000100100, 0b010010000, 0b000010010, 0b001001000, 0b000001001, # vertical
)

# Zobrist tables (deterministic via seed)
random.seed(ZOBRIST_SEED)
ZOBRIST_PIECES = [
    [random.getrandbits(64) for _ in range(NUM_PIECES)]
    for _ in range(TOTAL_CELLS)
]
ZOBRIST_CONSTRAINTS = [random.getrandbits(64) for _ in range(NUM_CONSTRAINTS)]


class TTFlag(IntEnum):
    """Transposition table entry type for alpha-beta bounds."""

    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2


class TranspositionTable:
    """Depth-preferred transposition table backed by a dict.
    Lock-free for maximum performance in free-threaded environments.
    """

    __slots__ = ("table",)

    def __init__(self) -> None:
        self.table: dict[int, tuple[int, float, TTFlag, tuple[int, int] | None]] = {}

    def get(self, hash_key: int) -> tuple[int, float, TTFlag, tuple[int, int] | None] | None:
        return self.table.get(hash_key)

    def store(
        self,
        hash_key: int,
        depth: int,
        value: float,
        flag: TTFlag,
        best_move: tuple[int, int] | None,
    ) -> None:
        entry = self.table.get(hash_key)
        if entry is None or entry[0] <= depth:
            self.table[hash_key] = (depth, value, flag, best_move)


class BitboardState:
    """Memory-efficient UTTT board using 9-bit integers per micro-board."""

    __slots__ = (
        "boards_p1",
        "boards_p2",
        "macro_p1",
        "macro_p2",
        "macro_draw",
        "active_macro",
        "hash",
    )

    def __init__(self) -> None:
        self.boards_p1: list[int] = [0] * BOARD_CELLS
        self.boards_p2: list[int] = [0] * BOARD_CELLS
        self.macro_p1: int = 0
        self.macro_p2: int = 0
        self.macro_draw: int = 0
        self.active_macro: int = FREE_MOVE
        self.hash: int = ZOBRIST_CONSTRAINTS[WILDCARD_CONSTRAINT]

    def clone(self) -> "BitboardState":
        new = BitboardState.__new__(BitboardState)
        new.boards_p1 = list(self.boards_p1)
        new.boards_p2 = list(self.boards_p2)
        new.macro_p1 = self.macro_p1
        new.macro_p2 = self.macro_p2
        new.macro_draw = self.macro_draw
        new.active_macro = self.active_macro
        new.hash = self.hash
        return new

    @staticmethod
    def check_win(board: int) -> bool:
        for mask in WIN_MASKS:
            if (board & mask) == mask:
                return True
        return False

    @staticmethod
    def is_full(board_p1: int, board_p2: int) -> bool:
        return (board_p1 | board_p2) == FULL_BOARD

    def get_legal_moves(self) -> list[tuple[int, int]]:
        moves: list[tuple[int, int]] = []

        if self.active_macro != FREE_MOVE:
            bit = 1 << (8 - self.active_macro)
            if (self.macro_p1 | self.macro_p2 | self.macro_draw) & bit:
                macros_to_check = range(BOARD_CELLS)
                self.active_macro = FREE_MOVE
            else:
                macros_to_check = [self.active_macro]
        else:
            macros_to_check = range(BOARD_CELLS)

        resolved = self.macro_p1 | self.macro_p2 | self.macro_draw
        for m_idx in macros_to_check:
            if resolved & (1 << (8 - m_idx)):
                continue
            combined = self.boards_p1[m_idx] | self.boards_p2[m_idx]
            for cell in range(BOARD_CELLS):
                if not (combined & (1 << (8 - cell))):
                    moves.append((m_idx, cell))

        return moves

    def apply_move(self, player_id: int, macro_idx: int, micro_idx: int) -> None:
        global_idx = macro_idx * BOARD_CELLS + micro_idx

        # XOR out old constraint
        constraint_idx = WILDCARD_CONSTRAINT if self.active_macro == FREE_MOVE else self.active_macro
        self.hash ^= ZOBRIST_CONSTRAINTS[constraint_idx]

        # XOR in piece
        self.hash ^= ZOBRIST_PIECES[global_idx][player_id - 1]

        bit_mask = 1 << (8 - micro_idx)
        if player_id == 1:
            self.boards_p1[macro_idx] |= bit_mask
            if self.check_win(self.boards_p1[macro_idx]):
                self.macro_p1 |= 1 << (8 - macro_idx)
            elif self.is_full(self.boards_p1[macro_idx], self.boards_p2[macro_idx]):
                self.macro_draw |= 1 << (8 - macro_idx)
        else:
            self.boards_p2[macro_idx] |= bit_mask
            if self.check_win(self.boards_p2[macro_idx]):
                self.macro_p2 |= 1 << (8 - macro_idx)
            elif self.is_full(self.boards_p1[macro_idx], self.boards_p2[macro_idx]):
                self.macro_draw |= 1 << (8 - macro_idx)

        # Update active constraint
        resolved = self.macro_p1 | self.macro_p2 | self.macro_draw
        if resolved & (1 << (8 - micro_idx)):
            self.active_macro = FREE_MOVE
        else:
            self.active_macro = micro_idx

        # XOR in new constraint
        constraint_idx = WILDCARD_CONSTRAINT if self.active_macro == FREE_MOVE else self.active_macro
        self.hash ^= ZOBRIST_CONSTRAINTS[constraint_idx]

    def is_terminal(self) -> tuple[bool, int]:
        if self.check_win(self.macro_p1):
            return True, 1
        if self.check_win(self.macro_p2):
            return True, 2
        if (self.macro_p1 | self.macro_p2 | self.macro_draw) == FULL_BOARD:
            return True, 3
        return False, 0

    def compute_hash(self) -> int:
        h = 0
        for m_idx in range(BOARD_CELLS):
            for micro_idx in range(BOARD_CELLS):
                bit_pos = 8 - micro_idx
                if self.boards_p1[m_idx] & (1 << bit_pos):
                    h ^= ZOBRIST_PIECES[m_idx * BOARD_CELLS + micro_idx][0]
                elif self.boards_p2[m_idx] & (1 << bit_pos):
                    h ^= ZOBRIST_PIECES[m_idx * BOARD_CELLS + micro_idx][1]

        constraint_idx = WILDCARD_CONSTRAINT if self.active_macro == FREE_MOVE else self.active_macro
        h ^= ZOBRIST_CONSTRAINTS[constraint_idx]
        return h


class SearchContext:
    """Encapsulates mutable search state passed through the recursion."""

    __slots__ = ("start_time", "time_limit", "tt", "weights")

    def __init__(self, time_limit: float, tt: TranspositionTable | None = None, weights: tuple[float, ...] | None = None) -> None:
        self.start_time: float = time.time()
        self.time_limit: float = time_limit
        self.tt: TranspositionTable = tt if tt is not None else TranspositionTable()
        self.weights: tuple[float, ...] | None = weights

    def timed_out(self) -> bool:
        return time.time() - self.start_time >= self.time_limit


def evaluate_state(state: BitboardState, player_id: int, weights: tuple[float, ...] | None = None) -> float:
    """Heuristic scoring from the perspective of player_id."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    w_macro = weights[0:9]
    w_unblocked_pair = weights[9]
    w_blocked_pair = weights[10]
    w_micro_center = weights[11]
    w_micro_two = weights[12]
    w_adjacent_macro = weights[13]
    w_free_move = weights[14]

    is_over, winner = state.is_terminal()
    if is_over:
        if winner == 3:
            return 0
        return WIN_SCORE if winner == player_id else -WIN_SCORE

    score = 0.0

    # Macro-board control
    for i in range(BOARD_CELLS):
        bit = 1 << (8 - i)
        if state.macro_p1 & bit:
            score += w_macro[i] if player_id == 1 else -w_macro[i]
        elif state.macro_p2 & bit:
            score += w_macro[i] if player_id == 2 else -w_macro[i]

    # Unblocked macro sequences
    for mask in WIN_MASKS:
        p1_cnt = (state.macro_p1 & mask).bit_count()
        p2_cnt = (state.macro_p2 & mask).bit_count()
        draw_cnt = (state.macro_draw & mask).bit_count()

        if p1_cnt == 2 and p2_cnt == 0 and draw_cnt == 0:
            score += w_unblocked_pair if player_id == 1 else -w_unblocked_pair
        elif p2_cnt == 2 and p1_cnt == 0 and draw_cnt == 0:
            score += w_unblocked_pair if player_id == 2 else -w_unblocked_pair
        elif p1_cnt == 2 and p2_cnt > 0:
            score -= w_blocked_pair if player_id == 1 else -w_blocked_pair
        elif p2_cnt == 2 and p1_cnt > 0:
            score -= w_blocked_pair if player_id == 2 else -w_blocked_pair

    # Adjacent macro-board synergies (Non-linear)
    for mask in ADJACENT_MASKS:
        if (state.macro_p1 & mask) == mask:
            score += w_adjacent_macro if player_id == 1 else -w_adjacent_macro
        elif (state.macro_p2 & mask) == mask:
            score += w_adjacent_macro if player_id == 2 else -w_adjacent_macro

    # Tactical micro-board scoring (unresolved boards only)
    resolved = state.macro_p1 | state.macro_p2 | state.macro_draw
    for i in range(BOARD_CELLS):
        if resolved & (1 << (8 - i)):
            continue
        p1_m = state.boards_p1[i]
        p2_m = state.boards_p2[i]

        if p1_m & CENTER_MASK:
            score += w_micro_center if player_id == 1 else -w_micro_center
        if p2_m & CENTER_MASK:
            score += w_micro_center if player_id == 2 else -w_micro_center

        for mask in WIN_MASKS:
            if (p1_m & mask).bit_count() == 2 and not (p2_m & mask):
                score += w_micro_two if player_id == 1 else -w_micro_two
            if (p2_m & mask).bit_count() == 2 and not (p1_m & mask):
                score += w_micro_two if player_id == 2 else -w_micro_two

    # Late-game exponential penalty for giving opponent a FREE_MOVE
    if state.active_macro == FREE_MOVE:
        p1_pieces = sum(b.bit_count() for b in state.boards_p1)
        p2_pieces = sum(b.bit_count() for b in state.boards_p2)
        player_to_move = 1 if p1_pieces == p2_pieces else 2
        
        # Calculate game stage roughly based on pieces placed (0 to 1)
        # Assuming typical game ends around 40-60 moves. Let's use 60 as scale factor.
        stage = min(1.0, (p1_pieces + p2_pieces) / 60.0)
        
        # Exponential scaling: penalty gets much larger late game
        penalty = w_free_move * (2.718 ** (stage * 2) - 1.0)
        
        if player_to_move == player_id:
            score += penalty # Good for us
        else:
            score -= penalty # Bad for us

    return score


def minimax(
    state: BitboardState,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    player_id: int,
    ctx: SearchContext,
) -> tuple[float, tuple[int, int] | None]:
    """Alpha-beta minimax with transposition table lookup."""
    if ctx.timed_out():
        return evaluate_state(state, player_id, ctx.weights), None

    alpha_orig = alpha
    entry = ctx.tt.get(state.hash)
    if entry is not None and entry[0] >= depth:
        _, tt_val, tt_flag, tt_move = entry
        if tt_flag == TTFlag.EXACT:
            return tt_val, tt_move
        elif tt_flag == TTFlag.LOWERBOUND:
            alpha = max(alpha, tt_val)
        elif tt_flag == TTFlag.UPPERBOUND:
            beta = min(beta, tt_val)
        if alpha >= beta:
            return tt_val, tt_move

    is_over, _ = state.is_terminal()
    if depth == 0 or is_over:
        return evaluate_state(state, player_id, ctx.weights), None

    legal_moves = state.get_legal_moves()
    if not legal_moves:
        return evaluate_state(state, player_id, ctx.weights), None

    # Move ordering: TT best-move first
    if entry is not None and entry[3] is not None and entry[3] in legal_moves:
        legal_moves.remove(entry[3])
        legal_moves.insert(0, entry[3])

    best_move = legal_moves[0]
    opponent_id = BOARD_SIDE - player_id  # 3 - player_id

    if maximizing:
        val = float("-inf")
        for move in legal_moves:
            child = state.clone()
            child.apply_move(player_id, move[0], move[1])
            score, _ = minimax(child, depth - 1, alpha, beta, False, player_id, ctx)
            if score > val:
                val = score
                best_move = move
            alpha = max(alpha, score)
            if beta <= alpha:
                break
    else:
        val = float("inf")
        for move in legal_moves:
            child = state.clone()
            child.apply_move(opponent_id, move[0], move[1])
            score, _ = minimax(child, depth - 1, alpha, beta, True, player_id, ctx)
            if score < val:
                val = score
                best_move = move
            beta = min(beta, score)
            if beta <= alpha:
                break

    # Determine TT flag
    if val <= alpha_orig:
        flag = TTFlag.UPPERBOUND
    elif val >= beta:
        flag = TTFlag.LOWERBOUND
    else:
        flag = TTFlag.EXACT
    ctx.tt.store(state.hash, depth, val, flag, best_move)

    return val, best_move


def run_search(
    state: BitboardState,
    time_limit: float,
    player_id: int,
    executor: ThreadPoolExecutor | None = None,
    weights: tuple[float, ...] | None = None,
) -> tuple[int, int]:
    """Iterative deepening search with optional root-level parallelism."""
    shared_tt = TranspositionTable()
    ctx = SearchContext(time_limit, tt=shared_tt, weights=weights)

    legal_moves = state.get_legal_moves()
    if not legal_moves:
        return (0, 0)
    best_move = legal_moves[0]

    depth = 1
    while depth <= MAX_SEARCH_DEPTH:
        if ctx.timed_out():
            break

        if executor and len(legal_moves) > 1:
            # Parallel root evaluation
            futures = []
            for move in legal_moves:
                child = state.clone()
                child.apply_move(player_id, move[0], move[1])
                # Search child as minimizing player
                futures.append(
                    (move, executor.submit(
                        minimax, child, depth - 1, float("-inf"), float("inf"),
                        False, player_id, ctx
                    ))
                )

            best_val = float("-inf")
            current_best_move = best_move
            for move, future in futures:
                try:
                    val, _ = future.result()
                    if val > best_val:
                        best_val = val
                        current_best_move = move
                except Exception:
                    continue

            if not ctx.timed_out():
                best_move = current_best_move
        else:
            # Sequential root evaluation
            _, move = minimax(
                state, depth, float("-inf"), float("inf"),
                True, player_id, ctx,
            )
            if move and not ctx.timed_out():
                best_move = move
            else:
                break

        # Move ordering: put best move first for next iteration
        if best_move in legal_moves:
            legal_moves.remove(best_move)
            legal_moves.insert(0, best_move)

        depth += 1

    return best_move
