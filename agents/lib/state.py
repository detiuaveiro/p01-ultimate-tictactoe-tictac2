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

# Precomputed micro-board features, indexed by (p1 << 9) | p2
# Packed: (p1_two << 12) | (p2_two << 8) | (p1_center << 4) | p2_center
MICRO_LUT = [0] * (512 * 512)

def _precompute_micro_lut():
    for p1 in range(512):
        for p2 in range(512):
            if (p1 & p2) != 0:
              continue
            p1_two = 0
            p2_two = 0
            for mask in WIN_MASKS:
                if (p1 & mask).bit_count() == 2 and not (p2 & mask):
                    p1_two += 1
                if (p2 & mask).bit_count() == 2 and not (p1 & mask):
                    p2_two += 1
            p1_center = 1 if (p1 & CENTER_MASK) else 0
            p2_center = 1 if (p2 & CENTER_MASK) else 0
            MICRO_LUT[(p1 << 9) | p2] = (p1_two << 12) | (p2_two << 8) | (p1_center << 4) | p2_center

_precompute_micro_lut()

# Macro LUT for control and synergistic patterns
# Stores (macro_control_sum, unblocked_p1, unblocked_p2, blocked_p1, blocked_p2, synergy_p1, synergy_p2)
# Since weights change, we store the RAW counts and multiply in evaluate_state.
# Index: (macro_p1 << 9) | macro_p2
MACRO_CONTROL_P1 = [0] * (512 * 512)
MACRO_CONTROL_P2 = [0] * (512 * 512)
MACRO_UNBLOCKED_P1 = [0] * (512 * 512)
MACRO_UNBLOCKED_P2 = [0] * (512 * 512)
MACRO_BLOCKED_P1 = [0] * (512 * 512)
MACRO_BLOCKED_P2 = [0] * (512 * 512)
MACRO_SYNERGY_P1 = [0] * (512 * 512)
MACRO_SYNERGY_P2 = [0] * (512 * 512)

def _precompute_macro_luts():
    for mp1 in range(512):
        for mp2 in range(512):
            if (mp1 & mp2) != 0:
                continue
            idx = (mp1 << 9) | mp2
            
            # Unblocked/Blocked pairs (Ignoring draws for the LUT, handled in eval)
            u1, u2, b1, b2 = 0, 0, 0, 0
            for mask in WIN_MASKS:
                c1 = (mp1 & mask).bit_count()
                c2 = (mp2 & mask).bit_count()
                if c1 == 2 and c2 == 0:
                    u1 += 1
                elif c2 == 2 and c1 == 0:
                    u2 += 1
                elif c1 == 2 and c2 > 0:
                    b1 += 1
                elif c2 == 2 and c1 > 0:
                    b2 += 1
            MACRO_UNBLOCKED_P1[idx], MACRO_UNBLOCKED_P2[idx] = u1, u2
            MACRO_BLOCKED_P1[idx], MACRO_BLOCKED_P2[idx] = b1, b2
            
            # Synergies
            s1, s2 = 0, 0
            for mask in ADJACENT_MASKS:
                if (mp1 & mask) == mask:
                    s1 += 1
                elif (mp2 & mask) == mask:
                    s2 += 1
            MACRO_SYNERGY_P1[idx], MACRO_SYNERGY_P2[idx] = s1, s2

_precompute_macro_luts()

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

    __slots__ = ("table", "max_size")

    def __init__(self, max_size: int = 100_000) -> None:
        self.table: dict[int, tuple[int, float, TTFlag, tuple[int, int] | None]] = {}
        self.max_size = max_size

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
        if len(self.table) >= self.max_size:
            self.table.clear()  # Prevent OOM by flushing when full

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
        "p1_two_total",
        "p2_two_total",
        "p1_center_total",
        "p2_center_total",
    )

    def __init__(self) -> None:
        self.boards_p1: list[int] = [0] * BOARD_CELLS
        self.boards_p2: list[int] = [0] * BOARD_CELLS
        self.macro_p1: int = 0
        self.macro_p2: int = 0
        self.macro_draw: int = 0
        self.active_macro: int = FREE_MOVE
        self.hash: int = ZOBRIST_CONSTRAINTS[WILDCARD_CONSTRAINT]
        
        # Incremental micro-features
        self.p1_two_total = 0
        self.p2_two_total = 0
        self.p1_center_total = 0
        self.p2_center_total = 0

    def clone(self) -> "BitboardState":
        new = BitboardState.__new__(BitboardState)
        new.boards_p1 = list(self.boards_p1)
        new.boards_p2 = list(self.boards_p2)
        new.macro_p1 = self.macro_p1
        new.macro_p2 = self.macro_p2
        new.macro_draw = self.macro_draw
        new.active_macro = self.active_macro
        new.hash = self.hash
        new.p1_two_total = self.p1_two_total
        new.p2_two_total = self.p2_two_total
        new.p1_center_total = self.p1_center_total
        new.p2_center_total = self.p2_center_total
        return new

    @staticmethod
    def check_win(board: int) -> bool:
        # Optimized bitwise win check
        return (
            (board & (board << 1) & (board << 2) & 0b100100100) != 0 or # Rows
            (board & (board << 3) & (board << 6) & 0b000000111) != 0 or # Cols
            (board & (board << 4) & (board << 8) & 0b000000001) != 0 or # Diag 1
            (board & (board << 2) & (board << 4) & 0b000000100) != 0    # Diag 2
        )

    @staticmethod
    def is_full(board_p1: int, board_p2: int) -> bool:
        return (board_p1 | board_p2) == FULL_BOARD

    def get_legal_moves(self) -> list[tuple[int, int]]:
        moves: list[tuple[int, int]] = []

        if self.active_macro != FREE_MOVE:
            bit = 1 << (8 - self.active_macro)
            if (self.macro_p1 | self.macro_p2 | self.macro_draw) & bit:
                macros_to_check = range(BOARD_CELLS)
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
        # Protect against invalid moves (e.g. from external solvers)
        macro_idx = max(0, min(8, macro_idx))
        micro_idx = max(0, min(8, micro_idx))
        player_id = max(1, min(2, player_id))

        global_idx = macro_idx * BOARD_CELLS + micro_idx

        # XOR out old constraint
        constraint_idx = WILDCARD_CONSTRAINT if self.active_macro == FREE_MOVE else self.active_macro
        self.hash ^= ZOBRIST_CONSTRAINTS[constraint_idx]

        # XOR in piece
        self.hash ^= ZOBRIST_PIECES[global_idx][player_id - 1]

        # Update incremental features
        old_idx = (self.boards_p1[macro_idx] << 9) | self.boards_p2[macro_idx]
        old_feat = MICRO_LUT[old_idx]

        bit_mask = 1 << (8 - micro_idx)
        if player_id == 1:
            self.boards_p1[macro_idx] |= bit_mask
        else:
            self.boards_p2[macro_idx] |= bit_mask

        new_idx = (self.boards_p1[macro_idx] << 9) | self.boards_p2[macro_idx]
        new_feat = MICRO_LUT[new_idx]
        
        # Delta update
        self.p1_two_total += ((new_feat >> 12) & 0xF) - ((old_feat >> 12) & 0xF)
        self.p2_two_total += ((new_feat >> 8) & 0xF) - ((old_feat >> 8) & 0xF)
        self.p1_center_total += ((new_feat >> 4) & 0xF) - ((old_feat >> 4) & 0xF)
        self.p2_center_total += (new_feat & 0xF) - (old_feat & 0xF)

        # Check if macro board is resolved
        resolved_at_start = (self.macro_p1 | self.macro_p2 | self.macro_draw) & (1 << (8 - macro_idx))
        
        if player_id == 1:
            if self.check_win(self.boards_p1[macro_idx]):
                self.macro_p1 |= 1 << (8 - macro_idx)
        else:
            if self.check_win(self.boards_p2[macro_idx]):
                self.macro_p2 |= 1 << (8 - macro_idx)

        if not ((self.macro_p1 | self.macro_p2 | self.macro_draw) & (1 << (8 - macro_idx))) and \
           self.is_full(self.boards_p1[macro_idx], self.boards_p2[macro_idx]):
            self.macro_draw |= 1 << (8 - macro_idx)

        # If it was just resolved, remove its micro-contribution
        resolved_at_end = (self.macro_p1 | self.macro_p2 | self.macro_draw) & (1 << (8 - macro_idx))
        if resolved_at_end and not resolved_at_start:
            self.p1_two_total -= (new_feat >> 12) & 0xF
            self.p2_two_total -= (new_feat >> 8) & 0xF
            self.p1_center_total -= (new_feat >> 4) & 0xF
            self.p2_center_total -= new_feat & 0xF

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
    
    mp1, mp2 = state.macro_p1, state.macro_p2
    idx = (mp1 << 9) | mp2
    
    # Macro control
    for i in range(BOARD_CELLS):
        bit = 1 << (8 - i)
        if mp1 & bit:
            score += w_macro[i]
        elif mp2 & bit:
            score -= w_macro[i]

    # Macro sequences via LUT (unblocked/blocked/synergy)
    score += MACRO_UNBLOCKED_P1[idx] * w_unblocked_pair
    score -= MACRO_UNBLOCKED_P2[idx] * w_unblocked_pair
    score -= MACRO_BLOCKED_P1[idx] * w_blocked_pair
    score += MACRO_BLOCKED_P2[idx] * w_blocked_pair
    score += MACRO_SYNERGY_P1[idx] * w_adjacent_macro
    score -= MACRO_SYNERGY_P2[idx] * w_adjacent_macro

    # Correct macro sequences for drawn boards
    if state.macro_draw:
        mdraw = state.macro_draw
        for mask in WIN_MASKS:
            if mdraw & mask:
                c1 = (mp1 & mask).bit_count()
                c2 = (mp2 & mask).bit_count()
                if c1 == 2 and c2 == 0:
                    score -= w_unblocked_pair
                elif c2 == 2 and c1 == 0:
                    score += w_unblocked_pair

    # Micro-board features (incrementally tracked)
    score += state.p1_center_total * w_micro_center
    score -= state.p2_center_total * w_micro_center
    score += state.p1_two_total * w_micro_two
    score -= state.p2_two_total * w_micro_two

    if state.active_macro == FREE_MOVE:
        p1_pieces = sum(b.bit_count() for b in state.boards_p1)
        p2_pieces = sum(b.bit_count() for b in state.boards_p2)
        player_to_move = 1 if p1_pieces == p2_pieces else 2
        stage = min(1.0, (p1_pieces + p2_pieces) / 60.0)
        penalty = w_free_move * (1.0 + stage * 2.0)
        score += penalty if player_to_move == 1 else -penalty

    return score if player_id == 1 else -score


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
        raise TimeoutError("Search timed out")

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
    opponent_id = BOARD_SIDE - player_id

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
    tt: TranspositionTable | None = None,
) -> tuple[int, int]:
    """Iterative deepening search with optional root-level parallelism."""
    shared_tt = tt if tt is not None else TranspositionTable()
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
                except TimeoutError:
                    continue
                except Exception:
                    continue

            if not ctx.timed_out():
                best_move = current_best_move
        else:
            try:
                _, move = minimax(
                    state, depth, float("-inf"), float("inf"),
                    True, player_id, ctx,
                )
                if move and not ctx.timed_out():
                    best_move = move
                else:
                    break
            except TimeoutError:
                break

        # Move ordering: put best move first for next iteration
        if best_move in legal_moves:
            legal_moves.remove(best_move)
            legal_moves.insert(0, best_move)

        depth += 1

    return best_move
