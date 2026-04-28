"""WebSocket client agent using Alpha-Beta Minimax search."""

import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union

# Add project root to sys.path to support direct execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseUTTTAgent
from agents.lib.state import (
    BOARD_CELLS,
    BOARD_SIDE,
    FREE_MOVE,
    BitboardState,
    run_search,
)

SEARCH_TIME_LIMIT = 4.0


class AdvancedAlphaBetaAgent(BaseUTTTAgent):
    """UTTT agent: Minimax + Alpha-Beta + Iterative Deepening + Bitboards."""

    def __init__(self) -> None:
        super().__init__()
        self.executor = ThreadPoolExecutor()

    def _translate_state(
        self,
        board_2d: list[list[int]],
        active_macro_2d: Optional[list[int]],
    ) -> BitboardState:
        """Convert the server's 2D board into a BitboardState."""
        state = BitboardState()

        if active_macro_2d is None:
            state.active_macro = FREE_MOVE
        else:
            my, mx = active_macro_2d
            state.active_macro = my * BOARD_SIDE + mx

        for y in range(BOARD_CELLS):
            for x in range(BOARD_CELLS):
                val = board_2d[y][x]
                if val == 0:
                    continue
                my, mx = y // BOARD_SIDE, x // BOARD_SIDE
                micro_y, micro_x = y % BOARD_SIDE, x % BOARD_SIDE
                macro_idx = my * BOARD_SIDE + mx
                micro_idx = micro_y * BOARD_SIDE + micro_x
                bit_pos = 8 - micro_idx
                if val == 1:
                    state.boards_p1[macro_idx] |= 1 << bit_pos
                else:
                    state.boards_p2[macro_idx] |= 1 << bit_pos

        # Sync macro board from micro boards
        for i in range(BOARD_CELLS):
            if state.boards_p1[i] and state.check_win(state.boards_p1[i]):
                state.macro_p1 |= 1 << (8 - i)
            elif state.boards_p2[i] and state.check_win(state.boards_p2[i]):
                state.macro_p2 |= 1 << (8 - i)
            elif state.is_full(state.boards_p1[i], state.boards_p2[i]):
                state.macro_draw |= 1 << (8 - i)

        state.hash = state.compute_hash()
        return state

    async def deliberate(
        self,
        board: list[list[int]],
        macro_board: list[list[int]],
        active_macro: Optional[list[int]],
        valid_actions: list[list[int]],
    ) -> Optional[Union[list[int], tuple[int, int]]]:
        if not valid_actions:
            return None

        if self.player_id is None:
            p1_count = sum(row.count(1) for row in board)
            p2_count = sum(row.count(2) for row in board)
            self.player_id = 1 if p1_count == p2_count else 2

        # Opening book: center of center board
        if all(board[y][x] == 0 for y in range(BOARD_CELLS) for x in range(BOARD_CELLS)):
            if self.player_id == 1:
                return [4, 4]

        state = self._translate_state(board, active_macro)

        loop = asyncio.get_running_loop()
        best_macro, best_micro = await loop.run_in_executor(
            self.executor, run_search, state, SEARCH_TIME_LIMIT, self.player_id, self.executor
        )

        # Map internal indices back to global [x, y] coordinates
        my, mx = best_macro // BOARD_SIDE, best_macro % BOARD_SIDE
        micro_y, micro_x = best_micro // BOARD_SIDE, best_micro % BOARD_SIDE
        return [mx * BOARD_SIDE + micro_x, my * BOARD_SIDE + micro_y]


if __name__ == "__main__":
    agent = AdvancedAlphaBetaAgent()
    asyncio.run(agent.run())
