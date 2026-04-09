import asyncio
import random
from typing import List, Optional, Tuple, Union

from agents.base_agent import BaseUTTTAgent


class DummyUTTTAgent(BaseUTTTAgent):
    """
    A simple Ultimate Tic-Tac-Toe agent that picks moves randomly.
    """

    async def deliberate(
        self,
        board: List[List[int]],
        macro_board: List[List[int]],
        active_macro: Optional[List[int]],
        valid_actions: List[List[int]],
    ) -> Optional[Union[List[int], Tuple[int, int]]]:
        """
        Randomly selects a move from the available valid actions.

        Args:
            board (List[List[int]]): The current 9x9 board state.
            macro_board (List[List[int]]): The current 3x3 macro board state.
            active_macro (Optional[List[int]]): The active macro board coordinates [my, mx].
            valid_actions (List[List[int]]): A list of valid moves [x, y].

        Returns:
            Optional[Union[List[int], Tuple[int, int]]]: The chosen move [x, y] or None.
        """
        await asyncio.sleep(0.5)
        if not valid_actions:
            return None
        return random.choice(valid_actions)


if __name__ == "__main__":
    agent = DummyUTTTAgent()
    asyncio.run(agent.run())
