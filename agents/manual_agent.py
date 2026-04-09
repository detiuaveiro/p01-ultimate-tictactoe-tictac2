import asyncio
from typing import List, Optional, Tuple, Union

from agents.base_agent import BaseUTTTAgent


class ManualUTTTAgent(BaseUTTTAgent):
    """
    An agent that allows manual control via the CLI.
    """

    async def deliberate(
        self,
        board: List[List[int]],
        macro_board: List[List[int]],
        active_macro: Optional[List[int]],
        valid_actions: List[List[int]],
    ) -> Optional[Union[List[int], Tuple[int, int]]]:
        """
        Prompts the user to enter a move via the command line.

        Args:
            board (List[List[int]]): The current 9x9 board state.
            macro_board (List[List[int]]): The current 3x3 macro board state.
            active_macro (Optional[List[int]]): The active macro board coordinates [my, mx].
            valid_actions (List[List[int]]): A list of valid moves [x, y].

        Returns:
            Optional[Union[List[int], Tuple[int, int]]]: The chosen move [x, y] or None.
        """
        print(f"\n--- YOUR TURN (Player {self.player_id}) ---")

        # If no active macro, we default to showing the whole board,
        # but for a 'Free Move', relative coordinates are harder to define.
        # We will allow the user to pick the macro-board first.
        if active_macro is None:
            print(">>> FREE MOVE! Pick any macro-board.")
            while True:
                m_input = await asyncio.to_thread(
                    input, "Select Macro-Board [mx,my] (0-2): "
                )
                try:
                    parts = m_input.strip().split(",")
                    if len(parts) != 2:
                        raise ValueError
                    mx, my = [int(i) for i in parts]
                    if 0 <= mx <= 2 and 0 <= my <= 2 and macro_board[my][mx] == 0:
                        active_macro = [my, mx]
                        break
                    print("That macro-board is already finished or invalid.")
                except ValueError:
                    print("Use format 'mx,my' (e.g., 1,1 for center).")

        my, mx = active_macro
        print(f">>> Playing in Macro-Board [{mx}, {my}]")

        while True:
            user_input = await asyncio.to_thread(
                input, "Enter local move 'x,y' (0-2): "
            )

            try:
                parts = user_input.strip().split(",")
                if len(parts) != 2:
                    raise ValueError
                lx, ly = [int(i) for i in parts]
                # Convert Relative (0-2) to Global (0-8)
                gx = mx * 3 + lx
                gy = my * 3 + ly

                if [gx, gy] in valid_actions:
                    return [gx, gy]
                else:
                    print(
                        f"Invalid local move. Cell ({lx},{ly}) is either taken or out of bounds."
                    )
            except (ValueError, IndexError):
                print("Invalid format. Use 'x,y' where x and y are 0, 1, or 2.")


if __name__ == "__main__":
    agent = ManualUTTTAgent()
    print("Starting Relative-Coordinate Manual Agent...")
    asyncio.run(agent.run())
