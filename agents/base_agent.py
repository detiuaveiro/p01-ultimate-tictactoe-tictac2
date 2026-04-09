import json
import logging
from typing import List, Optional, Tuple, Union

import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - AGENT - %(message)s")


class BaseUTTTAgent:
    """
    Abstract base class for all Ultimate Tic-Tac-Toe agents.
    The game is played on a $9 \times 9$ micro-board, divided into $3 \times 3$ macro-boards.
    """

    def __init__(self, server_uri: str = "ws://localhost:8765") -> None:
        """
        Initializes the BaseUTTTAgent.

        Args:
            server_uri (str): The URI of the UTTT server.
        """
        self.server_uri: str = server_uri
        self.player_id: Optional[int] = None

    async def run(self) -> None:
        """
        Connects to the server and enters the main communication loop.
        """
        try:
            async with websockets.connect(self.server_uri) as websocket:
                await websocket.send(json.dumps({"client": "agent"}))

                async for message in websocket:
                    if isinstance(message, bytes):
                        message = message.decode("utf-8")
                    data = json.loads(message)

                    if data.get("type") == "setup":
                        self.player_id = data.get("player_id")
                        logging.info(f"Connected! Assigned Player {self.player_id}")

                    elif data.get("type") == "state":
                        current_turn = data.get("current_turn")
                        board = data.get("board")
                        macro_board = data.get("macro_board")
                        active_macro = data.get("active_macro")
                        valid_actions = data.get("valid_actions")

                        if current_turn == self.player_id:
                            action = await self.deliberate(
                                board, macro_board, active_macro, valid_actions
                            )

                            if action is not None:
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "action": "move",
                                            "x": action[0],
                                            "y": action[1],
                                        }
                                    )
                                )

                    elif data.get("type") == "game_over":
                        logging.info(f"Match Over: {data.get('message')}")

        except Exception as e:
            logging.error(f"Connection lost: {e}")

    async def deliberate(
        self,
        board: List[List[int]],
        macro_board: List[List[int]],
        active_macro: Optional[List[int]],
        valid_actions: List[List[int]],
    ) -> Optional[Union[List[int], Tuple[int, int]]]:
        """
        Deliberates on the next move. MUST be implemented by subclasses.

        Args:
            board (List[List[int]]): The current 9x9 board state.
            macro_board (List[List[int]]): The current 3x3 macro board state.
            active_macro (Optional[List[int]]): The active macro board coordinates [my, mx].
            valid_actions (List[List[int]]): A list of valid moves [x, y].

        Returns:
            Optional[Union[List[int], Tuple[int, int]]]: The chosen move [x, y] or None.
        """
        raise NotImplementedError("Subclasses must implement deliberate()")
