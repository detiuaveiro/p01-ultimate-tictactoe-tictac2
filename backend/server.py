import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import websockets
from websockets.server import WebSocketServerProtocol

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class UTTTServer:
    """
    Ultimate Tic-Tac-Toe (UTTT) Server.

    Handles map loading, agent movement, and state broadcasting.
    The game is played on a $9 \times 9$ micro-board, divided into $3 \times 3$ macro-boards.
    """

    def __init__(self) -> None:
        """
        Initializes the UTTTServer.
        """
        self.frontend_ws: Optional[WebSocketServerProtocol] = None
        self.agent1_ws: Optional[WebSocketServerProtocol] = None
        self.agent2_ws: Optional[WebSocketServerProtocol] = None

        # 9x9 Micro Board (0=Empty, 1=P1, 2=P2)
        self.board: List[List[int]] = [[0] * 9 for _ in range(9)]
        # 3x3 Macro Board (0=Ongoing, 1=P1 Win, 2=P2 Win, 3=Draw)
        self.macro_board: List[List[int]] = [[0] * 3 for _ in range(3)]

        # (my, mx) indicating which macro-board the current player MUST play in.
        # None means the player can play in ANY available macro-board.
        self.active_macro: Optional[List[int]] = None

        self.first_player_this_round: int = 1
        self.current_turn: int = 1
        self.running: bool = False
        self.match_scores: Dict[int, int] = {1: 0, 2: 0}

    async def start(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        """
        Starts the UTTT server.

        Args:
            host (str): The host address to bind to.
            port (int): The port to listen on.
        """
        logging.info(f"UTTT Server started on ws://{host}:{port}")
        async with websockets.serve(self.handle_client, host, port):
            await asyncio.Future()

    async def handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """
        Handles incoming WebSocket connections.

        Args:
            websocket (WebSocketServerProtocol): The connected WebSocket client.
        """
        client_type = "Unknown"
        try:
            init_msg = await websocket.recv()
            if isinstance(init_msg, bytes):
                init_msg = init_msg.decode("utf-8")
            data: Dict[str, Any] = json.loads(init_msg)
            client_type = data.get("client", "Unknown")

            if client_type == "frontend":
                logging.info("Frontend connected.")
                self.frontend_ws = websocket
                await self.update_frontend()
                await self.frontend_loop(websocket)
            elif client_type == "agent":
                if not self.agent1_ws:
                    self.agent1_ws = websocket
                    logging.info("Player 1 (X) connected.")
                    await websocket.send(json.dumps({"type": "setup", "player_id": 1}))
                    # Start the agent loop and check conditions in parallel
                    await asyncio.gather(
                        self.agent_loop(websocket, 1),
                        self.check_start_conditions()
                    )
                elif not self.agent2_ws:
                    self.agent2_ws = websocket
                    logging.info("Player 2 (O) connected.")
                    await websocket.send(json.dumps({"type": "setup", "player_id": 2}))
                    # Start the agent loop and check conditions in parallel
                    await asyncio.gather(
                        self.agent_loop(websocket, 2),
                        self.check_start_conditions()
                    )
                else:
                    await websocket.close()
        except Exception as e:
            logging.error(f"Error: {e}")
        finally:
            if websocket == self.frontend_ws:
                self.frontend_ws = None
            elif websocket == self.agent1_ws:
                self.agent1_ws = None
                self.running = False
                await self.update_frontend()
            elif websocket == self.agent2_ws:
                self.agent2_ws = None
                self.running = False
                await self.update_frontend()

    async def frontend_loop(self, websocket: WebSocketServerProtocol) -> None:
        """
        Main loop for handling frontend communication.

        Args:
            websocket (WebSocketServerProtocol): The connected frontend client.
        """
        async for _ in websocket:
            pass

    async def agent_loop(self, websocket: WebSocketServerProtocol, player_id: int) -> None:
        """
        Main loop for handling agent communication.

        Args:
            websocket (WebSocketServerProtocol): The connected agent client.
            player_id (int): The ID of the player (1 or 2).
        """
        async for message in websocket:
            if not self.running or self.current_turn != player_id:
                continue
            try:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                data: Dict[str, Any] = json.loads(message)
                if data.get("action") == "move":
                    x, y = data.get("x"), data.get("y")
                    if x is not None and y is not None and self.process_move(player_id, x, y):
                        await self.check_game_over()
                        if self.running:
                            self.current_turn = 3 - self.current_turn
                            await self.broadcast_state()
                            await self.update_frontend()
            except Exception as e:
                logging.error(f"Error processing move: {e}")

    async def check_start_conditions(self) -> None:
        """
        Checks if both agents are connected and starts the game if so.
        """
        if self.agent1_ws and self.agent2_ws and not self.running:
            self.running = True
            self.board = [[0] * 9 for _ in range(9)]
            self.macro_board = [[0] * 3 for _ in range(3)]
            self.active_macro = None
            self.current_turn = self.first_player_this_round
            await self.update_frontend()
            # Give enough time for the loops to start and agents to be ready
            await asyncio.sleep(1.0)
            await self.broadcast_state()

    def get_valid_actions(self) -> List[List[int]]:
        """
        Returns a list of all currently valid actions for the current player.

        Returns:
            List[List[int]]: A list of [x, y] coordinates representing valid moves.
        """
        actions: List[List[int]] = []
        for y in range(9):
            for x in range(9):
                my, mx = y // 3, x // 3
                # Cannot play in a resolved macro-board
                if self.macro_board[my][mx] != 0:
                    continue
                # Cannot play in an occupied cell
                if self.board[y][x] != 0:
                    continue
                # Must play in the active macro-board, unless free move is granted
                if self.active_macro is not None and self.active_macro != [my, mx]:
                    continue

                actions.append([x, y])
        return actions

    def process_move(self, player_id: int, x: int, y: int) -> bool:
        """
        Processes a move from a player.

        Args:
            player_id (int): The ID of the player making the move.
            x (int): The x-coordinate of the move.
            y (int): The y-coordinate of the move.

        Returns:
            bool: True if the move was processed successfully, False otherwise.
        """
        if [x, y] not in self.get_valid_actions():
            return False

        self.board[y][x] = player_id
        my, mx = y // 3, x // 3
        micro_y, micro_x = y % 3, x % 3

        # Check if this move won the local macro-board
        local_winner = self.check_3x3_win(self.board, mx * 3, my * 3)
        if local_winner:
            self.macro_board[my][mx] = local_winner
        elif self.is_3x3_full(self.board, mx * 3, my * 3):
            self.macro_board[my][mx] = 3  # Draw

        # Set next active macro-board
        next_my, next_mx = micro_y, micro_x
        if self.macro_board[next_my][next_mx] != 0:
            self.active_macro = None  # Free move!
        else:
            self.active_macro = [next_my, next_mx]

        return True

    def check_3x3_win(self, grid: List[List[int]], start_x: int, start_y: int) -> int:
        """
        Checks for a win in a specific 3x3 subset of a grid.

        Args:
            grid (List[List[int]]): The grid to check (9x9 or 3x3).
            start_x (int): The starting x-coordinate of the 3x3 subset.
            start_y (int): The starting y-coordinate of the 3x3 subset.

        Returns:
            int: The ID of the winning player (1 or 2), 0 if no winner, 3 if draw.
        """
        for i in range(3):
            # Rows
            if (
                grid[start_y + i][start_x] != 0
                and grid[start_y + i][start_x]
                == grid[start_y + i][start_x + 1]
                == grid[start_y + i][start_x + 2]
            ):
                return grid[start_y + i][start_x]
            # Cols
            if (
                grid[start_y][start_x + i] != 0
                and grid[start_y][start_x + i]
                == grid[start_y + 1][start_x + i]
                == grid[start_y + 2][start_x + i]
            ):
                return grid[start_y][start_x + i]
        # Diagonals
        if (
            grid[start_y][start_x] != 0
            and grid[start_y][start_x]
            == grid[start_y + 1][start_x + 1]
            == grid[start_y + 2][start_x + 2]
        ):
            return grid[start_y][start_x]
        if (
            grid[start_y + 2][start_x] != 0
            and grid[start_y + 2][start_x]
            == grid[start_y + 1][start_x + 1]
            == grid[start_y][start_x + 2]
        ):
            return grid[start_y + 2][start_x]
        return 0

    def is_3x3_full(self, grid: List[List[int]], start_x: int, start_y: int) -> bool:
        """
        Checks if a 3x3 subset of a grid is full.

        Args:
            grid (List[List[int]]): The grid to check.
            start_x (int): The starting x-coordinate of the 3x3 subset.
            start_y (int): The starting y-coordinate of the 3x3 subset.

        Returns:
            bool: True if the 3x3 subset is full, False otherwise.
        """
        for y in range(3):
            for x in range(3):
                if grid[start_y + y][start_x + x] == 0:
                    return False
        return True

    async def check_game_over(self) -> None:
        """
        Checks if the game is over and handles the results.
        """
        # Treat the macro_board as a standard Tic-Tac-Toe board
        winner = self.check_3x3_win(self.macro_board, 0, 0)
        is_draw = self.is_3x3_full(self.macro_board, 0, 0)

        if winner in [1, 2]:
            self.match_scores[winner] += 1
            await self.end_round(f"Player {winner} Wins!")
        elif is_draw or winner == 3:
            await self.end_round("Global Draw!")

    async def end_round(self, message: str) -> None:
        """
        Ends the current round.

        Args:
            message (str): The message to display at the end of the round.
        """
        self.running = False
        payload = {"type": "game_over", "message": message}
        if self.agent1_ws:
            await self.agent1_ws.send(json.dumps(payload))
        if self.agent2_ws:
            await self.agent2_ws.send(json.dumps(payload))
        await self.update_frontend()

        await asyncio.sleep(3.0)
        self.first_player_this_round = 3 - self.first_player_this_round
        await self.check_start_conditions()

    async def broadcast_state(self) -> None:
        """
        Broadcasts the current game state to both agents.
        """
        payload = {
            "type": "state",
            "current_turn": self.current_turn,
            "board": self.board,
            "macro_board": self.macro_board,
            "active_macro": self.active_macro,
            "valid_actions": self.get_valid_actions(),
        }
        msg = json.dumps(payload)
        if self.agent1_ws:
            await self.agent1_ws.send(msg)
        if self.agent2_ws:
            await self.agent2_ws.send(msg)

    async def update_frontend(self) -> None:
        """
        Sends an update to the frontend.
        """
        if self.frontend_ws:
            await self.frontend_ws.send(
                json.dumps(
                    {
                        "type": "update",
                        "current_turn": self.current_turn,
                        "board": self.board,
                        "macro_board": self.macro_board,
                        "active_macro": self.active_macro,
                        "match_scores": self.match_scores,
                        "p1_connected": self.agent1_ws is not None,
                        "p2_connected": self.agent2_ws is not None,
                    }
                )
            )


if __name__ == "__main__":
    server = UTTTServer()
    asyncio.run(server.start())


if __name__ == "__main__":
    server = UTTTServer()
    asyncio.run(server.start())
