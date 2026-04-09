# <img src="frontend/favicon.svg" alt="logo" width="128" height="128" align="middle"> SI2 - Ultimate-TicTacToe

Ultimate Tic-Tac-Toe is a strategic variation of the classic Tic-Tac-Toe game. It is played on a 9x9 grid, which is composed of nine 3x3 smaller grids, called "micro-boards". The objective is to win three micro-boards in a row, column, or diagonal on the larger 3x3 "macro-board".

Players take turns placing their mark in one of the 81 empty cells. However, the position of a move in a micro-board determines which micro-board the next player must play in. For example, if a player moves in the top-right cell of a micro-board, the next player must play in the top-right micro-board. If a micro-board is already won or full, the next player is granted a "free move" and can play in any available cell on the entire board.

## Game Rules

1.  **Macro-board vs. Micro-boards**: The game is won by getting three micro-boards in a row, column, or diagonal on the 3x3 macro-board.
2.  **Next Micro-board**: The cell chosen in a micro-board dictates the micro-board the next player must play in.
3.  **Local Win**: A micro-board is won by getting three marks in a row within that 3x3 grid. Once won, that micro-board belongs to the player and cannot be played in again.
4.  **Free Move**: If a player is sent to a micro-board that is already won or full, they can play in any empty cell in any other micro-board that is not yet resolved.
5.  **Draw**: If the macro-board becomes full without a winner, the game is a draw.

### Game State Example
The game state is broadcasted as a JSON object:
```json
{
  "type": "state",
  "current_turn": 1,
  "board": [[0, 0, ...], ...],
  "macro_board": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
  "active_macro": [1, 1],
  "valid_actions": [[3, 3], [3, 4], [3, 5], [4, 3], [4, 4], [4, 5], [5, 3], [5, 4], [5, 5]]
}
```

### Possible Actions
An agent responds with a move:
```json
{
  "action": "move",
  "x": 4,
  "y": 4
}
```

## Setup

1.  **Launch the Simulation**:
    Start the backend and frontend using Docker Compose:
    ```bash
    docker compose up
    ```
    The frontend will be available at [http://localhost:8080](http://localhost:8080).

2.  **Run Agents Locally**:
    Create a virtual environment and install dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
    Execute the agents:
    ```bash
    python agents/dummy_agent.py
    # and in another terminal
    python agents/manual_agent.py
    ```

## Project Structure

- `backend/`: Python server using `websockets`.
  - `server.py`: Main simulation engine. Handles game logic, move validation, and state broadcasting.
  - `Dockerfile`: Containerization setup for the backend.
- `frontend/`: HTML5 Canvas-based visualization and map editor.
  - `index.html`: UI structure.
  - `script.js`: Frontend logic, WebSocket client, and Canvas rendering.
  - `styles.css`: Visual styling.
  - `favicon.svg`: Project logo.
- `agents/`: Autonomous agents that connect to the backend.
  - `base_agent.py`: Abstract base class for all agents.
  - `dummy_agent.py`: A simple agent that picks moves randomly.
  - `manual_agent.py`: An agent for manual control via the CLI.
- `compose.yml`: Docker Compose configuration for running the full stack.
- `requirements.txt`: Python dependencies.

## Development

To develop a new agent, inherit from `BaseUTTTAgent` and implement the `deliberate` method:

```python
import asyncio
from agents.base_agent import BaseUTTTAgent

class MyAgent(BaseUTTTAgent):
    async def deliberate(self, board, macro_board, active_macro, valid_actions):
        # Your logic here
        return valid_actions[0]

if __name__ == "__main__":
    agent = MyAgent()
    asyncio.run(agent.run())
```

Detailed documentation can be found [here](https://mariolpantunes.github.io/si2-ultimate-tictactoe/).

## Authors

* **Mário Antunes** - [mariolpantunes](https://github.com/mariolpantunes)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
