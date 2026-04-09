class App {
  constructor() {
    const serverHost = window.location.hostname || "localhost";
    this.ws = new WebSocket(`ws://${serverHost}:8765`);
    this.canvas = document.getElementById("sim-canvas");
    this.ctx = this.canvas.getContext("2d");

    this.cellSize = 60;
    this.macroSize = 180;
    this.state = null;

    this.setupWebsocket();
    this.drawEmptyBoard();
  }

  setupWebsocket() {
    this.ws.onopen = () => this.ws.send(JSON.stringify({ client: "frontend" }));

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "update") {
        this.state = data;

        document.getElementById("p1-status").innerText = data.p1_connected
          ? "Player 1 (Red X)"
          : "Player 1 (Disconnected)";
        document.getElementById("p2-status").innerText = data.p2_connected
          ? "Player 2 (Yellow O)"
          : "Player 2 (Disconnected)";
        document.getElementById("p1-score").innerText = data.match_scores[1];
        document.getElementById("p2-score").innerText = data.match_scores[2];

        const turnText = document.getElementById("turn-indicator");
        if (data.p1_connected && data.p2_connected) {
          if (data.active_macro === null) {
            turnText.innerText = `P${data.current_turn} Turn (Free Move!)`;
          } else {
            turnText.innerText = `Current Turn: Player ${data.current_turn}`;
          }
          turnText.style.color =
            data.current_turn === 1 ? "#BF616A" : "#EBCB8B";
        } else {
          turnText.innerText = "Waiting for agents...";
          turnText.style.color = "#D8DEE9";
        }

        this.draw();
      }
    };
  }

  drawEmptyBoard() {
    this.ctx.fillStyle = "#2E3440";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw Micro Grid Lines
    this.ctx.strokeStyle = "#434C5E";
    this.ctx.lineWidth = 2;
    for (let i = 0; i <= 9; i++) {
      this.ctx.beginPath();
      this.ctx.moveTo(i * this.cellSize, 0);
      this.ctx.lineTo(i * this.cellSize, this.canvas.height);
      this.ctx.stroke();
      this.ctx.beginPath();
      this.ctx.moveTo(0, i * this.cellSize);
      this.ctx.lineTo(this.canvas.width, i * this.cellSize);
      this.ctx.stroke();
    }

    // Draw Macro Grid Lines (Thick)
    this.ctx.strokeStyle = "#D8DEE9";
    this.ctx.lineWidth = 6;
    for (let i = 0; i <= 3; i++) {
      this.ctx.beginPath();
      this.ctx.moveTo(i * this.macroSize, 0);
      this.ctx.lineTo(i * this.macroSize, this.canvas.height);
      this.ctx.stroke();
      this.ctx.beginPath();
      this.ctx.moveTo(0, i * this.macroSize);
      this.ctx.lineTo(this.canvas.width, i * this.macroSize);
      this.ctx.stroke();
    }
  }

  drawX(x, y, size, color, lineWidth = 4) {
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = lineWidth;
    this.ctx.beginPath();
    this.ctx.moveTo(x + size * 0.2, y + size * 0.2);
    this.ctx.lineTo(x + size * 0.8, y + size * 0.8);
    this.ctx.moveTo(x + size * 0.8, y + size * 0.2);
    this.ctx.lineTo(x + size * 0.2, y + size * 0.8);
    this.ctx.stroke();
  }

  drawO(x, y, size, color, lineWidth = 4) {
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = lineWidth;
    this.ctx.beginPath();
    this.ctx.arc(x + size / 2, y + size / 2, size * 0.3, 0, Math.PI * 2);
    this.ctx.stroke();
  }

  draw() {
    if (!this.state || !this.state.board) return;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Highlight Active Macro Board
    this.drawEmptyBoard();
    if (this.state.active_macro) {
      const [my, mx] = this.state.active_macro;
      this.ctx.fillStyle = "rgba(136, 192, 208, 0.15)"; // Nord8 subtle highlight
      this.ctx.fillRect(
        mx * this.macroSize,
        my * this.macroSize,
        this.macroSize,
        this.macroSize,
      );
    }

    // Draw Micro Moves
    for (let y = 0; y < 9; y++) {
      for (let x = 0; x < 9; x++) {
        const cell = this.state.board[y][x];
        const cx = x * this.cellSize;
        const cy = y * this.cellSize;

        if (cell === 1) this.drawX(cx, cy, this.cellSize, "#BF616A");
        else if (cell === 2) this.drawO(cx, cy, this.cellSize, "#EBCB8B");
      }
    }

    // Draw Giant Macro Wins Overlays
    for (let my = 0; my < 3; my++) {
      for (let mx = 0; mx < 3; mx++) {
        const macroCell = this.state.macro_board[my][mx];
        const cx = mx * this.macroSize;
        const cy = my * this.macroSize;

        if (macroCell !== 0) {
          // Darken the background behind the giant letter
          this.ctx.fillStyle = "rgba(46, 52, 64, 0.85)";
          this.ctx.fillRect(
            cx + 3,
            cy + 3,
            this.macroSize - 6,
            this.macroSize - 6,
          );

          if (macroCell === 1)
            this.drawX(cx, cy, this.macroSize, "#BF616A", 15);
          else if (macroCell === 2)
            this.drawO(cx, cy, this.macroSize, "#EBCB8B", 15);
          else if (macroCell === 3) {
            this.ctx.fillStyle = "#4C566A";
            this.ctx.font = "bold 80px monospace";
            this.ctx.textAlign = "center";
            this.ctx.textBaseline = "middle";
            this.ctx.fillText(
              "-",
              cx + this.macroSize / 2,
              cy + this.macroSize / 2,
            );
          }
        }
      }
    }
  }
}
const app = new App();
