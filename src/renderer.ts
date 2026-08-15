import { GRID_SIZE, type CoinGameState, type Position } from "./game";

const COLORS = {
  background: "#f7f3e9",
  grid: "#c7bfae",
  red: "#df4545",
  blue: "#4169d8",
  redCoin: "#f1a0a0",
  blueCoin: "#9ab4f2",
  outline: "#332f29",
};

function canvasPosition(position: Position, cellSize: number): [number, number] {
  const x = position[1] * cellSize + cellSize / 2;
  const y = (GRID_SIZE - 1 - position[0]) * cellSize + cellSize / 2;
  return [x, y];
}

function circle(
  context: CanvasRenderingContext2D,
  position: Position,
  radius: number,
  fill: string,
  cellSize: number,
): void {
  const [x, y] = canvasPosition(position, cellSize);
  context.beginPath();
  context.arc(x, y, radius, 0, Math.PI * 2);
  context.fillStyle = fill;
  context.fill();
  context.lineWidth = 2;
  context.strokeStyle = COLORS.outline;
  context.stroke();
}

export function renderGame(canvas: HTMLCanvasElement, state: CoinGameState): void {
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas 2D rendering is unavailable.");

  const pixelRatio = window.devicePixelRatio || 1;
  const displaySize = Math.min(canvas.clientWidth || 560, 560);
  canvas.width = displaySize * pixelRatio;
  canvas.height = displaySize * pixelRatio;
  context.scale(pixelRatio, pixelRatio);

  const cellSize = displaySize / GRID_SIZE;
  context.fillStyle = COLORS.background;
  context.fillRect(0, 0, displaySize, displaySize);

  context.strokeStyle = COLORS.grid;
  context.lineWidth = 2;
  for (let index = 0; index <= GRID_SIZE; index += 1) {
    const offset = index * cellSize;
    context.beginPath();
    context.moveTo(offset, 0);
    context.lineTo(offset, displaySize);
    context.stroke();
    context.beginPath();
    context.moveTo(0, offset);
    context.lineTo(displaySize, offset);
    context.stroke();
  }

  circle(context, state.redCoinPosition, cellSize * 0.13, COLORS.redCoin, cellSize);
  circle(context, state.blueCoinPosition, cellSize * 0.13, COLORS.blueCoin, cellSize);
  circle(context, state.redPosition, cellSize * 0.27, COLORS.red, cellSize);
  circle(context, state.bluePosition, cellSize * 0.27, COLORS.blue, cellSize);
}
