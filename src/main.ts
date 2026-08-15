import "./style.css";
import { requestAiAction } from "./api";
import {
  ACTIONS,
  createInitialState,
  stepGame,
  type Action,
  type CoinGameState,
} from "./game";
import { renderGame } from "./renderer";

function getApp(): HTMLElement {
  const element = document.querySelector<HTMLElement>("#app");
  if (!element) throw new Error("The application root was not found.");
  return element;
}

let state: CoinGameState | null = null;
let totalRewards = { red: 0, blue: 0 };
let turnPending = false;

function showStartScreen(): void {
  getApp().innerHTML = `<button class="play-button" id="play-now" type="button">Play Now</button>`;
  document.querySelector<HTMLButtonElement>("#play-now")?.addEventListener("click", startGame);
}

function startGame(): void {
  state = createInitialState();
  totalRewards = { red: 0, blue: 0 };
  showGameScreen();
}

function showGameScreen(): void {
  getApp().innerHTML = `
    <section class="game-layout" aria-label="Coin Game">
      <div class="game-panel">
        <header class="status-bar">
          <span id="step-status"></span>
          <span id="reward-status"></span>
        </header>
        <canvas id="game-canvas" aria-label="Five by five coin-game grid"></canvas>
        <div class="legend" aria-label="Game legend">
          <span><i class="marker marker-red"></i>You</span>
          <span><i class="marker marker-blue"></i>AI</span>
          <span><i class="coin coin-red"></i>Your coin</span>
          <span><i class="coin coin-blue"></i>AI coin</span>
        </div>
      </div>
      <aside class="controls-panel">
        <div>
          <p class="eyebrow">Your turn</p>
          <h1>Choose an action</h1>
          <p class="instructions">The AI moves at the same time. The game waits until you choose.</p>
        </div>
        <div class="action-grid" aria-label="Player actions">
          <button class="action-button action-up" data-action="${ACTIONS.up}" type="button">↑<span>Up</span></button>
          <button class="action-button action-left" data-action="${ACTIONS.left}" type="button">←<span>Left</span></button>
          <button class="action-button action-stay" data-action="${ACTIONS.stay}" type="button">•<span>Stay</span></button>
          <button class="action-button action-right" data-action="${ACTIONS.right}" type="button">→<span>Right</span></button>
          <button class="action-button action-down" data-action="${ACTIONS.down}" type="button">↓<span>Down</span></button>
        </div>
        <p class="turn-message" id="turn-message" role="status">Choose your move.</p>
      </aside>
    </section>
  `;

  document.querySelectorAll<HTMLButtonElement>("[data-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const action = Number(button.dataset.action) as Action;
      void takeTurn(action);
    });
  });

  window.addEventListener("resize", renderCurrentState, { passive: true });
  renderCurrentState();
}

function renderCurrentState(): void {
  if (!state) return;
  const canvas = document.querySelector<HTMLCanvasElement>("#game-canvas");
  const stepStatus = document.querySelector<HTMLElement>("#step-status");
  const rewardStatus = document.querySelector<HTMLElement>("#reward-status");
  if (!canvas || !stepStatus || !rewardStatus) return;

  renderGame(canvas, state);
  stepStatus.textContent = `Episode ${state.episode + 1} · Step ${state.innerStep}`;
  rewardStatus.textContent = `Score ${totalRewards.red} · AI ${totalRewards.blue}`;
}

function setTurnPending(pending: boolean, message: string): void {
  turnPending = pending;
  document.querySelectorAll<HTMLButtonElement>("[data-action]").forEach((button) => {
    button.disabled = pending;
  });
  const turnMessage = document.querySelector<HTMLElement>("#turn-message");
  if (turnMessage) turnMessage.textContent = message;
}

async function takeTurn(playerAction: Action): Promise<void> {
  if (!state || turnPending) return;

  setTurnPending(true, "The AI is choosing…");
  try {
    const aiAction = await requestAiAction(state);
    const result = stepGame(state, playerAction, aiAction);
    state = result.state;
    totalRewards.red += result.rewards.red;
    totalRewards.blue += result.rewards.blue;
    renderCurrentState();
    setTurnPending(
      false,
      result.episodeEnded ? "New episode. Choose your move." : "Choose your next move.",
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : "The AI request failed.";
    setTurnPending(false, `${message} Try your move again.`);
  }
}

showStartScreen();
