import type { Action, CoinGameState } from "./game";
import { observationForAgent } from "./game";

interface AiActionResponse {
  action: number;
}

function isAction(value: number): value is Action {
  return Number.isInteger(value) && value >= 0 && value <= 4;
}

export async function requestAiAction(
  state: CoinGameState,
  signal?: AbortSignal,
): Promise<Action> {
  const response = await fetch("/api/action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      observation: observationForAgent(state, "blue"),
      agent: "blue",
      agent_id: 1,
      opponent_id: 0,
    }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`AI request failed with HTTP ${response.status}.`);
  }

  const data = (await response.json()) as AiActionResponse;
  if (!isAction(data.action)) {
    throw new Error("The AI response did not contain an action from 0 to 4.");
  }

  return data.action;
}
