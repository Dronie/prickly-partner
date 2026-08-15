import { describe, expect, it } from "vitest";
import {
  ACTIONS,
  createInitialState,
  observationForAgent,
  stepGame,
  type CoinGameState,
} from "../src/game";

const fixedRandom = (value: number) => () => value;

describe("coin-game environment", () => {
  it("starts players and coins in distinct cells", () => {
    const state = createInitialState(fixedRandom(0));
    const positions = [
      state.redPosition.join(","),
      state.bluePosition.join(","),
      state.redCoinPosition.join(","),
      state.blueCoinPosition.join(","),
    ];

    expect(new Set(positions).size).toBe(4);
    expect(state.redPosition).toEqual([0, 0]);
    expect(state.bluePosition).toEqual([4, 4]);
  });

  it("does not advance until stepGame is called", () => {
    const state = createInitialState(fixedRandom(0.2));
    expect(state.innerStep).toBe(0);
  });

  it("moves both players simultaneously and advances one step", () => {
    const state = createInitialState(fixedRandom(0.3));
    const result = stepGame(state, ACTIONS.right, ACTIONS.left, fixedRandom(0.8));

    expect(result.state.redPosition).toEqual([0, 1]);
    expect(result.state.bluePosition).toEqual([4, 3]);
    expect(result.state.innerStep).toBe(1);
  });

  it("uses the Python-compatible flattened observation layout", () => {
    const state: CoinGameState = {
      redPosition: [0, 0],
      bluePosition: [4, 4],
      redCoinPosition: [1, 2],
      blueCoinPosition: [3, 1],
      innerStep: 0,
      episode: 0,
      redCooperations: 0,
      redDefections: 0,
      blueCooperations: 0,
      blueDefections: 0,
    };

    const redObservation = observationForAgent(state, "red");
    const blueObservation = observationForAgent(state, "blue");

    expect(redObservation).toHaveLength(100);
    expect(redObservation[0]).toBe(1);
    expect(redObservation[(4 * 5 + 4) * 4 + 1]).toBe(1);
    expect(blueObservation[(4 * 5 + 4) * 4]).toBe(1);
    expect(blueObservation[1]).toBe(1);
  });

  it("applies the original payoff matrix when the AI takes the user's coin", () => {
    const state: CoinGameState = {
      redPosition: [0, 0],
      bluePosition: [2, 2],
      redCoinPosition: [2, 1],
      blueCoinPosition: [4, 0],
      innerStep: 0,
      episode: 0,
      redCooperations: 0,
      redDefections: 0,
      blueCooperations: 0,
      blueDefections: 0,
    };

    const result = stepGame(state, ACTIONS.stay, ACTIONS.left, fixedRandom(0));
    expect(result.rewards).toEqual({ red: -2, blue: 1 });
    expect(result.state.blueDefections).toBe(1);
  });
});
