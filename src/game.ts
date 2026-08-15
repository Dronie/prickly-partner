export const GRID_SIZE = 5;
export const EPISODE_LENGTH = 100;

export const ACTIONS = {
  right: 0,
  left: 1,
  up: 2,
  down: 3,
  stay: 4,
} as const;

export type Action = (typeof ACTIONS)[keyof typeof ACTIONS];
export type Position = readonly [row: number, column: number];
export type RandomSource = () => number;

export interface CoinGameState {
  redPosition: Position;
  bluePosition: Position;
  redCoinPosition: Position;
  blueCoinPosition: Position;
  innerStep: number;
  episode: number;
  redCooperations: number;
  redDefections: number;
  blueCooperations: number;
  blueDefections: number;
}

export interface StepResult {
  state: CoinGameState;
  rewards: { red: number; blue: number };
  episodeEnded: boolean;
}

const MOVES: Readonly<Record<Action, Position>> = {
  [ACTIONS.right]: [0, 1],
  [ACTIONS.left]: [0, -1],
  [ACTIONS.up]: [1, 0],
  [ACTIONS.down]: [-1, 0],
  [ACTIONS.stay]: [0, 0],
};

function positionsEqual(a: Position, b: Position): boolean {
  return a[0] === b[0] && a[1] === b[1];
}

function clip(value: number): number {
  return Math.max(0, Math.min(GRID_SIZE - 1, value));
}

function move(position: Position, action: Action): Position {
  const delta = MOVES[action];
  return [clip(position[0] + delta[0]), clip(position[1] + delta[1])];
}

function sampleOpenCell(forbidden: readonly Position[], random: RandomSource): Position {
  const candidates: Position[] = [];
  for (let row = 0; row < GRID_SIZE; row += 1) {
    for (let column = 0; column < GRID_SIZE; column += 1) {
      const candidate: Position = [row, column];
      if (!forbidden.some((position) => positionsEqual(position, candidate))) {
        candidates.push(candidate);
      }
    }
  }

  if (candidates.length === 0) {
    throw new Error("Cannot sample a coin position: the grid has no open cells.");
  }

  const rawIndex = Math.floor(random() * candidates.length);
  return candidates[Math.min(rawIndex, candidates.length - 1)];
}

export function createInitialState(random: RandomSource = Math.random): CoinGameState {
  const redPosition: Position = [0, 0];
  const bluePosition: Position = [GRID_SIZE - 1, GRID_SIZE - 1];
  const redCoinPosition = sampleOpenCell([redPosition, bluePosition], random);
  const blueCoinPosition = sampleOpenCell(
    [redPosition, bluePosition, redCoinPosition],
    random,
  );

  return {
    redPosition,
    bluePosition,
    redCoinPosition,
    blueCoinPosition,
    innerStep: 0,
    episode: 0,
    redCooperations: 0,
    redDefections: 0,
    blueCooperations: 0,
    blueDefections: 0,
  };
}

export function observationForAgent(
  state: CoinGameState,
  agent: "red" | "blue",
): number[] {
  const observation = new Array<number>(GRID_SIZE * GRID_SIZE * 4).fill(0);
  const channels =
    agent === "red"
      ? [state.redPosition, state.bluePosition, state.redCoinPosition, state.blueCoinPosition]
      : [state.bluePosition, state.redPosition, state.blueCoinPosition, state.redCoinPosition];

  channels.forEach((position, channel) => {
    const index = (position[0] * GRID_SIZE + position[1]) * 4 + channel;
    observation[index] = 1;
  });

  return observation;
}

export function stepGame(
  state: CoinGameState,
  redAction: Action,
  blueAction: Action,
  random: RandomSource = Math.random,
): StepResult {
  const proposedRedPosition = move(state.redPosition, redAction);
  const proposedBluePosition = move(state.bluePosition, blueAction);

  const redHitsBlue = positionsEqual(proposedRedPosition, state.bluePosition);
  const blueHitsRed = positionsEqual(proposedBluePosition, state.redPosition);
  const blockedRedPosition = redHitsBlue ? state.redPosition : proposedRedPosition;
  const blockedBluePosition = blueHitsRed ? state.bluePosition : proposedBluePosition;

  const sameTarget = positionsEqual(proposedRedPosition, proposedBluePosition);
  const redWinsTie = random() < 0.5;
  const redPosition = sameTarget && !redWinsTie ? state.redPosition : blockedRedPosition;
  const bluePosition = sameTarget && redWinsTie ? state.bluePosition : blockedBluePosition;

  const redCollectedRed = positionsEqual(redPosition, state.redCoinPosition);
  const redCollectedBlue = positionsEqual(redPosition, state.blueCoinPosition);
  const blueCollectedRed = positionsEqual(bluePosition, state.redCoinPosition);
  const blueCollectedBlue = positionsEqual(bluePosition, state.blueCoinPosition);

  let redReward = 0;
  let blueReward = 0;
  if (redCollectedRed) redReward += 1;
  if (redCollectedBlue) {
    redReward += 1;
    blueReward -= 2;
  }
  if (blueCollectedRed) {
    blueReward += 1;
    redReward -= 2;
  }
  if (blueCollectedBlue) blueReward += 1;

  const redCoinTaken = redCollectedRed || blueCollectedRed;
  const blueCoinTaken = redCollectedBlue || blueCollectedBlue;

  const redCoinPosition = redCoinTaken
    ? sampleOpenCell([redPosition, bluePosition, state.blueCoinPosition], random)
    : state.redCoinPosition;
  const blueCoinPosition = blueCoinTaken
    ? sampleOpenCell([redPosition, bluePosition, redCoinPosition], random)
    : state.blueCoinPosition;

  const nextStep = state.innerStep + 1;
  const episodeEnded = nextStep === EPISODE_LENGTH;
  const updatedState: CoinGameState = {
    redPosition,
    bluePosition,
    redCoinPosition,
    blueCoinPosition,
    innerStep: nextStep,
    episode: state.episode,
    redCooperations: state.redCooperations + Number(redCollectedRed),
    redDefections: state.redDefections + Number(redCollectedBlue),
    blueCooperations: state.blueCooperations + Number(blueCollectedBlue),
    blueDefections: state.blueDefections + Number(blueCollectedRed),
  };

  if (!episodeEnded) {
    return {
      state: updatedState,
      rewards: { red: redReward, blue: blueReward },
      episodeEnded: false,
    };
  }

  const resetState = createInitialState(random);
  return {
    state: {
      ...resetState,
      episode: state.episode + 1,
      redCooperations: updatedState.redCooperations,
      redDefections: updatedState.redDefections,
      blueCooperations: updatedState.blueCooperations,
      blueDefections: updatedState.blueDefections,
    },
    // The Python environment zeros rewards on its automatic episode reset.
    rewards: { red: 0, blue: 0 },
    episodeEnded: true,
  };
}
