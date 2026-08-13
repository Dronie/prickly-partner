# Coin Game Viewer

A standalone version of the JAX coin-game environment and its Pygame viewer.
It supports two-player keyboard play and playback using two Flax/Orbax policy
checkpoints.

## Set up

Python 3.10 is recommended for compatibility with the pinned JAX stack.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Run

```bash
python coin_game.py
```

The start screen allows keyboard play or model playback. In model mode, select
one checkpoint directory for each player. Checkpoints may live anywhere and do
not need to be copied into this repository.

For the most reliable loading, put a `renderer_policy_config.json` beside each
checkpoint directory. Older checkpoints without this manifest are supported by
best-effort inspection of their Orbax metadata.

## Controls

- Red player: W/A/S/D
- Blue player: I/J/K/L
- Model playback: Space pauses/resumes; N advances one step
- Escape or Q exits

The optional graphical folder chooser uses Zenity, KDialog, or Tkinter when one
is available. Paths can also be typed directly into the launcher.
