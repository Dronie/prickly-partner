# This file implements modex execution and action selection

from pathlib import Path

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIRECTORY = PROJECT_ROOT / "web" / "models"

class ModelUnavailableError(Exception):
    pass

# ONNX session handling
class InferenceService:
    def __init__(self):
        # ONNX session cache - let's us avoid re-initialising models
        # all the time
        self.sessions = {}

        for agent_id in range(10):
            model_path = (
                MODEL_DIRECTORY
                / f"z=-1.0_agent_{agent_id}_model.onnx"
            )

            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Model file not found: {model_path}"
                )
            # set up ONNX inference session on the CPU
            session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )

            self._check_interface(session, agent_id)
            self.sessions[agent_id] = session

    def _check_interface(self, session, agent_id):
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                f"Agent {agent_id}: expected inputs and outputs to be of length 1."
            )

        model_input = inputs[0]
        model_output = outputs[0]

        if (
            model_input.name != "Input"
            or model_input.type != "tensor(int8)"
            or model_input.shape != [110]
        ):
            raise ValueError(
                f"Agent {agent_id}: expected Input with type int8 and shape [110]."
            )
        
        if (
            model_output.name != "Output"
            or model_output.type != "tensor(float)"
            or model_output.shape != [5]
        ):
            raise ValueError(
                f"Agent {agent_id}: expected Output with type float32 and shape [5]."
            )

    def choose_action(
        self,
        observation: list[int],
        agent_id: int,
        opponent_id: int,
    ) -> int:
        if agent_id not in self.sessions:
            raise ModelUnavailableError(
                f"Agent {agent_id} is not loaded."
            )

        if not 0 <= opponent_id < 10:
            raise ValueError(
                "Opponent ID must be between 0 and 9."
            )

        observation_array = np.asarray(observation)

        if observation_array.shape != (100,):
            raise ValueError("Observation must contain 100 values.")

        if not np.isin(observation_array, [0, 1]).all():
            raise ValueError("Observation values must be 0 or 1.")

        observation_array = observation_array.astype(np.int8)

        opponent_encoding = np.zeros(10, dtype=np.int8)
        opponent_encoding[opponent_id] = 1

        model_input = np.concatenate(
            [observation_array, opponent_encoding]
        )

        session = self.sessions[agent_id]

        outputs = session.run(
            ["Output"],
            {"Input": model_input}
        )

        logits = np.asarray(outputs[0])

        if logits.shape != (5,):
            raise RuntimeError("Expected five action logits.")

        if not np.isfinite(logits).all():
            raise RuntimeError("The model returned non-finite logit values.")

        scores = logits.astype(np.float64)
        shifted_scores = scores - np.max(scores)

        weights = np.exp(shifted_scores)
        probs = weights / np.sum(weights)

        rng = np.random.default_rng()
        return int(rng.choice(5, p=probs))
























