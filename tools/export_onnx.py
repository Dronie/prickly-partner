import sys
import json
from os.path import abspath
from pathlib import Path
from pprint import pprint

# remove cli warning alerts
from absl import logging
logging.set_verbosity(logging.ERROR)

import jax
import jax.numpy as jnp
from flax.training import checkpoints
from flax.training.train_state import TrainState
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import optax
import numpy as np

from coin_game import CoinGame

root_dir = Path(__file__).resolve().parent.parent

# process model path
modelPathS = "".join(sys.argv[-1])

if modelPathS[-1] != "/":
    modelName = modelPathS.split("/")[-1]
    modelPathS+="/"
else:
    modelName = modelPathS.split("/")[-2]

absModelPathS = abspath(modelPathS)

modelPathS_json = modelPathS + "renderer_policy_config.json"

modelPath = Path(modelPathS_json)

# get model manifest data
with open(modelPath) as json_data:
    data = json.load(json_data)
    json_data.close()

input_dim = data["policy_input_dim"]
activation = data["activation"]
out_dim = data["action_dim"]
units = 64
hidden_layers = data["actor_hidden_layers"]

# define flax version of model
class Actor(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Dense(units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        return logits
        #return distrax.Categorical(logits=logits)

class Critic(nn.Module):
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return jnp.squeeze(v, axis=-1)

class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    def setup(self):
        self.actor = Actor(self.action_dim, self.activation)
        self.critic = Critic(self.activation)

    def __call__(self, x):
        pi = self.actor(x)
        v = self.critic(x)
        return pi, v

# re-create train_state
network = ActorCritic(out_dim, activation)

env = CoinGame(
    num_inner_steps=100,
    num_outer_steps=10,
    cnn=False
)
obs, state = env.reset(jax.random.PRNGKey(0))

init_x = jax.numpy.concat([obs['0'], jax.nn.one_hot(1, 10, dtype="int8")])

params = network.init(jax.random.PRNGKey(0), init_x)

tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(2.5e-4, eps=1e-5))

dummy_state = TrainState.create(
    apply_fn=network.apply,
    params=params,
    tx=tx
)

# restore flax checkpoint
restored_state = checkpoints.restore_checkpoint(absModelPathS, dummy_state)

actor_params = restored_state.params['params']['actor']

# define equivalent onnx model
from onnx import TensorProto, numpy_helper, save
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

X = make_tensor_value_info('Input', TensorProto.INT8, [input_dim])
DK0 = make_tensor_value_info('Dense_0', TensorProto.FLOAT, [input_dim, units])
DB0 = make_tensor_value_info('Bias_0', TensorProto.FLOAT, [units])
DK1 = make_tensor_value_info('Dense_1', TensorProto.FLOAT, [units, units])
DB1 = make_tensor_value_info('Bias_1', TensorProto.FLOAT, [units])
DK2 = make_tensor_value_info('Dense_2', TensorProto.FLOAT, [units, out_dim])
DB2 = make_tensor_value_info('Bias_2', TensorProto.FLOAT, [out_dim])
Y = make_tensor_value_info('Output', TensorProto.FLOAT, [out_dim])

node0 = make_node('Cast', inputs=['Input'], outputs=['InputFloat'], to=TensorProto.FLOAT)
node1 = make_node('MatMul', ['InputFloat', 'Dense_0'], ['XDK0'])
node2 = make_node('Add', ['XDK0', 'Bias_0'], ['L1'])
node3 = make_node('Relu', ['L1'], ['R1'])

node4 = make_node('MatMul', ['R1', 'Dense_1'], ['L1DK1'])
node5 = make_node('Add', ['L1DK1', 'Bias_1'], ['L2'])
node6 = make_node('Relu', ['L2'], ['R2'])

node7 = make_node('MatMul', ['R2', 'Dense_2'], ['L2DK2'])
node8 = make_node('Add', ['L2DK2', 'Bias_2'], ['Output'])

# prepare flax model params for injection into onnx model
onnx_initializers = []

for layer_index in range(hidden_layers+1):
    layer = actor_params[f"Dense_{layer_index}"]

    kernel = np.asarray(layer["kernel"], dtype=np.float32)
    bias = np.asarray(layer["bias"], dtype=np.float32)

    onnx_initializers.append(
        numpy_helper.from_array(
            kernel,
            name=f"Dense_{layer_index}"
        )
    )
    onnx_initializers.append(
        numpy_helper.from_array(
            bias,
            name=f"Bias_{layer_index}"
        )
    )

# make onnx model
onnx_graph = make_graph(nodes=[node0, node1, node2, node3, node4, node5, node6, node7, node8],
                        name='actor', # a name
                        inputs=[X],
                        outputs=[Y],
                        initializer=onnx_initializers)

onnx_model = make_model(onnx_graph)
check_model(onnx_model)

# Now verify that the ONNX model and the Flax model give the same outputs
from onnxruntime import InferenceSession

# define extra test cases
all_zeros = np.zeros_like(init_x).astype(np.int8)
random = np.random.randint(low=-128, high=127, size=init_x.shape[0], dtype=np.int8)

# get flax model outputs on test cases
flax_all_zeros_out, _ = dummy_state.apply_fn(
    restored_state.params, all_zeros
)
flax_rand_out, _ = dummy_state.apply_fn(
    restored_state.params, random
)
flax_example_out, _ = dummy_state.apply_fn(
    restored_state.params, init_x
)

# ensure onnxruntime using same version as onnx
onnx_model.ir_version = 11
del onnx_model.opset_import[:]
opset = onnx_model.opset_import.add()
opset.domain = ""
opset.version=23

# get onnx model outputs on test cases
session = InferenceSession(onnx_model.SerializeToString())

onnx_all_zeros_out = np.squeeze(session.run(['Output'], {'Input': np.asarray(all_zeros)}))
onnx_rand_out = np.squeeze(session.run(['Output'], {'Input': np.asarray(random)}))
onnx_example_out = np.squeeze(session.run(['Output'], {'Input': np.asarray(init_x)}))

# ensure outputs match closely
np.testing.assert_allclose(
    flax_all_zeros_out,
    onnx_all_zeros_out,
    rtol=1e-5,
    atol=1e-5
)
np.testing.assert_allclose(
    flax_rand_out,
    onnx_rand_out,
    rtol=1e-5,
    atol=1e-5
)
np.testing.assert_allclose(
    flax_example_out,
    onnx_example_out,
    rtol=1e-5,
    atol=1e-5
)

# finally, save the model
output_path = Path(f"{root_dir}/web/models/{modelName}.onnx")
output_path.parent.mkdir(parents=True, exist_ok=True)
save(onnx_model, output_path)

