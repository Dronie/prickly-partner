import jax

from coin_game import CoinGame

def test_reset_returns_valid_observations():
    env = CoinGame(
        num_inner_steps=10,
        num_outer_steps=2,
        cnn=False,
    )

    obs, state = env.reset(jax.random.PRNGKey(0))

    assert set(obs) == {"0", "1"}
    assert obs["0"].shape == (100,)
    assert obs["1"].shape == (100,)
    assert int(state.inner_t) == 0

def test_environment_can_take_a_step():
    env = CoinGame(
        num_inner_steps=10,
        num_outer_steps=2,
        cnn=False,
    )

    key = jax.random.PRNGKey(0)
    key, reset_key, step_key = jax.random.split(key, 3)

    obs, state = env.reset(reset_key)
    obs, state, rewards, dones, info = env.step(
        step_key,
        state,
        {"0": 4, "1": 4}
    )

    assert obs["0"].shape == (100,)
    assert set(rewards) == {"0", "1"}
    assert set(dones) == {"0", "1", "__all__"}
    assert int(state.inner_t) == 1
