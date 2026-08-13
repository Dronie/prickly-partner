import os
import shutil
import subprocess
import sys
import logging
import warnings
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from typing import Tuple, Dict
import chex
from typing import Optional, Tuple


class MultiAgentEnv:
    """Minimal base required by the standalone coin-game environment."""

    def __init__(self, num_agents: int):
        self.num_agents = num_agents


class Discrete:
    def __init__(self, n: int):
        self.n = n
        self.shape = ()


class Box:
    def __init__(self, low, high, shape, dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

@chex.dataclass
class EnvState:
    red_pos: jnp.ndarray
    blue_pos: jnp.ndarray
    red_coin_pos: jnp.ndarray
    blue_coin_pos: jnp.ndarray
    inner_t: int
    outer_t: int
    # stats
    red_coop: jnp.ndarray
    red_defect: jnp.ndarray
    blue_coop: jnp.ndarray
    blue_defect: jnp.ndarray
    counter: jnp.ndarray  # 9
    coop1: jnp.ndarray  # 9
    coop2: jnp.ndarray  # 9
    last_state: jnp.ndarray  # 2

MOVES = jnp.array(
    [
        [0, 1],  # right
        [0, -1],  # left
        [1, 0],  # up
        [-1, 0],  # down
        [0, 0],  # stay
    ],
    dtype=jnp.int8,
)


class CoinGame(MultiAgentEnv):
    """
    JAX Compatible version of coin game environment.
    """

    def __init__(
        self,
        num_inner_steps: int = 100, # timesteps per episode
        num_outer_steps: int = 10, # num episodes
        cnn: bool = False,
        egocentric: bool = False, # what does 'egocentric' mean here?
        shared_rewards: bool = False,
        payoff_matrix=[[1, 1, -2], [1, 1, -2]], # how is this used?
    ):

        super().__init__(num_agents=2) # always 2 agents
        self.agents = [str(i) for i in list(range(2))] 
        self.payoff_matrix = payoff_matrix
        self.grid_size = 5

        def _clip_position(pos: jnp.ndarray) -> jnp.ndarray:
            return jnp.clip(pos, 0, self.grid_size - 1).astype(jnp.int8)

        def _sample_open_cell(
            key: chex.PRNGKey,
            forbidden_positions: jnp.ndarray,
        ) -> jnp.ndarray:
            grid_indices = jnp.arange(self.grid_size * self.grid_size, dtype=jnp.int32)
            grid_positions = jnp.stack(
                [grid_indices // self.grid_size, grid_indices % self.grid_size], axis=-1
            )
            forbidden = jnp.any(
                jnp.all(
                    grid_positions[:, None, :] == forbidden_positions[None, :, :],
                    axis=-1,
                ),
                axis=1,
            )
            logits = jnp.where(
                forbidden,
                jnp.full((self.grid_size * self.grid_size,), -jnp.inf, dtype=jnp.float32),
                jnp.zeros((self.grid_size * self.grid_size,), dtype=jnp.float32),
            )
            idx = jax.random.categorical(key, logits)
            return grid_positions[idx].astype(jnp.int8)
        

        '''
        Helper functions:
        
        1) [ ] update stats
        2) [x] abs position
        3) [x] relative position
        4) [x] state to obs
        5) [ ] step 
        6) [x] reset (more or less understand)
        
        
        '''


        # this is causing issues
        def _update_stats(
            state: EnvState,
            rr: jnp.ndarray, # did red pick up a red coin?
            rb: jnp.ndarray, # did red pick up a blue coin?
            br: jnp.ndarray, # did blue pick up a red coin?
            bb: jnp.ndarray, # did blue pick up a blue coin?
        ):
            def state2idx(s: jnp.ndarray) -> int:
                # if s == given array then set idx to 1, 2, 3 etc. else set it to idx
                # e.g., if s = [0, 1], the below will set idx to 5
                idx = 0
                idx = jnp.where((s == jnp.array([1, 1])).all(), 1, idx)
                idx = jnp.where((s == jnp.array([1, 2])).all(), 2, idx)
                idx = jnp.where((s == jnp.array([2, 1])).all(), 3, idx)
                idx = jnp.where((s == jnp.array([2, 2])).all(), 4, idx)
                idx = jnp.where((s == jnp.array([0, 1])).all(), 5, idx)
                idx = jnp.where((s == jnp.array([0, 2])).all(), 6, idx)
                idx = jnp.where((s == jnp.array([2, 0])).all(), 7, idx)
                idx = jnp.where((s == jnp.array([1, 0])).all(), 8, idx)
                return idx

            # actions are X, C, D
            # a1 = 0 if red didn't pick up any coins (X)
            # a1 = 1 if red picked up a red coin (C)
            # a1 = 2 if red picked up a blue coin (D)
            
            a1 = 0
            a1 = jnp.where(rr, 1, a1)
            a1 = jnp.where(rb, 2, a1)

            # a2 = 0 if blue didn't pick up any coins (X)
            # a2 = 1 if blue picked up a blue coin (C)
            # a2 = 2 if blue picked up a red coin (D)
            a2 = 0
            a2 = jnp.where(bb, 1, a2)
            a2 = jnp.where(br, 2, a2)

            # if we didn't get a coin this turn, use the last convention
            # last state essentially tracks if the last coin you picked up was yours or not
            # i.e. it tracks if each player's last coin pickup was cooperation or defection
            convention_1 = jnp.where(a1 > 0, a1, state.last_state[0])
            convention_2 = jnp.where(a2 > 0, a2, state.last_state[1])

            idx = state2idx(state.last_state)
            
            counter = state.counter + jnp.zeros_like(
                state.counter, dtype=jnp.int16
            ).at[idx].set(1)
            #print(rr)
            
            coop1 = state.coop1 + jnp.zeros_like(
                state.counter, dtype=jnp.int16
            ).at[idx].set(rr) # ValueError: Cannot broadcast to shape with fewer dimensions: arr_shape=(1,) shape=()
            
            coop2 = state.coop2 + jnp.zeros_like(
                state.counter, dtype=jnp.int16
            ).at[idx].set(bb)
            
            convention = jnp.stack([convention_1, convention_2]).reshape(2)
            return counter, coop1, coop2, convention

        def _abs_position(state: EnvState) -> jnp.ndarray:
            '''
            returns an observation of shape (3, 3, 4) for each agent:
            i.e., a 4-stack of 3x3 binary matricies:
             - stack 0 tells the ego player's location
             - stack 1 tells the other player's location
             - stack 2 tells the red coin's location
             - stack 3 tells the blue coin's location
            '''
            obs1 = jnp.zeros((self.grid_size, self.grid_size, 4), dtype=jnp.int8)
            obs2 = jnp.zeros((self.grid_size, self.grid_size, 4), dtype=jnp.int8)

            # obs channels are [red_player, blue_player, red_coin, blue_coin]
            obs1 = obs1.at[state.red_pos[0], state.red_pos[1], 0].set(1)
            obs1 = obs1.at[state.blue_pos[0], state.blue_pos[1], 1].set(1)
            obs1 = obs1.at[
                state.red_coin_pos[0], state.red_coin_pos[1], 2
            ].set(1)
            obs1 = obs1.at[
                state.blue_coin_pos[0], state.blue_coin_pos[1], 3
            ].set(1)

            # each agent has egotistic color (so thinks they are red)
            obs2 = jnp.stack(
                [obs1[:, :, 1], obs1[:, :, 0], obs1[:, :, 3], obs1[:, :, 2]],
                axis=-1,
            )
            obs = {self.agents[0]: obs1, self.agents[1]: obs2}
            return obs # dictionary like {'0': agent_0_obs, '1': agent_1_obs}

        def _relative_position(state: EnvState) -> jnp.ndarray: # TODO: make canonical player an input to this function instead of assuming red is cannonical
            '''
            returns an observation of shape (3, 3, 4):
            i.e., a 4-stack of 3x3 binary matricies:
             - stack 0 tells the ego player's location
             - stack 1 tells the other player's location
             - stack 2 tells the red coin's location
             - stack 3 tells the blue coin's location

            here, the ego player will always be at (1, 1).
            '''
            
            """Assume canonical agent is red player"""
            # (x) redplayer at (4, 4)
            # (y) redcoin at   (0 ,0)
            #
            #  o o x        o o y
            #  o o o   ->   o x o
            #  y o o        o o o
            #
            # redplayer goes to (1, 1)
            # redcoing goes to  (2, 2)
            # offset = (-1, -1)
            # new_redcoin = (0, 0) + (-2, -2)

            agent_loc = jnp.array([state.red_pos[0], state.red_pos[1]]) # absolute player location
            
            # ego offset is where we want the respective player to see themself in the obs
            ego_offset = jnp.array([2, 2], dtype=jnp.int8) - agent_loc
            
            # alter everything according to the offset
            rel_other_player = state.blue_pos + ego_offset
            rel_red_coin = state.red_coin_pos + ego_offset
            rel_blue_coin = state.blue_coin_pos + ego_offset

            # create observation
            obs = jnp.zeros((self.grid_size, self.grid_size, 4), dtype=jnp.int8)
            obs = obs.at[2, 2, 0].set(1) # set player location
            other_valid = jnp.all((rel_other_player >= 0) & (rel_other_player < self.grid_size))
            red_coin_valid = jnp.all((rel_red_coin >= 0) & (rel_red_coin < self.grid_size))
            blue_coin_valid = jnp.all((rel_blue_coin >= 0) & (rel_blue_coin < self.grid_size))
            obs = jnp.where(
                other_valid, obs.at[rel_other_player[0], rel_other_player[1], 1].set(1), obs
            )
            obs = jnp.where(
                red_coin_valid, obs.at[rel_red_coin[0], rel_red_coin[1], 2].set(1), obs
            )
            obs = jnp.where(
                blue_coin_valid, obs.at[rel_blue_coin[0], rel_blue_coin[1], 3].set(1), obs
            )
            return obs

        def _state_to_obs(state: EnvState) -> jnp.ndarray:
            # If egocentric then observations are relative
            # Else, observation and absolute
            # if not using a cnn, flatten the obs
            '''
            Relative observations:
            each player's observation is a jnp.ndarray of shape (3, 3, 4):
            i.e., a 4-stack of 3x3 binary matricies:
             - stack 0 tells the ego player's location
             - stack 1 tells the other player's location
             - stack 2 tells the red coin's location
             - stack 3 tells the blue coin's location
            
            ego player always located at (1, 1)

            '''
            
            '''
            Absolute observations:
            
            
            '''
            if egocentric:
                obs1 = _relative_position(state)

                # flip red and blue coins for second agent
                obs2 = _relative_position(
                    EnvState(
                        red_pos=state.blue_pos,
                        blue_pos=state.red_pos,
                        red_coin_pos=state.blue_coin_pos,
                        blue_coin_pos=state.red_coin_pos,
                        inner_t=0, # set to zero here because it's not important for getting the obs probably?
                        outer_t=0, # set to zero here because it's not important for getting the obs probably?
                        red_coop=state.blue_coop,
                        red_defect=state.blue_defect,
                        blue_coop=state.red_coop,
                        blue_defect=state.red_defect,
                        last_state=state.last_state,
                        counter=state.counter,
                        coop1=state.coop1,
                        coop2=state.coop2,
                    )
                )
                obs = (obs1, obs2)
                obs = {agent: obs for agent, obs in zip(self.agents, obs)}
            else:
                obs = _abs_position(state)

            if not cnn:
                return {agent: obs[agent].flatten() for agent in obs}
            return obs # dictionary like {'0': agent_0_obs, '1': agent_1_obs}

        def _step(
            key: chex.PRNGKey,
            state: EnvState,
            actions: Dict[str, int],
        ):
            # get actions and update positions based on those actions, then reset rewards
            action_0, action_1 = actions.values() # get player actions
            proposed_red_pos = _clip_position(state.red_pos + MOVES[action_0])
            proposed_blue_pos = _clip_position(state.blue_pos + MOVES[action_1])
            red_hits_blue = jnp.all(proposed_red_pos == state.blue_pos)
            blue_hits_red = jnp.all(proposed_blue_pos == state.red_pos)
            blocked_red_pos = jnp.where(red_hits_blue, state.red_pos, proposed_red_pos)
            blocked_blue_pos = jnp.where(blue_hits_red, state.blue_pos, proposed_blue_pos)
            same_target = jnp.all(proposed_red_pos == proposed_blue_pos)
            key, tie_key = jax.random.split(key)
            red_wins_tie = jax.random.bernoulli(tie_key)
            new_red_pos = jnp.where(
                same_target & (~red_wins_tie), state.red_pos, blocked_red_pos
            )
            new_blue_pos = jnp.where(
                same_target & red_wins_tie, state.blue_pos, blocked_blue_pos
            )
            red_reward, blue_reward = 0, 0 # reset rewards

            # get matches where a 'match' denotes a player's position being the same as a coin's position
            # of type: jnp.array(bool)

            # is true if red picked up red coin
            red_red_matches = jnp.all( # returns a single bool value
                new_red_pos == state.red_coin_pos, axis=-1
            )
            # is true if red picked up blue coin
            red_blue_matches = jnp.all(
                new_red_pos == state.blue_coin_pos, axis=-1
            )

            # is true if blue picked up red coin
            blue_red_matches = jnp.all(
                new_blue_pos == state.red_coin_pos, axis=-1
            )
            # is true if blue picked up blue coin
            blue_blue_matches = jnp.all(
                new_blue_pos == state.blue_coin_pos, axis=-1
            )
            step_coop = red_red_matches.astype(jnp.float32) + blue_blue_matches.astype(
                jnp.float32
            )
            step_defect = red_blue_matches.astype(jnp.float32) + blue_red_matches.astype(
                jnp.float32
            )

            # assigns rewards based on matches
            ### [[1, 1, -2],[1, 1, -2]]
            _rr_reward = self.payoff_matrix[0][0]
            _rb_reward = self.payoff_matrix[0][1]
            _r_penalty = self.payoff_matrix[0][2]
            _br_reward = self.payoff_matrix[1][0]
            _bb_reward = self.payoff_matrix[1][1]
            _b_penalty = self.payoff_matrix[1][2]

            # red_reward and blue_reward are always zero before these lines
            red_reward = jnp.where(
                red_red_matches, red_reward + _rr_reward, red_reward
            )
            red_reward = jnp.where(
                red_blue_matches, red_reward + _rb_reward, red_reward
            )
            red_reward = jnp.where(
                blue_red_matches, red_reward + _r_penalty, red_reward
            )

            blue_reward = jnp.where(
                blue_red_matches, blue_reward + _br_reward, blue_reward
            )
            blue_reward = jnp.where(
                blue_blue_matches, blue_reward + _bb_reward, blue_reward
            )
            blue_reward = jnp.where(
                red_blue_matches, blue_reward + _b_penalty, blue_reward
            )
            
            '''
            (counter, coop1, coop2, last_state) = _update_stats(
                state,
                red_red_matches,
                red_blue_matches,
                blue_red_matches,
                blue_blue_matches,
            )
            '''
            # allocates new random location for coin

            key, red_coin_key, blue_coin_key = jax.random.split(key, 3)
            red_coin_taken = jnp.logical_or(red_red_matches, blue_red_matches)
            blue_coin_taken = jnp.logical_or(red_blue_matches, blue_blue_matches)

            sampled_red_coin_pos = _sample_open_cell(
                red_coin_key,
                jnp.stack([new_red_pos, new_blue_pos, state.blue_coin_pos], axis=0),
            )
            red_coin_base = jnp.where(red_coin_taken, sampled_red_coin_pos, state.red_coin_pos)
            sampled_blue_coin_pos = _sample_open_cell(
                blue_coin_key,
                jnp.stack([new_red_pos, new_blue_pos, red_coin_base], axis=0),
            )
            new_red_coin_pos = red_coin_base
            new_blue_coin_pos = jnp.where(
                blue_coin_taken, sampled_blue_coin_pos, state.blue_coin_pos
            )

            # gets stats to do with number of times cooperated / defected
            
            next_red_coop = state.red_coop + jnp.zeros(
                num_outer_steps, dtype=jnp.int8
            ).at[state.outer_t].set(red_red_matches)
            next_red_defect = state.red_defect + jnp.zeros(
                num_outer_steps, dtype=jnp.int8
            ).at[state.outer_t].set(red_blue_matches)
            next_blue_coop = state.blue_coop + jnp.zeros(
                num_outer_steps, dtype=jnp.int8
            ).at[state.outer_t].set(blue_blue_matches)
            next_blue_defect = state.blue_defect + jnp.zeros(
                num_outer_steps, dtype=jnp.int8
            ).at[state.outer_t].set(blue_red_matches)
            
            # construct the next state based on information derived above
            
            next_state = EnvState(
                red_pos=new_red_pos,
                blue_pos=new_blue_pos,
                red_coin_pos=new_red_coin_pos,
                blue_coin_pos=new_blue_coin_pos,
                inner_t=state.inner_t + 1,
                outer_t=state.outer_t,
                # red_coop=jnp.zeros((num_outer_steps), dtype=jnp.int8),#next_red_coop,
                # red_defect=jnp.zeros((num_outer_steps), dtype=jnp.int8),#next_red_defect,
                # blue_coop=jnp.zeros((num_outer_steps), dtype=jnp.int8),#next_blue_coop,
                # blue_defect=jnp.zeros((num_outer_steps), dtype=jnp.int8),#next_blue_defect,
                red_coop=next_red_coop,
                red_defect=next_red_defect,
                blue_coop=next_blue_coop,
                blue_defect=next_blue_defect,
                counter=jnp.zeros(9),#counter,
                coop1=jnp.zeros(9),#coop1,
                coop2=jnp.zeros(9),#coop2,
                last_state=jnp.zeros(2)#last_state,
            )

            # get observation from state (will be dictionary like {'0': agent_0_obs, '1': agent_1_obs})
            obs = _state_to_obs(next_state)

            # now calculate if done for inner or outer episode
            inner_t = next_state.inner_t # the timestep for this episode
            outer_t = next_state.outer_t # the current episode
            reset_inner = inner_t == num_inner_steps # reset_inner is a flag for if episode done

            # if inner episode is done, return start state for next game
            # i.e., reset the game to a random starting state and zero out all the
            #       saved stats            print(rewards)
            reset_obs, reset_state = _reset(key) 
            
            
            next_state = EnvState(
                red_pos=jnp.where(
                    reset_inner, reset_state.red_pos, next_state.red_pos
                ),
                blue_pos=jnp.where(
                    reset_inner, reset_state.blue_pos, next_state.blue_pos
                ),
                red_coin_pos=jnp.where(
                    reset_inner,
                    reset_state.red_coin_pos,
                    next_state.red_coin_pos,
                ),
                blue_coin_pos=jnp.where(
                    reset_inner,
                    reset_state.blue_coin_pos,
                    next_state.blue_coin_pos,
                ),
                inner_t=jnp.where(
                    reset_inner, jnp.zeros_like(inner_t), next_state.inner_t
                ),
                outer_t=jnp.where(reset_inner, outer_t + 1, outer_t),
                red_coop=next_state.red_coop,
                red_defect=next_state.red_defect,
                blue_coop=next_state.blue_coop,
                blue_defect=next_state.blue_defect,
                counter=jnp.zeros(9),#counter,
                coop1=jnp.zeros(9),#coop1,
                coop2=jnp.zeros(9),#coop2,
                last_state=jnp.zeros(2)#jnp.where(reset_inner, jnp.zeros(2), last_state),
            )

            obs = {agent: obs for agent, obs in zip(self.agents, [jnp.where(reset_inner, reset_obs[i], obs[i]) for i in obs])} # ??

            blue_reward = jnp.where(reset_inner, 0.0, blue_reward)
            red_reward = jnp.where(reset_inner, 0.0, red_reward)

            if shared_rewards:
                # shared reward (social welfare\sum of agents individual rewards)
                rewards = {agent: reward for agent, reward in zip(self.agents, (sum((red_reward, blue_reward)),  sum((red_reward, blue_reward))))}
            else:
                # individual reward
                rewards = {agent: reward for agent, reward in zip(self.agents, (red_reward, blue_reward))}

            dones = {agent: reset_inner for agent in self.agents}
            dones['__all__'] = reset_inner # needs to be a jnp array stating whether or not the episode is done?
            
            infos = {
                "step_coop": jnp.full((self.num_agents,), step_coop, dtype=jnp.float32),
                "step_defect": jnp.full(
                    (self.num_agents,), step_defect, dtype=jnp.float32
                ),
            }
            return (
                obs,
                next_state,
                rewards,
                dones,
                infos,
            )

        def _reset(
            key: jnp.ndarray
        ) -> Tuple[jnp.ndarray, EnvState]:
            # reset stats to all zeros - but why these shapes?
            empty_stats = jnp.zeros((num_outer_steps), dtype=jnp.int8)
            state_stats = jnp.zeros(9)
            key, red_coin_key, blue_coin_key = jax.random.split(key, 3)
            red_start = jnp.array([0, 0], dtype=jnp.int8)
            blue_start = jnp.array([self.grid_size - 1, self.grid_size - 1], dtype=jnp.int8)
            red_coin_start = _sample_open_cell(
                red_coin_key, jnp.stack([red_start, blue_start], axis=0)
            )
            blue_coin_start = _sample_open_cell(
                blue_coin_key,
                jnp.stack([red_start, blue_start, red_coin_start], axis=0),
            )

            # construct new state with random positions for all mcguffins and zeroed stats
            state = EnvState(
                red_pos=red_start,
                blue_pos=blue_start,
                red_coin_pos=red_coin_start,
                blue_coin_pos=blue_coin_start,
                inner_t=0,
                outer_t=0,
                red_coop=empty_stats,
                red_defect=empty_stats,
                blue_coop=empty_stats,
                blue_defect=empty_stats,
                counter=state_stats,
                coop1=state_stats,
                coop2=state_stats,
                last_state=jnp.zeros(2),# ?
            )
            # get obs for this new state
            obs = _state_to_obs(state)
            return obs, state

        # other than doing the jitting, I don't know why the below needs to happen
        # even the jitting doesn't make sense as the environment is vmapped later
        
        # overwrite Gymnax as it makes single-agent assumptions
        self.step = jax.jit(_step)
        self.reset = jax.jit(_reset)
        self.cnn = cnn # ?

        # will this not overwrite the jitted version?
        self.step = _step
        self.reset = _reset

    @property
    def name(self) -> str:
        """Environment name."""
        return "CoinGame-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 5

    def action_space(self, agent_id=None) -> Discrete:
        """Action space of the environment."""
        return Discrete(5)

    def get_legal_moves(self) -> Discrete:
        """All actions are always legal"""
        return jnp.ones(5)

    def observation_space(self) -> Box:
        """Observation space of the environment."""
        _shape = (5, 5, 4) if self.cnn else (100,)
        return Box(low=0, high=1, shape=_shape, dtype=jnp.uint8)

    def state_space(self) -> Box:
        """State space of the environment."""
        _shape = (5, 5, 4) if self.cnn else (100,)
        return Box(low=0, high=1, shape=_shape, dtype=jnp.uint8)

    def render(self, state: EnvState):
        import numpy as np
        from matplotlib.backends.backend_agg import (
            FigureCanvasAgg as FigureCanvas,
        )
        from matplotlib.figure import Figure
        from PIL import Image

        """Small utility for plotting the agent's state."""
        fig = Figure((6, 3))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(121)
        ax.imshow(
            np.zeros((self.grid_size, self.grid_size)),
            cmap="Greys",
            vmin=0,
            vmax=1,
            aspect="equal",
            interpolation="none",
            origin="lower",
            extent=[0, self.grid_size, 0, self.grid_size],
        )
        ax.set_aspect("equal")

        # ax.margins(0)
        ax.set_xticks(jnp.arange(1, self.grid_size + 1))
        ax.set_yticks(jnp.arange(1, self.grid_size + 1))
        ax.grid()
        red_pos = jnp.squeeze(state.red_pos)
        blue_pos = jnp.squeeze(state.blue_pos)
        red_coin_pos = jnp.squeeze(state.red_coin_pos)
        blue_coin_pos = jnp.squeeze(state.blue_coin_pos)
        ax.annotate(
            "R",
            fontsize=20,
            color="red",
            xy=(red_pos[0], red_pos[1]),
            xycoords="data",
            xytext=(red_pos[0] + 0.5, red_pos[1] + 0.5),
        )
        ax.annotate(
            "B",
            fontsize=20,
            color="blue",
            xy=(blue_pos[0], blue_pos[1]),
            xycoords="data",
            xytext=(blue_pos[0] + 0.5, blue_pos[1] + 0.5),
        )
        ax.annotate(
            "Rc",
            fontsize=20,
            color="red",
            xy=(red_coin_pos[0], red_coin_pos[1]),
            xycoords="data",
            xytext=(red_coin_pos[0] + 0.3, red_coin_pos[1] + 0.3),
        )
        ax.annotate(
            "Bc",
            color="blue",
            fontsize=20,
            xy=(blue_coin_pos[0], blue_coin_pos[1]),
            xycoords="data",
            xytext=(
                blue_coin_pos[0] + 0.3,
                blue_coin_pos[1] + 0.3,
            ),
        )

        ax2 = fig.add_subplot(122)
        ax2.text(0.0, 0.95, "Timestep: %s" % (state.inner_t))
        ax2.text(0.0, 0.75, "Episode: %s" % (state.outer_t))
        ax2.text(
            0.0, 0.45, "Red Coop: %s" % (state.red_coop[state.outer_t].sum())
        )
        ax2.text(
            0.6,
            0.45,
            "Red Defects : %s" % (state.red_defect[state.outer_t].sum()),
        )
        ax2.text(
            0.0, 0.25, "Blue Coop: %s" % (state.blue_coop[state.outer_t].sum())
        )
        ax2.text(
            0.6,
            0.25,
            "Blue Defects : %s" % (state.blue_defect[state.outer_t].sum()),
        )
        ax2.text(
            0.0,
            0.05,
            "Red Total: %s"
            % (
                state.red_defect[state.outer_t].sum()
                + state.red_coop[state.outer_t].sum()
            ),
        )
        ax2.text(
            0.6,
            0.05,
            "Blue Total: %s"
            % (
                state.blue_defect[state.outer_t].sum()
                + state.blue_coop[state.outer_t].sum()
            ),
        )
        ax2.axis("off")
        canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        image = Image.fromarray(buf[..., :3])  # drop alpha channel
        return image


if __name__ == "__main__":
    import distrax
    import flax.linen as nn
    import numpy as np
    import optax
    import pygame
    from flax.linen.initializers import constant, orthogonal
    from flax.training import checkpoints
    from flax.training.train_state import TrainState

    class Actor(nn.Module):
        action_dim: int
        num_hidden_layers: int = 2
        activation: str = "relu"

        @nn.compact
        def __call__(self, x):
            activation = nn.relu if self.activation == "relu" else nn.tanh

            for _ in range(self.num_hidden_layers):
                x = nn.Dense(
                    64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
                )(x)
                x = activation(x)
            logits = nn.Dense(
                self.action_dim,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
            )(x)
            return distrax.Categorical(logits=logits)


    class Critic(nn.Module):
        num_hidden_layers: int = 2
        activation: str = "relu"

        @nn.compact
        def __call__(self, x):
            activation = nn.relu if self.activation == "relu" else nn.tanh

            for _ in range(self.num_hidden_layers):
                x = nn.Dense(
                    64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
                )(x)
                x = activation(x)
            value = nn.Dense(
                1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
            )(x)
            return jnp.squeeze(value, axis=-1)


    class ActorCritic(nn.Module):
        action_dim: int
        actor_hidden_layers: int = 2
        critic_hidden_layers: int = 2
        activation: str = "relu"

        def setup(self):
            self.actor = Actor(
                self.action_dim,
                num_hidden_layers=self.actor_hidden_layers,
                activation=self.activation,
            )
            self.critic = Critic(
                num_hidden_layers=self.critic_hidden_layers,
                activation=self.activation,
            )

        def __call__(self, x):
            pi = self.actor(x)
            value = self.critic(x)
            return pi, value


    class LegacyActorCritic(nn.Module):
        action_dim: int
        actor_hidden_layers: int = 2
        critic_hidden_layers: int = 2
        activation: str = "relu"

        @nn.compact
        def __call__(self, x):
            activation = nn.relu if self.activation == "relu" else nn.tanh

            actor_mean = x
            for _ in range(self.actor_hidden_layers):
                actor_mean = nn.Dense(
                    64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
                )(actor_mean)
                actor_mean = activation(actor_mean)
            actor_mean = nn.Dense(
                self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(actor_mean)
            pi = distrax.Categorical(logits=actor_mean)

            critic = x
            for _ in range(self.critic_hidden_layers):
                critic = nn.Dense(
                    64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
                )(critic)
                critic = activation(critic)
            critic = nn.Dense(
                1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
            )(critic)
            return pi, jnp.squeeze(critic, axis=-1)

    def infer_policy_input_format(expected_input_dim: int, obs_dim: int):
        if expected_input_dim == obs_dim:
            return "base", 0, 0
        extra_dim = expected_input_dim - obs_dim
        history_stride = obs_dim + 1
        if extra_dim > 0 and extra_dim % history_stride == 0:
            return "history", extra_dim // history_stride, 0
        if extra_dim > 0:
            return "opponent_id", 0, extra_dim
        raise ValueError(
            f"Checkpoint input dim={expected_input_dim} is incompatible with obs dim={obs_dim}."
        )

    def normalize_checkpoint_dir(path: str) -> str:
        expanded = os.path.expanduser(path.strip())
        if os.path.isfile(expanded):
            return os.path.dirname(expanded)
        return expanded

    def _resolve_checkpoint_step_dir(ckpt_dir: str) -> Path:
        ckpt_path = Path(ckpt_dir)
        if (ckpt_path / "_METADATA").exists():
            return ckpt_path
        latest = checkpoints.latest_checkpoint(ckpt_dir, prefix="checkpoint_")
        if latest is None:
            raise ValueError(f"No finalized checkpoint found under {ckpt_dir}.")
        return Path(latest)

    def _load_checkpoint_metadata(ckpt_dir: str) -> dict:
        step_dir = _resolve_checkpoint_step_dir(ckpt_dir)
        metadata_path = step_dir / "_METADATA"
        if not metadata_path.exists():
            raise ValueError(f"Checkpoint metadata file not found at {metadata_path}.")
        return json.loads(metadata_path.read_text())

    def _checkpoint_uses_modular_actor_critic(metadata: dict) -> bool:
        tree_keys = metadata.get("tree_metadata", {})
        return "('params', 'params', 'actor', 'Dense_0', 'kernel')" in tree_keys

    def _infer_hidden_layers_from_metadata(metadata: dict) -> tuple[bool, int, int]:
        tree_keys = metadata.get("tree_metadata", {})
        modular = _checkpoint_uses_modular_actor_critic(metadata)
        if modular:
            actor_layers = {
                int(key.split("'Dense_")[1].split("'")[0])
                for key in tree_keys
                if "('params', 'params', 'actor', 'Dense_" in key and key.endswith("'kernel')")
            }
            if not actor_layers:
                raise ValueError("Could not infer actor depth from modular checkpoint metadata.")
            critic_layers = {
                int(key.split("'Dense_")[1].split("'")[0])
                for key in tree_keys
                if "('params', 'params', 'critic', 'Dense_" in key and key.endswith("'kernel')")
            }
            if not critic_layers:
                raise ValueError("Could not infer critic depth from modular checkpoint metadata.")
            return True, max(actor_layers), max(critic_layers)

        flat_layers = {
            int(key.split("'Dense_")[1].split("'")[0])
            for key in tree_keys
            if "('params', 'params', 'Dense_" in key and key.endswith("'kernel')")
        }
        if not flat_layers:
            raise ValueError("Could not infer dense layer layout from checkpoint metadata.")
        num_total_dense = len(flat_layers)
        if num_total_dense % 2 != 0:
            raise ValueError(
                f"Unsupported flat checkpoint layout with {num_total_dense} dense layers."
            )
        hidden_layers = (num_total_dense // 2) - 1
        return False, hidden_layers, hidden_layers

    def _extract_input_dim_from_params(params_tree) -> int:
        if "actor" in params_tree:
            return int(params_tree["actor"]["Dense_0"]["kernel"].shape[0])
        return int(params_tree["Dense_0"]["kernel"].shape[0])

    def _candidate_input_dims(obs_dim: int) -> list[int]:
        dims = [obs_dim]
        dims.extend(obs_dim + n for n in range(1, 65))
        dims.extend(obs_dim + k * (obs_dim + 1) for k in range(1, 17))
        seen = set()
        ordered = []
        for dim in dims:
            if dim not in seen:
                ordered.append(dim)
                seen.add(dim)
        return ordered

    def _restore_checkpoint_quietly(ckpt_dir: str, target):
        # Orbax checkpoints saved on GPU can emit noisy topology/sharding
        # warnings when restored on a CPU-only host, even if restore succeeds.
        root_logger = logging.getLogger()
        prev_level = root_logger.level
        try:
            root_logger.setLevel(logging.CRITICAL)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Couldn't find sharding info under RestoreArgs.*",
                    category=UserWarning,
                )
                return checkpoints.restore_checkpoint(
                    ckpt_dir=ckpt_dir,
                    target=target,
                )
        finally:
            root_logger.setLevel(prev_level)

    def _try_restore_policy_state(network, ckpt_dir: str, input_dim: int):
        init_x = jnp.zeros((input_dim,), dtype=jnp.float32)
        init_params = network.init(jax.random.PRNGKey(0), init_x)
        tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(2.5e-4, eps=1e-5))
        empty_state = TrainState.create(
            apply_fn=network.apply,
            params=jax.tree_util.tree_map(jnp.zeros_like, init_params),
            tx=tx,
        )
        state = _restore_checkpoint_quietly(ckpt_dir=ckpt_dir, target=empty_state)
        same_params = jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                lambda restored, empty: jnp.array_equal(restored, empty),
                state.params,
                empty_state.params,
            )
        )
        return state, same_params

    def _load_policy_manifest(ckpt_dir: str) -> Optional[dict]:
        ckpt_path = Path(ckpt_dir)
        candidate_paths = [
            ckpt_path / "renderer_policy_config.json",
            _resolve_checkpoint_step_dir(ckpt_dir).parent / "renderer_policy_config.json",
            _resolve_checkpoint_step_dir(ckpt_dir) / "renderer_policy_config.json",
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                return json.loads(candidate.read_text())
        return None

    def _env_kwargs_from_manifest(manifest: dict) -> dict:
        return {
            "cnn": bool(manifest.get("cnn", False)),
            "egocentric": bool(manifest.get("egocentric", False)),
            "shared_rewards": bool(manifest.get("shared_rewards", False)),
        }

    def _infer_render_env_kwargs(red_ckpt_path: str, blue_ckpt_path: str) -> dict:
        default_kwargs = {
            "cnn": False,
            "egocentric": False,
            "shared_rewards": False,
        }
        manifests = []
        for ckpt_path in (red_ckpt_path, blue_ckpt_path):
            manifest = _load_policy_manifest(normalize_checkpoint_dir(ckpt_path))
            if manifest is not None:
                manifests.append(manifest)

        if not manifests:
            return default_kwargs

        env_names = {manifest.get("env_name", "coin_game") for manifest in manifests}
        if len(env_names) != 1:
            raise ValueError(
                "Loaded checkpoints disagree on env_name; they should come from the same environment."
            )

        render_kwargs = _env_kwargs_from_manifest(manifests[0])
        for manifest in manifests[1:]:
            other_kwargs = _env_kwargs_from_manifest(manifest)
            if other_kwargs != render_kwargs:
                raise ValueError(
                    "Loaded checkpoints disagree on observation settings "
                    f"(got {render_kwargs} vs {other_kwargs})."
                )
        return render_kwargs

    def load_policy_for_coin_game(env: CoinGame, ckpt_path: str):
        ckpt_dir = normalize_checkpoint_dir(ckpt_path)
        obs_dim = env.observation_space().shape[-1]
        manifest = _load_policy_manifest(ckpt_dir)
        if manifest is not None:
            network_type = manifest["network_type"]
            network_cls = (
                ActorCritic
                if network_type == "modular_actor_critic"
                else LegacyActorCritic
            )
            network = network_cls(
                env.action_space(env.agents[0]).n,
                actor_hidden_layers=int(manifest["actor_hidden_layers"]),
                critic_hidden_layers=int(manifest["critic_hidden_layers"]),
                activation=manifest.get("activation", "relu"),
            )
            expected_input_dim = int(manifest["policy_input_dim"])
            state, same_params = _try_restore_policy_state(
                network, ckpt_dir, expected_input_dim
            )
            if same_params:
                raise ValueError(
                    f"Checkpoint restore from {ckpt_dir} left parameters unchanged."
                )
            obs_mode = manifest.get("obs_mode", "base")
            history_k = int(manifest.get("history_k", 0))
            opponent_id_dim = int(manifest.get("opponent_id_dim", 0))
            if obs_mode == "base":
                policy_format = "base"
            elif obs_mode == "base_plus_opponent_id":
                policy_format = "opponent_id"
            elif obs_mode == "base_plus_history":
                policy_format = "history"
            else:
                raise ValueError(f"Unsupported obs_mode in manifest: {obs_mode}")
        else:
            metadata = _load_checkpoint_metadata(ckpt_dir)
            uses_modular_ac, actor_hidden_layers, critic_hidden_layers = (
                _infer_hidden_layers_from_metadata(metadata)
            )
            network_cls = ActorCritic if uses_modular_ac else LegacyActorCritic
            network = network_cls(
                env.action_space(env.agents[0]).n,
                actor_hidden_layers=actor_hidden_layers,
                critic_hidden_layers=critic_hidden_layers,
                activation="relu",
            )

            restore_errors = []
            state = None
            expected_input_dim = None
            for candidate_dim in _candidate_input_dims(obs_dim):
                try:
                    candidate_state, same_params = _try_restore_policy_state(
                        network, ckpt_dir, candidate_dim
                    )
                except Exception as exc:
                    restore_errors.append(f"{candidate_dim}: {type(exc).__name__}")
                    continue
                if same_params:
                    continue
                state = candidate_state
                expected_input_dim = _extract_input_dim_from_params(
                    candidate_state.params["params"]
                )
                break

            if state is None or expected_input_dim is None:
                hint = ", ".join(restore_errors[:6])
                raise ValueError(
                    "Could not match checkpoint to a supported coin-game policy input "
                    f"shape. Tried plausible dims derived from obs_dim={obs_dim}. "
                    f"First restore errors: {hint}"
                )

            policy_format, history_k, opponent_id_dim = infer_policy_input_format(
                expected_input_dim, obs_dim
            )

        def augment_obs(obs_vec, obs_hist, opp_act_hist, opponent_id: int):
            obs_vec = jnp.asarray(obs_vec, dtype=jnp.float32).reshape(-1)
            if policy_format == "base":
                return obs_vec
            if policy_format == "history":
                hist_obs = obs_hist.reshape((history_k * obs_dim,))
                hist_acts = opp_act_hist.reshape((history_k,))
                return jnp.concatenate([obs_vec, hist_obs, hist_acts], axis=-1)
            opponent_one_hot = jax.nn.one_hot(
                jnp.asarray(opponent_id, dtype=jnp.int32),
                opponent_id_dim,
                dtype=obs_vec.dtype,
            )
            return jnp.concatenate([obs_vec, opponent_one_hot], axis=-1)

        return (
            network,
            state,
            policy_format,
            history_k,
            opponent_id_dim,
            augment_obs,
        )

    def startup_menu(surface, font, small_font):
        tiny_font = pygame.font.SysFont("monospace", 16)

        def draw_button(rect, label, color):
            pygame.draw.rect(surface, color, rect, border_radius=6)
            txt = None
            for f in (font, small_font, tiny_font):
                candidate = f.render(label, True, (20, 20, 20))
                if candidate.get_width() <= rect.width - 12:
                    txt = candidate
                    break
            if txt is None:
                txt = tiny_font.render(label, True, (20, 20, 20))
            text_x = rect.x + (rect.width - txt.get_width()) // 2
            text_y = rect.y + (rect.height - txt.get_height()) // 2
            surface.blit(txt, (text_x, text_y))

        def pick_model_path_via_os() -> str:
            prev_grab = pygame.event.get_grab()
            pygame.event.set_grab(False)
            try:
                if sys.platform.startswith("linux"):
                    if shutil.which("zenity"):
                        proc = subprocess.run(
                            ["zenity", "--file-selection", "--directory", "--title=Select model checkpoint folder"],
                            capture_output=True,
                            text=True,
                        )
                        if proc.returncode == 0:
                            return proc.stdout.strip()
                    if shutil.which("kdialog"):
                        proc = subprocess.run(
                            ["kdialog", "--getexistingdirectory", "."],
                            capture_output=True,
                            text=True,
                        )
                        if proc.returncode == 0:
                            return proc.stdout.strip()

                import tkinter as tk
                from tkinter import filedialog

                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                root.lift()
                root.update()
                selected = filedialog.askdirectory(
                    title="Select model checkpoint folder",
                    parent=root,
                )
                root.destroy()
                return selected or ""
            except Exception:
                return ""
            finally:
                pygame.event.set_grab(prev_grab)

        mode = "play"
        model_paths = {"red": "", "blue": ""}
        model_ids = {"red": "0", "blue": "1"}
        policy_mode = "argmax"
        active = None
        input_error = ""
        win_w = surface.get_width()
        pad_x = 40
        content_w = max(220, win_w - 2 * pad_x)
        play_rect = pygame.Rect(pad_x, 100, min(360, content_w), 60)
        model_rect = pygame.Rect(pad_x, 180, min(360, content_w), 60)
        red_path_rect = pygame.Rect(pad_x, 270, content_w, 46)
        red_browse_rect = pygame.Rect(pad_x, 322, min(180, content_w), 40)
        blue_path_rect = pygame.Rect(pad_x, 372, content_w, 46)
        blue_browse_rect = pygame.Rect(pad_x, 424, min(180, content_w), 40)
        red_id_rect = pygame.Rect(pad_x, 474, min(120, content_w), 42)
        blue_id_rect = pygame.Rect(pad_x + min(140, content_w), 474, min(120, content_w), 42)
        argmax_rect = pygame.Rect(pad_x, 528, min(180, content_w), 44)
        sample_rect = pygame.Rect(pad_x + min(200, content_w), 528, min(180, content_w), 44)
        start_rect = pygame.Rect(pad_x, 586, min(200, content_w), 56)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None, None
                if event.type == pygame.KEYDOWN and active is not None and mode == "model":
                    if event.key == pygame.K_BACKSPACE:
                        if active == "red_path":
                            model_paths["red"] = model_paths["red"][:-1]
                        elif active == "blue_path":
                            model_paths["blue"] = model_paths["blue"][:-1]
                        elif active == "red_id":
                            model_ids["red"] = model_ids["red"][:-1]
                        elif active == "blue_id":
                            model_ids["blue"] = model_ids["blue"][:-1]
                    elif event.key == pygame.K_RETURN:
                        if (
                            model_paths["red"].strip()
                            and model_paths["blue"].strip()
                            and model_ids["red"].strip()
                            and model_ids["blue"].strip()
                        ):
                            return "model", {
                                "red_path": model_paths["red"].strip(),
                                "blue_path": model_paths["blue"].strip(),
                                "red_id": int(model_ids["red"]),
                                "blue_id": int(model_ids["blue"]),
                                "policy_mode": policy_mode,
                            }
                        input_error = "Fill both checkpoint paths and both population IDs."
                    elif event.key in (pygame.K_ESCAPE,):
                        pass
                    elif event.unicode and event.unicode.isprintable():
                        if active in ("red_id", "blue_id"):
                            if event.unicode.isdigit():
                                target = "red" if active == "red_id" else "blue"
                                model_ids[target] += event.unicode
                        else:
                            target = "red" if active == "red_path" else "blue"
                            model_paths[target] += event.unicode
                    continue
                if event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_ESCAPE,
                    pygame.K_q,
                ):
                    return None, None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if play_rect.collidepoint(event.pos):
                        mode = "play"
                        input_error = ""
                    elif model_rect.collidepoint(event.pos):
                        mode = "model"
                    elif red_path_rect.collidepoint(event.pos):
                        active = "red_path"
                    elif blue_path_rect.collidepoint(event.pos):
                        active = "blue_path"
                    elif red_id_rect.collidepoint(event.pos):
                        active = "red_id"
                    elif blue_id_rect.collidepoint(event.pos):
                        active = "blue_id"
                    elif red_browse_rect.collidepoint(event.pos) and mode == "model":
                        selected = pick_model_path_via_os()
                        if selected:
                            model_paths["red"] = selected
                            input_error = ""
                        active = None
                    elif blue_browse_rect.collidepoint(event.pos) and mode == "model":
                        selected = pick_model_path_via_os()
                        if selected:
                            model_paths["blue"] = selected
                            input_error = ""
                        active = None
                    elif argmax_rect.collidepoint(event.pos) and mode == "model":
                        policy_mode = "argmax"
                        active = None
                    elif sample_rect.collidepoint(event.pos) and mode == "model":
                        policy_mode = "sample"
                        active = None
                    else:
                        active = None
                    if start_rect.collidepoint(event.pos):
                        if mode == "play":
                            return "play", {}
                        if (
                            model_paths["red"].strip()
                            and model_paths["blue"].strip()
                            and model_ids["red"].strip()
                            and model_ids["blue"].strip()
                        ):
                            return "model", {
                                "red_path": model_paths["red"].strip(),
                                "blue_path": model_paths["blue"].strip(),
                                "red_id": int(model_ids["red"]),
                                "blue_id": int(model_ids["blue"]),
                                "policy_mode": policy_mode,
                            }
                        input_error = "Fill both checkpoint paths and both population IDs."

            surface.fill((245, 242, 230))
            title = font.render("Coin Game Launcher", True, (35, 35, 35))
            surface.blit(title, (40, 28))

            for rect, label, selected in (
                (play_rect, "Play manually", mode == "play"),
                (model_rect, "Load pre-trained model", mode == "model"),
            ):
                color = (100, 170, 110) if selected else (190, 190, 190)
                draw_button(rect, label, color)

            if mode == "model":
                def draw_input(rect, value, placeholder, is_active):
                    pygame.draw.rect(surface, (255, 255, 255), rect, border_radius=4)
                    border = (70, 120, 200) if is_active else (70, 70, 70)
                    pygame.draw.rect(surface, border, rect, 2, border_radius=4)
                    text_value = value if value else placeholder
                    text_color = (20, 20, 20) if value else (120, 120, 120)
                    display_value = text_value
                    while small_font.size(display_value)[0] > rect.width - 16 and len(display_value) > 4:
                        display_value = "..." + display_value[4:]
                    surface.blit(
                        small_font.render(display_value, True, text_color),
                        (rect.x + 8, rect.y + 12),
                    )

                surface.blit(small_font.render("Red checkpoint", True, (35, 35, 35)), (pad_x, red_path_rect.y - 20))
                draw_input(red_path_rect, model_paths["red"], "Type red checkpoint folder...", active == "red_path")
                draw_button(red_browse_rect, "Pick Red Folder", (205, 205, 205))

                surface.blit(small_font.render("Blue checkpoint", True, (35, 35, 35)), (pad_x, blue_path_rect.y - 20))
                draw_input(blue_path_rect, model_paths["blue"], "Type blue checkpoint folder...", active == "blue_path")
                draw_button(blue_browse_rect, "Pick Blue Folder", (205, 205, 205))

                surface.blit(small_font.render("Red population ID", True, (35, 35, 35)), (pad_x, red_id_rect.y - 20))
                draw_input(red_id_rect, model_ids["red"], "0", active == "red_id")
                surface.blit(small_font.render("Blue population ID", True, (35, 35, 35)), (blue_id_rect.x, blue_id_rect.y - 20))
                draw_input(blue_id_rect, model_ids["blue"], "1", active == "blue_id")

                draw_button(argmax_rect, "Deterministic", (100, 170, 110) if policy_mode == "argmax" else (205, 205, 205))
                draw_button(sample_rect, "Sample Actions", (100, 170, 110) if policy_mode == "sample" else (205, 205, 205))

                hint = "Use one checkpoint per player. z_experiments checkpoints also need the correct population IDs."
                surface.blit(
                    small_font.render(hint, True, (70, 70, 70)),
                    (pad_x, sample_rect.bottom + 8),
                )
                if input_error:
                    surface.blit(
                        small_font.render(input_error, True, (180, 30, 30)),
                        (pad_x, start_rect.bottom + 10),
                    )

            draw_button(start_rect, "Start", (130, 150, 220))
            pygame.display.flip()
            pygame.time.Clock().tick(60)

    rng = jax.random.PRNGKey(0)
    env = CoinGame(
        num_inner_steps=100,
        num_outer_steps=9999,
        cnn=False,
        egocentric=False,
        shared_rewards=False,
    )
    obs, state = env.reset(rng)

    pygame.init()
    cell = 110
    margin = 24
    hud_h = 96
    width = env.grid_size * cell + 2 * margin
    height = env.grid_size * cell + 2 * margin + hud_h
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Coin Game")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 24)
    small_font = pygame.font.SysFont("monospace", 18)

    mode, model_cfg = startup_menu(screen, font, small_font)
    if mode is None:
        pygame.quit()
        raise SystemExit(0)

    if mode == "model":
        try:
            render_env_kwargs = _infer_render_env_kwargs(
                model_cfg["red_path"], model_cfg["blue_path"]
            )
            env = CoinGame(
                num_inner_steps=100,
                num_outer_steps=9999,
                **render_env_kwargs,
            )
            obs, state = env.reset(rng)
        except Exception as exc:
            pygame.quit()
            raise SystemExit(f"Model/environment setup failed: {exc}")

    bg = (245, 242, 230)
    grid = (170, 170, 170)
    red = (220, 40, 40)
    blue = (50, 90, 220)
    red_coin = (210, 70, 70)
    blue_coin = (70, 120, 220)
    text = (40, 40, 40)

    def action_from_key(key, keymap):
        if key == keymap["up"]:
            return 2
        if key == keymap["down"]:
            return 3
        if key == keymap["left"]:
            return 1
        if key == keymap["right"]:
            return 0
        return 4

    def rotate_ccw_cell(pos):
        return int(pos[1]), int(env.grid_size - 1 - pos[0])

    p1_map = {
        "up": pygame.K_w,
        "down": pygame.K_s,
        "left": pygame.K_a,
        "right": pygame.K_d,
    }
    p2_map = {
        "up": pygame.K_i,
        "down": pygame.K_k,
        "left": pygame.K_j,
        "right": pygame.K_l,
    }

    model_loaded = mode == "model"
    paused = False
    step_once = False
    model_error: Optional[str] = None
    model_policy_mode = "argmax"
    total_rewards = {"0": 0.0, "1": 0.0}
    last_step_rewards = {"0": 0.0, "1": 0.0}

    if model_loaded:
        try:
            (
                red_network,
                red_policy_state,
                red_policy_format,
                red_history_k,
                red_opponent_id_dim,
                red_augment_obs,
            ) = load_policy_for_coin_game(
                env, model_cfg["red_path"]
            )
            (
                blue_network,
                blue_policy_state,
                blue_policy_format,
                blue_history_k,
                blue_opponent_id_dim,
                blue_augment_obs,
            ) = load_policy_for_coin_game(
                env, model_cfg["blue_path"]
            )
            obs_dim = env.observation_space().shape[-1]
            red_obs_hist = jnp.zeros((red_history_k, obs_dim), dtype=jnp.float32)
            blue_obs_hist = jnp.zeros((blue_history_k, obs_dim), dtype=jnp.float32)
            red_opp_act_hist = jnp.zeros((red_history_k,), dtype=jnp.float32)
            blue_opp_act_hist = jnp.zeros((blue_history_k,), dtype=jnp.float32)
            red_population_id = int(model_cfg["red_id"])
            blue_population_id = int(model_cfg["blue_id"])
            model_policy_mode = model_cfg.get("policy_mode", "argmax")
            paused = True
            pygame.display.set_caption("Coin Game - Model playback (Space pause/resume, N step)")
        except Exception as exc:
            model_loaded = False
            model_error = f"Model load failed: {exc}"
            pygame.display.set_caption("Coin Game - P1: WASD | P2: IJKL | Esc/Q: Quit")
    else:
        pygame.display.set_caption("Coin Game - P1: WASD | P2: IJKL | Esc/Q: Quit")

    hud_title_font = pygame.font.SysFont("monospace", 20)
    hud_body_font = pygame.font.SysFont("monospace", 16)
    pause_rect = pygame.Rect(margin, 8, 120, 34)
    step_rect = pygame.Rect(margin, 48, 120, 34)
    hud_height_total = margin + hud_h
    control_block_w = 130 if model_loaded else 0
    text_rect = pygame.Rect(
        margin + control_block_w,
        8,
        width - (2 * margin) - control_block_w,
        hud_height_total - 16,
    )

    def draw_centered_text_button(rect, label, color):
        pygame.draw.rect(screen, color, rect, border_radius=5)
        txt = small_font.render(label, True, (20, 20, 20))
        x = rect.x + (rect.width - txt.get_width()) // 2
        y = rect.y + (rect.height - txt.get_height()) // 2
        screen.blit(txt, (x, y))

    def wrap_text(msg: str, draw_font, max_width: int):
        words = msg.split()
        if not words:
            return [""]
        lines = []
        cur = words[0]
        for w in words[1:]:
            candidate = f"{cur} {w}"
            if draw_font.size(candidate)[0] <= max_width:
                cur = candidate
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        return lines

    running = True
    while running:
        a1 = 4
        a2 = 4
        should_step = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            if model_loaded and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_n:
                    step_once = True
            if model_loaded and event.type == pygame.MOUSEBUTTONDOWN:
                if pause_rect.collidepoint(event.pos):
                    paused = not paused
                elif step_rect.collidepoint(event.pos):
                    step_once = True
            if event.type == pygame.KEYDOWN:
                p1_action = action_from_key(event.key, p1_map)
                p2_action = action_from_key(event.key, p2_map)
                if (not model_loaded) and p1_action != 4:
                    a1 = p1_action
                    should_step = True
                if (not model_loaded) and p2_action != 4:
                    a2 = p2_action
                    should_step = True

        rewards = last_step_rewards
        dones = {"__all__": False}
        if model_loaded:
            if (not paused) or step_once:
                red_obs_aug = red_augment_obs(
                    obs[env.agents[0]],
                    red_obs_hist,
                    red_opp_act_hist,
                    blue_population_id,
                )
                blue_obs_aug = blue_augment_obs(
                    obs[env.agents[1]],
                    blue_obs_hist,
                    blue_opp_act_hist,
                    red_population_id,
                )
                red_pi, _ = red_network.apply(red_policy_state.params, red_obs_aug)
                blue_pi, _ = blue_network.apply(blue_policy_state.params, blue_obs_aug)
                if model_policy_mode == "sample":
                    rng, red_act_key, blue_act_key = jax.random.split(rng, 3)
                    red_action = int(red_pi.sample(seed=red_act_key))
                    blue_action = int(blue_pi.sample(seed=blue_act_key))
                else:
                    red_action = int(jnp.argmax(red_pi.logits))
                    blue_action = int(jnp.argmax(blue_pi.logits))
                env_actions = {env.agents[0]: red_action, env.agents[1]: blue_action}
                rng, step_key = jax.random.split(rng)
                obs, state, rewards, dones, _ = env.step(step_key, state, env_actions)
                last_step_rewards = {
                    "0": float(rewards["0"]),
                    "1": float(rewards["1"]),
                }
                total_rewards["0"] += float(rewards["0"])
                total_rewards["1"] += float(rewards["1"])
                if red_history_k > 0:
                    red_obs_hist = jnp.concatenate(
                        [red_obs_hist[1:], jnp.asarray(obs[env.agents[0]], dtype=jnp.float32)[None, :]],
                        axis=0,
                    )
                    red_opp_act_hist = jnp.concatenate(
                        [red_opp_act_hist[1:], jnp.asarray([blue_action], dtype=jnp.float32)],
                        axis=0,
                    )
                if blue_history_k > 0:
                    blue_obs_hist = jnp.concatenate(
                        [blue_obs_hist[1:], jnp.asarray(obs[env.agents[1]], dtype=jnp.float32)[None, :]],
                        axis=0,
                    )
                    blue_opp_act_hist = jnp.concatenate(
                        [blue_opp_act_hist[1:], jnp.asarray([red_action], dtype=jnp.float32)],
                        axis=0,
                    )
                step_once = False
        elif should_step:
            rng, step_key = jax.random.split(rng)
            obs, state, rewards, dones, _ = env.step(step_key, state, {"0": a1, "1": a2})
            last_step_rewards = {
                "0": float(rewards["0"]),
                "1": float(rewards["1"]),
            }
            total_rewards["0"] += float(rewards["0"])
            total_rewards["1"] += float(rewards["1"])

        screen.fill(bg)
        grid_origin_y = margin + hud_h
        pygame.draw.rect(screen, (236, 233, 223), (0, 0, width, hud_height_total))

        for gx in range(env.grid_size + 1):
            x = margin + gx * cell
            pygame.draw.line(
                screen, grid, (x, grid_origin_y), (x, grid_origin_y + env.grid_size * cell), 2
            )
        for gy in range(env.grid_size + 1):
            y = grid_origin_y + gy * cell
            pygame.draw.line(
                screen, grid, (margin, y), (margin + env.grid_size * cell, y), 2
            )

        rcx, rcy = rotate_ccw_cell(state.red_coin_pos)
        bcx, bcy = rotate_ccw_cell(state.blue_coin_pos)
        pygame.draw.circle(
            screen,
            red_coin,
            (rcx * cell + margin + cell // 2, rcy * cell + grid_origin_y + cell // 2),
            cell // 7,
        )
        pygame.draw.circle(
            screen,
            blue_coin,
            (bcx * cell + margin + cell // 2, bcy * cell + grid_origin_y + cell // 2),
            cell // 7,
        )

        rpx, rpy = rotate_ccw_cell(state.red_pos)
        bpx, bpy = rotate_ccw_cell(state.blue_pos)
        pygame.draw.circle(
            screen,
            red,
            (rpx * cell + margin + cell // 2, rpy * cell + grid_origin_y + cell // 2),
            cell // 4,
        )
        pygame.draw.circle(
            screen,
            blue,
            (bpx * cell + margin + cell // 2, bpy * cell + grid_origin_y + cell // 2),
            cell // 4,
        )

        episode_idx = int(state.outer_t)
        line1 = (
            f"Step {int(state.inner_t)}  Episode {episode_idx}  "
            f"Total rewards: {total_rewards['0']:.0f}, {total_rewards['1']:.0f}"
        )
        line2 = (
            f"Last rewards: {float(rewards['0']):.0f}, {float(rewards['1']):.0f}"
        )
        if bool(dones["__all__"]):
            line2 += "   [Episode reset]"
        if model_loaded:
            pause_label = "Resume" if paused else "Pause"
            draw_centered_text_button(pause_rect, pause_label, (230, 210, 120))
            draw_centered_text_button(step_rect, "Step", (170, 200, 220))
            line2 += f"  Model: {'paused' if paused else 'running'} {model_policy_mode} (Space, N)"
        if model_error:
            line2 += f"  {model_error}"

        title_lines = wrap_text(line1, hud_title_font, text_rect.width)
        body_lines = wrap_text(line2, hud_body_font, text_rect.width)
        y_cursor = text_rect.y + 4
        for ln in title_lines:
            rendered = hud_title_font.render(ln, True, text)
            if y_cursor + rendered.get_height() > text_rect.bottom:
                break
            screen.blit(rendered, (text_rect.x, y_cursor))
            y_cursor += rendered.get_height() + 2

        y_cursor += 2
        for ln in body_lines:
            rendered = hud_body_font.render(ln, True, text)
            if y_cursor + rendered.get_height() > text_rect.bottom:
                break
            screen.blit(rendered, (text_rect.x, y_cursor))
            y_cursor += rendered.get_height() + 2

        pygame.display.flip()
        clock.tick(8 if model_loaded else 60)

    pygame.quit()
