# `cyberFly/QuadX-Pole-Balance-v3`

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/quadx_pole_balance.gif
    :width: 50%
```

## Task Description

The goal of this environment is to hover a quadrotor drone for as long as possible while balancing a 1 meter long pole.

## Usage

```python
import gymnasium
import cyberFly.gym_envs

env = gymnasium.make("cyberFly/QuadX-Pole-Balance-v3", render_mode="human")

term, trunc = False, False
obs, _ = env.reset()
while not (term or trunc):
    obs, rew, term, trunc, _ = env.step(env.action_space.sample())
```

## Environment Options

```{eval-rst}
.. autoclass:: cyberFly.gym_envs.quadx_envs.quadx_pole_balance_env.QuadXPoleBalanceEnv
```
