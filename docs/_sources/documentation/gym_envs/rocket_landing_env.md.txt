# `cyberFly/Rocket-Landing-v3`

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/rocket_landing_2.gif
    :width: 50%
```

## Task Description

The goal of this environment is to land a rocket falling at terminal velocity on a landing pad, with only 1% of fuel remaining.

## Usage

```python
import gymnasium
import cyberFly.gym_envs

env = gymnasium.make("cyberFly/Rocket-Landing-v4", render_mode="human")

term, trunc = False, False
obs, _ = env.reset()
while not (term or trunc):
    obs, rew, term, trunc, _ = env.step(env.action_space.sample())
```

## Environment Options

```{eval-rst}
.. autoclass:: cyberFly.gym_envs.rocket_envs.rocket_landing_env.RocketLandingEnv
```
