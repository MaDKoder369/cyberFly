# Gymnasium Beginner's Guide

## What is Gymnasium?

[Gymnasium](https://gymnasium.farama.org/) is a Python library that provides a standard API for reinforcement learning (RL) environments. It is the maintained fork of OpenAI Gym. You define an **agent** that interacts with an **environment** by taking **actions** and receiving **observations** and **rewards**.

---

## Installation

```bash
pip install gymnasium
```

For environments that need extra dependencies (e.g., physics simulators):

```bash
pip install gymnasium[box2d]    # Box2D environments (LunarLander, CarRacing, …)
pip install gymnasium[atari]    # Atari games
pip install gymnasium[all]      # Everything
```

---

## Core Concepts

| Term | Meaning |
|---|---|
| **Environment** | The world the agent lives in |
| **Observation** | What the agent *sees* each step |
| **Action** | What the agent *does* each step |
| **Reward** | A scalar signal telling the agent how well it did |
| **Episode** | One full run from reset to termination |
| **Step** | One tick of the environment loop |

---

## The Basic Loop

Every Gymnasium environment follows the same pattern:

```python
import gymnasium as gym

# 1. Create the environment
env = gym.make("CartPole-v1", render_mode="human")

# 2. Reset to get the first observation
obs, info = env.reset()

for step in range(1000):
    # 3. Choose an action (random here, replace with your policy)
    action = env.action_space.sample()

    # 4. Step the environment
    obs, reward, terminated, truncated, info = env.step(action)

    # 5. Check if the episode is over
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### What `env.step()` returns

| Return value | Type | Meaning |
|---|---|---|
| `obs` | array / dict | New observation after the action |
| `reward` | float | Reward for the last action |
| `terminated` | bool | Episode ended naturally (e.g., goal reached / fell over) |
| `truncated` | bool | Episode ended due to a time limit |
| `info` | dict | Extra diagnostic data (environment-specific) |

---

## Action and Observation Spaces

Spaces describe the *shape* and *type* of actions and observations.

```python
env = gym.make("CartPole-v1")

print(env.action_space)        # Discrete(2)  → 0 or 1
print(env.observation_space)   # Box([-4.8 … 4.8], shape=(4,), float32)
```

### Common space types

| Space | Description | Example |
|---|---|---|
| `Discrete(n)` | Integer from 0 to n-1 | Jump / No jump |
| `Box(low, high, shape)` | Continuous float array | Joint angles, velocity |
| `Dict(…)` | Dictionary of spaces | Multiple sensor readings |
| `MultiBinary(n)` | Array of 0/1 flags | Multi-button gamepad |

### Sampling a random action

```python
action = env.action_space.sample()
```

### Checking if an action is valid

```python
assert env.action_space.contains(action)
```

---

## Resetting the Environment

`env.reset()` starts a new episode and returns the first observation.

```python
# Simple reset
obs, info = env.reset()

# Reset with a fixed random seed (for reproducibility)
obs, info = env.reset(seed=42)
```

---

## Render Modes

Control how the environment is displayed:

```python
env = gym.make("CartPole-v1", render_mode="human")    # Opens a window
env = gym.make("CartPole-v1", render_mode="rgb_array") # Returns pixel array
env = gym.make("CartPole-v1", render_mode=None)        # No rendering (fastest)
```

---

## Wrappers

Wrappers modify an existing environment without changing its source code.

```python
import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordVideo

env = gym.make("CartPole-v1")
env = TimeLimit(env, max_episode_steps=200)       # Cut episodes short
env = RecordVideo(env, video_folder="./videos")   # Save video
```

### Useful built-in wrappers

| Wrapper | Purpose |
|---|---|
| `TimeLimit` | Truncate episodes after N steps |
| `RecordVideo` | Save episode videos to disk |
| `RecordEpisodeStatistics` | Track cumulative reward and length |
| `FlattenObservation` | Flatten dict/tuple observations to a 1-D array |
| `ClipAction` | Clip continuous actions to the valid range |
| `NormalizeObservation` | Normalize observations to zero mean / unit variance |

---

## Running Multiple Episodes

```python
import gymnasium as gym

env = gym.make("CartPole-v1")

for episode in range(5):
    obs, info = env.reset(seed=episode)
    total_reward = 0
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Episode {episode + 1}: total reward = {total_reward}")

env.close()
```

---

## Using a Custom Environment (like cyberFly)

Third-party packages register their own environments. After importing the package, `gym.make` finds them automatically:

```python
import gymnasium
import cyberFly.gym_envs  # registers cyberFly environments

env = gymnasium.make(
    "cyberFly/QuadX-DotTouch-v1",
    render_mode="human",
    flight_mode=6,
)

obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

---

## Reproducibility Tips

```python
import numpy as np

# Fix all random seeds
seed = 42
obs, info = env.reset(seed=seed)
np.random.seed(seed)

# Always use env.action_space.sample() for baselines
# so the environment's RNG is consumed consistently
```

---

## Common Mistakes

| Mistake | Fix |
|---|---|
| Forgetting `env.reset()` before the loop | Always call `reset()` before the first `step()` |
| Ignoring `truncated` | Check `terminated or truncated` to detect episode end |
| Calling `env.step()` after the episode ended | Reset when `terminated or truncated` is `True` |
| Not calling `env.close()` | Call `close()` to free resources and close render windows |

---

## How Gymnasium Works Under the Hood

### The `Env` Base Class

Every environment — built-in or custom — is a Python class that inherits from `gymnasium.Env`:

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        # Define spaces
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)   # seeds self.np_random
        obs = self.observation_space.sample()
        info = {}
        return obs, info

    def step(self, action):
        obs = self.observation_space.sample()
        reward = 1.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self): ...
    def close(self): ...
```

The four required methods are `reset`, `step`, `render`, and `close`. `observation_space` and `action_space` are required attributes.

---

### The MDP Model

Gymnasium models the world as a **Markov Decision Process (MDP)**:

$$s_{t+1} \sim P(s_{t+1} \mid s_t, a_t)$$

$$r_t = R(s_t, a_t, s_{t+1})$$

| Symbol | Gymnasium name | Description |
|---|---|---|
| $s_t$ | internal env state | True state of the world (hidden) |
| $o_t$ | `obs` | Observation the agent receives |
| $a_t$ | `action` | Action the agent picks |
| $r_t$ | `reward` | Scalar reward signal |
| $P$ | `step()` internals | Transition dynamics |
| $R$ | `step()` internals | Reward function |

The agent only ever sees observations $o_t$, not the true state $s_t$ (unless the environment is fully observable, where $o_t = s_t$).

---

### What Happens Inside `env.step(action)`

```
agent picks action
       │
       ▼
  env.step(action)
  ┌──────────────────────────────────────────┐
  │  1. Apply action to internal simulator   │
  │     (physics engine, game logic, etc.)   │
  │                                          │
  │  2. Compute next observation from state  │
  │                                          │
  │  3. Compute reward                       │
  │                                          │
  │  4. Check termination conditions         │
  │     - terminated  (natural end)          │
  │     - truncated   (time limit hit)       │
  │                                          │
  │  5. Collect diagnostic info dict         │
  └──────────────────────────────────────────┘
       │
       ▼
  returns (obs, reward, terminated, truncated, info)
```

---

### Random Number Generator — `self.np_random`

When you call `env.reset(seed=42)`, the base class seeds a `numpy.random.Generator` stored at `self.np_random`. Always use this generator inside your environment instead of global `np.random` so that results are reproducible:

```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)          # sets self.np_random
    x = self.np_random.uniform(-1, 1) # reproducible randomness
    ...
```

---

### Environment Registration

`gym.make("CartPole-v1")` works because environments are registered in a global **registry**. Third-party packages do this with `gymnasium.register()` — usually in their `__init__.py`:

```python
# Inside cyberFly/gym_envs/__init__.py (simplified)
import gymnasium

gymnasium.register(
    id="cyberFly/QuadX-DotTouch-v1",
    entry_point="cyberFly.gym_envs.quadx_envs:QuadXDotTouchEnv",
    max_episode_steps=1000,
)
```

After `import cyberFly.gym_envs`, the registry knows the id → class mapping and `gym.make` can find it.

You can inspect every registered environment:

```python
import gymnasium as gym

for env_id, spec in gym.envs.registry.items():
    print(env_id)
```

---

### How Wrappers Work

A `Wrapper` is itself a `gym.Env` that holds a reference to the original (`self.env`) and delegates calls through:

```
RecordVideo
  └── TimeLimit
        └── CartPoleEnv   ← the real environment
```

Each wrapper intercepts only what it cares about and forwards everything else unchanged:

```python
class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True          # force truncation
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
```

You can access the unwrapped environment at any time:

```python
base_env = env.unwrapped
```

---

### `terminated` vs `truncated` — Why Two Flags?

This distinction matters for correctly computing returns in RL algorithms:

| Flag | Meaning | Bootstrap value? |
|---|---|---|
| `terminated = True` | Natural end — the MDP reached an absorbing state (goal/failure). Future reward is **zero**. | No — $V(s_{T+1}) = 0$ |
| `truncated = True` | Artificial cut-off — time limit or out-of-bounds check. The MDP is *not* truly over. | Yes — $V(s_{T+1}) \neq 0$ |

Algorithms like PPO and TD3 use `truncated` to decide whether to bootstrap the value of the last state.

---

### The Observation and Action Space Objects

Spaces are not just descriptions — they are objects with useful methods:

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

sample = space.sample()           # random point inside the space
space.contains(sample)            # True — validates membership
space.shape                       # (3,)
space.dtype                       # float32
space.low, space.high             # bounds arrays

discrete = spaces.Discrete(5)
discrete.n                        # 5 — number of valid actions (0..4)
```

The space's internal RNG is seeded when you call `env.reset(seed=…)`, which is why `action_space.sample()` is also reproducible after a seed.

---

### Vectorised Environments (`VectorEnv`)

For training speed, Gymnasium supports running *N* environments in parallel under a single API:

```python
import gymnasium as gym

# Run 4 CartPole environments at once
envs = gym.make_vec("CartPole-v1", num_envs=4, vectorization_mode="sync")

obs, info = envs.reset(seed=0)    # obs.shape == (4, 4)
actions = envs.action_space.sample()  # shape (4,)
obs, rewards, terminated, truncated, info = envs.step(actions)

envs.close()
```

Each call to `step` advances all 4 environments simultaneously, returning batched arrays. This is how most modern RL training frameworks (Stable-Baselines3, CleanRL, etc.) achieve high sample throughput.

---

## The Mathematics Behind Gymnasium

This section builds the math from scratch. No prior RL knowledge is required.

---

### 1. What Is a Markov Decision Process?

A **Markov Decision Process (MDP)** is the formal framework that every Gymnasium environment implements. It is a tuple:

$$\mathcal{M} = \langle \mathcal{S},\, \mathcal{A},\, P,\, R,\, \gamma,\, \rho_0 \rangle$$

| Symbol | Name | Meaning |
|---|---|---|
| $\mathcal{S}$ | State space | All possible states the world can be in |
| $\mathcal{A}$ | Action space | All possible actions the agent can take |
| $P(s' \mid s, a)$ | Transition function | Probability of landing in state $s'$ after taking action $a$ in state $s$ |
| $R(s, a, s')$ | Reward function | Scalar signal received after the transition |
| $\gamma \in [0,1)$ | Discount factor | How much future rewards are worth compared to immediate ones |
| $\rho_0(s)$ | Initial state distribution | Probability of starting in each state (what `reset()` samples from) |

**The Markov property** is the key assumption: the next state depends *only* on the current state and action — not on any history before that.

$$P(s_{t+1} \mid s_0, a_0, \ldots, s_t, a_t) = P(s_{t+1} \mid s_t, a_t)$$

---

### 2. Trajectories and Returns

An **episode** is a sequence (called a trajectory) of states, actions, and rewards:

$$\tau = (s_0, a_0, r_0,\; s_1, a_1, r_1,\; \ldots,\; s_T)$$

The **return** $G_t$ is the total reward collected from time step $t$ to the end of the episode:

$$G_t = r_t + r_{t+1} + r_{t+2} + \cdots + r_T = \sum_{k=0}^{T-t} r_{t+k}$$

But rewards far in the future are less reliable, so we usually **discount** them:

$$G_t = r_t + \gamma\, r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k\, r_{t+k}$$

**Why discount?**
- $\gamma = 0$ → the agent cares only about the immediate reward (greedy).
- $\gamma = 1$ → the agent treats all future rewards equally (only valid for finite episodes).
- $\gamma = 0.99$ → a reward 100 steps away is worth $0.99^{100} \approx 0.37$ of a reward now.

In practice Gymnasium environments use $\gamma$ values like $0.99$ or $0.999$.

---

### 3. Policies

A **policy** $\pi$ is the agent's decision rule — it maps observations to actions.

**Deterministic policy** — always picks the same action for a given state:

$$a_t = \pi(s_t)$$

**Stochastic policy** — samples an action from a probability distribution:

$$a_t \sim \pi(a \mid s_t)$$

where $\pi(a \mid s_t)$ is a probability distribution over actions given state $s_t$.

> In Gymnasium code, `action = env.action_space.sample()` is a **uniform random** policy — every action is equally likely.

---

### 4. Value Functions

Value functions tell us *how good* a state (or state-action pair) is under a policy $\pi$.

#### State-Value Function $V^\pi(s)$

Expected discounted return starting from state $s$ and following policy $\pi$ forever:

$$V^\pi(s) = \mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty} \gamma^k\, r_{t+k} \;\middle|\; s_t = s\right]$$

In plain English: "If I am in state $s$ and follow policy $\pi$, how much total (discounted) reward do I expect to collect?"

#### Action-Value Function $Q^\pi(s, a)$

Expected discounted return after taking action $a$ in state $s$, then following $\pi$:

$$Q^\pi(s,a) = \mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty} \gamma^k\, r_{t+k} \;\middle|\; s_t = s,\; a_t = a\right]$$

The relationship between the two:

$$V^\pi(s) = \sum_{a} \pi(a \mid s)\; Q^\pi(s, a)$$

For a deterministic policy this simplifies to $V^\pi(s) = Q^\pi(s, \pi(s))$.

---

### 5. The Bellman Equations

The **Bellman equations** are recursive identities that express the value of a state in terms of the value of the *next* state. They are the mathematical backbone of nearly every RL algorithm.

#### Bellman Expectation Equation (for $V^\pi$)

$$V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\!\left[R(s,a,s') + \gamma\, V^\pi(s')\right]$$

Reading it step by step:
1. $\sum_{a} \pi(a \mid s)$ — average over all actions the policy might take.
2. $\sum_{s'} P(s' \mid s, a)$ — average over all states the environment might transition to.
3. $R(s,a,s') + \gamma V^\pi(s')$ — immediate reward plus discounted value of the next state.

#### Bellman Optimality Equation (for $V^*$)

The **optimal value function** $V^*(s)$ is achieved by the best possible policy:

$$V^*(s) = \max_{a}\; \sum_{s'} P(s' \mid s, a)\!\left[R(s,a,s') + \gamma\, V^*(s')\right]$$

The optimal policy simply picks the action that maximises this:

$$\pi^*(s) = \arg\max_{a}\; \sum_{s'} P(s' \mid s, a)\!\left[R(s,a,s') + \gamma\, V^*(s')\right]$$

> Most RL algorithms (Q-learning, DQN, PPO, …) are different ways of solving — or approximating — these equations when the transition model $P$ is **unknown**.

---

### 6. The Objective — What the Agent Is Maximising

The formal goal of RL is to find a policy $\pi$ that maximises the **expected return** from the initial state distribution:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi}\!\left[G_0\right] = \mathbb{E}_{\tau \sim \pi}\!\left[\sum_{t=0}^{T} \gamma^t\, r_t\right]$$

Every call to `env.step()` adds one term $r_t$ to this sum. Every training algorithm is trying to push $J(\pi)$ as high as possible.

---

### 7. Temporal Difference (TD) Error

Algorithms learn by comparing what they *predicted* to what actually *happened*. This error is called the **TD error** $\delta$:

$$\delta_t = r_t + \gamma\, V(s_{t+1}) - V(s_t)$$

- $r_t + \gamma V(s_{t+1})$ is the **TD target** — a one-step estimate of how good state $s_t$ really is.
- $V(s_t)$ is the **current prediction**.
- $\delta_t$ is the surprise: positive means things went better than expected, negative means worse.

The agent updates its value estimate in the direction of the TD error:

$$V(s_t) \leftarrow V(s_t) + \alpha\, \delta_t$$

where $\alpha \in (0,1]$ is the **learning rate**.

---

### 8. Policy Gradient Intuition

Instead of learning a value function and deriving a policy from it, **policy gradient** methods directly optimise $J(\pi_\theta)$ with respect to the policy parameters $\theta$.

The **policy gradient theorem** states:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t\right]$$

In plain English: increase the probability of actions that led to high returns, decrease the probability of actions that led to low returns. The term $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ points in the direction that makes action $a_t$ more likely under the current policy.

---

### 9. Putting It All Together — One Training Step

```
┌─────────────────────────────────────────────────────────────┐
│  t=0: obs = env.reset()                                     │
│                                                             │
│  for each step t:                                           │
│    1. a_t  ~ π_θ(· | obs_t)         ← sample from policy  │
│    2. obs_{t+1}, r_t = env.step(a_t) ← Gymnasium API       │
│    3. δ_t = r_t + γ·V(obs_{t+1}) - V(obs_t)  ← TD error   │
│    4. θ ← θ + α · ∇_θ log π_θ(a_t|obs_t) · G_t            │
│                                                             │
│  if terminated or truncated:  env.reset()                   │
└─────────────────────────────────────────────────────────────┘
```

Gymnasium provides steps 2 and the episode boundary detection. Steps 1, 3, and 4 are handled by the RL algorithm (Stable-Baselines3, CleanRL, custom code, etc.).

---

### 10. Key Numbers to Know

| Quantity | Typical range | What happens at extremes |
|---|---|---|
| Discount factor $\gamma$ | 0.95 – 0.999 | $\gamma \to 0$: shortsighted agent; $\gamma \to 1$: may never converge in infinite episodes |
| Learning rate $\alpha$ | 1e-4 – 3e-3 | Too high → unstable; too low → very slow learning |
| Episode length $T$ | 200 – 10 000 steps | Longer episodes → more variance in return estimates |
| Reward scale | Roughly −1 to +1 is ideal | Very large rewards make learning unstable |

---

## Further Reading

- Official docs: <https://gymnasium.farama.org/>
- API reference: <https://gymnasium.farama.org/api/env/>
- Built-in environments: <https://gymnasium.farama.org/environments/classic_control/>
- Creating custom environments: <https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/>
