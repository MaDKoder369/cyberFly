# Dot Touch Game – Deep Explanation

## Overview

The **Dot Touch Game** (`cyberFly/QuadX-DotTouch-v1`) is a Gymnasium
reinforcement-learning environment built on top of the cyberFly UAV simulator.
A quadrotor drone is spawned in a 3-D arena that contains several randomly
scattered "dots" (small coloured spheres). The drone's objective is to fly to
each dot and **touch it exactly once**. Touching every dot completes the
episode with a large reward; touching an already-visited dot triggers a heavy
penalty and immediately ends the episode.

---

## Game Rules

| Rule | Detail |
|------|--------|
| **Touch a new dot** | +50 reward, dot turns grey |
| **Touch an already-visited dot** | −100 reward, episode **terminates** |
| **All dots touched** | +200 bonus reward, episode **completes** |
| **Crash into the ground** | Base −100 penalty and episode terminates; the final step reward may still include other same-step reward components |
| **Leave the flight dome** | Base −100 penalty and episode terminates; the final step reward may still include other same-step reward components |
| **Time runs out** | Episode truncated; no added timeout penalty, though the final step reward may still include other reward terms already computed |

---

## Environment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_dots` | 6 | Number of dots placed in the arena |
| `dot_touch_distance` | 0.5 | Distance (metres) at which a dot counts as "touched" |
| `flight_mode` | 0 | PyBullet flight-control mode (0 = angular-rate + thrust) |
| `flight_dome_size` | 5.0 | Radius of the invisible dome the drone must stay inside |
| `max_duration_seconds` | 20.0 | Wall-clock simulation limit before the episode is truncated |
| `angle_representation` | `"quaternion"` | `"euler"` or `"quaternion"` for orientation encoding |
| `agent_hz` | 30 | How many times per second the RL agent gets to act |
| `render_mode` | `None` | `"human"` for real-time 3-D view, `"rgb_array"` for camera images |
| `render_resolution` | `(480, 480)` | Resolution of the rendered image |

---

## Observation Space (Dict)

The observation returned at every step is a dictionary with three keys:

### 1. `"attitude"` — Drone State Vector

A flat numpy array combining:

| Component | Size | Notes |
|-----------|------|-------|
| Angular velocity | 3 | rad/s around body axes |
| Angular position | 3 (euler) or 4 (quaternion) | Orientation |
| Linear velocity | 3 | m/s in world frame |
| Linear position | 3 | (x, y, z) in world frame |
| Previous action | 4 | Last motor command sent |
| Auxiliary state | 4 | Extra sensor data from the drone |

Total size: **20** (quaternion) or **19** (euler).

### 2. `"dot_deltas"` — Body-Frame Vectors to Every Dot

Shape: `(num_dots, 3)`

Each row is a 3-D vector **in the drone's body frame** pointing from the
drone's current position to the corresponding dot. Body-frame representation
is important because it is invariant to the drone's absolute position and
heading, making it much easier for a neural-network policy to learn spatial
relationships.

**How it is computed:**

```
rotation = quaternion_to_rotation_matrix(drone_quaternion)   # 3×3
dot_deltas = (dot_positions - drone_position) @ rotation     # (N, 3)
```

### 3. `"dot_touched"` — Binary Touch Mask

Shape: `(num_dots,)`

A vector of `0.0` and `1.0` values.
- `0.0` → the dot has **not** been touched yet (still a valid target).
- `1.0` → the dot has already been touched (must be avoided).

This mask gives the agent explicit memory of which dots it has visited,
which is critical for learning to avoid double-touches.

---

## Action Space

Continuous `Box(4,)`:

| Index | Meaning | Range |
|-------|---------|-------|
| 0 | Roll rate (vp) | [−π, π] |
| 1 | Pitch rate (vq) | [−π, π] |
| 2 | Yaw rate (vr) | [−π, π] |
| 3 | Thrust (T) | [0, 0.8] |

These correspond to desired angular rates around the body axes plus a
normalised collective thrust.

---

## Reward Design

The reward function is shaped to guide the agent towards touching all dots
while avoiding double-touches:

### Per-Step Baseline

Every step starts with a small **living penalty** of **−0.1** (inherited from
the base environment). This encourages the agent to finish quickly.

### Proximity Shaping

At each step, the distance to the **nearest untouched dot** is computed:

```
shaping_reward = 0.1 / max(nearest_untouched_distance, 0.01)
```

This reward is highest when the drone is very close to an untouched dot and
vanishes as the drone moves far away, creating a smooth gradient that guides
the policy toward the closest unvisited target.

### Touch Bonus

When the drone enters the `dot_touch_distance` radius of an untouched dot it
receives **+50** reward. The dot is marked as touched, and its visual changes
to grey so the human viewer can see progress.

### Double-Touch Penalty

If the drone enters the radius of a **previously touched** dot, it receives
**−100** reward and the episode **terminates immediately**. This makes
avoidance of visited dots a hard constraint.

### All-Dots-Touched Bonus

When every dot has been visited, the agent receives an additional **+200**
bonus and the episode is flagged as complete.

### Safety Penalties (inherited)

- **Ground collision**: −100, termination.
- **Out of flight dome**: −100, termination.

---

## Episode Lifecycle

```
┌──────────────┐
│   env.reset()│
└──────┬───────┘
       │
       ▼
 ┌─────────────────────────────┐
 │ Scatter num_dots random dots│
 │ touched[] = all False       │
 │ Render green spheres        │
 └──────────┬──────────────────┘
            │
            ▼
 ┌──────────────────────────────┐◄────────────────┐
 │  Agent observes state dict   │                  │
 │  Agent picks action          │                  │
 └──────────┬───────────────────┘                  │
            │                                      │
            ▼                                      │
 ┌──────────────────────────────┐                  │
 │  Simulation advances         │                  │
 │  (env_step_ratio sub-steps)  │                  │
 └──────────┬───────────────────┘                  │
            │                                      │
            ▼                                      │
 ┌──────────────────────────────┐                  │
 │  For each dot:               │                  │
 │   distance < threshold?      │                  │
 │     YES & untouched → +50    │                  │
 │     YES & touched   → −100   │──► TERMINATE     │
 │  All touched? → +200         │──► COMPLETE       │
 │  Crash / OOB? → −100        │──► TERMINATE     │
 │  Time up?                    │──► TRUNCATE      │
 └──────────┬───────────────────┘                  │
            │ (none of the above)                  │
            └──────────────────────────────────────┘
```

---

## Dot Placement Algorithm

Dots are placed using **spherical coordinates** sampled uniformly:

1. Sample `θ ∈ [0, 2π)` (azimuth) and `φ ∈ [0, π]` (polar angle).
2. Sample `r ∈ [1.0, 0.9 × flight_dome_size]`.
3. Convert to Cartesian:
   - `x = r · sin(φ) · cos(θ)`
   - `y = r · sin(φ) · sin(θ)`
   - `z = |r · cos(φ)|`
4. Clamp `z` to a minimum of 0.3 m so dots are never on the ground.

This distributes dots throughout the dome with a guaranteed minimum distance
from the drone's spawn point (r ≥ 1.0 m).

---

## How to Use

### Quick Start

```python
import gymnasium
import cyberFly.gym_envs  # registers all cyberFly environments

env = gymnasium.make("cyberFly/QuadX-DotTouch-v1", render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Training with Stable-Baselines3

```python
from stable_baselines3 import PPO
import gymnasium
import cyberFly.gym_envs

env = gymnasium.make("cyberFly/QuadX-DotTouch-v1")
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
model.save("dot_touch_ppo")
```

### Custom Parameters

```python
env = gymnasium.make(
    "cyberFly/QuadX-DotTouch-v1",
    num_dots=10,
    dot_touch_distance=0.3,
    flight_dome_size=8.0,
    max_duration_seconds=30.0,
)
```

---

## File Structure

| File | Purpose |
|------|---------|
| `cyberFly/gym_envs/quadx_envs/quadx_dot_touch_env.py` | Environment implementation |
| `cyberFly/gym_envs/__init__.py` | Gymnasium registration (`cyberFly/QuadX-DotTouch-v1`) |
| `DOT_TOUCH_GAME.md` | This explanation document |

---

## Key Design Decisions

1. **Touched mask in the observation** – The `dot_touched` vector gives the
   agent explicit knowledge of which dots have already been visited. Without
   it, the environment would be partially observable and much harder to solve.

2. **Body-frame deltas** – Expressing dot positions relative to the drone's
   body frame (rather than world frame) means the policy does not need to
   learn to rotate coordinate systems internally. This greatly accelerates
   learning.

3. **Hard termination on double-touch** – Rather than just giving a negative
   reward, the episode ends. This creates a very strong signal that
   double-touching is unacceptable and prevents the agent from learning
   "touch everything repeatedly" strategies.

4. **Proximity shaping** – The `1 / distance` shaping term creates a smooth
   reward landscape that points the agent toward the nearest untouched dot at
   every step, preventing the sparse-reward problem that would otherwise make
   this task very difficult.

5. **Generous time limit** – The default 20-second episode is long enough for
   a well-trained policy to visit 6 dots without excessive time pressure.

---

## Extending the Game

- **Increase difficulty**: raise `num_dots`, shrink `dot_touch_distance`, or
  reduce `max_duration_seconds`.
- **Order constraint**: modify the environment to require dots to be touched
  in a specific order (like a TSP tour).
- **Moving dots**: update `self.dots` positions each step for a dynamic
  variant.
- **Multi-agent**: use the PettingZoo wrappers in `cyberFly/pz_envs` to let
  multiple drones cooperate on collecting dots.
