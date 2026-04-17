# How This Project Works — A Detailed Technical Guide

This document explains the inner workings of the **UAV Multi-Agent Reinforcement Learning Library** in plain language with full technical depth. Whether you're a newcomer to RL or drones, this guide walks through every major system.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [The Big Picture — How Everything Connects](#2-the-big-picture--how-everything-connects)
3. [Physics Engine — How Drones Fly in Simulation](#3-physics-engine--how-drones-fly-in-simulation)
4. [Drone Types — The Virtual Aircraft](#4-drone-types--the-virtual-aircraft)
5. [Flight Controllers — How Drones Are Steered](#5-flight-controllers--how-drones-are-steered)
6. [Component System — Modular Drone Parts](#6-component-system--modular-drone-parts)
7. [Single-Agent Environments (Gymnasium)](#7-single-agent-environments-gymnasium)
8. [Multi-Agent Environments (PettingZoo)](#8-multi-agent-environments-pettingzoo)
9. [Observation Spaces — What the AI "Sees"](#9-observation-spaces--what-the-ai-sees)
10. [Action Spaces — What the AI "Does"](#10-action-spaces--what-the-ai-does)
11. [Reward Systems — How the AI Learns What's Good](#11-reward-systems--how-the-ai-learns-whats-good)
12. [Training Strategies — Teaching Agents to Compete](#12-training-strategies--teaching-agents-to-compete)
13. [Self-Play Architecture — Agents Play Against Themselves](#13-self-play-architecture--agents-play-against-themselves)
14. [Training Infrastructure — Running Experiments](#14-training-infrastructure--running-experiments)
15. [Evaluation and Model Saving](#15-evaluation-and-model-saving)
16. [Wind Simulation](#16-wind-simulation)
17. [Custom UAV Support](#17-custom-uav-support)
18. [Docker Setup](#18-docker-setup)
19. [Project File Structure](#19-project-file-structure)

---

## 1. What Is This Project?

This is a **research library for training AI agents to fly drones** using reinforcement learning (RL). Think of it like a video game where AI pilots learn to hover, navigate, and even dogfight — but the physics are realistic.

It builds on [PyFlyt](https://github.com/jjshoots/PyFlyt), a UAV flight simulator, and extends it with:

- **Multi-agent environments** where multiple drones interact (fight, race, capture flags)
- **Training pipelines** for teaching agents using advanced strategies like self-play
- **Evaluation tools** for comparing how well different training strategies perform

**Key technologies used:**

| Technology | Role |
|---|---|
| **PyBullet** | Realistic rigid-body physics simulation at 240 Hz |
| **Gymnasium** | Standard API for single-agent RL environments |
| **PettingZoo** | Standard API for multi-agent RL environments |
| **Stable-Baselines3 (SB3)** | RL algorithm implementations (e.g., SAC) |
| **Numba** | Just-in-time compiler for speeding up PID controllers (~1.3x faster) |
| **Weights & Biases (W&B)** | Experiment tracking, logging, and visualization |
| **TensorBoard** | Training curve visualization |

---

## 2. The Big Picture — How Everything Connects

Here is the architecture at a high level:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Scripts                         │
│         run_sa_policy.py  |  run_ma_policy.py  |  test_*.py     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ uses
┌──────────────────────────▼──────────────────────────────────────┐
│                    RL Framework Layer                            │
│           Stable-Baselines3 (SAC algorithm)                     │
│           + Callbacks (Checkpoint, W&B, Eval)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ wraps
┌──────────────────────────▼──────────────────────────────────────┐
│              Multi-Agent Wrappers (Self-Play etc.)              │
│    SelfPlayEnv | MASelfPlayEnv | FictitiousPlayEnv              │
└──────────────────────────┬──────────────────────────────────────┘
                           │ wraps
┌──────────────────────────▼──────────────────────────────────────┐
│                   Environment Layer                              │
│    Gymnasium (single-agent)  |  PettingZoo (multi-agent)        │
│    Hover, Waypoints, Gates   |  Dogfight, Combat, CTF           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ uses
┌──────────────────────────▼──────────────────────────────────────┐
│                      Core Physics (Aviary)                      │
│              PyBullet @ 240 Hz  |  Drone models                 │
│      QuadX  |  Fixedwing  |  Rocket  |  Custom UAVs             │
│       Motors | Lifting Surfaces | PID | Camera | Wind           │
└─────────────────────────────────────────────────────────────────┘
```

**The flow is:**
1. A **training script** (e.g., `run_ma_policy.py`) sets up the experiment
2. It creates an **RL algorithm** (SAC from Stable-Baselines3) 
3. The algorithm interacts with an **environment** through the standard `step()` / `reset()` API
4. The environment uses the **Aviary** physics engine to simulate the real physics
5. The Aviary uses **PyBullet** to compute forces, collisions, and movements at 240 Hz

---

## 3. Physics Engine — How Drones Fly in Simulation

### The Aviary Class

The `Aviary` class (in `PyFlyt/core/aviary.py`) is the heart of the simulation. It inherits from PyBullet's `BulletClient`, which means it **is** the physics engine, plus extra drone management on top.

**What the Aviary manages:**

- Spawning and tracking multiple drones (of different types simultaneously)
- Running the physics loop at a fixed **240 Hz** (240 updates per second)
- Applying forces (gravity, thrust, aerodynamics, wind) every physics step
- Detecting collisions between drones and the environment
- Rendering a 3D visualization (optional)

**Key timing concept — Two clocks running at different speeds:**

```
Physics clock:    240 Hz  (every 4.17 ms) — forces, collisions, movement
Control clock:    30-40 Hz (every 25-33 ms) — AI makes decisions

For every 1 AI decision, 6-8 physics updates happen.
This is called the "interpolation ratio."
```

This matters because real physics needs fine-grained updates, but AI agents don't need to decide that fast. The drone just holds its last command between AI decisions.

**Constructor parameters:**

```python
Aviary(
    start_pos=np.array([[0, 0, 1]]),     # Where drones spawn (x, y, z)
    start_orn=np.array([[0, 0, 0]]),     # Starting orientation (roll, pitch, yaw)
    drone_type="quadx",                   # "quadx", "fixedwing", "rocket", or custom
    render=False,                         # Show 3D view?
    physics_hz=240,                       # Physics update rate
    wind_type=None,                       # Optional wind model
)
```

### How a Physics Step Works

Each time the simulation advances one step:

1. **Apply forces** — Each drone calculates its motor thrust, aerodynamic forces, drag
2. **PyBullet steps** — The physics engine resolves all forces, updates positions/velocities
3. **Collision detection** — PyBullet checks for contacts between every pair of objects
4. **State update** — Each drone reads back its new position, velocity, orientation from PyBullet

The state of each drone is a vector containing:
```
[angular_velocity(3), angular_position(3), linear_velocity(3), linear_position(3), quaternion(4)]
```

---

## 4. Drone Types — The Virtual Aircraft

### QuadX (Quadcopter — CrazyFlie 2.x)

The most common drone type. An X-configuration quadcopter with 4 motors.

```
     Motor 0 (CCW)        Motor 2 (CW)
          ╲                  ╱
           ╲                ╱
            ┌──────────────┐
            │              │
            │    Body      │
            │              │
            └──────────────┘
           ╱                ╲
          ╱                  ╲
     Motor 3 (CW)         Motor 1 (CCW)
```

**How it moves:**
- **Thrust**: All 4 motors spin faster → goes up
- **Roll** (tilt left/right): One side spins faster than the other
- **Pitch** (tilt forward/back): Front or back pair spins faster
- **Yaw** (rotate): CW motors spin faster or CCW motors spin faster (uses torque reaction)

**Motor model:**
- Each motor has a maximum RPM, thrust coefficient, and torque coefficient
- Motor response isn't instant — there's a time constant (`motor_tau`) simulating how quickly the motor spins up
- The thrust formula: `Force = thrust_coeff × RPM²`

### Fixedwing

A traditional airplane with wings, stabilizers, and a single propeller.

**5 lifting surfaces:**
1. Left wing (with flap)
2. Right wing (with flap)  
3. Horizontal stabilizer (elevator)
4. Vertical stabilizer (rudder)
5. Main wing body

**How it flies:**
- Air flows over the wings, generating lift based on the **angle of attack** and **airspeed**
- The AI controls 4 things: **aileron** (roll), **elevator** (pitch), **rudder** (yaw), **throttle** (speed)
- It starts with a forward velocity of ~20 m/s — it has to keep moving to stay airborne (unlike a quadcopter which can hover)
- The flight dome is much larger (800m vs 5m for quadcopters) because fixed-wings need space to turn

### Rocket

A rocket with thrust-vectoring control — like a SpaceX Falcon 9 landing.

**Actuators (7 total):**
- 4 finlets (deflectable fins at the top, split into X and Y axes) — 3 controllable deflections
- 1 yaw control
- 1 ignition toggle
- 1 throttle
- 2 gimbal axes (pointing the main engine bell)

**Unique feature:** Fuel ratio simulation — as the rocket burns fuel, its mass decreases, affecting the physics.

---

## 5. Flight Controllers — How Drones Are Steered

The QuadX supports **8+ flight modes**, each providing a different level of abstraction between the AI agent and the raw motor commands:

| Mode | Input | What It Controls | Who Does the Hard Work |
|------|-------|------------------|------------------------|
| **0** | 4 motor throttles | Direct motor power | The AI does everything |
| **1** | Roll rate, Pitch rate, Yaw rate, Thrust | Angular velocities | Inner PID loop |
| **2** | Roll, Pitch, Yaw, Thrust | Target angles | Inner + middle PID |
| **7** | X, Y, Z, Yaw | Target position | All three PID loops |
| **8+** | Custom | Custom | Custom controller class |

**The PID controller hierarchy (for QuadX):**

```
AI Decision → [Outer Loop: Position → Velocity]
            → [Middle Loop: Velocity → Attitude]
            → [Inner Loop: Attitude → Motor Commands]
            → Motor voltages
```

- **Inner loop**: Controls angular *rates* (how fast the drone rotates)  
- **Middle loop**: Controls *attitude* (what angle the drone is tilted to)
- **Outer loop**: Controls *position* (where the drone is in space)

Each loop runs PID (Proportional-Integral-Derivative) controllers compiled with **Numba JIT** for 1.3x speed improvement. This matters because PID runs at 240 Hz for every drone.

**Why different modes matter for RL:**
- **Mode 0** (raw motors) is the hardest for RL — the agent must learn basic flight physics
- **Mode 7** (position control) is the easiest — the agent just says "go here" and the PID does the flying
- Research often uses Mode 0 to study if AI can learn low-level control from scratch

---

## 6. Component System — Modular Drone Parts

Every drone is built from pluggable components. This makes it easy to create new drone types by mixing and matching parts.

### Motors
```python
Motor(
    thrust_coeff=1.4e-7,      # How much thrust per RPM²
    torque_coeff=1e-9,         # How much torque per RPM²
    max_rpm=55000,             # Maximum rotation speed
    motor_tau=0.01,            # Time constant (how fast motor responds)
)
```

The motor model simulates the delay between commanding a motor speed and actually reaching it. This is realistic — real motors can't change speed instantly.

### Lifting Surfaces
Used by fixedwing and rocket. Computes aerodynamic lift and drag based on:
- The **relative airflow** over the surface (speed + wind)
- The **angle of attack** (angle between the wing and the airflow)
- The **surface area** and **lift coefficient**

### Boring Bodies (Drag)
Simple drag simulation for the main body of the aircraft. Applies a force opposing the direction of travel, proportional to airspeed squared.

### Camera
An on-board camera sensor:
- Configurable resolution (default 64×64 or 128×128)
- Field of view (FOV) setting
- Optional gimbal lock (camera always points down regardless of drone tilt)
- Returns RGB images that could be used for vision-based RL

### Gimbals
Provides thrust vector control. The engine nozzle can be pointed in different directions, used by the rocket for steering during powered flight.

---

## 7. Single-Agent Environments (Gymnasium)

These environments have **one drone** performing a task. They follow the standard [Gymnasium](https://gymnasium.farama.org/) API (`reset()`, `step(action)`, returns `observation, reward, terminated, truncated, info`).

### QuadX-Hover-v4
**Task:** Keep the drone hovering at position (0, 0, 1) — one meter above the ground.

```
Episode: up to 15 seconds (600 steps at 40 Hz)
Flight dome: 3m radius (drone is terminated if it flies outside)
Termination: crash (z ≤ 0) or out of bounds
```

**Reward formula (every step):**
```
reward = 1.0 
       - 3.0 × distance_from_target     (penalize being far from [0,0,1])
       - angular_distance                (penalize being tilted)  
       - 0.01 × yaw_rate²               (penalize spinning)
```

The agent gets a small positive reward each step for being alive, minus penalties. A perfect hover gives ~1.0 per step. Crashing far away might give -10 or worse.

### QuadX-Waypoints-v4
**Task:** Navigate through 4 waypoints in sequence.

- Waypoints are randomly generated on a sphere
- The agent sees the relative positions of the next waypoints in its observation
- Reward uses "potential shaping" — the agent gets rewarded for getting closer to the next waypoint, not just for reaching it
- Bonus reward when a waypoint is reached

### QuadX-Gates-v3
**Task:** Fly through racing gates. Similar to waypoints but gates are physical objects the drone must pass through.

### QuadX-Pole-Balance-v4
**Task:** Balance a pole on top of the drone (like balancing a broomstick on your hand — but flying).

Additional physics: A pole is attached via a joint on top of the drone. The drone must hover while keeping the pole upright.

### QuadX-Ball-In-Cup-v4
**Task:** A ball on a string is attached to the drone. The drone must swing and catch the ball in a cup.

### Fixedwing-Waypoints-v4
**Task:** Navigate a fixed-wing aircraft through waypoints. Much larger play area (800m dome).

### Rocket-Landing-v4
**Task:** Land a rocket precisely on a landing pad — similar to SpaceX's booster landings.

---

## 8. Multi-Agent Environments (PettingZoo)

These environments have **multiple drones** interacting. They use [PettingZoo](https://pettingzoo.farama.org/)'s **Parallel API** — all agents act simultaneously each step (as opposed to taking turns).

### MAQuadXHoverEnv — Cooperative Hover
**4 drones try to hover at assigned positions.**

- Each agent has its own target altitude
- Agents can see their teammates' positions (encourages awareness)
- Everyone gets rewarded based on their own hovering performance
- Good for testing basic multi-agent coordination

### CombatWaypointPursuitEnv — Chase Game
**2 drones: one tries to reach waypoints, the other tries to catch it.**

```
Ego (agent 0):  Navigate to waypoints without getting caught
Adversary (agent 1):  Catch the ego drone
```

**Key parameters:**
```
Flight dome: 5m radius
Episode: up to 10 seconds
Goal reach distance: 0.1m
Waypoint bonus: +100
Catch bonus: +200 (for adversary)
Crash penalty: -200
```

**Reward breakdown:**
- **Ego** gets rewarded for making progress toward waypoints, penalized when the adversary gets close
- **Adversary** gets rewarded for closing distance to ego, big bonus for catching it

This is an **asymmetric** game — the two agents have different goals, which makes training interesting.

### MAFixedwingDogfightEnv — Aerial Combat
**Teams of fixed-wing aircraft fighting each other.**

```
Default: 1v1 (configurable up to 4v4)
Flight dome: 800m radius
Episode: up to 60 seconds
```

**Health system:**
- Each aircraft starts with 0.28 health
- Takes damage when an enemy has you in their "cone of fire" (within 25m distance AND within 11.46° angle)
- Damage: 0.003 per physics step while being hit
- Dead when health ≤ 0.001

**Weapons model:**
To score a hit, three conditions must all be true:
1. **In range**: Distance to target < 25m
2. **In cone**: Angle to target < 0.2 radians (~11.5°)
3. **Chasing**: Angle is < 90° (you're facing toward them, not away)

Friendly fire is disabled — you can only damage the other team.

**Spawn pattern:** Aircraft start in a circle facing outward, with random radius (10-50m) and height (20-50m). This creates varied starting scenarios.

**Reward tuning knobs:**
- `aggressiveness` (0-1): Higher = more reward for attacking vs. avoiding damage
- `cooperativeness` (0-1): Higher = more reward for team performance vs. individual

### MAQuadXCaptureFlagEnv — Capture the Flag
**5 drones split into Red and Blue teams compete for flags.**

```
Flags: 4 flag stations spread across the arena
Teams: Alternating assignment (drone 0=red, 1=blue, 2=red, ...)
Flag reach distance: 0.2m
```

Agents spawn in a circle and race to capture flag positions.

---

## 9. Observation Spaces — What the AI "Sees"

Every RL agent needs to **observe** the world to make decisions. Here's what each drone sees:

### Single-Agent Observation (QuadX)

A flat vector of ~17-18 numbers:

| Component | Size | Meaning |
|---|---|---|
| Angular velocity (p, q, r) | 3 | How fast the drone is rotating around each axis |
| Angular position | 3 or 4 | Euler angles (roll, pitch, yaw) OR quaternion |
| Linear velocity (vx, vy, vz) | 3 | Speed in each direction |
| Linear position (x, y, z) | 3 | Where the drone is in space |
| Previous action | 4 | What the agent commanded last step |
| Auxiliary state | ~1-4 | Battery level, collision flag, etc. |

**Euler vs Quaternion:** The agent can "see" orientation as either Euler angles (3 numbers: roll, pitch, yaw — intuitive but has a mathematical problem called "gimbal lock") or quaternions (4 numbers — avoids gimbal lock but less intuitive). Most experiments use quaternions.

### Multi-Agent Observations

Multi-agent environments use **dictionary observations** with structured data:

```python
{
    "attitude": [ang_vel(3), ang_pos(3/4), lin_vel(3), lin_pos(3), action(4), aux(?), 
                 relative_positions_of_others(3×N)],
    "target_deltas": [[dx, dy, dz], [dx, dy, dz], ...]  # vectors to each target
}
```

**Why dictionaries?** Because different parts of the observation have different meanings and scales. Separating them lets the neural network process them appropriately.

### Dogfight Observation

The dogfight environment gives each agent:
- **Self observation**: Full attitude + health (14D)
- **Others observation**: For each other aircraft — their attitude + health + team flag (14D each)
- These can be flattened into one vector or kept as a dictionary

The "team flag" tells the agent which aircraft are allies vs enemies. This is important — you don't want to attack your teammates.

---

## 10. Action Spaces — What the AI "Does"

### QuadX Actions (Mode 0 — Raw Motors)
```
4D continuous: [motor_0, motor_1, motor_2, motor_3]
Range: [0, 1] — proportional throttle for each motor
```

### QuadX Actions (Mode 1 — Rate Control)
```
4D continuous: [roll_rate, pitch_rate, yaw_rate, thrust]
Range: [-π, π] for rates, [0, 0.8] for thrust
```

### Fixedwing Actions
```
4D continuous: [aileron, elevator, rudder, throttle]
Range: [-1, 1] for control surfaces, [0, 1] for throttle
```

### Rocket Actions
```
7D continuous: [fin_x, fin_y, fin_z, yaw, ignition, throttle, gimbal_x, gimbal_y]
Range: Various [-1, 1] and [0, 1]
```

All actions are **continuous** (real numbers, not discrete choices). This is why the project uses **SAC** (Soft Actor-Critic), which is designed for continuous action spaces.

---

## 11. Reward Systems — How the AI Learns What's Good

Reward design is one of the most critical (and difficult) parts of RL. Here's how rewards work in each environment:

### Hover Reward
Simple and intuitive:
```
reward = 1.0 - 3.0 × ||position - target|| - ||tilt|| - 0.01 × yaw_rate²
```
- You get +1 for existing each step
- Minus 3× the distance from the target (strongly penalizes being far)
- Minus the angular tilt (wants the drone level)
- Minus a small yaw rate penalty (don't spin)

### Combat Pursuit Rewards

**For the Ego drone (waypoint navigator):**
```
+100      when reaching a waypoint
+3 × progress_toward_waypoint      (getting closer each step)
+0.1 / distance_to_waypoint        (closer = more reward)
-200      if crashed or out of bounds
-0.001 × yaw_rate²                 (stability penalty)
```

**For the Adversary drone (chaser):**
```
+200      when catching the ego
+3 × progress_toward_ego           (getting closer each step)  
+0.1 / distance_to_ego             (closer = more reward)
-200      if crashed or out of bounds
-0.001 × yaw_rate²                 (stability penalty)
```

### Dogfight Rewards (Most Complex)

The dogfight reward has multiple components tuned by `aggressiveness` and `cooperativeness`:

**Engagement rewards:**
```
+4.0 × distance_closed        — reward for closing distance to enemies
+30.0 × angle_improvement     — reward for getting enemies in your sights  
+3.0 × (1/angle_to_enemy)     — continuous bonus for tracking enemies
+20.0 × hits_landed           — big reward for scoring hits
−20.0 × (1-aggressiveness) × hits_received  — penalty for being hit (scaled by aggressiveness)
```

**Boundary rewards:**
```
+tanh(0.1 × altitude - 1.0)             — encourages flying at reasonable altitude
−tanh(0.0025 × dist_from_origin - 1.0)  — penalty for flying too far from arena
−10 × (5.0 - dist) for each nearby ally  — collision avoidance (when < 5m from ally)
```

**Terminal rewards:**
```
+300    team victory (all enemies dead, some allies alive)
−1000   crash, collision, or out of bounds
```

### Dense vs Sparse Rewards

Most environments support both:
- **Dense** (default): Reward every step based on progress (easier for RL to learn from)
- **Sparse**: Only reward at the very end (harder but more realistic objective)

---

## 12. Training Strategies — Teaching Agents to Compete

Multi-agent training is harder than single-agent because the "environment" includes other learning agents. If both agents are learning simultaneously, the training signal is non-stationary (the goalpost keeps moving). This project implements several strategies to handle this:

### Vanilla Play (VP)
The simplest approach. Train one agent. The opponent always uses the **latest** version of its policy. This is like playing against yourself where you both level up simultaneously.

**Pros:** Simple, fast  
**Cons:** Can be unstable — if one agent finds an exploit, the other might collapse

### Fictitious Play (FP)
Each agent plays against a **weighted mixture** of all past versions of the opponent. The idea: instead of chasing a moving target, play against the "average" opponent.

**How the weights work:**
```
When adding the k-th policy to the pool:
    Average policy weight = (k-1)/(k+1)
    New policy weight     = 2/(k+1)
    Then normalize so all weights sum to 1.0
```

This gives more weight to recent policies while still considering old ones. Over time, this provably converges to a Nash equilibrium in certain game types.

### Delta-Uniform Play (DP)
Keep only the **last N** (default 10) policies and weight them equally.

```
Pool = [model_k-9, model_k-8, ..., model_k]
Each has weight = 1/10 = 0.1
```

**Why?** Pure fictitious play averages over ALL history, so very old (bad) policies dilute the pool. Delta-uniform keeps a sliding window of recent competitive policies.

### Self-Play (SP)
Train one agent against **copies of its own past self**. This creates a self-improving loop:

```
Iteration 1: Agent v1 trains against v0 (random)
Iteration 2: Agent v2 trains against v1
Iteration 3: Agent v3 trains against v2
...each version gets slightly better
```

---

## 13. Self-Play Architecture — Agents Play Against Themselves

The self-play system (in `PyFlyt/marl_wrappers/selfplay.py`) is a set of wrapper classes that convert multi-agent environments into single-agent environments that SB3 can train on.

### SelfPlayEnv
The basic wrapper for **1v1 games**:

```
Multi-agent env: expects actions for agent_0 AND agent_1 each step
SelfPlayEnv:     only asks the RL algorithm for agent_0's action
                 agent_1's action comes from a frozen "opponent policy"
```

**Step-by-step walkthrough:**

1. RL algorithm calls `env.step(action)` with one action
2. The wrapper queries the opponent policy: `opp_action = opp_policy.predict(opp_obs)`
3. Both actions are sent to the real multi-agent environment
4. The wrapper returns only the training agent's observation and reward

The opponent policy is **frozen** — it doesn't learn during this phase. After some training, the new trained model becomes the next opponent.

### MASelfPlayEnv
For games with **more than 2 agents**. One agent trains; all others use frozen policies.

```python
opp_policies = {
    "agent_1": frozen_model_1,
    "agent_2": frozen_model_2,
    "agent_3": frozen_model_3,
}
# Only agent_0 is being trained
```

If no policy is provided for an opponent, it takes **random actions**.

### FictitiousPlayEnv
The most sophisticated wrapper. Manages the full fictitious self-play training loop:

- Maintains a **pool of models** for each agent: `self.models[agent_id] = [model_v0, model_v1, ...]`
- Maintains a **probability distribution** over those models: `self.policy_dist[agent_id] = [0.1, 0.3, 0.6]`
- Each training round, the opponent is **sampled** from this distribution
- After training, the new best-response model is added to the pool
- The distribution is updated (using FP or delta-uniform weights)

### The Training Loop (Alternating)

```python
for iteration in range(100):
    # Phase 1: Train Ego against frozen Adversary
    env_ego.opp_policy = model_adv          # Adversary uses last model
    model_ego.learn(50000 steps)            # Ego improves
    
    # Phase 2: Train Adversary against frozen Ego  
    env_adv.opp_policy = model_ego          # Ego uses last model
    model_adv.learn(50000 steps)            # Adversary improves
    
    # Both have improved one step — repeat
```

This alternating structure ensures both agents keep improving, creating an arms race.

---

## 14. Training Infrastructure — Running Experiments

### Single-Agent Training (`run_sa_policy.py`)

```bash
python run_sa_policy.py --env hover --flight_mode 0 --output results/
```

**What happens:**
1. Creates a `QuadXHoverEnv` or `QuadXWaypointsEnv`
2. Initializes an SAC model with an MLP (multi-layer perceptron) policy
3. Trains for a configurable number of timesteps
4. Saves the model as a `.zip` file

**SAC (Soft Actor-Critic)** is the default algorithm because:
- It works well with continuous actions (drone controls are continuous)
- It's sample-efficient (learns fast relative to data collected)
- It has an entropy bonus that encourages exploration (important for drones that might crash immediately)

### Multi-Agent Training (`run_ma_policy.py`)

```bash
python run_ma_policy.py --env dogfight --strategy FP --total_timesteps 5000000
```

**What happens:**
1. Creates the chosen multi-agent environment (dogfight, combat, or hover)
2. Wraps it with the chosen strategy wrapper (VP, FP, DP, or SP)
3. Runs the alternating self-play training loop
4. Logs to W&B and saves checkpoints

**Available environments:**
```python
ENV_REGISTRY = {
    "dogfight":  MAFixedwingDogfightEnvV2,    # Fixed-wing aerial combat  
    "combat":    CombatWaypointPursuitEnv,     # QuadX chase game
    "hover":     MAQuadXHoverEnv,              # Cooperative hover
}
```

### Callbacks

During training, several callbacks fire at regular intervals:

| Callback | Purpose |
|---|---|
| **CheckpointCallback** | Saves model snapshots every N steps (for self-play pool) |
| **WandbCallback** | Logs rewards, losses, gradients to W&B dashboard |
| **EvalCallback** | Runs test episodes to measure current performance |

### Remote Training

Training is typically done on a remote GPU machine via SSH:

```bash
ssh -L 6006:localhost:6006 user@server    # Connect with TensorBoard port forwarding
source venv/bin/activate                   # Activate Python environment
nohup python run_ma_policy.py > out.log 2>&1 &   # Run in background
tensorboard --logdir=. --port=6006         # Monitor training curves
```

---

## 15. Evaluation and Model Saving

### Model Format
Models are saved as **SB3 `.zip` files** containing:
- Neural network weights (actor and critic)
- Policy architecture definition
- Optimizer state (for resuming training)
- Observation/action space specifications

### Evaluation Scripts

**`eval_drones.py`** — Evaluates QuadX combat models:
1. Loads two saved SAC models (ego and adversary)
2. Runs episodes in the combat environment with rendering enabled
3. Records videos of the episodes
4. Tracks cumulative rewards for both agents

**`eval_fwing.py`** — Evaluates fixed-wing dogfight models:
1. Loads fixed-wing dogfight models
2. Uses a custom debug camera (overhead view) for clear visualization
3. Records combat videos showing the full flight dome

### Competitive Evaluation

```python
results = evaluate_competitive_game(env, [model_ego, model_adv], num_episodes=100)
# Returns: {"team_0_wins": 67, "team_1_wins": 28, "ties": 5}
```

This runs many episodes and counts wins/losses to measure which strategy trains better agents.

### Model Compatibility

A custom `load_sac_compat()` function handles loading models across different versions of SB3:
- Converts old `net_arch` format (list of dicts) to new format (single dict)
- Handles optimizer state differences between SB3 versions
- Ensures models trained months ago still load correctly

---

## 16. Wind Simulation

The wind system is architecturally in place but currently minimal. The framework supports custom wind fields:

```python
Aviary(
    wind_type="custom_wind",
    wind_options={"speed": 5.0, "direction": [1, 0, 0]},
)
```

Wind forces are applied as velocity perturbations to aerodynamic surfaces (wings, fins). This mainly affects **fixedwing** and **rocket** drones, since quadcopters don't have large lifting surfaces (though drag on the body is still affected).

The example `10_custom_wind.py` shows how to implement a custom wind field class.

---

## 17. Custom UAV Support

You can define entirely new drone types:

**Step 1: Create a YAML configuration file**
```yaml
# models/vehicles/my_drone/my_drone.yaml
motor:
  thrust_coeff: 1.4e-7
  torque_coeff: 1e-9
  max_rpm: 55000
  motor_tau: 0.01
drag:
  coefficient: 0.1
control:
  angular_rate_pid: [1.0, 0.0, 0.5]
  attitude_pid: [5.0, 0.0, 2.0]
```

**Step 2: Create a URDF file** (defines the physical shape and joints)
```xml
<!-- models/vehicles/my_drone/my_drone.urdf -->
<robot name="my_drone">
  <link name="base_link">
    <inertial> ... </inertial>
    <visual> ... </visual>
    <collision> ... </collision>
  </link>
</robot>
```

**Step 3: Create a Python class inheriting from `DroneClass`**
```python
class MyDrone(DroneClass):
    def reset(self): ...
    def state_update(self): ...       # Read sensors
    def physics_update(self): ...     # Apply forces
```

**Step 4: Register and use it**
```python
env = Aviary(
    drone_type=["quadx", "my_drone"],
    drone_type_mappings={"my_drone": MyDrone},
    start_pos=np.array([[0,0,1], [2,0,1]]),
)
```

You can even mix custom drones with built-in ones in the same simulation.

---

## 18. Docker Setup

A Dockerfile is provided for reproducible environments:

```dockerfile
FROM python:3.11-slim

# System dependencies for 3D rendering and physics
RUN apt-get install -y \
    libgl1-mesa-glx libgl1-mesa-dev \    # OpenGL rendering
    libglu1-mesa-dev libglew-dev \        # OpenGL utilities
    libosmesa6-dev \                       # Off-screen rendering
    libglfw3 libglfw3-dev \               # Window management
    x11-apps                               # X11 display

# Python packages
RUN pip install -e .                       # Install project in editable mode
RUN pip install "pybullet>=3.2.0"          # Physics engine

ENV DISPLAY=:0                             # X11 display for GUI
ENV PYTHONUNBUFFERED=1                     # Real-time logging
```

The container includes everything needed to train and evaluate models headlessly (without a monitor), or with X11 forwarding for visualization.

---

## 19. Project File Structure

```
UAV-MARL-Lib/
│
├── PyFlyt/                          # Main library package
│   ├── core/                        # Physics simulation core
│   │   ├── aviary.py                # Main physics engine (Aviary class)
│   │   ├── abstractions/            # Reusable components
│   │   │   ├── pid.py               # Numba-JIT compiled PID controllers
│   │   │   ├── camera.py            # On-board camera sensor
│   │   │   ├── motors.py            # Brushless motor model
│   │   │   ├── lifting_surfaces.py  # Aerodynamic force computation
│   │   │   ├── boring_bodies.py     # Simple drag bodies
│   │   │   └── gimbals.py           # Thrust vectoring
│   │   ├── drones/                  # Drone implementations
│   │   │   ├── quadx.py             # QuadX quadcopter (CrazyFlie 2.x)
│   │   │   ├── fixedwing.py         # Fixed-wing airplane
│   │   │   └── rocket.py            # Thrust-vectored rocket
│   │   ├── wind/                    # Wind field models
│   │   └── utils/                   # Utility functions
│   │
│   ├── gym_envs/                    # Single-agent Gymnasium environments
│   │   ├── quadx_envs/              # Quadcopter tasks
│   │   │   ├── quadx_hover_env.py   # Hovering
│   │   │   ├── quadx_waypoints_env.py  # Waypoint navigation
│   │   │   └── ...                  # Gates, pole balance, ball-in-cup
│   │   ├── fixedwing_envs/          # Fixed-wing tasks
│   │   └── rocket_envs/             # Rocket landing
│   │
│   ├── pz_envs/                     # Multi-agent PettingZoo environments
│   │   ├── quadx_envs/              # Multi-drone quadcopter scenarios
│   │   │   ├── ma_quadx_hover_env.py     # Cooperative hover
│   │   │   ├── ma_combat_env.py          # Pursuit-evasion
│   │   │   └── ma_quadx_dogfight_env.py  # Quadcopter dogfight
│   │   └── fixedwing_envs/          # Multi-drone fixed-wing scenarios
│   │       └── ma_fixedwing_dogfight_env.py  # Team aerial combat
│   │
│   ├── CL2_envs/                    # Custom lab environments
│   │   └── quadx_capture_flag.py    # Capture the flag (5 drones, 2 teams)
│   │
│   ├── marl_wrappers/               # Multi-agent training wrappers
│   │   └── selfplay.py              # Self-play, fictitious play, etc.
│   │
│   └── models/                      # URDF files and vehicle configs
│       └── vehicles/                # Drone model definitions (YAML + URDF)
│
├── run_sa_policy.py                 # Single-agent training entry point
├── run_ma_policy.py                 # Multi-agent training entry point
├── eval_drones.py                   # QuadX model evaluation + video
├── eval_fwing.py                    # Fixed-wing model evaluation + video
├── test_ma_ctf.py                   # Capture-the-flag testing
├── test_ma_envs.py                  # Multi-agent env testing
├── test_ma_fixedwing.py             # Fixed-wing self-play testing
├── test_sa_envs.py                  # Single-agent env testing
│
├── examples/                        # Usage examples
│   └── core/                        # Core simulation examples (01-10)
│
├── results/                         # Saved training results and models
├── dogFight/                        # Saved dogfight models and logs
├── docs/                            # Generated documentation
├── docs_source/                     # Documentation source files
├── Dockerfile                       # Container definition
└── pyproject.toml                   # Package configuration and dependencies
```

---

## Quick Reference — Key Numbers

| Parameter | Value | Context |
|---|---|---|
| Physics rate | 240 Hz | PyBullet simulation frequency |
| Control rate | 30-40 Hz | How often the AI makes decisions |
| QuadX observation | 17-18D | State vector size |
| QuadX action | 4D | Motor commands |
| Fixedwing action | 4D | Aileron, elevator, rudder, throttle |
| Rocket action | 7D | Fins, gimbal, throttle, ignition |
| Dogfight dome | 800m | Play area radius for fixed-wing |
| Combat dome | 5m | Play area radius for QuadX chase |
| Lethal distance | 25m | Weapon range in dogfight |
| Lethal angle | 0.2 rad (~11.5°) | Cone of fire in dogfight |
| Damage per hit | 0.003/step | Health reduction in dogfight |
| Starting health | 0.28 | Per aircraft in dogfight |
| Crash penalty | -200 to -1000 | Depending on environment |
| Waypoint bonus | +100 | Reaching a waypoint |
| Team victory bonus | +300 | Winning a dogfight |
