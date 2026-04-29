"""Obstacle Maze – QuadX Dot Touch with PyBullet box obstacles.

Same dot-touch game, but 8 solid box walls are spawned each episode
that the drone must navigate around to reach every dot.

Navigation uses a 3-D potential-field controller:
  • Attraction toward the nearest untouched dot
  • Repulsion away from nearby obstacle surfaces
  • Random escape kick when the drone gets stuck in a local minimum

Usage:
    python _run_dot_touch_obstacles.py
"""

import gymnasium          # Standard RL environment interface
import numpy as np        # Numerical arrays and math operations
import pybullet as p      # Only used for quaternion → rotation-matrix helper

import cyberFly.gym_envs  # Registers cyberFly environments with gymnasium


# ── Environment ────────────────────────────────────────────────────────────────
DOME_SIZE = 5.0   # Radius (m) of the spherical flight zone; drone is penalised if it exits

env = gymnasium.make(
    "cyberFly/QuadX-DotTouch-v1",   # Environment ID registered in cyberFly.gym_envs
    render_mode="human",              # Open a real-time 3-D visualisation window
    flight_mode=6,                    # Local velocity control [vx, vy, vr, vz]
    flight_dome_size=DOME_SIZE,       # Pass dome radius to the environment
    truncate_on_completion=False,     # Don't auto-end the episode when all dots are touched
)


# ── Controller parameters ──────────────────────────────────────────────────────
TOUCH_RADIUS           = 0.5   # m – distance at which the env registers a dot as touched
BRAKE_DISTANCE         = 1.5   # m – start slowing down when drone is this close to target
MAX_VELOCITY           = 2.0   # m/s – hard cap on any velocity component
VELOCITY_GAIN          = 1.5   # proportional gain: position error (m) → velocity (m/s)

RETURN_TO_BASE         = True                        # Fly back to spawn after all dots done
BASE_POSITION          = np.array([0.0, 0.0, 1.0])  # Spawn point: origin at 1 m altitude
BASE_ARRIVAL_THRESHOLD = 0.3   # m – within this distance the drone is considered home


# ── Potential-field parameters ─────────────────────────────────────────────────
K_ATT = 1.5   # Attractive gain: how strongly the drone is pulled toward the target dot
K_REP = 8.0   # Repulsive gain: how strongly the drone is pushed away from obstacles
D0    = 1.4   # Influence radius (m): obstacles beyond this distance have no repulsive effect


# ── Stuck-detection parameters ────────────────────────────────────────────────
STUCK_HISTORY      = 40    # Number of past positions stored in the ring buffer
STUCK_THRESH       = 0.10  # m – if the drone’s spread over STUCK_HISTORY steps < this, it’s stuck
ESCAPE_STEPS       = 20    # How many steps to apply the random escape kick before resuming normal nav


# ── Obstacle configuration ─────────────────────────────────────────────────────
NUM_OBSTACLES    = 8      # Number of solid boxes spawned per episode
MIN_SPAWN_DIST   = 1.2   # Minimum clearance (m) between a block centre and any dot or the base


def _get_aviary(gym_env):
    """Return the underlying Aviary (BulletClient) from a wrapped gymnasium env."""
    return gym_env.unwrapped.env   # .unwrapped strips all wrappers; .env is the BulletClient


def spawn_obstacles(gym_env, dots_world: np.ndarray) -> list[dict]:
    """Spawn NUM_OBSTACLES solid boxes inside the flight dome.

    Blocks are placed randomly but are guaranteed to be at least
    MIN_SPAWN_DIST away from every dot and from the spawn base position.
    Dimensions are varied to create a mix of thin walls and chunky cubes.

    Args:
        gym_env:    The wrapped gymnasium env (Aviary is accessed via unwrapped).
        dots_world: (N, 3) world-frame dot positions – used for clearance checks.

    Returns:
        List of dicts with keys:
            'pos'     – np.ndarray (3,) centre position
            'half'    – np.ndarray (3,) half-extents of the box
            'body_id' – PyBullet body id (int)
    """
    aviary = _get_aviary(gym_env)        # Grab the PyBullet physics client
    rng = np.random.default_rng()        # Create a local random number generator

    # Obstacle shapes: (half_x, half_y, half_z) pairs
    # Mix of thin vertical walls, thin horizontal slabs, and cube blocks
    shape_presets = [
        np.array([0.10, 0.70, 0.70]),   # thin vertical wall (x-thin)
        np.array([0.70, 0.10, 0.70]),   # thin vertical wall (y-thin)
        np.array([0.70, 0.70, 0.10]),   # thin horizontal slab
        np.array([0.35, 0.35, 0.35]),   # cube block
        np.array([0.50, 0.15, 0.65]),   # rectangular pillar
        np.array([0.15, 0.50, 0.65]),   # rectangular pillar (rotated)
    ]

    obstacles = []   # Accumulates metadata dicts for every successfully spawned block

    for _ in range(NUM_OBSTACLES):   # Try to place NUM_OBSTACLES boxes
        half_ext = shape_presets[rng.integers(len(shape_presets))].copy()   # Pick a random shape
        # Add a small random scale jitter so each instance looks slightly different
        half_ext *= rng.uniform(0.8, 1.3)   # Scale uniformly between 80 % and 130 % of preset size

        col_id = aviary.createCollisionShape(    # Register a box collision shape in the physics engine
            aviary.GEOM_BOX,
            halfExtents=half_ext.tolist(),
        )
        vis_id = aviary.createVisualShape(       # Register a matching visual shape for rendering
            aviary.GEOM_BOX,
            halfExtents=half_ext.tolist(),
            rgbaColor=[0.85, 0.30, 0.10, 1.0],   # vivid orange-red
        )

        # Try up to 300 times to find a position with enough clearance
        for _attempt in range(300):   # 300 attempts is plenty for a sparse DOME_SIZE=5 m space
            radius = rng.uniform(0.8, DOME_SIZE * 0.78)   # Random distance from centre
            angle  = rng.uniform(0.0, 2.0 * np.pi)        # Random azimuth angle (radians)
            height = rng.uniform(0.35, DOME_SIZE * 0.55)  # Random height above ground
            pos = np.array([
                radius * np.cos(angle),   # Convert polar → Cartesian x
                radius * np.sin(angle),   # Convert polar → Cartesian y
                height,
            ])

            # Reject if too close to any dot or to the spawn base
            dot_dists = np.linalg.norm(dots_world - pos, axis=1)   # Distance to every dot
            if np.any(dot_dists < MIN_SPAWN_DIST):   # Too close to at least one dot — retry
                continue
            if np.linalg.norm(BASE_POSITION - pos) < MIN_SPAWN_DIST:   # Too close to base — retry
                continue
            break   # Position passes all clearance checks — use it
        else:
            # Could not find a free spot — skip this obstacle
            aviary.removeCollisionShape(col_id) if hasattr(aviary, "removeCollisionShape") else None
            continue

        body_id = aviary.createMultiBody(
            baseMass=0,                         # static (immovable) block
            baseCollisionShapeIndex=col_id,     # Link the collision shape
            baseVisualShapeIndex=vis_id,        # Link the visual shape
            basePosition=pos.tolist(),          # Place it at the validated position
        )
        obstacles.append({            # Store metadata so the navigation code can query positions
            "pos":     pos.copy(),    # World-frame centre position (3,)
            "half":    half_ext.copy(),   # Half-extents of the box — unused by nav, kept for debug
            "body_id": body_id,       # PyBullet body id — could be used to remove the block later
        })

    # Update the aviary’s contact-array so new bodies participate in collision tracking
    aviary.register_all_new_bodies()

    print(f"  [obstacles] spawned {len(obstacles)} blocks")
    return obstacles


# ── Potential-field navigation ─────────────────────────────────────────────────

def compute_potential_field_action(
    drone_pos:    np.ndarray,
    target_pos:   np.ndarray,
    obs_positions: np.ndarray,
    quaternion:   np.ndarray,
) -> np.ndarray:
    """Compute a [vx, vy, vr, vz] body-frame velocity via potential fields.

    Attractive component pulls toward *target_pos*.
    Repulsive component pushes away from obstacles within D0 metres.
    The combined world-frame velocity is rotated into the drone body frame.

    Args:
        drone_pos:     (3,) drone position in world frame.
        target_pos:    (3,) target dot position in world frame.
        obs_positions: (M, 3) obstacle centre positions in world frame.
        quaternion:    (4,) drone orientation quaternion [x, y, z, w].

    Returns:
        np.ndarray: action [vx, vy, vr, vz] in drone body frame.
    """
    # Attractive force – proportional with distance-based braking near target
    delta_att = target_pos - drone_pos   # World-frame vector from drone to target (m)
    dist_att  = float(np.linalg.norm(delta_att))   # Distance to target (m)

    if dist_att > 1e-6:   # Guard against division by zero when already at the target
        if dist_att < BRAKE_DISTANCE:   # Inside braking zone: scale down speed
            min_brake = TOUCH_RADIUS * 0.5   # Below this distance the speed reaches zero
            speed_scale = max(
                0.0,
                (dist_att - min_brake) / (BRAKE_DISTANCE - min_brake),
            )   # Linear ramp from 0 (at min_brake) to 1 (at BRAKE_DISTANCE)
        else:
            speed_scale = 1.0   # Full speed beyond the braking zone
        f_att = K_ATT * speed_scale * (delta_att / dist_att)   # Unit vector scaled by gain
    else:
        f_att = np.zeros(3)   # Already at target — no attractive force needed

    # Repulsive forces – one contribution per obstacle within influence radius D0
    f_rep = np.zeros(3)   # Accumulate repulsive forces from all nearby obstacles
    for obs_pos in obs_positions:   # Loop over every known obstacle centre
        delta_rep = drone_pos - obs_pos         # Vector pointing FROM obstacle TO drone
        dist_rep  = float(np.linalg.norm(delta_rep))   # Distance to this obstacle (m)
        if 1e-6 < dist_rep < D0:   # Only obstacles inside influence radius contribute
            # Classical potential-field repulsion formula:
            # magnitude grows as 1/dist^2 and vanishes at D0
            magnitude = K_REP * (1.0 / dist_rep - 1.0 / D0) / (dist_rep ** 2)
            f_rep += magnitude * (delta_rep / dist_rep)   # Push away along the obstacle→drone vector

    # Combine forces and clip to max velocity
    f_total = f_att + f_rep   # Net force = attraction + repulsion
    vx_w = float(np.clip(f_total[0], -MAX_VELOCITY, MAX_VELOCITY))   # World-frame X velocity
    vy_w = float(np.clip(f_total[1], -MAX_VELOCITY, MAX_VELOCITY))   # World-frame Y velocity
    vz_w = float(np.clip(f_total[2], -MAX_VELOCITY, MAX_VELOCITY))   # World-frame Z velocity

    # Rotate world-frame velocity into drone body frame (R maps world → body)
    R = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)   # 3×3 rotation matrix
    vel_body = R @ np.array([vx_w, vy_w, vz_w])   # Matrix-vector product: world → body

    return np.array([vel_body[0], vel_body[1], 0.0, vel_body[2]])   # [vx, vy, vr=0, vz]


# ── Observation helpers ────────────────────────────────────────────────────────

def get_drone_world_pos(obs: dict) -> np.ndarray:
    """Extract drone world-frame position from observation."""
    return obs["attitude"][10:13]   # attitude array layout: [roll,pitch,yaw, qx,qy,qz,qw, vx,vy,vz, x,y,z]


def get_quaternion(obs: dict) -> np.ndarray:
    """Extract drone orientation quaternion from observation."""
    return obs["attitude"][3:7]   # Quaternion [x, y, z, w] at indices 3–6


def get_nearest_dot(obs: dict, dots_world: np.ndarray):
    """Return (world_pos, index) of the nearest untouched dot, or (None, -1)."""
    distances = np.linalg.norm(obs["dot_deltas"], axis=1)   # Euclidean dist to each dot (body frame)
    distances[obs["dot_touched"] == 1.0] = np.inf            # Exclude already-touched dots
    if np.all(np.isinf(distances)):   # All dots done
        return None, -1
    idx = int(np.argmin(distances))   # Index of the nearest untouched dot
    return dots_world[idx], idx       # Return world-frame position and its index


def compute_return_to_base_action(obs: dict, obs_positions: np.ndarray) -> np.ndarray:
    """Potential-field action toward the spawn base, avoiding obstacles."""
    return compute_potential_field_action(   # Reuse the same PF function with base as the target
        get_drone_world_pos(obs),   # Current drone position in world frame
        BASE_POSITION,              # Target is the spawn base
        obs_positions,              # Still avoid obstacles on the way home
        get_quaternion(obs),        # Current drone orientation
    )


# ── Logging ────────────────────────────────────────────────────────────────────

def log_progress(step: int, obs: dict, info: dict, target_idx: int) -> None:
    pos    = get_drone_world_pos(obs).round(2)   # World-frame position rounded for readability
    n_dots = len(obs["dot_touched"])             # Total dots in this episode
    done   = int(obs["dot_touched"].sum())       # Number of dots visited so far
    if target_idx >= 0:   # Still navigating to a dot
        dist = float(np.linalg.norm(obs["dot_deltas"][target_idx]))   # Distance to current target
        print(f"  step={step:4d}  pos={pos}  dots={done}/{n_dots}"
              f"  → dot#{target_idx + 1}  dist={dist:.2f}m")
    else:   # All dots visited
        print(f"  step={step:4d}  pos={pos}  dots={done}/{n_dots}  ALL DONE")


def log_episode_end(episode_num: int, step: int, info: dict, n_dots: int) -> None:
    if info.get("out_of_bounds"):   # Drone exited the flight dome
        reason = "OOB"
    elif info.get("collision"):     # Drone hit the ground or an obstacle
        reason = "CRASH"
    else:   # Completed (or double-touch penalty)
        reason = (f"double={info.get('double_touch')} "
                  f"complete={info.get('all_dots_touched')}")
    print(f"Ep {episode_num}  |  dots={info['dots_touched']}/{n_dots}"
          f"  |  {reason}  |  step={step}")


# ── Main loop ─────────────────────────────────────────────────────────────────
obs, info = env.reset()   # Start first episode; receive initial observation and metadata

# Cache dot world positions (needed by potential field; not directly in obs)
dots_world = env.unwrapped.dots.copy()   # shape: (num_dots, 3) — fixed until next reset
n_dots = len(obs["dot_touched"])         # Total dot count for this episode

# Spawn the first set of obstacles after reset
obstacles    = spawn_obstacles(env, dots_world)   # Returns list of {'pos', 'half', 'body_id'}
obs_centres  = np.array([o["pos"] for o in obstacles]) if obstacles else np.zeros((0, 3))   # (M,3) array for PF

episode           = 0      # Counts completed episodes
step              = 0      # Counts steps within the current episode
returning_to_base = False  # True while drone is flying back to spawn after finishing all dots

# Ring buffer for stuck detection (stores recent drone positions)
pos_history      = np.zeros((STUCK_HISTORY, 3))   # (STUCK_HISTORY, 3) rolling position buffer
escape_countdown = 0   # Counts down the remaining steps of the active escape kick

print(f"Starting obstacle maze – {n_dots} dots, {len(obstacles)} blocks per episode")
print("=" * 60)

for _ in range(10_000):   # Run for at most 10,000 steps across all episodes
    drone_pos = get_drone_world_pos(obs)   # Current world-frame position of the drone

    # ── Stuck detection ──────────────────────────────────────────────────────
    pos_history = np.roll(pos_history, 1, axis=0)   # Shift all rows down by 1 (oldest falls off)
    pos_history[0] = drone_pos                       # Insert current position at the front

    if escape_countdown > 0:   # Currently executing an escape kick — ignore normal nav
        # Apply a random body-frame kick to escape the local minimum
        kick = np.random.uniform(-MAX_VELOCITY, MAX_VELOCITY, 3)   # Random 3-D velocity direction
        action = np.array([kick[0], kick[1], 0.0, kick[2]])        # vr always 0 (no yaw)
        escape_countdown -= 1   # Decrement countdown; reverts to normal nav when it hits 0

    else:
        # Measure how much the drone has moved over the last STUCK_HISTORY steps
        if step > STUCK_HISTORY:   # Wait until the buffer is fully populated before checking
            spread = float(
                np.max(np.linalg.norm(pos_history - pos_history.mean(axis=0), axis=1))
            )   # Max distance of any buffered position from the centroid — small = stuck
            if spread < STUCK_THRESH:
                print(f"  step={step:4d}  STUCK (spread={spread:.3f}m) – applying escape kick")
                escape_countdown = ESCAPE_STEPS   # Trigger the escape countdown
                action = np.zeros(4)              # Neutral action this step before kick begins
                obs, reward, terminated, truncated, info = env.step(action)   # Step sim once
                step += 1
                continue   # Skip the rest of the loop body and start the kick next iteration

        # ── Normal navigation ────────────────────────────────────────────────
        target_world, target_idx = get_nearest_dot(obs, dots_world)   # Find closest untouched dot

        if target_world is None:   # All dots are done
            if RETURN_TO_BASE:
                dist_base = float(np.linalg.norm(BASE_POSITION - drone_pos))   # Distance to home
                if dist_base > BASE_ARRIVAL_THRESHOLD:   # Not home yet — keep flying
                    returning_to_base = True
                    action = compute_return_to_base_action(obs, obs_centres)   # PF toward base
                    if step % 60 == 0:   # Print return progress every 60 steps
                        print(f"  step={step:4d}  RETURNING TO BASE  dist={dist_base:.2f}m")
                else:   # Arrived at base
                    action = np.zeros(4)   # Hover in place
                    if returning_to_base:  # Print arrival message exactly once
                        print(f"  step={step:4d}  ARRIVED AT BASE!")
                        returning_to_base = False   # Reset so message won’t repeat
            else:
                action = np.zeros(4)   # Return-to-base disabled: hover after completion
        else:
            returning_to_base = False   # Still navigating — not in return mode
            action = compute_potential_field_action(
                drone_pos,       # Current drone position
                target_world,    # World-frame position of the nearest dot
                obs_centres,     # Obstacle centres for repulsion
                get_quaternion(obs),   # Drone orientation for world→body rotation
            )

    obs, reward, terminated, truncated, info = env.step(action)   # Advance simulation by 1 step
    step += 1   # Increment per-episode counter

    if step % 60 == 0:   # Periodic progress report (≈60 Hz → roughly every second)
        _, t_idx = get_nearest_dot(obs, dots_world)   # Re-query nearest dot for the log
        log_progress(step, obs, info, t_idx)

    if terminated or truncated:   # Episode ended: crash, OOB, time-limit, or completion
        episode += 1
        log_episode_end(episode, step, info, n_dots)

        # Reset the environment – resetSimulation() inside wipes old obstacles
        obs, info = env.reset()

        # Refresh dot positions (new random layout each episode)
        dots_world = env.unwrapped.dots.copy()

        # Spawn a fresh set of obstacles for the new episode
        obstacles   = spawn_obstacles(env, dots_world)
        obs_centres = np.array([o["pos"] for o in obstacles]) if obstacles else np.zeros((0, 3))

        step              = 0      # Reset per-episode counter
        returning_to_base = False  # Clear return flag
        pos_history[:]    = 0.0    # Wipe the stuck-detection buffer
        escape_countdown  = 0      # Cancel any active escape kick

env.close()   # Release PyBullet resources and close the render window
