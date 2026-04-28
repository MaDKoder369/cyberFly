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

import gymnasium
import numpy as np
import pybullet as p   # only used for quaternion→rotation-matrix helper

import cyberFly.gym_envs


# ── Environment ────────────────────────────────────────────────────────────────
DOME_SIZE = 5.0

env = gymnasium.make(
    "cyberFly/QuadX-DotTouch-v1",
    render_mode="human",
    flight_mode=6,                   # Local velocity control [vx, vy, vr, vz]
    flight_dome_size=DOME_SIZE,
    truncate_on_completion=False,
)


# ── Controller parameters ──────────────────────────────────────────────────────
TOUCH_RADIUS           = 0.5   # m  – matches env default
BRAKE_DISTANCE         = 1.5   # m  – start braking when this close
MAX_VELOCITY           = 2.0   # m/s
VELOCITY_GAIN          = 1.5   # proportional gain

RETURN_TO_BASE         = True
BASE_POSITION          = np.array([0.0, 0.0, 1.0])
BASE_ARRIVAL_THRESHOLD = 0.3   # m


# ── Potential-field parameters ─────────────────────────────────────────────────
K_ATT = 1.5   # attractive gain (toward dot)
K_REP = 8.0   # repulsive gain (away from obstacle)
D0    = 1.4   # obstacle influence radius (m)


# ── Stuck-detection parameters ────────────────────────────────────────────────
STUCK_HISTORY      = 40    # steps of position history to keep
STUCK_THRESH       = 0.10  # m – declare stuck if total spread < this
ESCAPE_STEPS       = 20    # steps of random kick to apply when stuck


# ── Obstacle configuration ─────────────────────────────────────────────────────
NUM_OBSTACLES    = 8
MIN_SPAWN_DIST   = 1.2    # minimum clearance (m) between block centre and any dot / base


def _get_aviary(gym_env):
    """Return the underlying Aviary (BulletClient) from a wrapped gymnasium env."""
    return gym_env.unwrapped.env


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
    aviary = _get_aviary(gym_env)
    rng = np.random.default_rng()

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

    obstacles = []

    for _ in range(NUM_OBSTACLES):
        half_ext = shape_presets[rng.integers(len(shape_presets))].copy()
        # Add a small random scale jitter so each instance looks slightly different
        half_ext *= rng.uniform(0.8, 1.3)

        col_id = aviary.createCollisionShape(
            aviary.GEOM_BOX,
            halfExtents=half_ext.tolist(),
        )
        vis_id = aviary.createVisualShape(
            aviary.GEOM_BOX,
            halfExtents=half_ext.tolist(),
            rgbaColor=[0.85, 0.30, 0.10, 1.0],   # vivid orange-red
        )

        # Try up to 300 times to find a position with enough clearance
        for _attempt in range(300):
            radius = rng.uniform(0.8, DOME_SIZE * 0.78)
            angle  = rng.uniform(0.0, 2.0 * np.pi)
            height = rng.uniform(0.35, DOME_SIZE * 0.55)
            pos = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                height,
            ])

            # Reject if too close to any dot or to the spawn base
            dot_dists = np.linalg.norm(dots_world - pos, axis=1)
            if np.any(dot_dists < MIN_SPAWN_DIST):
                continue
            if np.linalg.norm(BASE_POSITION - pos) < MIN_SPAWN_DIST:
                continue
            break
        else:
            # Could not find a free spot – skip this obstacle
            aviary.removeCollisionShape(col_id) if hasattr(aviary, "removeCollisionShape") else None
            continue

        body_id = aviary.createMultiBody(
            baseMass=0,                         # static (immovable) block
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=pos.tolist(),
        )
        obstacles.append({
            "pos":     pos.copy(),
            "half":    half_ext.copy(),
            "body_id": body_id,
        })

    # Update the aviary's contact-array so new bodies participate in collision tracking
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
    delta_att = target_pos - drone_pos
    dist_att  = float(np.linalg.norm(delta_att))

    if dist_att > 1e-6:
        if dist_att < BRAKE_DISTANCE:
            min_brake = TOUCH_RADIUS * 0.5
            speed_scale = max(
                0.0,
                (dist_att - min_brake) / (BRAKE_DISTANCE - min_brake),
            )
        else:
            speed_scale = 1.0
        f_att = K_ATT * speed_scale * (delta_att / dist_att)
    else:
        f_att = np.zeros(3)

    # Repulsive forces – one contribution per obstacle within influence radius D0
    f_rep = np.zeros(3)
    for obs_pos in obs_positions:
        delta_rep = drone_pos - obs_pos
        dist_rep  = float(np.linalg.norm(delta_rep))
        if 1e-6 < dist_rep < D0:
            magnitude = K_REP * (1.0 / dist_rep - 1.0 / D0) / (dist_rep ** 2)
            f_rep += magnitude * (delta_rep / dist_rep)

    # Combine forces and clip to max velocity
    f_total = f_att + f_rep
    vx_w = float(np.clip(f_total[0], -MAX_VELOCITY, MAX_VELOCITY))
    vy_w = float(np.clip(f_total[1], -MAX_VELOCITY, MAX_VELOCITY))
    vz_w = float(np.clip(f_total[2], -MAX_VELOCITY, MAX_VELOCITY))

    # Rotate world-frame velocity into drone body frame (R maps world → body)
    R = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
    vel_body = R @ np.array([vx_w, vy_w, vz_w])

    return np.array([vel_body[0], vel_body[1], 0.0, vel_body[2]])


# ── Observation helpers ────────────────────────────────────────────────────────

def get_drone_world_pos(obs: dict) -> np.ndarray:
    """Extract drone world-frame position from observation."""
    return obs["attitude"][10:13]


def get_quaternion(obs: dict) -> np.ndarray:
    """Extract drone orientation quaternion from observation."""
    return obs["attitude"][3:7]


def get_nearest_dot(obs: dict, dots_world: np.ndarray):
    """Return (world_pos, index) of the nearest untouched dot, or (None, -1)."""
    distances = np.linalg.norm(obs["dot_deltas"], axis=1)
    distances[obs["dot_touched"] == 1.0] = np.inf
    if np.all(np.isinf(distances)):
        return None, -1
    idx = int(np.argmin(distances))
    return dots_world[idx], idx


def compute_return_to_base_action(obs: dict, obs_positions: np.ndarray) -> np.ndarray:
    """Potential-field action toward the spawn base, avoiding obstacles."""
    return compute_potential_field_action(
        get_drone_world_pos(obs),
        BASE_POSITION,
        obs_positions,
        get_quaternion(obs),
    )


# ── Logging ────────────────────────────────────────────────────────────────────

def log_progress(step: int, obs: dict, info: dict, target_idx: int) -> None:
    pos    = get_drone_world_pos(obs).round(2)
    n_dots = len(obs["dot_touched"])
    done   = int(obs["dot_touched"].sum())
    if target_idx >= 0:
        dist = float(np.linalg.norm(obs["dot_deltas"][target_idx]))
        print(f"  step={step:4d}  pos={pos}  dots={done}/{n_dots}"
              f"  → dot#{target_idx + 1}  dist={dist:.2f}m")
    else:
        print(f"  step={step:4d}  pos={pos}  dots={done}/{n_dots}  ALL DONE")


def log_episode_end(episode_num: int, step: int, info: dict, n_dots: int) -> None:
    if info.get("out_of_bounds"):
        reason = "OOB"
    elif info.get("collision"):
        reason = "CRASH"
    else:
        reason = (f"double={info.get('double_touch')} "
                  f"complete={info.get('all_dots_touched')}")
    print(f"Ep {episode_num}  |  dots={info['dots_touched']}/{n_dots}"
          f"  |  {reason}  |  step={step}")


# ── Main loop ─────────────────────────────────────────────────────────────────
obs, info = env.reset()

# Cache dot world positions (needed by potential field; not directly in obs)
dots_world = env.unwrapped.dots.copy()   # shape: (num_dots, 3)
n_dots = len(obs["dot_touched"])

# Spawn the first set of obstacles after reset
obstacles    = spawn_obstacles(env, dots_world)
obs_centres  = np.array([o["pos"] for o in obstacles]) if obstacles else np.zeros((0, 3))

episode           = 0
step              = 0
returning_to_base = False

# Ring buffer for stuck detection (stores recent drone positions)
pos_history      = np.zeros((STUCK_HISTORY, 3))
escape_countdown = 0

print(f"Starting obstacle maze – {n_dots} dots, {len(obstacles)} blocks per episode")
print("=" * 60)

for _ in range(10_000):
    drone_pos = get_drone_world_pos(obs)

    # ── Stuck detection ──────────────────────────────────────────────────────
    pos_history = np.roll(pos_history, 1, axis=0)
    pos_history[0] = drone_pos

    if escape_countdown > 0:
        # Apply a random body-frame kick to escape the local minimum
        kick = np.random.uniform(-MAX_VELOCITY, MAX_VELOCITY, 3)
        action = np.array([kick[0], kick[1], 0.0, kick[2]])
        escape_countdown -= 1

    else:
        # Measure how much the drone has moved over the last STUCK_HISTORY steps
        if step > STUCK_HISTORY:
            spread = float(
                np.max(np.linalg.norm(pos_history - pos_history.mean(axis=0), axis=1))
            )
            if spread < STUCK_THRESH:
                print(f"  step={step:4d}  STUCK (spread={spread:.3f}m) – applying escape kick")
                escape_countdown = ESCAPE_STEPS
                action = np.zeros(4)
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1
                continue

        # ── Normal navigation ────────────────────────────────────────────────
        target_world, target_idx = get_nearest_dot(obs, dots_world)

        if target_world is None:
            # All dots touched
            if RETURN_TO_BASE:
                dist_base = float(np.linalg.norm(BASE_POSITION - drone_pos))
                if dist_base > BASE_ARRIVAL_THRESHOLD:
                    returning_to_base = True
                    action = compute_return_to_base_action(obs, obs_centres)
                    if step % 60 == 0:
                        print(f"  step={step:4d}  RETURNING TO BASE  dist={dist_base:.2f}m")
                else:
                    action = np.zeros(4)
                    if returning_to_base:
                        print(f"  step={step:4d}  ARRIVED AT BASE!")
                        returning_to_base = False
            else:
                action = np.zeros(4)
        else:
            returning_to_base = False
            action = compute_potential_field_action(
                drone_pos,
                target_world,
                obs_centres,
                get_quaternion(obs),
            )

    obs, reward, terminated, truncated, info = env.step(action)
    step += 1

    if step % 60 == 0:
        _, t_idx = get_nearest_dot(obs, dots_world)
        log_progress(step, obs, info, t_idx)

    if terminated or truncated:
        episode += 1
        log_episode_end(episode, step, info, n_dots)

        # Reset the environment – resetSimulation() inside wipes old obstacles
        obs, info = env.reset()

        # Refresh dot positions (new random layout each episode)
        dots_world = env.unwrapped.dots.copy()

        # Spawn a fresh set of obstacles for the new episode
        obstacles   = spawn_obstacles(env, dots_world)
        obs_centres = np.array([o["pos"] for o in obstacles]) if obstacles else np.zeros((0, 3))

        step              = 0
        returning_to_base = False
        pos_history[:]    = 0.0
        escape_countdown  = 0

env.close()
