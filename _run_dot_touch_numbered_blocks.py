"""Numbered Dots + Guarded Blocks – QuadX Dot Touch game.

Rules
-----
  • Dots are numbered 1..N in the 3-D scene (floating white labels).
  • The drone must touch them IN ORDER: 1 → 2 → 3 → …
  • Every dot is guarded by 1-2 solid orange blocks placed directly on
    the straight-line path between the spawn base and that dot.
    The drone must navigate AROUND each guard block to reach the dot.
  • Completing all dots in order ends the episode successfully.

Navigation uses a potential-field controller:
  • Attraction  – toward the current target dot
  • Repulsion   – away from every block within influence radius D0
  • Escape kick – random burst when the drone gets stuck

Usage:
    python _run_dot_touch_numbered_blocks.py
"""

import gymnasium
import numpy as np
import pybullet as p

import cyberFly.gym_envs


# ── Environment ────────────────────────────────────────────────────────────────
DOME_SIZE = 5.0

env = gymnasium.make(
    "cyberFly/QuadX-DotTouch-v1",
    render_mode="human",
    flight_mode=6,                   # local velocity control [vx, vy, vr, vz]
    flight_dome_size=DOME_SIZE,
    truncate_on_completion=False,
)


# ── Controller parameters ──────────────────────────────────────────────────────
TOUCH_RADIUS           = 0.5    # m  – matches env default dot_touch_distance
BRAKE_DISTANCE         = 1.5    # m  – start braking when this close to target
MAX_VELOCITY           = 2.0    # m/s
RETURN_TO_BASE         = True
BASE_POSITION          = np.array([0.0, 0.0, 1.0])
BASE_ARRIVAL_THRESHOLD = 0.3    # m


# ── Potential-field parameters ─────────────────────────────────────────────────
K_ATT = 1.6   # attractive gain
K_REP = 7.0   # repulsive gain (reduced to avoid excessive trapping)
D0    = 1.2   # obstacle influence radius (m) (smaller to allow closer approach)


# ── Stuck detection ────────────────────────────────────────────────────────────
STUCK_HISTORY = 45    # length of position ring-buffer
STUCK_THRESH  = 0.10  # m – total spread below this → stuck
ESCAPE_STEPS  = 40    # steps to apply the escape kick


# ── Guard block configuration ──────────────────────────────────────────────────
# For every dot the game places GUARDS_PER_DOT blocks on the path from base.
GUARDS_PER_DOT       = 2       # blocks per dot
GUARD_OFFSETS        = [0.45, 0.70]  # fraction along base→dot segment (0=base, 1=dot)
GUARD_LATERAL_MAX    = 0.45    # m – max random sideways shift so blocks aren't perfectly centred
GUARD_HALF_EXT       = np.array([0.15, 0.55, 0.55])  # default half-extents (thin wall)
MIN_GUARD_DIST_BASE  = 1.3     # m – blocks must be at least this far from spawn base


# ── Helpers ────────────────────────────────────────────────────────────────────

def _aviary(gym_env):
    """Return the Aviary (BulletClient) from a wrapped gym env."""
    return gym_env.unwrapped.env


def spawn_guard_blocks(gym_env, dots_world: np.ndarray):
    """Spawn guard blocks on the path from base to each dot.

    For each dot i, GUARDS_PER_DOT boxes are placed at GUARD_OFFSETS
    fractions along the straight line from BASE_POSITION to dot i.
    Each block is shifted slightly sideways so the drone must go around.

    Args:
        gym_env:    Wrapped gymnasium environment.
        dots_world: (N, 3) world-frame dot positions.

    Returns:
        List of dicts {pos, half, body_id}.
    """
    aviary = _aviary(gym_env)
    rng    = np.random.default_rng()

    blocks = []

    for dot_idx, dot_pos in enumerate(dots_world):
        direction = dot_pos - BASE_POSITION          # vector base→dot
        length    = float(np.linalg.norm(direction))

        if length < 1e-6:
            continue                                  # dot at base – skip

        unit_dir = direction / length

        # Build an orthogonal lateral direction for the sideways shift
        # Pick any vector not parallel to unit_dir, then cross-product it
        not_parallel = np.array([0, 0, 1]) if abs(unit_dir[2]) < 0.9 else np.array([1, 0, 0])
        lateral = np.cross(unit_dir, not_parallel)
        lateral /= np.linalg.norm(lateral)

        for frac in GUARD_OFFSETS:
            # Centre on the path at fraction 'frac'
            centre = BASE_POSITION + frac * direction

            # Small random lateral shift so guard isn't perfectly centred
            shift  = rng.uniform(-GUARD_LATERAL_MAX, GUARD_LATERAL_MAX)
            centre = centre + shift * lateral

            # Keep block above the floor and away from the spawn base
            centre[2] = max(centre[2], 0.30)
            if float(np.linalg.norm(centre - BASE_POSITION)) < MIN_GUARD_DIST_BASE:
                continue  # too close to spawn – skip this guard position

            # Vary shape slightly: alternate between wall orientations
            if (dot_idx + int(frac * 10)) % 2 == 0:
                half = GUARD_HALF_EXT.copy() * rng.uniform(0.85, 1.25)
            else:
                half = GUARD_HALF_EXT[[1, 0, 2]].copy() * rng.uniform(0.85, 1.25)  # rotated 90°

            col_id = aviary.createCollisionShape(
                aviary.GEOM_BOX,
                halfExtents=half.tolist(),
            )
            vis_id = aviary.createVisualShape(
                aviary.GEOM_BOX,
                halfExtents=half.tolist(),
                rgbaColor=[0.90, 0.35, 0.05, 1.0],   # vivid orange
            )
            body_id = aviary.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=centre.tolist(),
            )
            blocks.append({"pos": centre.copy(), "half": half.copy(), "body_id": body_id})

    aviary.register_all_new_bodies()
    print(f"  [blocks] spawned {len(blocks)} guard blocks ({GUARDS_PER_DOT} per dot)")
    return blocks


def add_dot_labels(gym_env, dots_world: np.ndarray) -> list[int]:
    """Add a white number label above every dot using PyBullet debug text.

    Args:
        gym_env:    Wrapped gymnasium environment.
        dots_world: (N, 3) world-frame dot positions.

    Returns:
        List of debug-text item IDs (needed to remove them on reset).
    """
    aviary = _aviary(gym_env)
    label_ids = []
    for i, pos in enumerate(dots_world):
        label_pos = [pos[0], pos[1], pos[2] + 0.55]   # float above the dot sphere
        item_id = aviary.addUserDebugText(
            text=str(i + 1),
            textPosition=label_pos,
            textColorRGB=[1.0, 1.0, 1.0],
            textSize=2.0,
        )
        label_ids.append(item_id)
    return label_ids


def remove_dot_labels(gym_env, label_ids: list[int]) -> None:
    """Remove all number labels from the previous episode."""
    aviary = _aviary(gym_env)
    for item_id in label_ids:
        aviary.removeUserDebugItem(item_id)


# ── Potential-field navigation ─────────────────────────────────────────────────

def _rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    return np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)


def compute_pf_action(
    drone_pos:     np.ndarray,
    target_pos:    np.ndarray,
    obs_positions: np.ndarray,
    quaternion:    np.ndarray,
) -> np.ndarray:
    """Potential-field velocity command in drone body frame.

    Args:
        drone_pos:     (3,) world-frame drone position.
        target_pos:    (3,) world-frame target dot position.
        obs_positions: (M, 3) obstacle centre positions.
        quaternion:    (4,) drone orientation quaternion.

    Returns:
        np.ndarray: [vx, vy, vr, vz] body-frame velocity command.
    """
    # ── Attractive force ───────────────────────────────────────────────────
    delta_att = target_pos - drone_pos
    dist_att  = float(np.linalg.norm(delta_att))

    if dist_att > 1e-6:
        min_brake = TOUCH_RADIUS * 0.5
        if dist_att < BRAKE_DISTANCE:
            speed_scale = max(0.0, (dist_att - min_brake) / (BRAKE_DISTANCE - min_brake))
        else:
            speed_scale = 1.0
        f_att = K_ATT * speed_scale * (delta_att / dist_att)
    else:
        f_att = np.zeros(3)

    # ── Repulsive forces ───────────────────────────────────────────────────
    f_rep = np.zeros(3)
    for obs_pos in obs_positions:
        delta = drone_pos - obs_pos
        dist  = float(np.linalg.norm(delta))
        if 1e-6 < dist < D0:
            mag   = K_REP * (1.0 / dist - 1.0 / D0) / (dist ** 2)
            f_rep += mag * (delta / dist)

    # ── Combine and rotate into body frame ────────────────────────────────
    f_total = f_att + f_rep
    vx_w = float(np.clip(f_total[0], -MAX_VELOCITY, MAX_VELOCITY))
    vy_w = float(np.clip(f_total[1], -MAX_VELOCITY, MAX_VELOCITY))
    vz_w = float(np.clip(f_total[2], -MAX_VELOCITY, MAX_VELOCITY))

    R        = _rotation_matrix(quaternion)
    vel_body = R @ np.array([vx_w, vy_w, vz_w])

    return np.array([vel_body[0], vel_body[1], 0.0, vel_body[2]])


# ── Observation helpers ────────────────────────────────────────────────────────

def drone_pos(obs):    return obs["attitude"][10:13]
def quaternion(obs):   return obs["attitude"][3:7]


def next_dot_sequential(obs, dots_world):
    """Return (world_pos, index) of the next dot that must be touched in order.

    The drone must touch dot 0 before dot 1, dot 1 before dot 2, etc.
    Returns (None, -1) when all dots are done.
    """
    touched = obs["dot_touched"]
    for idx in range(len(touched)):
        if touched[idx] != 1.0:
            return dots_world[idx], idx
    return None, -1


# ── Logging ────────────────────────────────────────────────────────────────────

def log_step(step, obs, n_dots, target_idx):
    pos  = drone_pos(obs).round(2)
    done = int(obs["dot_touched"].sum())
    if target_idx >= 0:
        dist = float(np.linalg.norm(obs["dot_deltas"][target_idx]))
        print(f"  step={step:4d}  pos={pos}  dots={done}/{n_dots}"
              f"  → dot#{target_idx + 1}  dist={dist:.2f}m")
    else:
        print(f"  step={step:4d}  pos={pos}  dots={done}/{n_dots}  ALL DONE")


def log_episode(ep, step, info, n_dots):
    if info.get("out_of_bounds"):
        reason = "OOB"
    elif info.get("collision"):
        reason = "CRASH"
    else:
        reason = (f"double={info.get('double_touch')} "
                  f"complete={info.get('all_dots_touched')}")
    print(f"Ep {ep}  |  dots={info['dots_touched']}/{n_dots}"
          f"  |  {reason}  |  step={step}")


# ── Main loop ─────────────────────────────────────────────────────────────────
obs, info = env.reset()

dots_world = env.unwrapped.dots.copy()   # (N, 3) dot positions, fixed per episode
n_dots     = len(obs["dot_touched"])

# Spawn guard blocks and number labels for the first episode
blocks      = spawn_guard_blocks(env, dots_world)
obs_centres = np.array([b["pos"] for b in blocks]) if blocks else np.zeros((0, 3))
label_ids   = add_dot_labels(env, dots_world)

episode           = 0
step              = 0
returning_to_base = False

pos_history      = np.zeros((STUCK_HISTORY, 3))
escape_countdown = 0
escape_action    = np.zeros(4)   # fixed body-frame kick held for ESCAPE_STEPS steps

print(f"\nNumbered-blocks game started – {n_dots} dots, "
      f"{len(blocks)} guard blocks, sequential order")
print("=" * 60)

for _ in range(10_000):
    cur_pos = drone_pos(obs)

    # ── Stuck detection ──────────────────────────────────────────────────────
    pos_history    = np.roll(pos_history, 1, axis=0)
    pos_history[0] = cur_pos

    if escape_countdown > 0:
        # Hold the same kick direction chosen when escape started
        action = escape_action
        escape_countdown -= 1

    else:
        if step > STUCK_HISTORY:
            spread = float(
                np.max(np.linalg.norm(pos_history - pos_history.mean(axis=0), axis=1))
            )
            if spread < STUCK_THRESH:
                print(f"  step={step:4d}  STUCK (spread={spread:.3f}m) – escape kick")
                escape_countdown = ESCAPE_STEPS
                # Pick a fixed kick: strong upward + random horizontal to fly over the block
                hx = float(np.random.uniform(-MAX_VELOCITY, MAX_VELOCITY))
                hy = float(np.random.uniform(-MAX_VELOCITY, MAX_VELOCITY))
                escape_action = np.array([hx, hy, 0.0, MAX_VELOCITY])

        # ── Sequential dot selection ─────────────────────────────────────────
        target_world, target_idx = next_dot_sequential(obs, dots_world)

        if target_world is None:
            # All dots touched in order
            if RETURN_TO_BASE:
                d_base = float(np.linalg.norm(BASE_POSITION - cur_pos))
                if d_base > BASE_ARRIVAL_THRESHOLD:
                    returning_to_base = True
                    action = compute_pf_action(
                        cur_pos, BASE_POSITION, obs_centres, quaternion(obs)
                    )
                    if step % 60 == 0:
                        print(f"  step={step:4d}  RETURNING dist={d_base:.2f}m")
                else:
                    action = np.zeros(4)
                    if returning_to_base:
                        print(f"  step={step:4d}  ARRIVED AT BASE!")
                        returning_to_base = False
            else:
                action = np.zeros(4)
        else:
            returning_to_base = False
            action = compute_pf_action(
                cur_pos, target_world, obs_centres, quaternion(obs)
            )

    obs, reward, terminated, truncated, info = env.step(action)
    step += 1

    if step % 60 == 0:
        _, t_idx = next_dot_sequential(obs, dots_world)
        log_step(step, obs, n_dots, t_idx)

    if terminated or truncated:
        episode += 1
        log_episode(episode, step, info, n_dots)

        # Clean up labels before reset (reset wipes the PyBullet world)
        remove_dot_labels(env, label_ids)
        obs, info = env.reset()

        # Fresh dot layout and new guard blocks
        dots_world  = env.unwrapped.dots.copy()
        blocks      = spawn_guard_blocks(env, dots_world)
        obs_centres = np.array([b["pos"] for b in blocks]) if blocks else np.zeros((0, 3))
        label_ids   = add_dot_labels(env, dots_world)

        step              = 0
        returning_to_base = False
        pos_history[:]    = 0.0
        escape_countdown  = 0
        escape_action     = np.zeros(4)

env.close()
