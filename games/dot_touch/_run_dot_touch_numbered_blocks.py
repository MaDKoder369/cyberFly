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

import gymnasium          # Standard RL environment interface
import numpy as np        # Numerical arrays and math operations
import pybullet as p      # Used for quaternion → rotation-matrix conversion

import cyberFly.gym_envs  # Registers cyberFly environments with gymnasium


# ── Environment ────────────────────────────────────────────────────────────────
DOME_SIZE = 5.0   # Radius (m) of the spherical flight zone

env = gymnasium.make(
    "cyberFly/QuadX-DotTouch-v1",   # Environment ID registered in cyberFly.gym_envs
    render_mode="human",              # Open a real-time 3-D visualisation window
    flight_mode=6,                    # local velocity control [vx, vy, vr, vz]
    flight_dome_size=DOME_SIZE,       # Dome radius passed to the environment
    truncate_on_completion=False,     # Keep episode running after all dots are touched
)


# ── Controller parameters ──────────────────────────────────────────────────────
TOUCH_RADIUS           = 0.5    # m  – distance at which the env registers a dot as touched
BRAKE_DISTANCE         = 1.5    # m  – start slowing down when this close to the target
MAX_VELOCITY           = 2.0    # m/s – hard cap on any single velocity component
RETURN_TO_BASE         = True                        # Fly back to spawn after all dots done
BASE_POSITION          = np.array([0.0, 0.0, 1.0])  # Spawn point: origin at 1 m altitude
BASE_ARRIVAL_THRESHOLD = 0.3    # m – within this distance the drone is considered home


# ── Potential-field parameters ─────────────────────────────────────────────────
K_ATT = 1.6   # Attractive gain: how strongly the drone is pulled toward the target dot
K_REP = 7.0   # Repulsive gain: how strongly the drone is pushed away from guard blocks
D0    = 1.2   # Influence radius (m): guard blocks beyond this distance have no effect


# ── Stuck detection ────────────────────────────────────────────────────────────
STUCK_HISTORY = 45    # Number of past positions stored in the rolling buffer
STUCK_THRESH  = 0.10  # m – if positional spread over the buffer is below this, declare stuck
ESCAPE_STEPS  = 40    # Steps to apply the escape kick before resuming normal navigation


# ── Guard block configuration ──────────────────────────────────────────────────
# For every dot the game places GUARDS_PER_DOT blocks on the path from base.
GUARDS_PER_DOT       = 2          # Number of guard blocks placed per dot
GUARD_OFFSETS        = [0.45, 0.70]   # Fraction along the base→dot line (0 = base, 1 = dot)
GUARD_LATERAL_MAX    = 0.45       # m – max random sideways shift from the path centreline
GUARD_HALF_EXT       = np.array([0.15, 0.55, 0.55])  # Default half-extents (thin vertical wall)
MIN_GUARD_DIST_BASE  = 1.3        # m – guards must be at least this far from the spawn base


# ── Helpers ────────────────────────────────────────────────────────────────────

def _aviary(gym_env):
    """Return the Aviary (BulletClient) from a wrapped gym env."""
    return gym_env.unwrapped.env   # .unwrapped removes all wrappers; .env is the BulletClient


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
    aviary = _aviary(gym_env)               # Grab the PyBullet physics client
    rng    = np.random.default_rng()        # Local RNG for reproducible per-call randomness

    blocks = []   # Accumulates metadata dicts for every successfully placed guard block

    for dot_idx, dot_pos in enumerate(dots_world):   # One group of guards per dot
        direction = dot_pos - BASE_POSITION          # World-frame vector from base → dot
        length    = float(np.linalg.norm(direction)) # Total distance base → dot (m)

        if length < 1e-6:
            continue                                  # dot at base – skip (degenerate case)

        unit_dir = direction / length   # Unit vector along the base→dot line

        # Build an orthogonal lateral direction for the sideways shift
        # Pick any vector not parallel to unit_dir, then cross-product it
        not_parallel = np.array([0, 0, 1]) if abs(unit_dir[2]) < 0.9 else np.array([1, 0, 0])
        lateral = np.cross(unit_dir, not_parallel)   # Perpendicular to the path direction
        lateral /= np.linalg.norm(lateral)           # Normalise to unit length

        for frac in GUARD_OFFSETS:   # Place one block at each offset fraction along the path
            # Place the block centre at 'frac' of the way from base to dot
            centre = BASE_POSITION + frac * direction

            # Small random lateral shift so guard isn’t perfectly centred on the path
            shift  = rng.uniform(-GUARD_LATERAL_MAX, GUARD_LATERAL_MAX)   # Random sideways offset
            centre = centre + shift * lateral   # Shift centre perpendicular to path

            # Keep block above the floor and away from the spawn base
            centre[2] = max(centre[2], 0.30)   # Clamp z ≥ 0.30 m (never underground)
            if float(np.linalg.norm(centre - BASE_POSITION)) < MIN_GUARD_DIST_BASE:
                continue  # Too close to spawn – skip so the drone always has room to take off

            # Vary shape slightly: alternate between wall orientations
            if (dot_idx + int(frac * 10)) % 2 == 0:
                half = GUARD_HALF_EXT.copy() * rng.uniform(0.85, 1.25)           # Normal orientation
            else:
                half = GUARD_HALF_EXT[[1, 0, 2]].copy() * rng.uniform(0.85, 1.25)  # Rotated 90° in XY

            col_id = aviary.createCollisionShape(   # Register box collision shape in physics
                aviary.GEOM_BOX,
                halfExtents=half.tolist(),
            )
            vis_id = aviary.createVisualShape(      # Register matching visual shape for rendering
                aviary.GEOM_BOX,
                halfExtents=half.tolist(),
                rgbaColor=[0.90, 0.35, 0.05, 1.0],   # vivid orange
            )
            body_id = aviary.createMultiBody(
                baseMass=0,                          # Static (immovable) body
                baseCollisionShapeIndex=col_id,      # Link collision shape
                baseVisualShapeIndex=vis_id,         # Link visual shape
                basePosition=centre.tolist(),        # Place at computed position
            )
            blocks.append({"pos": centre.copy(), "half": half.copy(), "body_id": body_id})

    aviary.register_all_new_bodies()   # Inform the aviary so blocks participate in collision tracking
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
    aviary = _aviary(gym_env)   # Access the PyBullet client
    label_ids = []              # Will hold the debug-text item IDs returned by PyBullet
    for i, pos in enumerate(dots_world):   # One label per dot
        label_pos = [pos[0], pos[1], pos[2] + 0.55]   # Float the text 0.55 m above the dot sphere
        item_id = aviary.addUserDebugText(
            text=str(i + 1),              # Display 1-based number (dot 0 → “1”, dot 1 → “2”, …)
            textPosition=label_pos,       # World-frame position of the text anchor
            textColorRGB=[1.0, 1.0, 1.0], # White text for contrast
            textSize=2.0,                 # Large enough to be readable in the render window
        )
        label_ids.append(item_id)   # Save the id so we can delete this label on the next reset
    return label_ids


def remove_dot_labels(gym_env, label_ids: list[int]) -> None:
    """Remove all number labels from the previous episode."""
    aviary = _aviary(gym_env)   # Access the PyBullet client
    for item_id in label_ids:   # Delete each debug-text item by its id
        aviary.removeUserDebugItem(item_id)


# ── Potential-field navigation ─────────────────────────────────────────────────

def _rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a quaternion [x,y,z,w] to a 3×3 rotation matrix (world → body)."""
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

def drone_pos(obs):    return obs["attitude"][10:13]   # World-frame [x, y, z] at indices 10–12
def quaternion(obs):   return obs["attitude"][3:7]    # Orientation quaternion [x,y,z,w] at indices 3–6


def next_dot_sequential(obs, dots_world):
    """Return (world_pos, index) of the next dot that must be touched in order.

    The drone must touch dot 0 before dot 1, dot 1 before dot 2, etc.
    Returns (None, -1) when all dots are done.
    """
    touched = obs["dot_touched"]   # (N,) binary array: 1.0 = visited, 0.0 = pending
    for idx in range(len(touched)):   # Walk in ascending index order
        if touched[idx] != 1.0:       # First untouched dot found
            return dots_world[idx], idx   # Return its world position and 0-based index
    return None, -1   # All dots visited — signal completion to the caller


# ── Logging ────────────────────────────────────────────────────────────────────

def log_step(step, obs, n_dots, target_idx):
    pos  = drone_pos(obs).round(2)            # World-frame position rounded for readability
    done = int(obs["dot_touched"].sum())      # Count of dots visited so far
    if target_idx >= 0:   # Still navigating to a dot
        dist = float(np.linalg.norm(obs["dot_deltas"][target_idx]))   # Distance to current target
        print(f"  step={step:4d}  pos={pos}  dots={done}/{n_dots}"
              f"  → dot#{target_idx + 1}  dist={dist:.2f}m")
    else:   # All dots done
        print(f"  step={step:4d}  pos={pos}  dots={done}/{n_dots}  ALL DONE")


def log_episode(ep, step, info, n_dots):
    if info.get("out_of_bounds"):   # Drone flew outside the allowed flight dome
        reason = "OOB"
    elif info.get("collision"):     # Drone hit the ground or an obstacle
        reason = "CRASH"
    else:   # Completed successfully or triggered a double-touch penalty
        reason = (f"double={info.get('double_touch')} "
                  f"complete={info.get('all_dots_touched')}")
    print(f"Ep {ep}  |  dots={info['dots_touched']}/{n_dots}"
          f"  |  {reason}  |  step={step}")


# ── Main loop ─────────────────────────────────────────────────────────────────
obs, info = env.reset()   # Start first episode; receive initial observation and metadata

dots_world = env.unwrapped.dots.copy()   # (N, 3) dot world positions; fixed until next reset
n_dots     = len(obs["dot_touched"])     # Total dot count for this episode

# Spawn guard blocks and number labels for the first episode
blocks      = spawn_guard_blocks(env, dots_world)   # Returns list of {pos, half, body_id}
obs_centres = np.array([b["pos"] for b in blocks]) if blocks else np.zeros((0, 3))   # (M,3) for PF
label_ids   = add_dot_labels(env, dots_world)       # Returns debug-text ids for later cleanup

episode           = 0      # Counts completed episodes across the entire run
step              = 0      # Counts steps within the current episode
returning_to_base = False  # True while drone is flying back to spawn after all dots done

pos_history      = np.zeros((STUCK_HISTORY, 3))   # Rolling position buffer for stuck detection
escape_countdown = 0              # Steps remaining in the current escape kick
escape_action    = np.zeros(4)    # Fixed body-frame kick held for ESCAPE_STEPS steps

print(f"\nNumbered-blocks game started – {n_dots} dots, "
      f"{len(blocks)} guard blocks, sequential order")
print("=" * 60)

for _ in range(10_000):   # Run for at most 10,000 steps across all episodes
    cur_pos = drone_pos(obs)   # Current world-frame drone position

    # ── Stuck detection ─────────────────────────────────────────────────────────
    pos_history    = np.roll(pos_history, 1, axis=0)   # Shift buffer: oldest position falls off
    pos_history[0] = cur_pos                           # Insert latest position at front

    if escape_countdown > 0:   # Escape kick is active — apply the pre-chosen direction
        # Hold the same kick direction chosen when escape started
        action = escape_action         # Fixed kick direction held for ESCAPE_STEPS
        escape_countdown -= 1          # Count down; when zero, normal nav resumes

    else:
        if step > STUCK_HISTORY:   # Wait until the buffer is full before evaluating spread
            spread = float(
                np.max(np.linalg.norm(pos_history - pos_history.mean(axis=0), axis=1))
            )   # Max deviation of any buffered position from the centroid — small = stuck
            if spread < STUCK_THRESH:
                print(f"  step={step:4d}  STUCK (spread={spread:.3f}m) – escape kick")
                escape_countdown = ESCAPE_STEPS   # Trigger the escape phase
                # Strong upward + random horizontal kick to fly over the blocking wall
                hx = float(np.random.uniform(-MAX_VELOCITY, MAX_VELOCITY))   # Random x
                hy = float(np.random.uniform(-MAX_VELOCITY, MAX_VELOCITY))   # Random y
                escape_action = np.array([hx, hy, 0.0, MAX_VELOCITY])        # Full upward thrust

        # ── Sequential dot selection ─────────────────────────────────────────
        target_world, target_idx = next_dot_sequential(obs, dots_world)   # Next dot in order

        if target_world is None:   # All dots touched in order — no nav target left
            # All dots touched in order
            if RETURN_TO_BASE:
                d_base = float(np.linalg.norm(BASE_POSITION - cur_pos))   # Distance to base
                if d_base > BASE_ARRIVAL_THRESHOLD:   # Not home yet — keep flying
                    returning_to_base = True
                    action = compute_pf_action(        # PF still avoids blocks on the way home
                        cur_pos, BASE_POSITION, obs_centres, quaternion(obs)
                    )
                    if step % 60 == 0:   # Log return progress every 60 steps
                        print(f"  step={step:4d}  RETURNING dist={d_base:.2f}m")
                else:   # Arrived at base
                    action = np.zeros(4)   # Hover in place
                    if returning_to_base:  # Print arrival message exactly once
                        print(f"  step={step:4d}  ARRIVED AT BASE!")
                        returning_to_base = False   # Reset flag so message won't repeat
            else:
                action = np.zeros(4)   # Return-to-base disabled: hover after completion
        else:
            returning_to_base = False   # Still navigating dots — not in return mode
            action = compute_pf_action(   # PF action: attracted to dot, repelled by blocks
                cur_pos, target_world, obs_centres, quaternion(obs)
            )

    obs, reward, terminated, truncated, info = env.step(action)   # Advance sim by one step
    step += 1   # Increment per-episode counter

    if step % 60 == 0:   # Periodic log (≈60 Hz → roughly every second)
        _, t_idx = next_dot_sequential(obs, dots_world)   # Re-query current dot for the log
        log_step(step, obs, n_dots, t_idx)

    if terminated or truncated:   # Episode ended: crash, OOB, time-limit, or completion
        episode += 1
        log_episode(episode, step, info, n_dots)

        # Clean up labels before reset (reset wipes the PyBullet world)
        remove_dot_labels(env, label_ids)   # Remove floating number text from the scene
        obs, info = env.reset()             # Start new episode; PyBullet world is cleared

        # Fresh dot layout and new guard blocks
        dots_world  = env.unwrapped.dots.copy()      # New random dot positions
        blocks      = spawn_guard_blocks(env, dots_world)   # New guard blocks on new paths
        obs_centres = np.array([b["pos"] for b in blocks]) if blocks else np.zeros((0, 3))
        label_ids   = add_dot_labels(env, dots_world)  # New number labels above new dots

        step              = 0           # Reset per-episode counter
        returning_to_base = False       # Clear return flag
        pos_history[:]    = 0.0         # Wipe the stuck-detection buffer
        escape_countdown  = 0           # Cancel any active escape kick
        escape_action     = np.zeros(4) # Clear the stored escape direction

env.close()   # Release PyBullet resources and close the render window
