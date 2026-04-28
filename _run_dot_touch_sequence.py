"""Heuristic policy for QuadX-DotTouch-v1 environment - SEQUENTIAL mode.

Dots are numbered 1..N (by their array index order).
The drone must touch them in ascending number order: 1 → 2 → 3 → …
"""
import gymnasium
import numpy as np
import cyberFly.gym_envs


# Environment setup
env = gymnasium.make(
    "cyberFly/QuadX-DotTouch-v1",
    render_mode="human",
    flight_mode=6,                  # Local velocity control [vx, vy, vr, vz]
    truncate_on_completion=False
)

# Policy parameters
TOUCH_RADIUS = 0.5    # Distance threshold (m) at which a dot is "touched"
BRAKE_DISTANCE = 1.5  # Distance (m) at which the drone starts braking
MAX_VELOCITY = 2.0    # Maximum allowed velocity command (m/s)
VELOCITY_GAIN = 1.5   # Proportional gain: position error (m) → velocity (m/s)

# Return-to-base feature
RETURN_TO_BASE = True
BASE_POSITION = np.array([0.0, 0.0, 1.0])
BASE_ARRIVAL_THRESHOLD = 0.3


def get_next_dot_in_sequence(dot_deltas, dot_touched):
    """Find the next dot that must be touched in numerical sequence.

    Dots are numbered 1..N by their array index.  The drone must touch
    dot 1 before dot 2, dot 2 before dot 3, and so on.

    Args:
        dot_deltas:  (N, 3) array of body-frame vectors to each dot.
        dot_touched: (N,)  binary mask – 1.0 means already visited.

    Returns:
        tuple: (target_vector, distance, dot_number)
               or (None, inf, None) when all dots are touched.
    """
    n_dots = len(dot_touched)

    for idx in range(n_dots):
        if dot_touched[idx] != 1.0:
            # This is the lowest-numbered untouched dot – target it.
            vec = dot_deltas[idx]
            dist = np.linalg.norm(vec)
            dot_number = idx + 1          # Human-readable label (1-based)
            return vec, dist, dot_number

    # All dots have been touched
    return None, np.inf, None


def compute_velocity_command(target_vector, distance):
    """Compute velocity command with distance-based braking.

    Args:
        target_vector: (3,) body-frame vector to target [x, y, z]
        distance: distance to target (m)

    Returns:
        action: [vx, vy, vr, vz] velocity command
    """
    if distance < BRAKE_DISTANCE:
        min_brake_dist = TOUCH_RADIUS * 0.5
        speed_scale = max(0.0, (distance - min_brake_dist) / (BRAKE_DISTANCE - min_brake_dist))
    else:
        speed_scale = 1.0

    gain = VELOCITY_GAIN * speed_scale

    vx = np.clip(gain * target_vector[0], -MAX_VELOCITY, MAX_VELOCITY)
    vy = np.clip(gain * target_vector[1], -MAX_VELOCITY, MAX_VELOCITY)
    vr = 0.0
    vz = np.clip(gain * target_vector[2], -MAX_VELOCITY, MAX_VELOCITY)

    return np.array([vx, vy, vr, vz])


def compute_return_to_base_action(obs):
    """Compute velocity command to return to base after all dots are touched.

    Args:
        obs: observation dictionary from environment

    Returns:
        action: [vx, vy, vr, vz] velocity command
    """
    current_pos = obs["attitude"][10:13]
    world_delta = BASE_POSITION - current_pos
    distance_to_base = np.linalg.norm(world_delta)

    import pybullet as p
    quat = obs["attitude"][3:7]
    R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    body_delta = R @ world_delta

    if distance_to_base < BRAKE_DISTANCE:
        min_brake_dist = BASE_ARRIVAL_THRESHOLD
        speed_scale = max(0.0, (distance_to_base - min_brake_dist) / (BRAKE_DISTANCE - min_brake_dist))
    else:
        speed_scale = 1.0

    gain = VELOCITY_GAIN * speed_scale

    vx = np.clip(gain * body_delta[0], -MAX_VELOCITY, MAX_VELOCITY)
    vy = np.clip(gain * body_delta[1], -MAX_VELOCITY, MAX_VELOCITY)
    vr = 0.0
    vz = np.clip(gain * body_delta[2], -MAX_VELOCITY, MAX_VELOCITY)

    return np.array([vx, vy, vr, vz])


def log_progress(step, obs, info, target_dot_number):
    """Print progress update."""
    position = obs["attitude"][10:13]
    n_dots = len(obs["dot_touched"])
    dots_touched = int(obs["dot_touched"].sum())

    if target_dot_number is not None:
        target_vec = obs["dot_deltas"][target_dot_number - 1]
        dist = np.linalg.norm(target_vec)
        print(f"  step={step:4d} pos={position.round(2)} "
              f"dots={dots_touched}/{n_dots} "
              f"→ dot#{target_dot_number} dist={dist:.2f}m")
    else:
        print(f"  step={step:4d} pos={position.round(2)} "
              f"dots={dots_touched}/{n_dots} ALL DONE")


def log_episode_end(episode_num, step, info, n_dots):
    """Print episode summary."""
    if info.get("out_of_bounds"):
        reason = "OOB"
    elif info.get("collision"):
        reason = "CRASH"
    else:
        reason = f"double={info.get('double_touch')} complete={info.get('all_dots_touched')}"

    print(f"Ep {episode_num} | dots={info['dots_touched']}/{n_dots} | {reason} | step={step}")


# Main loop
obs, info = env.reset()

n_dots = len(obs["dot_touched"])   # How many dots exist in this environment

episode = 0
step = 0
returning_to_base = False

for _ in range(10000):
    # --- Sequential dot selection ---
    target_vec, dist, target_dot_number = get_next_dot_in_sequence(
        obs["dot_deltas"], obs["dot_touched"]
    )

    if target_vec is None:
        # All dots touched
        if RETURN_TO_BASE:
            current_pos = obs["attitude"][10:13]
            distance_to_base = np.linalg.norm(BASE_POSITION - current_pos)

            if distance_to_base > BASE_ARRIVAL_THRESHOLD:
                returning_to_base = True
                action = compute_return_to_base_action(obs)

                if step % 60 == 0:
                    print(f"  step={step:4d} RETURNING TO BASE - distance={distance_to_base:.2f}m")
            else:
                action = np.zeros(4)
                if returning_to_base:
                    print(f"  step={step:4d} ARRIVED AT BASE!")
                    returning_to_base = False
        else:
            action = np.zeros(4)
    else:
        returning_to_base = False
        action = compute_velocity_command(target_vec, dist)

    obs, reward, terminated, truncated, info = env.step(action)
    step += 1

    if step % 60 == 0:
        log_progress(step, obs, info, target_dot_number)

    if terminated or truncated:
        episode += 1
        log_episode_end(episode, step, info, n_dots)
        obs, info = env.reset()
        step = 0
        returning_to_base = False
