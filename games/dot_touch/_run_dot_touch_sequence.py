"""Heuristic policy for QuadX-DotTouch-v1 environment - SEQUENTIAL mode.

Dots are numbered 1..N (by their array index order).
The drone must touch them in ascending number order: 1 → 2 → 3 → …
"""
import gymnasium          # Standard RL environment interface (successor to OpenAI Gym)
import numpy as np        # Numerical arrays and math (vectors, norms, clipping)
import cyberFly.gym_envs  # Registers cyberFly environments so gymnasium.make() can find them


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
RETURN_TO_BASE = True                        # If True, drone flies back to spawn after all dots
BASE_POSITION = np.array([0.0, 0.0, 1.0])   # Spawn point: world origin at 1 m altitude
BASE_ARRIVAL_THRESHOLD = 0.3                # Distance (m) at which base is considered "reached"


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
    n_dots = len(dot_touched)   # Total number of dots placed in this episode

    for idx in range(n_dots):      # Walk the array in ascending index order (0, 1, 2, …)
        if dot_touched[idx] != 1.0:   # 0.0 = untouched; 1.0 = already visited
            # First untouched dot found — this is the required next target
            vec = dot_deltas[idx]              # (3,) body-frame vector pointing to this dot
            dist = np.linalg.norm(vec)         # Euclidean distance to the dot (metres)
            dot_number = idx + 1               # Convert 0-based index → 1-based human label
            return vec, dist, dot_number       # Return immediately; lower-index dots go first

    # Every dot has been touched — no target left
    return None, np.inf, None   # None target signals the main loop to switch to return-to-base


def compute_velocity_command(target_vector, distance):
    """Compute velocity command with distance-based braking.

    Args:
        target_vector: (3,) body-frame vector to target [x, y, z]
        distance: distance to target (m)

    Returns:
        action: [vx, vy, vr, vz] velocity command
    """
    if distance < BRAKE_DISTANCE:   # Inside braking zone — reduce speed to avoid overshoot
        min_brake_dist = TOUCH_RADIUS * 0.5   # Below this distance the commanded speed becomes 0
        # Linear ramp: speed_scale = 1.0 at BRAKE_DISTANCE, 0.0 at min_brake_dist
        speed_scale = max(0.0, (distance - min_brake_dist) / (BRAKE_DISTANCE - min_brake_dist))
        # max(0.0, …) prevents negative scaling if drone overshoots past the dot
    else:
        speed_scale = 1.0   # Beyond braking zone: fly at full commanded speed

    gain = VELOCITY_GAIN * speed_scale   # Effective gain = base gain × speed reduction factor

    # Proportional control: velocity = gain × position error, clipped for safety
    vx = np.clip(gain * target_vector[0], -MAX_VELOCITY, MAX_VELOCITY)  # Forward / back (body X)
    vy = np.clip(gain * target_vector[1], -MAX_VELOCITY, MAX_VELOCITY)  # Left / right  (body Y)
    vr = 0.0   # Yaw rate: zero — keep current heading throughout
    vz = np.clip(gain * target_vector[2], -MAX_VELOCITY, MAX_VELOCITY)  # Up / down     (body Z)

    return np.array([vx, vy, vr, vz])   # Pack into the 4-element action the env expects


def compute_return_to_base_action(obs):
    """Compute velocity command to return to base after all dots are touched.

    Args:
        obs: observation dictionary from environment

    Returns:
        action: [vx, vy, vr, vz] velocity command
    """
    current_pos = obs["attitude"][10:13]          # World-frame drone position [x, y, z] (metres)
    world_delta = BASE_POSITION - current_pos      # Vector from drone to base in world frame
    distance_to_base = np.linalg.norm(world_delta) # Straight-line distance remaining (metres)

    import pybullet as p   # PyBullet utility: only needed for quaternion→rotation-matrix helper
    quat = obs["attitude"][3:7]                            # Orientation quaternion [x, y, z, w]
    R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)  # 3×3 rotation matrix (world → body)
    body_delta = R @ world_delta   # Transform the world-frame delta into the drone's body frame

    if distance_to_base < BRAKE_DISTANCE:   # Inside braking zone — reduce speed to land softly
        min_brake_dist = BASE_ARRIVAL_THRESHOLD   # Below this the drone hovers in place
        # Same linear ramp as used for dot approach
        speed_scale = max(0.0, (distance_to_base - min_brake_dist) / (BRAKE_DISTANCE - min_brake_dist))
    else:
        speed_scale = 1.0   # Full speed until braking zone

    gain = VELOCITY_GAIN * speed_scale   # Scale down gain as we approach base

    # Proportional control in body frame, clipped to safe velocity limits
    vx = np.clip(gain * body_delta[0], -MAX_VELOCITY, MAX_VELOCITY)  # Body-frame forward/back
    vy = np.clip(gain * body_delta[1], -MAX_VELOCITY, MAX_VELOCITY)  # Body-frame left/right
    vr = 0.0   # No yaw rotation during return
    vz = np.clip(gain * body_delta[2], -MAX_VELOCITY, MAX_VELOCITY)  # Body-frame up/down

    return np.array([vx, vy, vr, vz])   # 4-element action vector sent to env.step()


def log_progress(step, obs, info, target_dot_number):
    """Print progress update."""
    position = obs["attitude"][10:13]          # World-frame position [x, y, z]
    n_dots = len(obs["dot_touched"])           # Total dots in this episode
    dots_touched = int(obs["dot_touched"].sum())  # Sum of 1.0 flags = count of visited dots

    if target_dot_number is not None:   # Still navigating toward a dot
        # Retrieve body-frame vector to the current target (1-based → 0-based index)
        target_vec = obs["dot_deltas"][target_dot_number - 1]
        dist = np.linalg.norm(target_vec)   # Distance to current target dot (metres)
        print(f"  step={step:4d} pos={position.round(2)} "
              f"dots={dots_touched}/{n_dots} "
              f"→ dot#{target_dot_number} dist={dist:.2f}m")
    else:   # All dots have been touched
        print(f"  step={step:4d} pos={position.round(2)} "
              f"dots={dots_touched}/{n_dots} ALL DONE")


def log_episode_end(episode_num, step, info, n_dots):
    """Print episode summary."""
    if info.get("out_of_bounds"):   # Drone flew outside the allowed flight dome
        reason = "OOB"
    elif info.get("collision"):     # Drone hit the ground or an obstacle
        reason = "CRASH"
    else:   # Episode ended normally: either all-done or a double-touch penalty
        reason = f"double={info.get('double_touch')} complete={info.get('all_dots_touched')}"

    # One-line summary: episode #, dots scored, termination reason, total steps taken
    print(f"Ep {episode_num} | dots={info['dots_touched']}/{n_dots} | {reason} | step={step}")


# Main loop
obs, info = env.reset()   # Start first episode; get initial observation and metadata dict

n_dots = len(obs["dot_touched"])   # How many dots exist in this environment

episode = 0             # Counter for completed episodes across the entire run
step = 0                # Counter for steps within the current episode
returning_to_base = False   # True once all dots are touched and drone is flying home

for _ in range(10000):   # Run for at most 10,000 steps across all episodes
    # --- Sequential dot selection ---
    # Ask the heuristic which dot to target next (enforces 1 → 2 → 3 → … order)
    target_vec, dist, target_dot_number = get_next_dot_in_sequence(
        obs["dot_deltas"],   # (N, 3) body-frame vectors to each dot
        obs["dot_touched"]   # (N,)  binary flags: 1.0 = already touched
    )

    if target_vec is None:   # All dots have been visited — no navigation target left
        if RETURN_TO_BASE:
            current_pos = obs["attitude"][10:13]                    # Current world-frame position
            distance_to_base = np.linalg.norm(BASE_POSITION - current_pos)  # Metres to base

            if distance_to_base > BASE_ARRIVAL_THRESHOLD:   # Not at base yet — keep flying
                returning_to_base = True   # Mark as returning (used to print arrival once)
                action = compute_return_to_base_action(obs)   # Fly toward spawn point

                if step % 60 == 0:   # Print status every 60 steps to avoid console spam
                    print(f"  step={step:4d} RETURNING TO BASE - distance={distance_to_base:.2f}m")
            else:   # Close enough to base — stop moving
                action = np.zeros(4)   # Hover: zero all velocity commands [vx=0, vy=0, vr=0, vz=0]
                if returning_to_base:   # Print the arrival message exactly once
                    print(f"  step={step:4d} ARRIVED AT BASE!")
                    returning_to_base = False   # Reset flag so message isn't repeated
        else:
            action = np.zeros(4)   # Return-to-base disabled: hover in place after completion
    else:
        returning_to_base = False   # Still navigating to a dot — not in return-home mode
        action = compute_velocity_command(target_vec, dist)   # Proportional velocity toward dot

    obs, reward, terminated, truncated, info = env.step(action)   # Apply action, advance sim by 1 step
    step += 1   # Increment per-episode step counter

    if step % 60 == 0:   # Log progress roughly every second (env runs at ~60 Hz)
        log_progress(step, obs, info, target_dot_number)

    if terminated or truncated:   # Episode ended: crash, OOB, time-limit, or completion
        episode += 1                                    # Tally completed episodes
        log_episode_end(episode, step, info, n_dots)    # Print one-line summary
        obs, info = env.reset()   # Start new episode with fresh random dot placement
        step = 0                  # Reset per-episode step counter
        returning_to_base = False # Clear return flag so new episode starts clean
