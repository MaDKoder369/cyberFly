"""Heuristic policy for QuadX-DotTouch-v1 environment.

Uses flight_mode=6 (local velocity control) with proportional control
and distance-based braking. Achieves 6/6 dots completion.
"""
import gymnasium
import numpy as np
import cyberFly.gym_envs


# Environment setup
env = gymnasium.make(
    "cyberFly/QuadX-DotTouch-v1",
    render_mode="human",
    flight_mode=6  # Local velocity control [vx, vy, vr, vz]
)

# Policy parameters
TOUCH_RADIUS = 0.5    # Dot touch distance (m)
BRAKE_DISTANCE = 1.5  # Start braking at this distance (m)
MAX_VELOCITY = 2.0    # Maximum commanded velocity (m/s)
VELOCITY_GAIN = 1.5   # Position-to-velocity proportional gain


def get_target_dot(dot_deltas, dot_touched):
    """Find the nearest untouched dot.
    
    Args:
        dot_deltas: (N, 3) array of body-frame vectors to dots
        dot_touched: (N,) binary mask (1.0 = already visited)
    
    Returns:
        tuple: (target_vector, distance) or (None, inf) if all touched
    """
    distances = np.linalg.norm(dot_deltas, axis=1)
    distances[dot_touched == 1.0] = np.inf
    
    if np.all(np.isinf(distances)):
        return None, np.inf
    
    target_idx = np.argmin(distances)
    return dot_deltas[target_idx], distances[target_idx]


def compute_velocity_command(target_vector, distance):
    """Compute velocity command with distance-based braking.
    
    Args:
        target_vector: (3,) body-frame vector to target [x, y, z]
        distance: Distance to target (m)
    
    Returns:
        action: [vx, vy, vr, vz] velocity command
    """
    # Ramp down speed as we approach the target
    if distance < BRAKE_DISTANCE:
        # Linear ramp from 1.0 to 0.0 as distance decreases
        min_brake_dist = TOUCH_RADIUS * 0.5
        speed_scale = max(0.0, (distance - min_brake_dist) / (BRAKE_DISTANCE - min_brake_dist))
    else:
        speed_scale = 1.0
    
    # Proportional velocity control with speed scaling
    gain = VELOCITY_GAIN * speed_scale
    vx = np.clip(gain * target_vector[0], -MAX_VELOCITY, MAX_VELOCITY)
    vy = np.clip(gain * target_vector[1], -MAX_VELOCITY, MAX_VELOCITY)
    vr = 0.0  # No yaw control
    vz = np.clip(gain * target_vector[2], -MAX_VELOCITY, MAX_VELOCITY)
    
    return np.array([vx, vy, vr, vz])


def log_progress(step, obs, info):
    """Print progress update."""
    position = obs["attitude"][10:13]
    distances = np.linalg.norm(obs["dot_deltas"], axis=1)
    distances[obs["dot_touched"] == 1.0] = np.inf
    nearest = distances.min() if not np.all(np.isinf(distances)) else np.nan
    
    print(f"  step={step:4d} pos={position.round(2)} "
          f"dots={info['dots_touched']}/6 nearest={nearest:.2f}m")


def log_episode_end(episode_num, step, info):
    """Print episode summary."""
    if info.get("out_of_bounds"):
        reason = "OOB"
    elif info.get("collision"):
        reason = "CRASH"
    else:
        reason = f"double={info.get('double_touch')} complete={info.get('all_dots_touched')}"
    
    print(f"Ep {episode_num} | dots={info['dots_touched']}/6 | {reason} | step={step}")


# Main loop
obs, info = env.reset()
episode = 0
step = 0

for _ in range(10000):
    # Get target and compute action
    target_vec, dist = get_target_dot(obs["dot_deltas"], obs["dot_touched"])
    
    if target_vec is None:
        action = np.zeros(4)  # Hover if all dots touched
    else:
        action = compute_velocity_command(target_vec, dist)
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1
    
    # Log progress every 60 steps
    if step % 60 == 0:
        log_progress(step, obs, info)
    
    # Handle episode end
    if terminated or truncated:
        episode += 1
        log_episode_end(episode, step, info)
        obs, info = env.reset()
        step = 0

env.close()
