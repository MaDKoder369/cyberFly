"""Heuristic policy for QuadX-DotTouch-v1 environment.

Uses flight_mode=6 (local velocity control) with proportional control
and distance-based braking. Achieves 6/6 dots completion.
"""
# Import gymnasium - the standard RL environment interface (OpenAI Gym successor)
import gymnasium
# Import numpy for numerical operations and array handling
import numpy as np
# Import cyberFly gym environments - this registers all cyberFly envs with gymnasium
import cyberFly.gym_envs


# Environment setup
# Create the dot-touch environment using gymnasium's make() factory function
env = gymnasium.make(
    "cyberFly/QuadX-DotTouch-v1",  # Environment ID registered in cyberFly.gym_envs
    render_mode="human",            # Enable real-time 3D visualization window
    flight_mode=6,                  # Use local velocity control [vx, vy, vr, vz]
                                    # Mode 6: commands are body-frame velocities (m/s)
                                    # instead of low-level angular rates + thrust
    truncate_on_completion=False    # Don't auto-end episode when dots done (allows return-to-base)
)

# Policy parameters - tuned for stable dot-touching behavior
TOUCH_RADIUS = 0.5    # Distance threshold (m) at which a dot is "touched" by the drone
BRAKE_DISTANCE = 1.5  # Distance (m) at which the drone starts slowing down to approach gently
MAX_VELOCITY = 2.0    # Maximum allowed velocity command (m/s) - prevents overly aggressive flight
VELOCITY_GAIN = 1.5   # Proportional gain: converts position error (m) to velocity command (m/s)

# Return-to-base feature
RETURN_TO_BASE = True          # If True, drone returns to spawn point after touching all dots
BASE_POSITION = np.array([0.0, 0.0, 1.0])  # Spawn position (origin at 1m altitude)
BASE_ARRIVAL_THRESHOLD = 0.3   # Distance (m) at which we consider base "reached"


def get_target_dot(dot_deltas, dot_touched):
    """Find the nearest untouched dot.
    
    Args:
        dot_deltas: (N, 3) array of body-frame vectors to dots
        dot_touched: (N,) binary mask (1.0 = already visited)
    
    Returns:
        tuple: (target_vector, distance) or (None, inf) if all touched
    """
    # Calculate Euclidean distance to each dot using L2 norm
    # np.linalg.norm computes sqrt(x^2 + y^2 + z^2) for each row
    distances = np.linalg.norm(dot_deltas, axis=1)
    
    # Set distances of already-touched dots to infinity so they're never selected
    # This creates a copy and modifies only the touched entries (where mask == 1.0)
    distances[dot_touched == 1.0] = np.inf
    
    # Check if all dots have been touched (all distances are infinite)
    # np.isinf returns boolean array, np.all checks if all elements are True
    if np.all(np.isinf(distances)):
        # Return None for target and infinite distance (signals completion)
        return None, np.inf
    
    # Find the index of the dot with minimum distance (nearest untouched dot)
    # np.argmin returns the index of the smallest value in the array
    target_idx = np.argmin(distances)
    
    # Return the body-frame vector to the target dot and its distance
    # dot_deltas[target_idx] is a 3D vector [x, y, z] in body frame
    return dot_deltas[target_idx], distances[target_idx]


def compute_velocity_command(target_vector, distance):
    """Compute velocity command with distance-based braking.
    
    Args:
        target_vector: (3,) body-frame vector to target [x, y, z]
        distance: Distance to target (m)
    
    Returns:
        action: [vx, vy, vr, vz] velocity command
    """
    # Implement distance-based speed scaling for smooth approach
    # As the drone gets closer to the target, reduce commanded speed to avoid overshoot
    if distance < BRAKE_DISTANCE:
        # Calculate minimum distance at which to start braking (half the touch radius)
        # This prevents the drone from stopping too early
        min_brake_dist = TOUCH_RADIUS * 0.5
        
        # Linear interpolation: ramp speed_scale from 1.0 (at BRAKE_DISTANCE) to 0.0 (at min_brake_dist)
        # Formula: (current - min) / (max - min) gives value in [0, 1]
        speed_scale = max(0.0, (distance - min_brake_dist) / (BRAKE_DISTANCE - min_brake_dist))
        # max(0.0, ...) ensures we never get negative scaling
    else:
        # Beyond braking distance: use full speed (no scaling)
        speed_scale = 1.0
    
    # Calculate effective proportional gain by multiplying base gain with speed scaling
    # This creates a distance-dependent gain that decreases as we approach the target
    gain = VELOCITY_GAIN * speed_scale
    
    # Compute forward velocity command (x-axis in body frame)
    # gain * target_vector[0] gives proportional control: velocity proportional to position error
    # np.clip limits the output to [-MAX_VELOCITY, MAX_VELOCITY] for safety
    vx = np.clip(gain * target_vector[0], -MAX_VELOCITY, MAX_VELOCITY)
    
    # Compute lateral velocity command (y-axis in body frame, right-positive)
    # Same proportional control as vx but for lateral motion
    vy = np.clip(gain * target_vector[1], -MAX_VELOCITY, MAX_VELOCITY)
    
    # Yaw rate command - set to 0.0 (no yaw control, keep current heading)
    # The drone will naturally orient itself through body-frame velocity tracking
    vr = 0.0
    
    # Compute vertical velocity command (z-axis in body frame, up-positive)
    # Uses same gain and clipping as horizontal velocities for consistent 3D motion
    vz = np.clip(gain * target_vector[2], -MAX_VELOCITY, MAX_VELOCITY)
    
    # Return the complete action as a numpy array [vx, vy, vr, vz]
    # This will be sent to the environment's step() function
    return np.array([vx, vy, vr, vz])


def compute_return_to_base_action(obs):
    """Compute velocity command to return to base after completing all dots.
    
    Args:
        obs: Observation dictionary from environment
    
    Returns:
        action: [vx, vy, vr, vz] velocity command to fly toward base
    """
    # Extract current world-frame position from observation
    # obs["attitude"][10:13] contains [x, y, z] position in meters
    current_pos = obs["attitude"][10:13]
    
    # Compute world-frame vector from current position to base
    # delta = target - current
    world_delta = BASE_POSITION - current_pos
    
    # Calculate distance to base for braking logic
    distance_to_base = np.linalg.norm(world_delta)
    
    # Import pybullet to get rotation matrix for world-to-body frame conversion
    import pybullet as p
    
    # Extract quaternion from observation (indices 3:7)
    quat = obs["attitude"][3:7]
    
    # Convert quaternion to 3x3 rotation matrix
    # This matrix transforms from world frame to body frame
    R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    
    # Transform world-frame delta to body-frame delta
    # body_delta = R @ world_delta (matrix-vector multiplication)
    body_delta = R @ world_delta
    
    # Use the same braking logic as for dots: slow down as we approach base
    if distance_to_base < BRAKE_DISTANCE:
        # Linear ramp: speed_scale goes from 1.0 to 0.0 as we approach
        min_brake_dist = BASE_ARRIVAL_THRESHOLD
        speed_scale = max(0.0, (distance_to_base - min_brake_dist) / (BRAKE_DISTANCE - min_brake_dist))
    else:
        # Beyond braking distance: use full speed
        speed_scale = 1.0
    
    # Apply proportional control with speed scaling
    gain = VELOCITY_GAIN * speed_scale
    
    # Compute velocity commands for each axis (body frame)
    vx = np.clip(gain * body_delta[0], -MAX_VELOCITY, MAX_VELOCITY)
    vy = np.clip(gain * body_delta[1], -MAX_VELOCITY, MAX_VELOCITY)
    vr = 0.0  # No yaw control
    vz = np.clip(gain * body_delta[2], -MAX_VELOCITY, MAX_VELOCITY)
    
    return np.array([vx, vy, vr, vz])


def log_progress(step, obs, info):
    """Print progress update."""
    # Extract the drone's world-frame position from the observation
    # obs["attitude"] is a flat array; indices [10:13] contain [x, y, z] position in meters
    position = obs["attitude"][10:13]
    
    # Calculate distances to all dots using Euclidean norm (same as in get_target_dot)
    distances = np.linalg.norm(obs["dot_deltas"], axis=1)
    
    # Exclude already-touched dots by setting their distances to infinity
    # This prevents them from showing up as "nearest" in the log
    distances[obs["dot_touched"] == 1.0] = np.inf
    
    # Find the distance to the nearest untouched dot
    # If all dots are touched, distances.min() will be inf, so we check and use nan for display
    nearest = distances.min() if not np.all(np.isinf(distances)) else np.nan
    
    # Print formatted progress line with:
    # - Current step count (4-digit field width for alignment)
    # - Drone position rounded to 2 decimal places as [x, y, z]
    # - Number of dots touched out of 6 total
    # - Distance to nearest untouched dot (2 decimal places)
    print(f"  step={step:4d} pos={position.round(2)} "
          f"dots={info['dots_touched']}/6 nearest={nearest:.2f}m")


def log_episode_end(episode_num, step, info):
    """Print episode summary."""
    # Determine the termination reason by checking info dictionary flags
    # Priority order: OOB > CRASH > double-touch/completion
    if info.get("out_of_bounds"):
        # Drone flew outside the allowed flight dome (radius > 5m)
        reason = "OOB"
    elif info.get("collision"):
        # Drone collided with the ground (z < 0 or similar collision detection)
        reason = "CRASH"
    else:
        # Normal termination: either double-touch penalty or successful completion
        # Format: "double=True/False complete=True/False" for debugging
        reason = f"double={info.get('double_touch')} complete={info.get('all_dots_touched')}"
    
    # Print episode summary with episode number, final dot count, reason, and total steps
    print(f"Ep {episode_num} | dots={info['dots_touched']}/6 | {reason} | step={step}")


# Main loop
# Reset the environment to get initial observation and info dictionary
# reset() returns (observation, info) tuple
obs, info = env.reset()

# Initialize episode counter (increments each time the drone resets)
episode = 0

# Initialize step counter within current episode (resets to 0 on each episode)
step = 0

# Track whether all dots are touched and we're in return-to-base mode
returning_to_base = False

# Main training/evaluation loop - run for 10,000 steps total across all episodes
for _ in range(10000):
    # Get target and compute action
    # Extract dot deltas and touched mask from the observation dictionary
    # get_target_dot returns (vector, distance) or (None, inf) if all dots touched
    target_vec, dist = get_target_dot(obs["dot_deltas"], obs["dot_touched"])
    
    # Decide on action based on whether there's a valid target
    if target_vec is None:
        # All dots touched - check if we should return to base
        if RETURN_TO_BASE:
            # Calculate distance to base to check if we've arrived
            current_pos = obs["attitude"][10:13]
            distance_to_base = np.linalg.norm(BASE_POSITION - current_pos)
            
            if distance_to_base > BASE_ARRIVAL_THRESHOLD:
                # Not at base yet - compute return action
                returning_to_base = True
                action = compute_return_to_base_action(obs)
                
                # Log return-to-base progress every 60 steps
                if step % 60 == 0:
                    print(f"  step={step:4d} RETURNING TO BASE - distance={distance_to_base:.2f}m")
            else:
                # Arrived at base - hover in place
                action = np.zeros(4)
                if returning_to_base:
                    # Just arrived - log it once
                    print(f"  step={step:4d} ARRIVED AT BASE!")
                    returning_to_base = False
        else:
            # Return-to-base disabled - just hover
            action = np.zeros(4)  # [vx=0, vy=0, vr=0, vz=0]
    else:
        # Valid target exists - compute velocity command with braking
        returning_to_base = False
        action = compute_velocity_command(target_vec, dist)
    
    # Step environment with the computed action
    # Returns: (next_observation, reward, terminated, truncated, info)
    # - terminated: episode ended due to failure condition (crash, OOB, double-touch)
    # - truncated: episode ended due to time limit or success condition (all dots touched)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Increment step counter for this episode
    step += 1
    
    # Log progress every 60 steps (approximately every 2 seconds at 30 Hz)
    # Modulo operator % gives remainder after division (0 when step is multiple of 60)
    if step % 60 == 0:
        log_progress(step, obs, info)
    
    # Handle episode end (either terminated or truncated flag is True)
    if terminated or truncated:
        # Increment episode counter
        episode += 1
        
        # Print episode summary with final statistics
        log_episode_end(episode, step, info)
        
        # Reset environment for next episode
        # Returns new initial observation and info dictionary
        obs, info = env.reset()
        
        # Reset step counter to 0 for the new episode
        step = 0
        
        # Reset return-to-base flag for new episode
        returning_to_base = False

# Clean up: close the environment and release resources (PyBullet window, etc.)
env.close()
