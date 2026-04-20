import gymnasium
import numpy as np
import cyberFly.gym_envs

env = gymnasium.make("cyberFly/QuadX-DotTouch-v1", render_mode="human", flight_mode=6)
obs, info = env.reset()

# flight_mode=6: action = [vx, vy, vr, vz]  (local linear velocities + yaw rate + vertical vel)
# attitude layout (quaternion mode):
#   [0:3]  ang_vel  (body frame)
#   [3:7]  quaternion
#   [7:10] lin_vel  (body frame, u/v/w)
#   [10:13] lin_pos (world frame)
#   [13:17] prev_action
#   [17:21] aux

TOUCH_D = 0.5    # dot_touch_distance
BRAKE_D = 1.5    # start braking this many metres out
MAX_VEL = 2.0    # max commanded velocity (m/s)
KP      = 1.5    # position → velocity gain
KZ      = 1.5    # altitude P gain


def policy(obs):
    deltas  = obs["dot_deltas"]    # (N,3) body-frame vectors to dots
    touched = obs["dot_touched"]   # (N,) 1.0 = visited

    dists = np.linalg.norm(deltas, axis=1)

    # Choose nearest untouched dot
    candidate_dists = dists.copy()
    candidate_dists[touched == 1.0] = np.inf

    if np.all(np.isinf(candidate_dists)):
        return np.array([0.0, 0.0, 0.0, 0.0])

    target_idx = np.argmin(candidate_dists)
    d = deltas[target_idx]
    dist = dists[target_idx]

    # Ramp speed to 0 near target
    if dist < BRAKE_D:
        speed_scale = max((dist - TOUCH_D * 0.5) / (BRAKE_D - TOUCH_D * 0.5), 0.0)
    else:
        speed_scale = 1.0

    kp_eff = KP * speed_scale

    vx = np.clip(kp_eff * d[0], -MAX_VEL, MAX_VEL)
    vy = np.clip(kp_eff * d[1], -MAX_VEL, MAX_VEL)
    vr = 0.0
    vz = np.clip(KZ * d[2], -MAX_VEL, MAX_VEL)

    return np.array([vx, vy, vr, vz])


resets = 0
step = 0

for _ in range(10000):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1
    if step % 60 == 0:
        lin_pos = obs["attitude"][10:13]
        dists = np.linalg.norm(obs["dot_deltas"], axis=1)
        dists[obs["dot_touched"] == 1.0] = np.inf
        print(f"  step={step:4d} pos={lin_pos.round(2)} dots={info['dots_touched']}/6 nearest={dists.min():.2f}m")
    if terminated or truncated:
        reason = "OOB" if info.get("out_of_bounds") else "CRASH" if info.get("collision") else f"double={info.get('double_touch')} complete={info.get('all_dots_touched')}"
        print(f"Ep {resets+1} | dots={info['dots_touched']}/6 | {reason} | step={step}")
        obs, info = env.reset()
        resets += 1
        step = 0

env.close()
