"""Dot Race — two-drone competitive game.

Two drones race to claim the most dots.  The first drone to fly within the
capture radius of an unclaimed dot owns it permanently.  The drone with more
claimed dots when time expires (or all dots are taken) wins.

Both drones use the same greedy heuristic: fly toward the nearest unclaimed
dot using proportional local-velocity control (flight_mode=6).

Run:
    python games/dot_race/_run_dot_race.py
"""

import numpy as np
import pybullet as p

from cyberFly.pz_envs.quadx_envs.ma_quadx_dot_race_env import MAQuadXDotRaceEnv


# ── Heuristic parameters ──────────────────────────────────────────────────────
MAX_VELOCITY   = 2.5   # m/s — maximum commanded speed
VELOCITY_GAIN  = 1.8   # proportional gain  (velocity = gain × position-error)
BRAKE_DISTANCE = 1.2   # m  — start slowing down within this range
MIN_BRAKE_DIST = 0.15  # m  — floor for the brake ramp


# ── Policy ────────────────────────────────────────────────────────────────────

def _nearest_unclaimed(dot_deltas: np.ndarray, dot_owner: np.ndarray, agent_id: int) -> tuple[np.ndarray | None, float]:
    """Return the body-frame vector and distance to the nearest unclaimed dot.

    A dot is "available" when its owner is -1 (unclaimed).

    Args:
        dot_deltas: (num_dots, 3) body-frame vectors to every dot.
        dot_owner:  (num_dots,) ownership array  (-1 | 0 | 1).
        agent_id:   index of this drone (0 or 1).

    Returns:
        (target_vector, distance) – or (None, inf) if all dots are claimed.
    """
    distances = np.linalg.norm(dot_deltas, axis=1)
    # mask out already-claimed dots
    distances[dot_owner != -1] = np.inf

    if np.all(np.isinf(distances)):
        return None, np.inf

    idx = np.argmin(distances)
    return dot_deltas[idx], distances[idx]


def _velocity_command(target_body: np.ndarray, distance: float) -> np.ndarray:
    """Proportional velocity command with distance-based braking.

    Args:
        target_body: (3,) body-frame vector toward the target.
        distance:    scalar distance to the target.

    Returns:
        action: [vx, vy, vr, vz]
    """
    if distance < BRAKE_DISTANCE:
        speed_scale = max(
            0.0,
            (distance - MIN_BRAKE_DIST) / (BRAKE_DISTANCE - MIN_BRAKE_DIST),
        )
    else:
        speed_scale = 1.0

    gain = VELOCITY_GAIN * speed_scale
    vx = float(np.clip(gain * target_body[0], -MAX_VELOCITY, MAX_VELOCITY))
    vy = float(np.clip(gain * target_body[1], -MAX_VELOCITY, MAX_VELOCITY))
    vr = 0.0  # no active yaw control
    vz = float(np.clip(gain * target_body[2], -MAX_VELOCITY, MAX_VELOCITY))
    return np.array([vx, vy, vr, vz])


def compute_action(obs: np.ndarray, agent_id: int, num_dots: int) -> np.ndarray:
    """Greedy heuristic: fly to the nearest unclaimed dot.

    The observation layout (from MAQuadXDotRaceEnv) is:
        [0:21]                 attitude (ang_vel + quat + lin_vel + lin_pos + aux + past_action)
        [21 : 21+D*3]          dot_deltas flattened  (D = num_dots)
        [21+D*3 : 21+D*4]      dot_owner
        [21+D*4 : 21+D*4+3]    opponent world-frame position
        [21+D*4+3 : 21+D*4+5]  scores [mine, opponent]

    Args:
        obs:      flat observation vector.
        agent_id: 0 or 1.
        num_dots: number of dots in the environment.

    Returns:
        action: [vx, vy, vr, vz]
    """
    att_size = 21  # quaternion attitude size in the MA env
    D = num_dots

    dot_deltas = obs[att_size : att_size + D * 3].reshape(D, 3)
    dot_owner  = obs[att_size + D * 3 : att_size + D * 4]

    target, dist = _nearest_unclaimed(dot_deltas, dot_owner, agent_id)

    if target is None:
        # All dots claimed – hover in place
        return np.zeros(4)

    return _velocity_command(target, dist)


# ── Episode runner ────────────────────────────────────────────────────────────

def print_scoreboard(step: int, scores: np.ndarray, num_dots: int, claimed: int) -> None:
    bar0 = "█" * scores[0]
    bar1 = "█" * scores[1]
    remaining = num_dots - claimed
    print(
        f"  step={step:4d} | "
        f"uav_0 (blue): {scores[0]:2d} {bar0:<{num_dots}}  "
        f"uav_1  (red): {scores[1]:2d} {bar1:<{num_dots}}  "
        f"| unclaimed: {remaining}"
    )


def run_episode(env: MAQuadXDotRaceEnv) -> dict[str, int]:
    """Run one full episode and return final scores."""
    observations, _ = env.reset(seed=42)
    step = 0
    last_scores = np.zeros(2, dtype=int)

    print("\n" + "=" * 72)
    print("  DOT RACE  — uav_0 (blue) vs uav_1 (red)")
    print("=" * 72)

    while env.agents:
        actions: dict[str, np.ndarray] = {}
        for ag in env.agents:
            ag_id = env.agent_name_mapping[ag]
            actions[ag] = compute_action(observations[ag], ag_id, env.num_dots)

        observations, rewards, terminations, truncations, infos = env.step(actions)
        step += 1

        # Print scoreboard whenever a dot is claimed
        current_scores = env.scores.copy()
        claimed_total = int(np.sum(env.dot_owner >= 0))
        if not np.array_equal(current_scores, last_scores) or step % 30 == 0:
            print_scoreboard(step, current_scores, env.num_dots, claimed_total)
            last_scores = current_scores.copy()

        # Report termination reasons
        for ag, term in terminations.items():
            if term:
                ag_id = env.agent_name_mapping[ag]
                reason = infos.get(ag, {})
                print(f"  !! {ag} eliminated  {reason}")

    # ── Final result ──────────────────────────────────────────────────
    print("\n" + "─" * 72)
    s0, s1 = int(env.scores[0]), int(env.scores[1])
    print(f"  FINAL SCORE  →  uav_0 (blue): {s0}   uav_1 (red): {s1}")
    if s0 > s1:
        print("  🏆  uav_0  WINS!")
    elif s1 > s0:
        print("  🏆  uav_1  WINS!")
    else:
        print("  🤝  It's a DRAW!")
    print("─" * 72 + "\n")

    return {"uav_0": s0, "uav_1": s1}


def main() -> None:
    env = MAQuadXDotRaceEnv(
        num_dots=8,
        dot_touch_distance=0.5,
        flight_dome_size=7.0,
        max_duration_seconds=30.0,
        agent_hz=30,
        render_mode="human",
    )

    try:
        run_episode(env)
        input("\nPress Enter to close the simulation…")
    finally:
        env.close()


if __name__ == "__main__":
    main()
