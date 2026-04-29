"""Multiagent QuadX Dot Race Environment.

Two drones compete to claim as many dots as possible by touching them first.
The first drone to fly within *dot_touch_distance* of an unclaimed dot claims
it permanently.  The drone with more claimed dots at the end of the episode
wins.

PettingZoo Parallel API.  Both agents act simultaneously each step.

Observation (flat Box per agent, flight_mode=6 local-velocity):
    attitude      – ang_vel(3) + quaternion(4) + lin_vel(3) + lin_pos(3) +
                    aux_state(4) + past_action(4)  =  21 values
    dot_deltas    – (num_dots × 3) body-frame vectors to every dot
    dot_owner     – (num_dots,)  −1 unclaimed | 0 owned by uav_0 | 1 owned by uav_1
    opponent_pos  – (3,)  world-frame position of the other drone
    scores        – (2,)  [my_score, opponent_score]

Actions: [vx, vy, vr, vz]  (flight_mode=6, local-velocity commands)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pybullet as p
from gymnasium import spaces

from cyberFly.pz_envs.quadx_envs.ma_quadx_base_env import MAQuadXBaseEnv


class MAQuadXDotRaceEnv(MAQuadXBaseEnv):
    """Two-drone competitive dot-claiming race.

    Args:
        num_dots (int): number of shared dots in the arena.
        dot_touch_distance (float): capture radius for a dot (metres).
        flight_dome_size (float): radius of the allowable flying area (metres).
        max_duration_seconds (float): maximum episode length in seconds.
        agent_hz (int): control loop rate (must divide 120 evenly).
        render_mode (None | str): ``"human"`` for live 3-D view, else ``None``.
    """

    metadata = {
        "render_modes": ["human"],
        "name": "ma_quadx_dot_race",
        "is_parallelizable": True,
    }

    # RGBA colours: index 0 = unclaimed (green), 1 = uav_0 (blue), 2 = uav_1 (red)
    _DOT_COLOURS: list[tuple[float, float, float, float]] = [
        (0.0, 0.85, 0.2, 1.0),
        (0.2, 0.45, 1.0, 1.0),
        (1.0, 0.25, 0.15, 1.0),
    ]

    def __init__(
        self,
        num_dots: int = 8,
        dot_touch_distance: float = 0.5,
        flight_dome_size: float = 7.0,
        max_duration_seconds: float = 30.0,
        agent_hz: int = 30,
        render_mode: None | str = None,
    ) -> None:
        # Place the two drones on opposite sides of the arena at 1.5 m altitude.
        # Both start flat (zero Euler angles) to avoid roll/pitch instability.
        start_pos = np.array([[-2.5, 0.0, 1.5], [2.5, 0.0, 1.5]])
        start_orn = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        super().__init__(
            start_pos=start_pos,
            start_orn=start_orn,
            flight_mode=6,              # local-velocity control
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation="quaternion",
            agent_hz=agent_hz,
            render_mode=render_mode,
        )

        self.num_dots = num_dots
        self.dot_touch_distance = dot_touch_distance

        # ── Flat observation space ────────────────────────────────────
        # combined_space: 13(quat-attitude) + 4(aux) + 4(action) = 21
        # dot_deltas:  num_dots * 3
        # dot_owner:   num_dots
        # opponent_pos: 3
        # scores:       2
        obs_size = (
            self.combined_space.shape[0]  # 21
            + num_dots * 3
            + num_dots
            + 3
            + 2
        )
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    # ──────────────────────────────────────────────────────────────────
    # PettingZoo required spaces
    # ──────────────────────────────────────────────────────────────────

    def observation_space(self, agent: Any = None) -> spaces.Box:  # noqa: ARG002
        return self._observation_space

    # ──────────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: None | int = None,
        options: None | dict[str, Any] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if options is None:
            options = {}

        super().begin_reset(seed, options)

        # Spawn dots (must happen after begin_reset so aviary exists)
        self._spawn_dots(seed)
        self._dot_visuals: list[int] = []
        if self.render_mode:
            self._render_dots()

        # ── Episode state (must be initialised BEFORE end_reset calls update_states) ──
        # dot_owner[i] = -1  →  unclaimed
        #              =  0  →  owned by uav_0
        #              =  1  →  owned by uav_1
        self.dot_owner = np.full(self.num_dots, -1, dtype=np.int8)
        self.scores = np.zeros(2, dtype=np.int32)

        # Entry-detection state: prevents multiple claims per linger
        self._was_inside = np.zeros((2, self.num_dots), dtype=bool)

        # Per-outer-step claim counters (reset before each outer step)
        self._step_claims = np.zeros(2, dtype=int)

        super().end_reset(seed, options)

        observations = {
            ag: self.compute_observation_by_id(self.agent_name_mapping[ag])
            for ag in self.agents
        }
        infos: dict[str, Any] = {ag: {} for ag in self.agents}
        return observations, infos

    # ──────────────────────────────────────────────────────────────────
    # Dot helpers
    # ──────────────────────────────────────────────────────────────────

    def _spawn_dots(self, seed: None | int) -> None:
        """Sample *num_dots* random positions inside the flight dome."""
        rng = np.random.default_rng(seed)
        self.dots = np.zeros((self.num_dots, 3))
        for i in range(self.num_dots):
            theta = rng.uniform(0.0, 2.0 * np.pi)
            phi = rng.uniform(0.0, np.pi)
            dist = rng.uniform(1.2, self.flight_dome_size * 0.85)
            x = dist * np.sin(phi) * np.cos(theta)
            y = dist * np.sin(phi) * np.sin(theta)
            z = max(abs(dist * np.cos(phi)), 0.5)
            self.dots[i] = [x, y, z]

    def _render_dots(self) -> None:
        """Load visual URDF spheres for every dot."""
        urdf_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../models/target.urdf",
        )
        for pos in self.dots:
            body_id = self.aviary.loadURDF(
                urdf_path,
                basePosition=pos.tolist(),
                useFixedBase=True,
                globalScaling=self.dot_touch_distance / 2.0,
            )
            self._dot_visuals.append(body_id)
            self.aviary.changeVisualShape(
                body_id, linkIndex=-1, rgbaColor=self._DOT_COLOURS[0]
            )

    def _paint_dot(self, dot_idx: int, owner: int) -> None:
        """Recolour a dot to show which drone claimed it (0=blue, 1=red)."""
        if self.render_mode and dot_idx < len(self._dot_visuals):
            self.aviary.changeVisualShape(
                self._dot_visuals[dot_idx],
                linkIndex=-1,
                rgbaColor=self._DOT_COLOURS[owner + 1],
            )

    # ──────────────────────────────────────────────────────────────────
    # Physics hook – claim dots on first entry
    # ──────────────────────────────────────────────────────────────────

    def update_states(self) -> None:
        """Check dot proximity and claim unclaimed dots on first entry."""
        for agent_id in range(2):
            _, _, _, lin_pos, quaternion = self.compute_attitude_by_id(agent_id)
            world_dists = np.linalg.norm(self.dots - lin_pos, axis=-1)
            is_inside = world_dists < self.dot_touch_distance

            # Only process dots newly entered this sub-step
            newly_entered = is_inside & ~self._was_inside[agent_id]
            for i in np.where(newly_entered)[0]:
                if self.dot_owner[i] == -1:
                    # First drone here → claim
                    self.dot_owner[i] = agent_id
                    self.scores[agent_id] += 1
                    self._step_claims[agent_id] += 1
                    self._paint_dot(i, agent_id)

            self._was_inside[agent_id] = is_inside

    # ──────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────

    def compute_observation_by_id(self, agent_id: int) -> np.ndarray:
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = self.compute_attitude_by_id(
            agent_id
        )
        aux_state = self.aviary.aux_state(agent_id)

        # Standard attitude vector (quaternion representation)
        attitude = np.concatenate(
            [ang_vel, quaternion, lin_vel, lin_pos, aux_state, self.past_actions[agent_id]]
        )

        # Body-frame vectors to every dot
        rotation = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        dot_deltas = np.matmul(self.dots - lin_pos, rotation).flatten()

        # Dot ownership as float: −1 unclaimed, 0 = uav_0, 1 = uav_1
        dot_owner = self.dot_owner.astype(np.float64)

        # Opponent world-frame position
        opp_id = 1 - agent_id
        _, _, _, opp_pos, _ = self.compute_attitude_by_id(opp_id)

        # Scores: [mine, opponent's]
        scores = np.array(
            [float(self.scores[agent_id]), float(self.scores[opp_id])]
        )

        return np.concatenate([attitude, dot_deltas, dot_owner, opp_pos, scores])

    # ──────────────────────────────────────────────────────────────────
    # Per-sub-step termination / truncation / reward
    # ──────────────────────────────────────────────────────────────────

    def compute_term_trunc_reward_info_by_id(
        self, agent_id: int
    ) -> tuple[bool, bool, float, dict[str, Any]]:
        reward = 0.0
        term = False
        trunc = self.step_count >= self.max_steps
        info: dict[str, Any] = {}

        # ── Collision (ignore first 10 steps to let physics settle) ──
        if self.step_count > 10 and np.any(
            self.aviary.contact_array[self.aviary.drones[agent_id].Id]
        ):
            reward -= 50.0
            term = True
            info["collision"] = True

        # ── Out-of-bounds ─────────────────────────────────────────────
        _, _, _, lin_pos, _ = self.compute_attitude_by_id(agent_id)
        if np.linalg.norm(lin_pos) > self.flight_dome_size:
            reward -= 50.0
            term = True
            info["out_of_bounds"] = True

        # ── Proximity shaping toward nearest unclaimed dot ────────────
        unclaimed_idx = np.where(self.dot_owner == -1)[0]
        if len(unclaimed_idx) > 0:
            dists = np.linalg.norm(self.dots[unclaimed_idx] - lin_pos, axis=-1)
            nearest = dists.min()
            reward += 0.05 / max(nearest, 0.01)

        # ── All dots claimed → end episode ────────────────────────────
        if np.all(self.dot_owner >= 0):
            trunc = True
            info["all_dots_claimed"] = True

        info["score"] = int(self.scores[agent_id])
        return term, trunc, reward, info

    # ──────────────────────────────────────────────────────────────────
    # Outer-step override – add dot-claim rewards once per outer step
    # ──────────────────────────────────────────────────────────────────

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        # Reset per-step claim counters before the inner physics loop
        self._step_claims[:] = 0

        obs, rews, terms, truncs, infos = super().step(actions)

        # Credit / debit dot-claim rewards once per outer step
        for ag, ag_id in self.agent_name_mapping.items():
            if ag not in rews:
                continue
            opp_id = 1 - ag_id
            claimed = self._step_claims[ag_id]
            stolen = self._step_claims[opp_id]
            if claimed > 0:
                rews[ag] += 10.0 * claimed
            if stolen > 0:
                rews[ag] -= 5.0 * stolen

        # Win / lose bonus when episode ends this step
        episode_done = all(terms.values()) or all(truncs.values())
        if episode_done and len(self.agents) == 0:
            # agents list has already been culled; iterate over infos instead
            for ag, ag_id in self.agent_name_mapping.items():
                if ag not in rews:
                    continue
                opp_id = 1 - ag_id
                if self.scores[ag_id] > self.scores[opp_id]:
                    rews[ag] += 50.0
                    infos[ag]["winner"] = True
                elif self.scores[ag_id] < self.scores[opp_id]:
                    rews[ag] -= 25.0
                    infos[ag]["loser"] = True

        return obs, rews, terms, truncs, infos
