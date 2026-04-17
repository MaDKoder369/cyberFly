"""QuadX Dot Touch Environment.

A game where dots are scattered in 3D space and the drone must touch each dot
exactly once. Touching a dot a second time incurs a penalty and terminates the
episode. The episode ends successfully when all dots have been touched once.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from cyberFly.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv


class QuadXDotTouchEnv(QuadXBaseEnv):
    """QuadX Dot Touch Environment.

    Actions are vp, vq, vr, T, ie: angular rates and thrust.
    The drone must fly through a set of randomly placed dots in 3D space.
    Each dot may only be touched **once**. Touching an already-visited dot
    gives a large penalty and terminates the episode. The episode is
    successfully completed when every dot has been touched exactly once.

    Args:
        num_dots (int): number of dots scattered in the environment.
        dot_touch_distance (float): distance threshold for a dot to be
            considered "touched".
        flight_mode (int): the flight mode of the UAV.
        flight_dome_size (float): size of the allowable flying area.
        max_duration_seconds (float): maximum simulation time of the
            environment.
        angle_representation (Literal["euler", "quaternion"]): can be
            "euler" or "quaternion".
        agent_hz (int): looprate of the agent to environment interaction.
        render_mode (None | Literal["human", "rgb_array"]): render_mode.
        render_resolution (tuple[int, int]): render_resolution.
    """

    def __init__(
        self,
        num_dots: int = 6,
        dot_touch_distance: float = 0.5,
        flight_mode: int = 0,
        flight_dome_size: float = 5.0,
        max_duration_seconds: float = 20.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 30,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
    ):
        """__init__.

        Args:
            num_dots (int): number of dots scattered in the environment.
            dot_touch_distance (float): distance threshold for a dot to be
                considered "touched".
            flight_mode (int): the flight mode of the UAV.
            flight_dome_size (float): size of the allowable flying area.
            max_duration_seconds (float): maximum simulation time of the
                environment.
            angle_representation (Literal["euler", "quaternion"]): can be
                "euler" or "quaternion".
            agent_hz (int): looprate of the agent to environment interaction.
            render_mode (None | Literal["human", "rgb_array"]): render_mode.
            render_resolution (tuple[int, int]): render_resolution.
        """
        super().__init__(
            start_pos=np.array([[0.0, 0.0, 1.0]]),
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        self.num_dots = num_dots
        self.dot_touch_distance = dot_touch_distance
        self.flight_dome_size = flight_dome_size

        # Observation space:
        #   "attitude"    – standard drone state vector
        #   "dot_deltas"  – (num_dots, 3) body-frame offsets to each dot
        #   "dot_touched" – (num_dots,) binary mask, 1.0 if already touched
        self.observation_space = spaces.Dict(
            {
                "attitude": self.combined_space,
                "dot_deltas": spaces.Box(
                    low=-2 * flight_dome_size,
                    high=2 * flight_dome_size,
                    shape=(num_dots, 3),
                    dtype=np.float64,
                ),
                "dot_touched": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(num_dots,),
                    dtype=np.float64,
                ),
            }
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Reset the environment and scatter new dots.

        Args:
            seed: seed to pass to the base environment.
            options: None.
        """
        if options is None:
            options = dict()
        super().begin_reset(seed, options)

        # Generate random dot positions inside the flight dome
        self._generate_dots()

        # Track which dots have been touched (boolean array)
        self.touched = np.zeros(self.num_dots, dtype=bool)

        # Info
        self.info["dots_touched"] = 0
        self.info["double_touch"] = False
        self.info["all_dots_touched"] = False

        # Render dots
        self._dot_visuals: list[int] = []
        if self.render_mode is not None:
            self._render_dots()

        super().end_reset()

        return self.state, self.info

    # ------------------------------------------------------------------
    # Dot generation helpers
    # ------------------------------------------------------------------
    def _generate_dots(self) -> None:
        """Sample *num_dots* random positions inside the flight dome."""
        self.dots = np.zeros((self.num_dots, 3))
        for i in range(self.num_dots):
            theta = self.np_random.uniform(0.0, 2.0 * np.pi)
            phi = self.np_random.uniform(0.0, np.pi)
            dist = self.np_random.uniform(1.0, self.flight_dome_size * 0.9)
            x = dist * np.sin(phi) * np.cos(theta)
            y = dist * np.sin(phi) * np.sin(theta)
            z = abs(dist * np.cos(phi))
            z = max(z, 0.3)  # keep dots above the ground
            self.dots[i] = np.array([x, y, z])

    def _render_dots(self) -> None:
        """Load visual URDF spheres for every dot."""
        import os

        targ_obj_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../models/target.urdf",
        )
        for i, pos in enumerate(self.dots):
            body_id = self.env.loadURDF(
                targ_obj_dir,
                basePosition=pos.tolist(),
                useFixedBase=True,
                globalScaling=self.dot_touch_distance / 2.0,
            )
            self._dot_visuals.append(body_id)
            # colour untouched dots green
            self.env.changeVisualShape(
                body_id,
                linkIndex=-1,
                rgbaColor=(0.0, 0.8, 0.2, 1.0),
            )

    def _update_dot_colour(self, idx: int) -> None:
        """Fade a touched dot to grey so the pilot can see it was visited."""
        if self.render_mode is not None and idx < len(self._dot_visuals):
            self.env.changeVisualShape(
                self._dot_visuals[idx],
                linkIndex=-1,
                rgbaColor=(0.5, 0.5, 0.5, 0.35),
            )

    # ------------------------------------------------------------------
    # State computation
    # ------------------------------------------------------------------
    def compute_state(self) -> None:
        """Compute observation state for the current timestep.

        State dict:
            "attitude"    – ang_vel, ang_pos, lin_vel, lin_pos, action, aux
            "dot_deltas"  – (num_dots, 3) body-frame vectors to each dot
            "dot_touched" – (num_dots,) 1.0 where dot was already touched
        """
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        # Build attitude vector
        new_state: dict[str, np.ndarray] = dict()
        if self.angle_representation == 0:
            new_state["attitude"] = np.concatenate(
                [ang_vel, ang_pos, lin_vel, lin_pos, self.action, aux_state], axis=-1
            )
        elif self.angle_representation == 1:
            new_state["attitude"] = np.concatenate(
                [ang_vel, quaternion, lin_vel, lin_pos, self.action, aux_state], axis=-1
            )

        # Body-frame deltas to every dot
        import pybullet as p

        rotation = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        dot_deltas = np.matmul((self.dots - lin_pos), rotation)
        new_state["dot_deltas"] = dot_deltas

        # Touched mask
        new_state["dot_touched"] = self.touched.astype(np.float64)

        # Cache distances for reward computation
        self._current_distances = np.linalg.norm(dot_deltas, axis=-1)

        self.state = new_state

    # ------------------------------------------------------------------
    # Reward / termination / truncation
    # ------------------------------------------------------------------
    def compute_term_trunc_reward(self) -> None:
        """Compute termination, truncation, and reward for the current step."""
        super().compute_base_term_trunc_reward()

        # Check each dot for proximity
        for i in range(self.num_dots):
            if self._current_distances[i] < self.dot_touch_distance:
                if self.touched[i]:
                    # Already touched – penalise and terminate
                    self.reward = -100.0
                    self.info["double_touch"] = True
                    self.termination |= True
                    return
                else:
                    # First touch – reward
                    self.touched[i] = True
                    self.reward += 50.0
                    self.info["dots_touched"] = int(np.sum(self.touched))
                    self._update_dot_colour(i)

        # Small shaping reward: encourage getting closer to the nearest
        # un-touched dot
        untouched_mask = ~self.touched
        if np.any(untouched_mask):
            nearest_dist = float(np.min(self._current_distances[untouched_mask]))
            self.reward += 0.1 / max(nearest_dist, 0.01)

        # All dots touched → success!
        if np.all(self.touched):
            self.reward += 200.0
            self.info["all_dots_touched"] = True
            self.info["env_complete"] = True
            self.truncation |= True
