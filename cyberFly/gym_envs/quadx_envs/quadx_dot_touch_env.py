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
        truncate_on_completion: bool = True,
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
            truncate_on_completion (bool): if True, episode ends immediately when
                all dots are touched. If False, episode continues (allows return-to-base).
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
        self.truncate_on_completion = truncate_on_completion

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
        self._init_touch_tracking()

        # Render dots
        self._dot_visuals: list[int] = []
        if self.render_mode is not None:
            self._render_dots()

        super().end_reset()

        return self.state, self.info

    def _init_touch_tracking(self) -> None:
        """Initialize all dot-touch tracking state."""
        # Boolean array tracking which dots have been touched at least once
        # Shape: (num_dots,), initialized to False (no dots touched yet)
        self.touched = np.zeros(self.num_dots, dtype=bool)
        
        # Boolean array tracking which dots were touched during THIS outer step
        # Cleared at the start of each step via step() override
        # Prevents false double-touch triggers within sub-steps (env_step_ratio=4)
        self._touched_this_step = np.zeros(self.num_dots, dtype=bool)
        
        # Boolean array tracking whether drone was inside each dot's radius at END of last step
        # Used to detect re-entry: only penalize double-touch if drone LEFT and came BACK
        # Prevents penalty while drone lingers inside a radius it just entered
        self._was_inside_radius = np.zeros(self.num_dots, dtype=bool)
        
        # Initialize info dictionary fields for episode tracking
        # dots_touched: integer count of how many unique dots have been touched
        self.info["dots_touched"] = 0
        # double_touch: flag indicating if episode terminated due to double-touch
        self.info["double_touch"] = False
        # all_dots_touched: flag indicating successful completion (all dots visited)
        self.info["all_dots_touched"] = False

    # ------------------------------------------------------------------
    # Dot generation helpers
    # ------------------------------------------------------------------
    def _generate_dots(self) -> None:
        """Sample *num_dots* random positions inside the flight dome."""
        # Pre-allocate array to store (x, y, z) positions for all dots
        self.dots = np.zeros((self.num_dots, 3))
        
        # Generate each dot position using spherical coordinates for uniform distribution
        for i in range(self.num_dots):
            # Azimuthal angle (horizontal rotation around z-axis): uniform in [0, 2π]
            theta = self.np_random.uniform(0.0, 2.0 * np.pi)
            
            # Polar angle (angle from z-axis): uniform in [0, π]
            # Sampling uniformly in [0, π] for phi gives uniform distribution on sphere surface
            phi = self.np_random.uniform(0.0, np.pi)
            
            # Radial distance from origin: uniform in [1.0m, 90% of dome size]
            # Minimum 1.0m ensures dots aren't too close to spawn point (origin)
            dist = self.np_random.uniform(1.0, self.flight_dome_size * 0.9)
            
            # Convert spherical coordinates (dist, theta, phi) to Cartesian (x, y, z)
            # Standard spherical-to-Cartesian formulas:
            x = dist * np.sin(phi) * np.cos(theta)  # x = r sin(φ) cos(θ)
            y = dist * np.sin(phi) * np.sin(theta)  # y = r sin(φ) sin(θ)
            z = abs(dist * np.cos(phi))             # z = |r cos(φ)| (abs ensures positive)
            
            # Clamp z to minimum 0.3m to keep dots above ground level
            # Without this, dots could be placed at z=0 (on the ground)
            z = max(z, 0.3)
            
            # Store the computed position in the dots array
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
    # Step override
    # ------------------------------------------------------------------
    def step(self, action):
        """Step the environment, clearing the per-step touch flags."""
        # Clear the per-step touch tracker at the start of each outer step
        # This array tracks touches within the CURRENT step only (resets every step)
        # fill(False) is more efficient than [:] = False for boolean arrays
        self._touched_this_step.fill(False)
        
        # Call the parent class's step method to execute the action
        # This runs the physics simulation, updates state, and computes rewards
        return super().step(action)
    # ------------------------------------------------------------------
    # Reward / termination / truncation
    # ------------------------------------------------------------------
    def compute_term_trunc_reward(self) -> None:
        """Compute termination, truncation, and reward for the current step."""
        # Call parent class method to compute base rewards and check OOB/collision
        # This sets up self.reward with living penalty and checks safety violations
        super().compute_base_term_trunc_reward()

        # Create boolean array indicating which dots the drone is currently inside
        # True where distance < threshold, False otherwise
        # Shape: (num_dots,)
        is_inside = self._current_distances < self.dot_touch_distance
        
        # Process only the dots the drone is currently inside
        # np.where(is_inside)[0] returns array of indices where is_inside==True
        # This is more efficient than looping over all dots
        for i in np.where(is_inside)[0]:
            # Check if this is a valid first touch or a double-touch violation
            # _handle_dot_touch returns True if double-touch detected
            if self._handle_dot_touch(i):
                # Double-touch detected - episode must terminate immediately
                # Update radius tracking before returning to maintain consistency
                self._was_inside_radius = is_inside
                # Early return - skip proximity reward and completion check
                return
        
        # Update radius tracking for next step
        # Store current inside/outside status to detect re-entry on next step
        # This enables the lingering-without-penalty behavior
        self._was_inside_radius = is_inside
        
        # Add proximity shaping reward to guide the agent toward untouched dots
        self._add_proximity_reward()
        
        # Check if all dots have been touched and award completion bonus
        self._check_completion()
    
    def _handle_dot_touch(self, dot_idx: int) -> bool:
        """Handle touching a dot. Returns True if double-touch detected.
        
        Args:
            dot_idx: Index of the dot being touched.
            
        Returns:
            True if this was a double-touch (terminate episode).
        """
        # Check if this dot has already been touched in a previous step
        if self.touched[dot_idx]:
            # Dot was already touched - check if this is a re-entry
            # Only penalize if the drone LEFT the radius and came BACK
            # self._was_inside_radius[dot_idx] tells us if we were inside LAST step
            if not self._was_inside_radius[dot_idx]:
                # Drone was OUTSIDE last step, now INSIDE - this is a re-entry!
                
                # If all dots are already touched, don't penalize (allows return-to-base)
                # The task is complete, so passing through old dots on the way home is okay
                if not np.all(self.touched):
                    # Not all dots touched yet - this is a genuine double-touch violation
                    # Apply large negative reward for double-touch violation
                    self.reward = -100.0
                    
                    # Set info flag so logs can report double-touch as termination reason
                    self.info["double_touch"] = True
                    
                    # Terminate the episode (this is a failure condition)
                    self.termination = True
                    
                    # Return True to signal double-touch detected (caller will exit early)
                    return True
            # else: drone was inside last step and still inside - allow lingering
        else:
            # First touch of this dot - this is good behavior!
            # Mark this dot as touched in the persistent tracking array
            self.touched[dot_idx] = True
            
            # Mark as touched in the per-step array (prevents sub-step re-trigger)
            self._touched_this_step[dot_idx] = True
            
            # Award positive reward for touching a new dot
            self.reward += 50.0
            
            # Update the touched count in info dict (for logging/monitoring)
            # np.sum on boolean array counts True values
            self.info["dots_touched"] = int(np.sum(self.touched))
            
            # Change the visual appearance of the dot (green -> grey)
            # This provides visual feedback to human observers
            self._update_dot_colour(dot_idx)
        
        # Return False - no double-touch, episode continues normally
        return False
    
    def _add_proximity_reward(self) -> None:
        """Add shaping reward for being close to nearest untouched dot."""
        # Create boolean mask for untouched dots
        # ~self.touched inverts the boolean array (True -> False, False -> True)
        untouched_dots = ~self.touched
        
        # Check if there are any untouched dots remaining
        # np.any returns True if at least one element is True
        if np.any(untouched_dots):
            # Find the distance to the nearest untouched dot
            # self._current_distances[untouched_dots] selects only untouched dot distances
            # np.min finds the smallest distance among them
            nearest_dist = np.min(self._current_distances[untouched_dots])
            
            # Add inverse-distance reward: reward = 0.1 / distance
            # This creates a smooth gradient guiding the agent toward the nearest dot
            # max(..., 0.01) prevents division by zero or excessive rewards at very close range
            # Reward is high when close (dist=0.1 -> reward=1.0) and low when far (dist=5 -> reward=0.02)
            self.reward += 0.1 / max(nearest_dist, 0.01)
    
    def _check_completion(self) -> None:
        """Check if all dots touched and award completion bonus."""
        # Check if all dots have been touched
        # np.all returns True only if all elements in self.touched are True
        if np.all(self.touched):
            # Award large bonus reward for completing the task
            # This is in addition to the 6 * 50 = 300 points from individual dot touches
            self.reward += 200.0
            
            # Set completion flag in info dict for logging/monitoring
            self.info["all_dots_touched"] = True
            
            # Set env_complete flag (used by parent class for tracking success)
            self.info["env_complete"] = True
            
            # Only truncate if the parameter allows it
            # If truncate_on_completion is False, episode continues (enables return-to-base)
            if self.truncate_on_completion:
                # Truncate the episode (success condition, not a failure)
                # truncation (not termination) signals successful completion
                self.truncation = True
