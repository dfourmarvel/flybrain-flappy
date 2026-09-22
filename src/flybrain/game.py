"""Flappy Bird game engine: deterministic, seeded, headless."""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class GameConfig:
    """Game physics and dimension constants."""

    height: float = 512.0  # world units (px); y grows downward, 0 = top
    bird_x: float = 80.0
    bird_radius: float = 12.0
    start_y: float = 256.0
    gravity: float = 0.5  # added to vy every frame
    flap_velocity: float = -8.0  # flap SETS vy to this (does not add)
    max_fall_speed: float = 10.0  # vy is capped at this
    pipe_width: float = 52.0
    gap_height: float = 140.0
    pipe_spacing: float = 200.0  # horizontal distance between consecutive pipes' left edges
    pipe_speed: float = 3.0  # px per frame, pipes move left
    first_pipe_x: float = 300.0  # left edge of first pipe at reset
    gap_margin: float = 40.0  # gap centre drawn uniformly from [gap_height/2 + margin, height - gap_height/2 - margin]
    max_frames: int = 1500
    frame_ms: float = 25.0  # simulated brain time per frame (used by later steps, not by physics)


@dataclass
class Obs:
    """Observation after a step."""

    bird_y: float
    bird_vy: float
    next_pipe_dx: float  # left edge of next pipe minus bird_x
    next_gap_y: float  # gap centre of next pipe
    score: int
    alive: bool
    frame: int


class Game:
    """Flappy Bird game simulation."""

    def __init__(self, config: GameConfig = GameConfig(), seed: int = 0):
        """Initialize the game.

        Args:
            config: Game configuration with physics constants.
            seed: RNG seed for deterministic gap placement.
        """
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.frame_log: list[Obs] = []

        # Game state
        self.bird_y: float = 0.0
        self.bird_vy: float = 0.0
        self.score: int = 0
        self.alive: bool = True
        self.frame: int = 0

        # Pipe state: list of (x, gap_top, gap_bottom) tuples
        self.pipes: list[tuple[float, float, float]] = []
        self._dropped: int = 0  # pipes removed off-screen so far (all were already scored)

        self.reset()

    def reset(self) -> Obs:
        """Reset the game to initial state. Same seed always gives the same pipe layout."""
        self.rng = np.random.default_rng(self.seed)
        self.bird_y = self.config.start_y
        self.bird_vy = 0.0
        self.score = 0
        self.alive = True
        self.frame = 0
        self.pipes = []
        self._dropped = 0
        self.frame_log = []

        # Spawn initial pipes
        for i in range(3):
            x = self.config.first_pipe_x + i * self.config.pipe_spacing
            gap_center = self._sample_gap_center()
            gap_top = gap_center - self.config.gap_height / 2
            gap_bottom = gap_center + self.config.gap_height / 2
            self.pipes.append((x, gap_top, gap_bottom))

        return self._make_obs()

    def step(self, flap: bool) -> tuple[Obs, int, bool]:
        """Simulate one frame.

        Args:
            flap: Whether the bird flaps this frame.

        Returns:
            (observation, points_gained_this_frame, done)
        """
        # (1) Flap
        if flap:
            self.bird_vy = self.config.flap_velocity

        # (2) Apply gravity, cap fall speed
        self.bird_vy = min(self.bird_vy + self.config.gravity, self.config.max_fall_speed)

        # (3) Update bird position
        self.bird_y += self.bird_vy

        # (4) Move pipes left
        self.pipes = [(x - self.config.pipe_speed, gap_top, gap_bottom)
                      for x, gap_top, gap_bottom in self.pipes]

        # (5) Drop pipes that are off-screen
        before = len(self.pipes)
        self.pipes = [(x, gap_top, gap_bottom) for x, gap_top, gap_bottom in self.pipes
                      if x + self.config.pipe_width >= 0]

        self._dropped += before - len(self.pipes)

        # (6) Spawn new pipes: keep at least 3 alive
        while len(self.pipes) < 3:
            if not self.pipes:
                x = self.config.first_pipe_x
            else:
                # Find the rightmost pipe's left edge
                last_x = max(p[0] for p in self.pipes)
                x = last_x + self.config.pipe_spacing
            gap_center = self._sample_gap_center()
            gap_top = gap_center - self.config.gap_height / 2
            gap_bottom = gap_center + self.config.gap_height / 2
            self.pipes.append((x, gap_top, gap_bottom))

        # (7) Scoring. Pipes move in lockstep, so they are passed strictly in list order:
        # the first unscored pipe sits at index (score - dropped).
        points_this_frame = 0
        i = self.score - self._dropped
        threshold = self.config.bird_x - self.config.bird_radius
        while i < len(self.pipes) and self.pipes[i][0] + self.config.pipe_width < threshold:
            points_this_frame += 1
            i += 1

        self.score += points_this_frame

        # (8) Collision detection
        if self._check_collision():
            self.alive = False

        # (9) Frame increment and termination
        self.frame += 1
        done = not self.alive or self.frame >= self.config.max_frames

        obs = self._make_obs()
        self.frame_log.append(obs)

        return obs, points_this_frame, done

    def _sample_gap_center(self) -> float:
        """Sample a gap centre from the uniform distribution."""
        min_center = self.config.gap_height / 2 + self.config.gap_margin
        max_center = self.config.height - self.config.gap_height / 2 - self.config.gap_margin
        return float(self.rng.uniform(min_center, max_center))

    def _make_obs(self) -> Obs:
        """Create an observation from the current state."""
        # Find next pipe: the one whose right edge >= bird_x - bird_radius
        next_pipe_idx = None
        for i, (x, gap_top, gap_bottom) in enumerate(self.pipes):
            if x + self.config.pipe_width >= self.config.bird_x - self.config.bird_radius:
                next_pipe_idx = i
                break

        if next_pipe_idx is not None:
            next_x, next_gap_top, next_gap_bottom = self.pipes[next_pipe_idx]
            next_pipe_dx = next_x - self.config.bird_x
            next_gap_y = (next_gap_top + next_gap_bottom) / 2
        else:
            # Unreachable with >= 3 pipes alive; fail loudly rather than feed inf to the brain.
            raise RuntimeError("no pipe ahead of the bird")

        return Obs(
            bird_y=self.bird_y,
            bird_vy=self.bird_vy,
            next_pipe_dx=next_pipe_dx,
            next_gap_y=next_gap_y,
            score=self.score,
            alive=self.alive,
            frame=self.frame,
        )

    def _check_collision(self) -> bool:
        """Check if the bird collides with obstacles."""
        # Ceiling collision
        if self.bird_y - self.config.bird_radius <= 0:
            return True

        # Floor collision
        if self.bird_y + self.config.bird_radius >= self.config.height:
            return True

        # Pipe collisions (circle-rectangle overlap)
        for x, gap_top, gap_bottom in self.pipes:
            # Top pipe rect: [x, x+w] × [0, gap_top]
            if self._circle_rect_overlap(x, 0, self.config.pipe_width, gap_top):
                return True

            # Bottom pipe rect: [x, x+w] × [gap_bottom, height]
            if self._circle_rect_overlap(x, gap_bottom, self.config.pipe_width,
                                         self.config.height - gap_bottom):
                return True

        return False

    def _circle_rect_overlap(self, rect_x: float, rect_y: float, rect_w: float,
                              rect_h: float) -> bool:
        """Check if circle (bird) overlaps with axis-aligned rectangle.

        Args:
            rect_x: Left edge of rectangle.
            rect_y: Top edge of rectangle.
            rect_w: Width of rectangle.
            rect_h: Height of rectangle.

        Returns:
            True if there is overlap.
        """
        # Closest point on rectangle to circle center
        closest_x = max(rect_x, min(self.config.bird_x, rect_x + rect_w))
        closest_y = max(rect_y, min(self.bird_y, rect_y + rect_h))

        # Distance from circle center to closest point
        dx = self.config.bird_x - closest_x
        dy = self.bird_y - closest_y
        distance = (dx * dx + dy * dy) ** 0.5

        return distance < self.config.bird_radius


def run_episode(policy: Callable[[Obs], bool],
                seed: int,
                config: GameConfig = GameConfig()) -> tuple[int, list[Obs]]:
    """Run one episode with a given policy.

    Args:
        policy: Callable that takes Obs and returns whether to flap.
        seed: RNG seed for the game.
        config: Game configuration.

    Returns:
        (final_score, frame_log)
    """
    game = Game(config, seed)
    obs = game.reset()

    while True:
        flap = policy(obs)
        obs, _, done = game.step(flap)
        if done:
            break

    return game.score, game.frame_log
