"""Tests for the Flappy Bird game engine."""

import pytest

from flybrain.game import Game, GameConfig, Obs, run_episode


class TestGameDeterminism:
    """Test that the game is deterministic."""

    def test_same_seed_same_actions_identical_outcome(self):
        """Same seed + same action sequence → identical score and frame_log."""
        config = GameConfig()
        action_sequence = [False, True, False, False, True] * 100  # 500 frames

        # First run
        game1 = Game(config, seed=42)
        game1.reset()
        obs1_list = []
        for flap in action_sequence:
            obs, _, done = game1.step(flap)
            obs1_list.append(obs)
            if done:
                break

        # Second run with same seed
        game2 = Game(config, seed=42)
        game2.reset()
        obs2_list = []
        for flap in action_sequence:
            obs, _, done = game2.step(flap)
            obs2_list.append(obs)
            if done:
                break

        # Check identical
        assert len(obs1_list) == len(obs2_list)
        assert game1.score == game2.score
        for o1, o2 in zip(obs1_list, obs2_list):
            assert o1.bird_y == o2.bird_y
            assert o1.bird_vy == o2.bird_vy
            assert o1.next_pipe_dx == o2.next_pipe_dx
            assert o1.next_gap_y == o2.next_gap_y
            assert o1.score == o2.score
            assert o1.alive == o2.alive
            assert o1.frame == o2.frame


class TestGapCentreVariation:
    """Test that different seeds produce different gap centres."""

    def test_different_seeds_different_gaps(self):
        """Each seed gives its own layout; no two of 10 seeds share a first-3-pipe layout."""
        layouts = set()
        for seed in range(10):
            game = Game(GameConfig(), seed=seed)
            game.reset()
            layouts.add(tuple(round((t + b) / 2, 9) for _, t, b in game.pipes))
        assert len(layouts) == 10

    def test_reset_reproduces_layout(self):
        """Same seed gives the same layout however many times reset() is called."""
        game = Game(GameConfig(), seed=3)
        first = list(game.pipes)
        game.reset()
        game.reset()
        assert game.pipes == first
        assert Game(GameConfig(), seed=3).pipes == first


class TestGamePlay:
    """Test basic gameplay mechanics."""

    def test_do_nothing_dies_within_40_frames(self):
        """Do-nothing agent dies (falls to floor) within ~40 frames."""
        config = GameConfig()
        game = Game(config, seed=0)
        game.reset()

        for _ in range(40):
            obs, _, done = game.step(False)
            if done:
                break

        assert not obs.alive, "Bird should be dead after falling"
        assert game.frame < 40, f"Bird died at frame {game.frame}, should be < 40"

    def test_flap_every_frame_dies_at_ceiling(self):
        """Flap-every-frame agent dies at ceiling within ~40 frames."""
        config = GameConfig()
        game = Game(config, seed=0)
        game.reset()

        for _ in range(40):
            obs, _, done = game.step(True)
            if done:
                break

        assert not obs.alive, "Bird should be dead after hitting ceiling"
        assert game.frame < 40, f"Bird died at frame {game.frame}, should be < 40"


class TestGapMarginBounds:
    """Test that gap centres respect margin bounds."""

    def test_gap_centres_within_bounds_1000_pipes(self):
        """Gap centres of >= 1,000 pipes actually spawned by step() stay within margin bounds."""
        cfg = GameConfig()
        lo = cfg.gap_height / 2 + cfg.gap_margin
        hi = cfg.height - cfg.gap_height / 2 - cfg.gap_margin

        class Immortal(Game):
            def _check_collision(self):
                return False

        seen = {}
        for seed in range(50):
            game = Immortal(cfg, seed=seed)
            game.reset()
            for _ in range(cfg.max_frames):
                for x, top, bottom in game.pipes:
                    seen[(seed, round(x + game.frame * cfg.pipe_speed, 6))] = (top + bottom) / 2
                _, _, done = game.step(False)
                if done:
                    break
        assert len(seen) >= 1000
        assert all(lo <= c <= hi for c in seen.values())


class TestScoring:
    """Test the scoring system."""

    def test_oracle_policy_median_at_least_5(self):
        """Game is playable: a naive oracle has median score >= 5 over seeds 0-9.

        A single seed is not used because the naive oracle fails some layouts badly.
        """
        def oracle(obs: Obs) -> bool:
            return obs.bird_y > obs.next_gap_y + 10 and obs.bird_vy >= 0

        scores = sorted(run_episode(oracle, seed=s)[0] for s in range(10))
        assert (scores[4] + scores[5]) / 2 >= 5, f"Oracle scores {scores}"

    def test_score_counts_every_pipe_passed(self):
        """Regression: score must keep counting after the first pipe scrolls off-screen."""
        def oracle(obs: Obs) -> bool:
            return obs.bird_y > obs.next_gap_y + 10 and obs.bird_vy >= 0

        score, log = run_episode(oracle, seed=0)
        cfg = GameConfig()
        # Pipes passed = pipes whose right edge crossed the bird by the final frame.
        travelled = log[-1].frame * cfg.pipe_speed
        passed = sum(
            1 for k in range(100)
            if cfg.first_pipe_x + k * cfg.pipe_spacing + cfg.pipe_width - travelled
            < cfg.bird_x - cfg.bird_radius
        )
        assert score == passed


class TestMaxFrames:
    """Test that episode stops at max_frames."""

    def test_episode_stops_at_max_frames(self):
        """Episode stops at max_frames if never dead."""
        # Use a config with large gap to avoid collisions
        # With gap_height=300 and margin=40:
        # min_center = 150 + 40 = 190
        # max_center = 512 - 150 - 40 = 322
        # Valid range: [190, 322]
        config = GameConfig(
            gravity=0.0,
            max_frames=100,
            gap_height=300.0,  # Large gap
            start_y=256.0,  # Start in middle of valid range
            gap_margin=40.0,
        )

        def never_flap(obs: Obs) -> bool:
            return False

        score, frame_log = run_episode(never_flap, seed=0, config=config)

        # Should reach max_frames without dying
        assert frame_log[-1].frame == config.max_frames, \
            f"Expected frame {config.max_frames}, got {frame_log[-1].frame}"
        assert frame_log[-1].alive, "Bird should still be alive at max_frames"


class TestFrameLog:
    """Test the frame log recording."""

    def test_frame_log_recorded(self):
        """Frame log is recorded correctly."""
        config = GameConfig()
        game = Game(config, seed=0)
        obs = game.reset()

        actions = [False, True, False, False, True]
        for flap in actions:
            obs, _, done = game.step(flap)
            if done:
                break

        assert len(game.frame_log) > 0, "Frame log should have entries"
        assert all(isinstance(o, Obs) for o in game.frame_log)

        # Verify frame numbers increment
        for i, o in enumerate(game.frame_log):
            assert o.frame == i + 1, f"Frame {i} has unexpected frame number {o.frame}"


class TestCollisionDetection:
    """Test collision mechanics."""

    def test_ceiling_collision(self):
        """Bird collides with ceiling."""
        config = GameConfig()
        game = Game(config, seed=0)
        game.reset()

        # Flap repeatedly to hit ceiling
        for _ in range(50):
            obs, _, done = game.step(True)
            if done:
                break

        assert game.bird_y - config.bird_radius <= 0, "Bird should hit ceiling"
        assert not obs.alive

    def test_floor_collision(self):
        """Bird collides with floor."""
        config = GameConfig()
        game = Game(config, seed=0)
        game.reset()

        # Don't flap to hit floor
        for _ in range(50):
            obs, _, done = game.step(False)
            if done:
                break

        assert game.bird_y + config.bird_radius >= config.height, "Bird should hit floor"
        assert not obs.alive


class TestPipeManagement:
    """Test pipe spawning and removal."""

    def test_at_least_3_pipes_alive(self):
        """At least 3 pipes are always alive."""
        config = GameConfig()
        game = Game(config, seed=0)
        game.reset()

        for _ in range(200):
            obs, _, done = game.step(False)
            assert len(game.pipes) >= 3, f"Only {len(game.pipes)} pipes alive"
            if done:
                break
