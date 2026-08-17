import sys
import unittest
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from analyze_sd_latent_distribution import (
    ForwardDiffusionSchedule,
    StreamingLatentStatistics,
    render_covariance_tables,
    render_statistics_table,
)
from adjscc_sd_vae_ffhq import DEFAULT_SD_CONFIG


class StreamingLatentStatisticsTests(unittest.TestCase):
    def test_statistics_match_direct_population_calculation(self):
        torch.manual_seed(7)
        latent = torch.randn(3, 4, 2, 3)
        accumulator = StreamingLatentStatistics(4)
        accumulator.update(latent)
        statistics = accumulator.finalize()

        values = latent.to(torch.float64)
        observations = values.permute(0, 2, 3, 1).reshape(-1, 4)
        centered = observations - observations.mean(dim=0)
        expected_covariance = centered.transpose(0, 1).matmul(centered)
        expected_covariance /= observations.shape[0]

        self.assertEqual(statistics["sample_count"], 3)
        self.assertEqual(statistics["spatial_position_count"], 18)
        self.assertEqual(statistics["element_count"], latent.numel())
        self.assertAlmostEqual(
            statistics["global_mean"], values.mean().item(), places=12
        )
        self.assertAlmostEqual(
            statistics["global_std"],
            values.std(unbiased=False).item(),
            places=12,
        )
        self.assertAlmostEqual(statistics["min"], values.min().item(), places=12)
        self.assertAlmostEqual(statistics["max"], values.max().item(), places=12)
        self.assertAlmostEqual(
            statistics["rms"], values.square().mean().sqrt().item(), places=12
        )
        self.assertAlmostEqual(
            statistics["l2_norm"], values.square().sum().sqrt().item(), places=12
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(statistics["channel_mean"], dtype=torch.float64),
                observations.mean(dim=0),
                atol=1e-12,
                rtol=0.0,
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(statistics["channel_std"], dtype=torch.float64),
                observations.std(dim=0, unbiased=False),
                atol=1e-12,
                rtol=0.0,
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(
                    statistics["channel_covariance"], dtype=torch.float64
                ),
                expected_covariance,
                atol=1e-12,
                rtol=0.0,
            )
        )

    def test_streaming_batches_match_single_update(self):
        torch.manual_seed(11)
        latent = torch.randn(5, 4, 3, 2)
        complete = StreamingLatentStatistics(4)
        complete.update(latent)
        streamed = StreamingLatentStatistics(4)
        streamed.update(latent[:2])
        streamed.update(latent[2:])

        expected = complete.finalize()
        actual = streamed.finalize()
        for key in ("global_mean", "global_std", "min", "max", "rms", "l2_norm"):
            self.assertAlmostEqual(actual[key], expected[key], places=12)
        for key in ("channel_mean", "channel_std", "channel_covariance"):
            self.assertTrue(
                torch.allclose(
                    torch.tensor(actual[key], dtype=torch.float64),
                    torch.tensor(expected[key], dtype=torch.float64),
                    atol=1e-12,
                    rtol=0.0,
                )
            )

    def test_rendered_tables_contain_requested_statistics(self):
        accumulator = StreamingLatentStatistics(4)
        accumulator.update(torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2))
        statistics = accumulator.finalize()
        series = [("clean z", statistics), ("ADJSCC z_hat", statistics)]

        summary = render_statistics_table("comparison", series)
        covariance = render_covariance_tables(series)
        for label in (
            "global mean",
            "global std",
            "channel mean [4]",
            "channel std [4]",
            "min",
            "max",
            "RMS",
            "L2 norm",
        ):
            self.assertIn(label, summary)
        self.assertIn("channel covariance 4x4", covariance)
        self.assertIn("| ch3 |", covariance)


class ForwardDiffusionScheduleTests(unittest.TestCase):
    def test_q_sample_matches_schedule_coefficients(self):
        schedule = ForwardDiffusionSchedule(str(DEFAULT_SD_CONFIG))
        latent = torch.ones(2, 4, 2, 2)
        noise = torch.full_like(latent, 2.0)
        timestep = 123
        expected = (
            schedule.sqrt_alphas_cumprod[timestep].float() * latent
            + schedule.sqrt_one_minus_alphas_cumprod[timestep].float() * noise
        )
        actual = schedule.q_sample(latent, timestep, noise)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(schedule.timesteps, 1000)
        with self.assertRaises(ValueError):
            schedule.validate_timesteps([1000])


if __name__ == "__main__":
    unittest.main()
