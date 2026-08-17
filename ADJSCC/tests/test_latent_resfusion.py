import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from infer_latent_resfusion_handover import (
    build_parser as build_inference_parser,
    decode_from_raw_timestep,
    raw_timestep_to_t_start,
    validate_args as validate_inference_args,
)
from latent_resfusion import (
    FiveStepResfusionSchedule,
    LatentResfusionUNet,
    generate_resfusion_states,
    load_latent_resfusion_checkpoint,
    match_stable_diffusion_timesteps,
)
from ldm.modules.diffusionmodules.util import make_beta_schedule
from train_latent_resfusion import (
    build_parser as build_training_parser,
    run_epoch,
)
from training import save_checkpoint


class FiveStepScheduleTests(unittest.TestCase):
    def setUp(self):
        self.schedule = FiveStepResfusionSchedule()

    def test_twelve_step_linear_pro_truncates_to_five_steps(self):
        expected = torch.tensor(
            [
                0.9916666667,
                0.8339015152,
                0.5755183942,
                0.3104311338,
                0.25,
            ],
            dtype=torch.float32,
        )
        self.assertEqual(self.schedule.total_steps, 12)
        self.assertEqual(self.schedule.num_steps, 5)
        self.assertEqual(self.schedule.acceleration_index, 4)
        torch.testing.assert_close(
            self.schedule.alphas_cumprod,
            expected,
            atol=1e-7,
            rtol=1e-7,
        )
        self.assertAlmostEqual(
            self.schedule.posterior_betas[0].item(),
            self.schedule.betas[0].item(),
            places=8,
        )
        self.assertAlmostEqual(
            self.schedule.alphas_cumprod[-1].sqrt().item(), 0.5, places=7
        )

    def test_forward_state_and_resnoise_match_closed_forms(self):
        clean = torch.randn(2, 4, 4, 4)
        degraded = torch.randn_like(clean)
        noise = torch.randn_like(clean)
        timesteps = torch.tensor([0, 4])

        alpha_bar = self.schedule.alphas_cumprod[timesteps].reshape(
            2, 1, 1, 1
        )
        alpha = self.schedule.alphas[timesteps].reshape(2, 1, 1, 1)
        beta = self.schedule.betas[timesteps].reshape(2, 1, 1, 1)
        residual = degraded - clean
        expected_state = (
            alpha_bar.sqrt() * clean
            + (1.0 - alpha_bar.sqrt()) * residual
            + (1.0 - alpha_bar).sqrt() * noise
        )
        expected_target = noise + (
            (1.0 - alpha.sqrt())
            * (1.0 - alpha_bar).sqrt()
            / beta
            * residual
        )

        actual_state = self.schedule.forward_state(
            clean, degraded, timesteps, noise
        )
        actual_target = self.schedule.residual_noise_target(
            clean, degraded, timesteps, noise
        )
        torch.testing.assert_close(actual_state, expected_state)
        torch.testing.assert_close(actual_target, expected_target)

    def test_initial_state_is_half_degraded_plus_sqrt_three_quarters_noise(self):
        degraded = torch.randn(1, 4, 4, 4)
        noise = torch.randn_like(degraded)
        actual = self.schedule.initial_state(degraded, noise)
        expected = 0.5 * degraded + (0.75 ** 0.5) * noise
        torch.testing.assert_close(actual, expected)

    def test_mapping_matches_easy_tex(self):
        betas = make_beta_schedule(
            "linear",
            1000,
            linear_start=0.00085,
            linear_end=0.012,
        )
        sd_alpha_bar = torch.cumprod(
            1.0 - torch.tensor(betas, dtype=torch.float64), dim=0
        )
        self.assertEqual(
            match_stable_diffusion_timesteps(
                self.schedule, sd_alpha_bar
            ),
            [520, 475, 309, 146, 9, 0],
        )

    def test_state_generation_stops_at_requested_handover(self):
        class ZeroPrediction(torch.nn.Module):
            def forward(self, state, degraded_latent, timesteps):
                del degraded_latent, timesteps
                return torch.zeros_like(state)

        degraded = torch.randn(1, 4, 4, 4)
        initial_noise = torch.zeros_like(degraded)
        stage_zero = generate_resfusion_states(
            ZeroPrediction(),
            self.schedule,
            degraded,
            initial_noise=initial_noise,
            max_completed_steps=0,
        )
        all_stages = generate_resfusion_states(
            ZeroPrediction(),
            self.schedule,
            degraded,
            initial_noise=initial_noise,
            max_completed_steps=5,
        )
        self.assertEqual(len(stage_zero), 1)
        self.assertEqual(len(all_stages), 6)
        torch.testing.assert_close(stage_zero[0], all_stages[0])
        self.assertTrue(all(torch.isfinite(state).all() for state in all_stages))


class LatentResfusionModelTests(unittest.TestCase):
    def test_small_unet_has_eight_input_and_four_output_channels(self):
        model = LatentResfusionUNet(
            dim=8,
            dim_mults=(1, 2),
            resnet_block_groups=4,
        )
        first_conv = model.denoiser.init_conv
        self.assertEqual(first_conv.in_channels, 8)
        self.assertEqual(model.denoiser.final_conv.out_channels, 4)

        state = torch.randn(1, 4, 8, 8)
        degraded = torch.randn_like(state)
        output = model(state, degraded, torch.tensor([3]))
        self.assertEqual(output.shape, state.shape)
        output.square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in gradients)
        )

    def test_checkpoint_restores_architecture_metadata(self):
        source = LatentResfusionUNet(
            dim=8,
            dim_mults=(1, 2),
            resnet_block_groups=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resfusion.pt"
            save_checkpoint(
                str(path),
                source,
                epoch=7,
                best_loss=0.125,
                metadata={
                    "model_dim": 8,
                    "dim_mults": [1, 2],
                    "resnet_block_groups": 4,
                    "resfusion_total_steps": 12,
                    "resfusion_reverse_steps": 5,
                },
            )
            restored, _, info = load_latent_resfusion_checkpoint(
                str(path), torch.device("cpu")
            )
            self.assertEqual(restored.dim, 8)
            self.assertEqual(restored.dim_mults, (1, 2))
            self.assertEqual(info["epoch"], 7)
            self.assertEqual(info["best_loss"], 0.125)
            self.assertFalse(
                any(parameter.requires_grad for parameter in restored.parameters())
            )
            for expected, actual in zip(
                source.parameters(), restored.parameters()
            ):
                self.assertTrue(torch.equal(expected, actual))


class DirectHandoverCLITests(unittest.TestCase):
    def test_raw_timestep_is_inclusive(self):
        self.assertEqual(raw_timestep_to_t_start(0), 1)
        self.assertEqual(raw_timestep_to_t_start(520), 521)
        with self.assertRaises(ValueError):
            raw_timestep_to_t_start(-1)
        with self.assertRaises(ValueError):
            raw_timestep_to_t_start(1000)

    def test_decode_uses_exact_raw_schedule_and_includes_selected_timestep(self):
        class RecordingSampler:
            ddpm_num_timesteps = 1000

            def decode(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                return args[0]

        sampler = RecordingSampler()
        latent = torch.randn(1, 4, 8, 8)
        conditioning = torch.randn(1, 2, 3)
        unconditional = torch.randn_like(conditioning)
        output = decode_from_raw_timestep(
            sampler,
            latent,
            conditioning,
            raw_timestep=146,
            guidance_scale=7.5,
            unconditional_conditioning=unconditional,
        )
        self.assertIs(output, latent)
        self.assertIs(sampler.args[0], latent)
        self.assertIs(sampler.args[1], conditioning)
        self.assertEqual(sampler.args[2], 147)
        self.assertEqual(
            sampler.kwargs["unconditional_guidance_scale"], 7.5
        )
        self.assertIs(
            sampler.kwargs["unconditional_conditioning"], unconditional
        )
        self.assertTrue(sampler.kwargs["use_original_steps"])

    def test_training_and_inference_are_separate_parsers(self):
        training = build_training_parser().parse_args(
            ["--adjscc-checkpoint", "adjscc.pt"]
        )
        self.assertEqual(training.epochs, 100)
        self.assertEqual(training.dim_mults, "1,2,4,8")

        inference = build_inference_parser().parse_args(
            [
                "--init-img",
                "input.png",
                "--adjscc-checkpoint",
                "adjscc.pt",
                "--resfusion-checkpoint",
                "resfusion.pt",
                "--handover-step",
                "3",
            ]
        )
        validate_inference_args(inference)
        self.assertEqual(inference.handover_step, 3)
        self.assertEqual(inference.guidance_scale, 1.0)
        self.assertEqual(inference.dim_mults, (1, 2, 4, 8))


class TrainingLoopTests(unittest.TestCase):
    def test_one_synthetic_epoch_updates_only_resfusion_model(self):
        class DummyVAE(torch.nn.Module):
            def encode(self, images):
                return torch.cat((images, images[:, :1]), dim=1)

        class DummyADJSCC(torch.nn.Module):
            def forward(self, latent, snr_db):
                scale = 0.8 + 0.0 * snr_db.reshape(-1, 1, 1, 1)
                return latent * scale

        model = LatentResfusionUNet(
            dim=8,
            dim_mults=(1, 2),
            resnet_block_groups=4,
        )
        schedule = FiveStepResfusionSchedule()
        vae = DummyVAE()
        adjscc = DummyADJSCC()
        loader = [
            {"image": torch.randn(1, 3, 8, 8)},
            {"image": torch.randn(1, 3, 8, 8)},
        ]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        before = model.denoiser.final_conv.weight.detach().clone()
        loss = run_epoch(
            model,
            schedule,
            vae,
            adjscc,
            loader,
            torch.device("cpu"),
            snr_low=-10.0,
            snr_high=20.0,
            precision="full",
            optimizer=optimizer,
            accumulation_steps=2,
            max_batches=2,
            log_every=0,
        )
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertGreaterEqual(loss, 0.0)
        self.assertFalse(
            torch.equal(before, model.denoiser.final_conv.weight.detach())
        )


if __name__ == "__main__":
    unittest.main()
