import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from adjscc_sd_img2img import (
    build_parser,
    load_adjscc,
    load_init_image,
    strength_schedule,
)
from adjscc_sd_vae_ffhq import LatentADJSCC
from training import save_checkpoint


class StableDiffusionADJSCCImg2ImgTests(unittest.TestCase):
    def test_strength_schedule(self):
        self.assertEqual(strength_schedule(0.0, 50), (0, None))
        self.assertEqual(strength_schedule(0.5, 50), (25, 24))
        self.assertEqual(strength_schedule(1.0, 50), (50, 49))
        with self.assertRaises(ValueError):
            strength_schedule(-0.1, 50)
        with self.assertRaises(ValueError):
            strength_schedule(1.1, 50)
        with self.assertRaises(ValueError):
            strength_schedule(0.5, 0)

    def test_input_image_is_rgb_square_and_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.png"
            Image.new("RGB", (80, 64), color=(0, 127, 255)).save(str(path))
            image = load_init_image(str(path), image_size=64)
            self.assertEqual(image.shape, (1, 3, 64, 64))
            self.assertGreaterEqual(image.min().item(), -1.0)
            self.assertLessEqual(image.max().item(), 1.0)

    def test_adjscc_checkpoint_metadata_controls_architecture(self):
        source = LatentADJSCC(transmit_channels=4, feature_channels=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adjscc.pt"
            save_checkpoint(
                str(path),
                source,
                epoch=2,
                best_loss=0.25,
                metadata={
                    "transmit_channel_num": 4,
                    "feature_channels": 8,
                    "checkpoint_kind": "intra_epoch",
                    "epoch_in_progress": 3,
                    "batch_in_epoch": 25,
                    "batches_in_epoch": 100,
                    "progress_percent": 25.0,
                },
            )
            restored, info = load_adjscc(
                str(path),
                torch.device("cpu"),
                fallback_transmit_channels=16,
                fallback_feature_channels=256,
            )
            self.assertEqual(restored.transmit_channels, 4)
            self.assertEqual(restored.feature_channels, 8)
            self.assertEqual(info["checkpoint_kind"], "intra_epoch")
            self.assertEqual(info["progress_percent"], 25.0)
            self.assertFalse(any(parameter.requires_grad for parameter in restored.parameters()))

    def test_parser_defaults(self):
        args = build_parser().parse_args(
            ["--init-img", "input.png", "--adjscc-checkpoint", "model.pt"]
        )
        self.assertEqual(args.image_size, 256)
        self.assertEqual(args.strength, 0.35)
        self.assertEqual(args.ddim_steps, 50)


if __name__ == "__main__":
    unittest.main()
