import io
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from dataset.common import ChannelConditionDataset, TensorImageDataset
from dataset.dataset_imagenet import TFRecordImageDataset
from adjscc_sd_vae_ffhq import (
    FFHQDataset,
    FrozenFirstStageVAE,
    LatentADJSCC,
    build_parser,
    train_one_epoch,
)
from training import load_checkpoint, save_checkpoint
from util_channel import Channel
from util_module import DeepJSCC, GDN


def _varint(value):
    output = bytearray()
    while value > 0x7F:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _bytes_field(number, value):
    return _varint((number << 3) | 2) + _varint(len(value)) + value


def _image_example(encoded):
    bytes_list = _bytes_field(1, encoded)
    feature = _bytes_field(1, bytes_list)
    entry = _bytes_field(1, b"image/encoded") + _bytes_field(2, feature)
    features = _bytes_field(1, entry)
    return _bytes_field(1, features)


class ModelTests(unittest.TestCase):
    def test_gdn_forward_and_backward(self):
        inputs = torch.randn(2, 4, 8, 8, requires_grad=True)
        output = GDN(4)(inputs)
        self.assertEqual(output.shape, inputs.shape)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_all_channel_models_preserve_shape(self):
        image = torch.rand(2, 3, 17, 19)
        snr = torch.full((2, 1), 10.0)
        for channel_type in Channel.VALID_CHANNELS:
            model = DeepJSCC(
                transmit_channels=4,
                channel_type=channel_type,
                attention=True,
                feature_channels=8,
            )
            conditions = {}
            if channel_type in ("slow_fading", "slow_fading_eq"):
                conditions.update(
                    h_real=torch.ones(2, 1), h_imag=torch.ones(2, 1)
                )
            if channel_type == "burst":
                conditions.update(
                    b_prob=torch.full((2, 1), 0.25),
                    b_stddev=torch.ones(2, 1),
                )
            output = model(image, snr, **conditions)
            self.assertEqual(output.shape, image.shape)
            self.assertTrue(torch.isfinite(output).all())
            self.assertGreaterEqual(output.min().item(), 0.0)
            self.assertLessEqual(output.max().item(), 1.0)

    def test_baseline_backward(self):
        model = DeepJSCC(
            transmit_channels=4,
            channel_type="awgn",
            attention=False,
            feature_channels=8,
        )
        image = torch.rand(1, 3, 16, 16)
        output = model(image, torch.tensor([[5.0]]))
        torch.nn.functional.mse_loss(output, image).backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_checkpoint_round_trip(self):
        model = DeepJSCC(4, "awgn", attention=False, feature_channels=8)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.pt")
            save_checkpoint(path, model, epoch=3, best_loss=0.25)
            restored = DeepJSCC(4, "awgn", attention=False, feature_channels=8)
            payload = load_checkpoint(path, restored, torch.device("cpu"))
            self.assertEqual(payload["epoch"], 3)
            for expected, actual in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(expected, actual))

    def test_latent_adjscc_forward_and_backward(self):
        model = LatentADJSCC(transmit_channels=4, feature_channels=8)
        latent = torch.randn(2, 4, 17, 19)
        output = model(latent, torch.full((2, 1), 10.0))
        self.assertEqual(output.shape, latent.shape)
        torch.nn.functional.mse_loss(output, latent).backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_frozen_vae_decoder_preserves_input_gradient(self):
        class DummyFirstStage(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Conv2d(3, 4, 1)
                self.decoder = torch.nn.Conv2d(4, 3, 1)

            def encode(self, images):
                return self.encoder(images)

            def decode(self, latent):
                return self.decoder(latent)

        vae = FrozenFirstStageVAE(DummyFirstStage(), scale_factor=0.5)
        latent = torch.randn(1, 4, 8, 8, requires_grad=True)
        reconstruction = vae.decode_first_stage(latent)
        reconstruction.mean().backward()
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertEqual(
            sum(parameter.numel() for parameter in vae.parameters() if parameter.requires_grad),
            0,
        )

    def test_training_progress_logging(self):
        class DummyADJSCC(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, latent, snr_db):
                del snr_db
                return self.scale * latent

        class DummyVAE(torch.nn.Module):
            def encode(self, images):
                return torch.cat((images, images[:, :1]), dim=1)

        model = DummyADJSCC()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = [
            {"image": torch.randn(1, 3, 8, 8)} for _ in range(3)
        ]
        output = io.StringIO()
        with redirect_stdout(output):
            train_one_epoch(
                model,
                DummyVAE(),
                loader,
                optimizer,
                torch.device("cpu"),
                snr_low=0.0,
                snr_high=0.0,
                latent_weight=1.0,
                image_weight=0.0,
                epoch=7,
                log_every=2,
            )
        progress = output.getvalue()
        self.assertIn("Epoch 7 train [2/3", progress)
        self.assertIn("Epoch 7 train [3/3 (100.00%)]", progress)
        self.assertIn("ETA=", progress)
        self.assertEqual(build_parser().parse_args(["train"]).log_every, 100)


class DatasetTests(unittest.TestCase):
    def test_channel_condition_dataset(self):
        images = TensorImageDataset(torch.rand(3, 3, 8, 8))
        dataset = ChannelConditionDataset(
            images, snr_low=0, snr_high=20, fading=True
        )
        sample = dataset[0]
        self.assertEqual(sample["image"].shape, (3, 8, 8))
        self.assertEqual(sample["snr_db"].shape, (1,))
        self.assertIn("h_real", sample)
        self.assertIn("h_imag", sample)

    def test_legacy_tfrecord_reader(self):
        image = Image.new("RGB", (8, 8), color=(10, 20, 30))
        encoded_buffer = io.BytesIO()
        image.save(encoded_buffer, format="PNG")
        example = _image_example(encoded_buffer.getvalue())
        record = (
            struct.pack("<Q", len(example))
            + b"\x00" * 4
            + example
            + b"\x00" * 4
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.record"
            path.write_bytes(record)
            dataset = TFRecordImageDataset(str(path), patch_size=8, augment=False)
            tensor = dataset[0]
            self.assertEqual(tensor.shape, (3, 8, 8))
            self.assertAlmostEqual(tensor[0, 0, 0].item(), 10 / 255.0, places=6)

    def test_ffhq_dataset_normalizes_to_minus_one_one(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "face.png"
            Image.new("RGB", (24, 20), color=(0, 127, 255)).save(str(path))
            dataset = FFHQDataset([path], image_size=32, augment=False)
            sample = dataset[0]
            self.assertEqual(sample["image"].shape, (3, 32, 32))
            self.assertGreaterEqual(sample["image"].min().item(), -1.0)
            self.assertLessEqual(sample["image"].max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
