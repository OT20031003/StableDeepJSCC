"""Train and evaluate baseline Deep JSCC on ImageNet/Kodak with PyTorch."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import dataset_imagenet
from dataset.common import ChannelConditionDataset, TensorImageDataset
from training import (
    evaluate,
    fit,
    load_checkpoint,
    make_loader,
    mse_to_psnr,
    parameter_count,
    resolve_device,
    save_image,
    save_json,
    seed_everything,
)
from util_module import DeepJSCC


PROJECT_DIR = Path(__file__).resolve().parent


def experiment_name(args, mixed: bool = False) -> str:
    snr = "mix" if mixed else str(args.snr_train)
    return "bdjscc_imagenet_{}_tcn{}_snrdb{}_bs{}_lr{}".format(
        args.channel_type,
        args.transmit_channel_num,
        snr,
        args.batch_size,
        args.learning_rate,
    )


def checkpoint_path(args, mixed: bool = False) -> Path:
    if args.load_model_path:
        return Path(args.load_model_path)
    return Path(args.model_dir) / (experiment_name(args, mixed) + ".pt")


def build_model(args) -> DeepJSCC:
    return DeepJSCC(
        transmit_channels=args.transmit_channel_num,
        channel_type=args.channel_type,
        attention=False,
        feature_channels=args.feature_channels,
    )


def training_dataset(args, mixed: bool):
    fading = args.channel_type in ("slow_fading", "slow_fading_eq")
    if mixed:
        return dataset_imagenet.get_dataset_snr_range(
            args.snr_low_mix,
            args.snr_up_mix,
            root=args.data_dir,
            patch_size=args.patch_size,
            fading=fading,
        )
    if fading:
        return dataset_imagenet.get_dataset_snr_and_h(
            args.snr_train, root=args.data_dir, patch_size=args.patch_size
        )
    return dataset_imagenet.get_dataset_snr(
        args.snr_train, root=args.data_dir, patch_size=args.patch_size
    )


def train(args, model, device, mixed: bool = False) -> None:
    dataset, _ = training_dataset(args, mixed)
    loader = make_loader(
        dataset, args.batch_size, True, args.num_workers, device
    )
    name = experiment_name(args, mixed)
    fit(
        model,
        loader,
        device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_path=str(Path(args.model_dir) / (name + ".pt")),
        history_path=str(Path(args.loss_dir) / (name + ".json")),
        load_path=args.load_model_path,
        metadata=vars(args),
        max_train_batches=args.max_train_batches,
    )


def _kodak_dataset(args, images, snr_db):
    return ChannelConditionDataset(
        TensorImageDataset(images),
        snr_db=snr_db,
        fading=args.channel_type in ("slow_fading", "slow_fading_eq"),
    )


def evaluate_mismatch(args, model, device) -> None:
    load_checkpoint(str(checkpoint_path(args, args.mixed_checkpoint)), model, device)
    kodak = dataset_imagenet.get_kodak(args.kodak_dir)
    results = {"snr": [], "mse": [], "psnr": []}
    destination = Path(args.eval_dir) / (
        experiment_name(args, args.mixed_checkpoint) + "_mismatch.json"
    )

    for snr_db in range(args.snr_low_eval, args.snr_up_eval + 1):
        dataset = _kodak_dataset(args, kodak, snr_db)
        loader = make_loader(
            dataset, args.eval_batch_size, False, args.num_workers, device
        )
        measurements = [
            evaluate(model, loader, device, args.max_eval_batches)
            for _ in range(args.eval_repeats)
        ]
        mse = float(np.mean(measurements))
        psnr = mse_to_psnr(mse)
        results["snr"].append(snr_db)
        results["mse"].append(mse)
        results["psnr"].append(psnr)
        save_json(str(destination), results)
        print("SNR={} dB, MSE={:.8f}, PSNR={:.4f} dB".format(snr_db, mse, psnr))


@torch.no_grad()
def evaluate_picture(args, model, device) -> None:
    load_checkpoint(str(checkpoint_path(args, args.mixed_checkpoint)), model, device)
    model.eval()
    image = dataset_imagenet.load_image(args.eval_image).unsqueeze(0).to(device)
    snr_db = torch.tensor([[args.snr_eval]], dtype=image.dtype, device=device)
    conditions = {}
    if args.channel_type in ("slow_fading", "slow_fading_eq"):
        scale = 2.0 ** -0.5
        conditions["h_real"] = torch.randn(1, 1, device=device) * scale
        conditions["h_imag"] = torch.randn(1, 1, device=device) * scale
    reconstruction = model(image, snr_db, **conditions)
    mse = F.mse_loss(reconstruction, image).item()
    psnr = mse_to_psnr(mse)
    if args.eval_output:
        output = Path(args.eval_output)
    else:
        output = Path(args.predict_dir) / (
            "{}_bdjscc_imagenet_snr{}dB.png".format(
                Path(args.eval_image).stem, args.snr_eval
            )
        )
    save_image(str(output), reconstruction[0])
    print("MSE={:.8f}, PSNR={:.4f} dB".format(mse, psnr))
    print("saved reconstruction to {}".format(output))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("train", "train_mix", "eval_mismatch", "eval_pic")
    )
    parser.add_argument(
        "-ct",
        "--channel-type",
        "--channel_type",
        choices=("awgn", "slow_fading", "slow_fading_eq"),
        default="awgn",
    )
    parser.add_argument(
        "-md", "--model-dir", "--model_dir", default=str(PROJECT_DIR / "model")
    )
    parser.add_argument("-lmp", "--load-model-path", "--load_model_path")
    parser.add_argument("-bs", "--batch-size", "--batch_size", default=16, type=int)
    parser.add_argument("--eval-batch-size", default=1, type=int)
    parser.add_argument("-e", "--epochs", default=2, type=int)
    parser.add_argument(
        "-lr", "--learning-rate", "--learning_rate", default=1e-4, type=float
    )
    parser.add_argument(
        "-tcn",
        "--transmit-channel-num",
        "--transmit_channel_num",
        default=16,
        type=int,
    )
    parser.add_argument("--feature-channels", default=256, type=int)
    parser.add_argument("--patch-size", default=128, type=int)
    parser.add_argument(
        "-snr_train",
        "--snr-train",
        "--snr_train",
        default=10,
        type=int,
    )
    parser.add_argument("--snr-low-mix", default=0, type=int)
    parser.add_argument("--snr-up-mix", default=20, type=int)
    parser.add_argument("--snr-low-eval", default=0, type=int)
    parser.add_argument("--snr-up-eval", default=20, type=int)
    parser.add_argument(
        "-snr_eval", "--snr-eval", "--snr_eval", default=10, type=float
    )
    parser.add_argument(
        "-ldd", "--loss-dir", "--loss_dir", default=str(PROJECT_DIR / "loss")
    )
    parser.add_argument(
        "-ed", "--eval-dir", "--eval_dir", default=str(PROJECT_DIR / "eval")
    )
    parser.add_argument("--data-dir", default=str(dataset_imagenet.DEFAULT_IMAGENET_ROOT))
    parser.add_argument("--kodak-dir", default=str(dataset_imagenet.DEFAULT_KODAK_ROOT))
    parser.add_argument(
        "--eval-image",
        default=str(dataset_imagenet.DEFAULT_KODAK_ROOT / "kodim03.png"),
    )
    parser.add_argument("--eval-output")
    parser.add_argument("--predict-dir", default=str(PROJECT_DIR / "predict_pic"))
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval-repeats", default=100, type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument(
        "--mixed-checkpoint",
        action="store_true",
        help="evaluate the checkpoint produced by train_mix",
    )
    return parser


def main(args) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    model = build_model(args).to(device)
    print("device: {}".format(device))
    print("trainable parameters: {:,}".format(parameter_count(model)))
    if args.command == "train":
        train(args, model, device, mixed=False)
    elif args.command == "train_mix":
        train(args, model, device, mixed=True)
    elif args.command == "eval_mismatch":
        evaluate_mismatch(args, model, device)
    else:
        evaluate_picture(args, model, device)


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    print("Current execution parameters:")
    for argument, value in sorted(vars(parsed_args).items()):
        print("{}: {}".format(argument, value))
    main(parsed_args)
