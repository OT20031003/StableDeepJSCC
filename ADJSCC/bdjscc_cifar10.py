"""Train and evaluate baseline Deep JSCC on CIFAR-10 with PyTorch."""

import argparse
from pathlib import Path

import numpy as np

from dataset import dataset_cifar10
from dataset.common import ChannelConditionDataset
from training import (
    evaluate,
    fit,
    load_checkpoint,
    make_loader,
    mse_to_psnr,
    parameter_count,
    resolve_device,
    save_json,
    seed_everything,
)
from util_module import DeepJSCC


PROJECT_DIR = Path(__file__).resolve().parent


def experiment_name(args, mixed: bool = False) -> str:
    snr = "mix" if mixed else str(args.snr_train)
    return "bdjscc_cifar10_{}_tcn{}_snrdb{}_bs{}_lr{}".format(
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


def training_datasets(args, mixed: bool):
    fading = args.channel_type in ("slow_fading", "slow_fading_eq")
    if mixed:
        getter = (
            dataset_cifar10.get_dataset_snr_range_and_h
            if fading
            else dataset_cifar10.get_dataset_snr_range
        )
        return getter(
            args.snr_low_mix,
            args.snr_up_mix,
            root=args.data_dir,
            download=args.download,
        )
    getter = (
        dataset_cifar10.get_dataset_snr_and_h
        if fading
        else dataset_cifar10.get_dataset_snr
    )
    return getter(args.snr_train, root=args.data_dir, download=args.download)


def train(args, model, device, mixed: bool = False) -> None:
    (train_dataset, _), (test_dataset, _) = training_datasets(args, mixed)
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, device
    )
    test_loader = make_loader(
        test_dataset, args.batch_size, False, args.num_workers, device
    )
    name = experiment_name(args, mixed)
    fit(
        model,
        train_loader,
        device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_path=str(Path(args.model_dir) / (name + ".pt")),
        history_path=str(Path(args.loss_dir) / (name + ".json")),
        validation_loader=test_loader,
        load_path=args.load_model_path,
        metadata=vars(args),
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )


def evaluate_mismatch(args, model, device) -> None:
    load_checkpoint(str(checkpoint_path(args, args.mixed_checkpoint)), model, device)
    images = dataset_cifar10.load_cifar10(
        root=args.data_dir, train=False, download=args.download
    )
    fading = args.channel_type in ("slow_fading", "slow_fading_eq")
    results = {"snr": [], "mse": [], "psnr": []}
    destination = Path(args.eval_dir) / (
        experiment_name(args, args.mixed_checkpoint) + "_mismatch.json"
    )

    for snr_db in range(args.snr_low_eval, args.snr_up_eval + 1):
        dataset = ChannelConditionDataset(images, snr_db=snr_db, fading=fading)
        loader = make_loader(
            dataset, args.batch_size, True, args.num_workers, device
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "train_mix", "eval_mismatch"))
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
    parser.add_argument("-bs", "--batch-size", "--batch_size", default=128, type=int)
    parser.add_argument("-e", "--epochs", default=1280, type=int)
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
    parser.add_argument(
        "-snr_train",
        "--snr-train",
        "--snr_train",
        "-snr_eval",
        "--snr_eval",
        default=10,
        type=int,
    )
    parser.add_argument("--snr-low-mix", default=0, type=int)
    parser.add_argument("--snr-up-mix", default=20, type=int)
    parser.add_argument("--snr-low-eval", default=0, type=int)
    parser.add_argument("--snr-up-eval", default=20, type=int)
    parser.add_argument(
        "-ldd", "--loss-dir", "--loss_dir", default=str(PROJECT_DIR / "loss")
    )
    parser.add_argument(
        "-ed", "--eval-dir", "--eval_dir", default=str(PROJECT_DIR / "eval")
    )
    parser.add_argument("--data-dir", default=str(dataset_cifar10.DEFAULT_ROOT))
    parser.add_argument("--no-download", action="store_false", dest="download")
    parser.set_defaults(download=True)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval-repeats", default=10, type=int)
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
    else:
        evaluate_mismatch(args, model, device)


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    print("Current execution parameters:")
    for argument, value in sorted(vars(parsed_args).items()):
        print("{}: {}".format(argument, value))
    main(parsed_args)
