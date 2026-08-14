"""Train and evaluate adaptive Deep JSCC on CIFAR-10 with PyTorch."""

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


def experiment_name(args) -> str:
    return (
        "adjscc_cifar10_{}_tcn{}_snrdb{}to{}_bs{}_lr{}".format(
            args.channel_type,
            args.transmit_channel_num,
            args.snr_low_train,
            args.snr_up_train,
            args.batch_size,
            args.learning_rate,
        )
    )


def model_path(args) -> Path:
    if args.load_model_path:
        return Path(args.load_model_path)
    return Path(args.model_dir) / (experiment_name(args) + ".pt")


def build_model(args, channel_type=None) -> DeepJSCC:
    return DeepJSCC(
        transmit_channels=args.transmit_channel_num,
        channel_type=channel_type or args.channel_type,
        attention=True,
        feature_channels=args.feature_channels,
        condition_on_fading=args.condition_on_fading,
    )


def training_datasets(args):
    if args.channel_type == "awgn":
        return dataset_cifar10.get_dataset_snr_range(
            args.snr_low_train,
            args.snr_up_train,
            root=args.data_dir,
            download=args.download,
        )
    return dataset_cifar10.get_dataset_snr_range_and_h(
        args.snr_low_train,
        args.snr_up_train,
        root=args.data_dir,
        download=args.download,
    )


def train(args, model, device) -> None:
    (train_dataset, _), (test_dataset, _) = training_datasets(args)
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, device
    )
    test_loader = make_loader(
        test_dataset, args.batch_size, False, args.num_workers, device
    )
    name = experiment_name(args)
    checkpoint = Path(args.model_dir) / (name + ".pt")
    history = Path(args.loss_dir) / (name + ".json")
    fit(
        model,
        train_loader,
        device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_path=str(checkpoint),
        history_path=str(history),
        validation_loader=test_loader,
        load_path=args.load_model_path,
        metadata=vars(args),
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )


def _test_dataset(args, images, snr_db: float):
    fading = args.channel_type in ("slow_fading", "slow_fading_eq")
    return ChannelConditionDataset(images, snr_db=snr_db, fading=fading)


def evaluate_snr(args, model, device) -> None:
    load_checkpoint(str(model_path(args)), model, device)
    images = dataset_cifar10.load_cifar10(
        root=args.data_dir, train=False, download=args.download
    )
    results = {"snr": [], "mse": [], "psnr": []}
    destination = Path(args.eval_dir) / (experiment_name(args) + ".json")

    for snr_db in range(args.snr_low_eval, args.snr_up_eval + 1):
        dataset = _test_dataset(args, images, snr_db)
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


def evaluate_burst(args, model, device) -> None:
    if args.channel_type != "awgn":
        raise ValueError("eval_burst expects an AWGN-trained checkpoint")
    load_checkpoint(str(model_path(args)), model, device)
    model.channel.channel_type = "burst"
    images = dataset_cifar10.load_cifar10(
        root=args.data_dir, train=False, download=args.download
    )
    results = {"prob": [], "mse": [], "psnr": []}
    destination = Path(args.eval_dir) / (
        "burst_snr{}dB_sigma{}_{}.json".format(
            args.burst_snr_eval, args.burst_stddev, experiment_name(args)
        )
    )

    probabilities = np.arange(
        args.burst_prob_min,
        args.burst_prob_max + args.burst_prob_step / 2.0,
        args.burst_prob_step,
    )
    for probability in probabilities:
        dataset = ChannelConditionDataset(
            images,
            snr_db=args.burst_snr_eval,
            burst_prob=float(probability),
            burst_stddev=args.burst_stddev,
        )
        loader = make_loader(
            dataset, args.batch_size, True, args.num_workers, device
        )
        measurements = [
            evaluate(model, loader, device, args.max_eval_batches)
            for _ in range(args.eval_repeats)
        ]
        mse = float(np.mean(measurements))
        psnr = mse_to_psnr(mse)
        results["prob"].append(float(probability))
        results["mse"].append(mse)
        results["psnr"].append(psnr)
        save_json(str(destination), results)
        print(
            "burst probability={:.3f}, MSE={:.8f}, PSNR={:.4f} dB".format(
                probability, mse, psnr
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "eval", "eval_burst"))
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
        "-snr_low_train", "--snr-low-train", "--snr_low_train", default=0, type=int
    )
    parser.add_argument(
        "-snr_up_train", "--snr-up-train", "--snr_up_train", default=20, type=int
    )
    parser.add_argument(
        "-snr_low_eval", "--snr-low-eval", "--snr_low_eval", default=0, type=int
    )
    parser.add_argument(
        "-snr_up_eval", "--snr-up-eval", "--snr_up_eval", default=20, type=int
    )
    parser.add_argument(
        "-ldd", "--loss-dir", "--loss_dir", default=str(PROJECT_DIR / "loss")
    )
    parser.add_argument(
        "-ed", "--eval-dir", "--eval_dir", default=str(PROJECT_DIR / "eval")
    )
    parser.add_argument(
        "--data-dir", default=str(dataset_cifar10.DEFAULT_ROOT)
    )
    parser.add_argument("--no-download", action="store_false", dest="download")
    parser.set_defaults(download=True)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval-repeats", default=10, type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument(
        "--snr-only-attention",
        action="store_false",
        dest="condition_on_fading",
        help="do not feed fading coefficients to AF modules",
    )
    parser.set_defaults(condition_on_fading=True)
    parser.add_argument(
        "-b_snr_eval",
        "--burst-snr-eval",
        "--burst_snr_eval",
        default=10,
        type=float,
    )
    parser.add_argument(
        "-b_stddev",
        "--burst-stddev",
        "--burst-standard-derivation",
        "--burst_standard_derivation",
        default=0.0,
        type=float,
    )
    parser.add_argument("--burst-prob-min", default=0.0, type=float)
    parser.add_argument("--burst-prob-max", default=0.2, type=float)
    parser.add_argument("--burst-prob-step", default=0.025, type=float)
    return parser


def main(args) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    model = build_model(args).to(device)
    print("device: {}".format(device))
    print("trainable parameters: {:,}".format(parameter_count(model)))
    if args.command == "train":
        train(args, model, device)
    elif args.command == "eval":
        evaluate_snr(args, model, device)
    else:
        evaluate_burst(args, model, device)


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    print("Current execution parameters:")
    for argument, value in sorted(vars(parsed_args).items()):
        print("{}: {}".format(argument, value))
    main(parsed_args)
