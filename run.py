import argparse
from RiT.training import train


parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str)
parser.add_argument("--pretrained", action="store_true")
# Logger
parser.add_argument("--comet", action="store_true", dest="use_comet")
parser.add_argument(
    "--comet-api-key", help="API Key for Comet.ml", dest="_comet_api_key", default=None
)
parser.add_argument("--wandb", action="store_true", dest="use_wandb")
parser.add_argument(
    "--wandb-api-key", help="API Key for WandB", dest="_wandb_api_key", default=None
)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument(
    "--dataset",
    default="cifar10",
    type=str,
    choices=["imagenet", "imagenet64", "tiny-imagenet", "cifar10", "cifar100", "svhn", "mnist", "fashionmnist"],
)
parser.add_argument("--image-size", default=None, type=int, help="Input image size. If not set, it will be set based on the dataset's default size.")
parser.add_argument("--data-root", default="~/data", type=str)
parser.add_argument("--profiler", default=None, type=str, choices=["simple", "advanced", "pytorch"], help="Profiler to use. If not set, no profiling will be done.")
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", default=256, type=int)
parser.add_argument("--accumulate-grad-batches", default=1, type=int, help="Accumulate gradients over N batches before doing a backward pass.")
# Optimizer
parser.add_argument(
    "--optimizer",
    default="adam",
    type=str,
    dest="opt"
)
parser.add_argument("--weight-decay", default=0.0, type=float)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
# Learning rate scheduler
parser.add_argument(
    "--lr-scheduler",
    default=None,
    type=str,
    choices=["cosine", "cosine_restart", "reduce_on_plateau"],
)
parser.add_argument(
    "--lr-warmup-epochs",
    default=0,
    type=int,
    help="Number of warmup epochs for the learning rate. Set to 0 to disable warmup.",
)
parser.add_argument("--min-lr", default=1e-5, type=float)
parser.add_argument("--lr-scheduler-stop-epoch", default=None, type=int)

parser.add_argument(
    "--off-benchmark",
    action="store_false",
    dest="trainer_benchmark",
    help="The value (True or False) to set torch.backends.cudnn.benchmark to. The value for torch.backends.cudnn.benchmark set in the current session will be used (False if not manually set). If deterministic is set to True, this will default to False. You can read more about the interaction of torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic. Setting this flag to True can increase the speed of your system if your input sizes don’t change. However, if they do, then it might make your system slower. The CUDNN auto-tuner will try to find the best algorithm for the hardware when a new input size is encountered. This might also increase the memory usage.",
)
parser.add_argument("--max-epochs", default=100, type=int)
parser.add_argument(
    "--precision",
    default="32-true",
    type=str,
    choices=[
        "16",
        "32",
        "64",
        "bf16",
        "16-true",
        "16-mixed",
        "bf16-true",
        "bf16-mixed",
        "32-true",
        "64-true",
    ],
)
# Augmentation
parser.add_argument("--autoaugment", action="store_true")
parser.add_argument("--randaugment", action="store_true")
parser.add_argument("--criterion", default="ce", type=str, choices=["ce", "margin", "wce", "fwce"])
parser.add_argument("--label-smoothing", default=0.0, type=float)
parser.add_argument("--rcpaste", action="store_true")
parser.add_argument("--cutmix", action="store_true")
parser.add_argument("--cutmix-beta", default=1.0, type=float)
parser.add_argument("--mixup", action="store_true")
parser.add_argument("--mixup-alpha", default=1.0, type=float)
parser.add_argument("--mixup-prob", default=0.5, type=float, help="Probability of applying mixup over cutmix. must be in [0, 1].")
parser.add_argument("--random-crop", action="store_true", default=False)
parser.add_argument("--random-crop-size", default=None, type=int)
parser.add_argument("--random-crop-padding", default=0, type=int)

parser.add_argument(
    "--default-dtype",
    default="float16",
    type=str,
    choices=["float16", "float32", "float64"],
    help="Default dtype for the model.",
)
parser.add_argument(
    "--matmul-precision",
    default="medium",
    type=str,
    choices=["medium", "high", "highest"],
)
# Logging
parser.add_argument("--log-layer-outputs", action="store_true")
parser.add_argument(
    "--log-gradients", action="store_true", help="Save gradients during training."
)
parser.add_argument(
    "--no-log-weights",
    action="store_false",
    dest="log_weights",
    help="Disable logging weights during training.",
)
parser.add_argument("--log-all", action="store_true", help="Log all available metrics.")
parser.add_argument("--model-summary-depth", default=-1, type=int)
parser.add_argument("--tags", default=None, type=str, help="Comma separated tags.", )
# Misc
parser.add_argument("--no-gpu", action="store_true")
parser.add_argument("--seed", default=9248, type=int)  # 92:48
parser.add_argument("--project-name", default="RiT", type=str)
parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
parser.add_argument("--no-shuffle", action="store_false", dest="shuffle")
parser.add_argument("--allow-download", action="store_true", dest="download_data")
parser.add_argument("--debug", action="store_true", help="Enable debug mode.")

# model parameters
parser.add_argument("--dropout", default=0.0, type=float)

# extra args
parser.add_argument("--iteration-loss", action="store_true")

args = parser.parse_args()


if __name__ == "__main__":
    train(args)
