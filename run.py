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
parser.add_argument("--img-size", default=None, type=int, help="Input image size. If not set, it will be set based on the dataset's default size.")
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
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 5e-4)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-4, metavar='LR',
                    help='warmup learning rate (default: 1e-4)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')
parser.add_argument('--lr-k-decay', type=float, default=1.0, metavar='K',
                    help='K decay factor for cosine annealing')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='number of warmup + cooldown epochs before restarting cycle')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='M',
                    help='cycle multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-decay', type=float, default=1.0, metavar='M',
                    help='cycle decay factor (default: 1.0)')
parser.add_argument('--lr-cycle-steps', type=int, default=0, metavar='N',
                    help='number of steps to take in each cycle (default: 0)')
parser.add_argument('--schedule-free', action="store_true", help="Use schedule-free optimizer.")
parser.add_argument(
    "--off-benchmark",
    action="store_false",
    dest="trainer_benchmark",
    help="The value (True or False) to set torch.backends.cudnn.benchmark to. The value for torch.backends.cudnn.benchmark set in the current session will be used (False if not manually set). If deterministic is set to True, this will default to False. You can read more about the interaction of torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic. Setting this flag to True can increase the speed of your system if your input sizes don’t change. However, if they do, then it might make your system slower. The CUDNN auto-tuner will try to find the best algorithm for the hardware when a new input size is encountered. This might also increase the memory usage.",
)
parser.add_argument("--epochs", default=100, type=int)
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
parser.add_argument("--gradient-clip-val", default=0.0, type=float, help="Gradient clipping value. 0 means don't clip.")
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
parser.add_argument("--log-iterations-conv", action="store_true")
# Misc
parser.add_argument("--no-gpu", action="store_true")
parser.add_argument("--seed", default=9248, type=int)  # 92:48
parser.add_argument("--project-name", default="RiT", type=str)
parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
parser.add_argument("--no-shuffle", action="store_false", dest="shuffle")
parser.add_argument("--allow-download", action="store_true", dest="download_data")
parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
# Distillation
parser.add_argument("--distill", action="store_true")
parser.add_argument("--teacher-model", default=None, type=str)
parser.add_argument("--distill-type", default="hard", type=str, choices=["soft", "hard"])
parser.add_argument("--distill-token", action="store_true", dest="use_distill_token")
parser.add_argument("--distill-alpha", default=0.1, type=float)
parser.add_argument("--distill-temperature", default=3.0, type=float)
parser.add_argument("--finetune-teacher", action="store_true")
parser.add_argument("--finetune-teacher-epochs", default=10, type=int)
# model args
# ViT args
parser.add_argument("--depth", default=12, type=int)
parser.add_argument("--global-pool", default="token", type=str, choices=["", "avg", "token", "map"])
parser.add_argument("--no-qkv-bias", action="store_false", dest="qkv_bias")
parser.add_argument("--qk-norm", action="store_true")
parser.add_argument("--init-values", default=None, type=float)
parser.add_argument("--no-class-token", action="store_false", dest="class_token")
parser.add_argument("--no-embed-class", action="store_true")
parser.add_argument("--reg-tokens", default=0, type=int)
parser.add_argument("--pre-norm", action="store_true")
parser.add_argument("--fc-norm", default=None, type=bool)
parser.add_argument("--dynamic-img-size", action="store_true")
parser.add_argument("--dynamic-img-pad", action="store_true")
parser.add_argument("--drop-rate", default=0.0, type=float)
parser.add_argument("--pos-drop-rate", default=0.0, type=float)
parser.add_argument("--patch-drop-rate", default=0.0, type=float)
parser.add_argument("--proj-drop-rate", default=0.0, type=float)
parser.add_argument("--attn-drop-rate", default=0.0, type=float)
parser.add_argument("--drop-path-rate", default=0.0, type=float)
parser.add_argument("--weight-init", default="", type=str)
parser.add_argument("--fix-init", action="store_true")
parser.add_argument("--use-v", action="store_true", help="CatViT use v projection.")
parser.add_argument("--norm-layer", type=str, default=None)
# Transit:
parser.add_argument("--iterations", default=12, type=int)
parser.add_argument("--n-deq-layers", default=1, type=int)
parser.add_argument("--block-type", type=str, default="prenorm_add")
parser.add_argument("--z-init-type", type=str, default="zero", choices=["zero", "input", "rand", "pre"])
parser.add_argument("--norm-type", default="none", type=str)
parser.add_argument("--prefix-filter-out", default=None, type=str)
parser.add_argument("--filter-out", default=None, type=str)
parser.add_argument("--jac-reg", action="store_true")
parser.add_argument("--jac-loss-weight", default=0.1, type=float)
parser.add_argument("--log-sradius", action="store_true")
parser.add_argument("--stochastic-depth-sigma", default=0.0, type=float)
parser.add_argument("--stability-reg", action="store_true")
parser.add_argument("--stability-reg-weight", default=1.0, type=float)
parser.add_argument("--trajectory-loss-steps", default=0, type=int)
parser.add_argument("--incremental-trajectory-loss", action="store_true")
parser.add_argument("--update-rate", default=1.0, type=float)
parser.add_argument("--injection", default="none", type=str, choices=["none", "input", "linear", "norm", "linear_norm", "block"])
parser.add_argument("--convergence-loss-threshold", default=0.0, type=float)
parser.add_argument("--incremental-iterations", action="store_true")
parser.add_argument("--incremental-iterations-min", default=1, type=int)
parser.add_argument("--incremental-iterations-max", default=12, type=int)
parser.add_argument("--use-head-vit", action="store_true")
parser.add_argument("--phantom-grad", action="store_true")
parser.add_argument("--phantom-grad-steps", default=5, type=int)
parser.add_argument("--phantom-grad-update-rate", default=0.5, type=float)
parser.add_argument("--convergence-threshold", default=0.0, type=float)
parser.add_argument("--stable-skip", action="store_true")
parser.add_argument("--n-pre-layers", default=0, type=int)
parser.add_argument("--n-post-layers", default=0, type=int)
parser.add_argument("--weight-distill", action="store_true")
parser.add_argument("--weight-distill-type", default="avg", type=str)
parser.add_argument("--svd-k", default=None, type=int)
parser.add_argument("--teacher-model-path", default=None, type=str)
parser.add_argument("--expand-tokens", action="store_true")
parser.add_argument("--expand-token-keep-input", action="store_true")
parser.add_argument("--expand-weights", action="store_true")

# deq args
parser.add_argument("--f-solver", default="fixed_point_iter", type=str)
parser.add_argument("--b-solver", default="fixed_point_iter", type=str)
parser.add_argument("--no-stat", default=None, type=bool)
parser.add_argument("--f-max-iter", default=40, type=int)
parser.add_argument("--b-max-iter", default=40, type=int)
parser.add_argument("--f-tol", default=1e-3, type=float)
parser.add_argument("--b-tol", default=1e-6, type=float)
parser.add_argument("--f-stop-mode", default="abs", type=str)
parser.add_argument("--b-stop-mode", default="abs", type=str)
parser.add_argument("--eval-factor", default=1.0, type=float)
parser.add_argument("--eval-f-max-iter", default=0, type=int)
parser.add_argument("--ift", action="store_true")
parser.add_argument("--hook-ift", action="store_true")
parser.add_argument("--grad", default=1, type=int)
parser.add_argument("--tau", default=1.0, type=float)
parser.add_argument("--sup-gap", default=-1, type=int)
parser.add_argument("--sup-loc", default=None, type=int)
parser.add_argument("--n-states", default=1, type=int)
parser.add_argument("--indexing", default=None, type=int)
# MOE args
parser.add_argument("--num-experts", default=10, type=int)
parser.add_argument("--gating-top-n", default=2, type=int)
parser.add_argument("--threshold-train", default=0.2, type=float)
parser.add_argument("--threshold-eval", default=0.2, type=float)
parser.add_argument("--capacity-factor-train", default=1.25, type=float)
parser.add_argument("--capacity-factor-eval", default=2.0, type=float)
parser.add_argument("--balance-loss-coef", default=1e-2, type=float)
parser.add_argument("--router-z-loss-coef", default=1e-3, type=float)
# nViT args
parser.add_argument("--manual-norm-weights", action="store_true")
# WTViT
# parser.add_argument("--distil-mode", default="avg", type=str, choices=["avg"])
parser.add_argument("--weight-tie-cycle", default=0, type=int)


args = parser.parse_args()



if __name__ == "__main__":
    train(args)
