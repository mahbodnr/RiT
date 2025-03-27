import random
import string
from datetime import datetime

import torch
import torch.nn as nn

from RiT.criterions import MarginLoss, WeightedCrossEntropyLoss, FixedpointWeightedCrossEntropyLoss
from RiT.lr_schedulers import GradualWarmupScheduler, StopScheduler


def get_layer_outputs(model, input):
    layer_outputs = {}

    def hook(module, input, output):
        layer_name = f"{module.__class__.__name__}_{module.parent_name}"
        layer_outputs[layer_name] = output.detach()

    # Add parent name attribute to each module
    for name, module in model.named_modules():
        module.parent_name = name
    # Register the hook to each layer in the model
    for module in model.modules():
        module.register_forward_hook(hook)
    # Pass the input through the model
    _ = model(input)
    # Remove the hooks and parent name attribute
    for module in model.modules():
        module._forward_hooks.clear()
        delattr(module, "parent_name")

    return layer_outputs


def get_criterion(args):
    if args.criterion == "ce":
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.criterion == "margin":
        criterion = MarginLoss(m_pos=0.9, m_neg=0.1, lambda_=0.5)
    elif args.criterion == "wce":
        criterion = WeightedCrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.criterion == "fwce":
        criterion = FixedpointWeightedCrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        raise ValueError(f"Criterion {args.criterion} not implemented.")

    return criterion

def get_scheduler(optimizer, args):
    if args.lr_scheduler == "reduce_on_plateau":
        # TODO: Add ReduceLROnPlateau parameters
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            # factor=args.lr_scheduler_factor,
            # patience=args.lr_scheduler_patience,
            verbose=True,
            # threshold=args.lr_scheduler_threshold,
            # threshold_mode="rel",
            # cooldown=args.lr_scheduler_cooldown,
            min_lr=args.min_lr,
        )
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=(
                args.epochs
                if args.get("lr_scheduler_T_max", None) is None
                else args.lr_scheduler_T_max
            ),
            eta_min=args.min_lr,
        )
    elif args.lr_scheduler == "cosine_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.lr_scheduler_T_0,
            T_mult=args.lr_scheduler_T_mult,
            eta_min=args.min_lr,
        )
    elif args.lr_scheduler is None or args.lr_scheduler.lower() == "none":
        return None
    else:
        raise NotImplementedError(
            f"Unknown lr_scheduler: {args.lr_scheduler}"
        )
    if args.lr_warmup_epochs > 0:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1.0,
            total_epoch=args.lr_warmup_epochs,
            after_scheduler=scheduler,
        )
    if args.lr_scheduler_stop_epoch is not None:
        scheduler = StopScheduler(
            optimizer,
            base_scheduler=scheduler,
            stop_epoch=args.lr_scheduler_stop_epoch,
        )

    return scheduler

def get_experiment_name(args):
    experiment_name = f"{args.model_name}_{args.dataset}"
    experiment_name += f"_{random_string(5)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return experiment_name


random_string = lambda n: "".join(
    [random.choice(string.ascii_lowercase) for i in range(n)]
)
