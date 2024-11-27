import os
from pprint import pprint

import torch
import pytorch_lightning as pl
import numpy as np

from RiT.network import Net
from RiT.utils import get_experiment_name
from RiT.datasets import get_dataloader

def train(args):
    # torch set default dtype
    if args.default_dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif args.default_dtype == "float32":
        torch.set_default_dtype(torch.float32)
        torch.set_float32_matmul_precision(args.matmul_precision)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.gpus = 0 if args.no_gpu else torch.cuda.device_count()
    args.num_workers = 4 * args.gpus if args.gpus else 8
    if not args.gpus:
        args.precision = 32
    if args.log_all:
        args.log_gradients = True
        args.log_weights = True
        args.log_layer_outputs = True


    train_dl, val_dl = get_dataloader(args)
    args._sample_input_data, args._sample_input_label = next(
        iter(train_dl)
    )
    args._sample_input_data = args._sample_input_data.to("cuda" if args.gpus else "cpu")
    args._sample_input_label = args._sample_input_label.to("cuda" if args.gpus else "cpu")



    print("Arguments:")
    pprint({k: v for k, v in vars(args).items() if not k.startswith("_")})
    experiment_name = get_experiment_name(args)
    args.experiment_name = experiment_name
    # Set up logger
    print(f"Experiment: {experiment_name}")
    if args.use_comet:
        if args.use_wandb:
            print("[WARNING] Both Comet.ml and WandB are enabled. Using Comet.ml.")
        print("[INFO] Log with Comet.ml!")
        logger = pl.loggers.CometLogger(
            api_key=args._comet_api_key,
            save_dir="logs",
            project_name=args.project_name,
            experiment_name=experiment_name,
        )
        del args._comet_api_key  # remove the API key from args
    elif args.use_wandb:
        print("[INFO] Log with WandB!")
        import wandb

        wandb.login(key=args._wandb_api_key)
        logger = pl.loggers.WandbLogger(
            log_model=True,
            save_dir="logs",
            project=args.project_name,
            name=experiment_name,
            tags=args.tags.split(",") if args.tags else None,
        )
        del args._wandb_api_key  # remove the API key from args
    else:
        print("[INFO] Log with CSV")
        logger = pl.loggers.CSVLogger(save_dir="logs", name=experiment_name)

    if args.jac_reg:
        os.environ["TIMM_FUSED_ATTN"] = "0"

    # Set up model
    assert args.model_name is not None, "Model name is required."
    net = Net(args)

    # Set up trainer
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename=experiment_name + "-{epoch:03d}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    if args.profiler is not None:
        if args.profiler == "pytorch":
            profiler = pl.profilers.PyTorchProfiler(
                output_filename=f"logs/profile/{experiment_name}.txt",
                use_cuda=True,
                # profile_memory=True,
                export_to_chrome=True,
                use_cpu=False,
            )
        elif args.profiler == "simple":
            profiler = pl.profilers.SimpleProfiler()
        elif args.profiler == "advanced":
            profiler = pl.profilers.AdvancedProfiler()
        else:
            raise ValueError(f"Invalid profiler: {args.profiler}")
        print(f"[INFO] Profiling with {args.profiler} profiler")
    else:
        profiler = None
    trainer = pl.Trainer(
        precision=args.precision,
        fast_dev_run=args.dry_run,
        accelerator="cpu" if args.no_gpu else "auto",
        devices=args.gpus if args.gpus else "auto",
        benchmark=args.trainer_benchmark,
        logger=logger,
        max_epochs=None,
        callbacks=[checkpoint_callback],
        enable_model_summary=False,  # Implemented seperately inside the Trainer
        profiler=profiler,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
    )

    # Train model
    trainer.fit(model=net, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Save model
    save_models_dir = "model_checkpoints"
    save_model_path = os.path.join(save_models_dir, experiment_name + ".ckpt")
    trainer.save_checkpoint(save_model_path)
    print(f"Model saved to {save_model_path}")

    # Save model to logger
    # Comet.ml
    if args.use_comet:
        urls = logger.experiment.log_model(
            experiment_name, save_model_path, overwrite=True
        )
        print(f"Model saved to comet: {urls}")
    # WandB
    elif args.use_wandb:
        wandb.save(save_model_path)
        print(f"Model saved to WandB: {save_model_path}")
    else:
        print(f"Model saved to {save_model_path}")
