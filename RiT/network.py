import time
import pytorch_lightning as pl
import torch
import wandb
from tqdm import tqdm

from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.models import create_model

from RiT.models import *
from RiT.utils import get_criterion, get_layer_outputs
from RiT.augmentation import CutMix, MixUp

from RiT.models.repeat_transformer import RiTHalt, SimpleRiT2
import matplotlib.pyplot as plt
from pytorch_lightning.utilities.exceptions import MisconfigurationException



class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        if hasattr(hparams, "_sample_input_data"):
            self._sample_input_data = hparams._sample_input_data
            del hparams._sample_input_data
            self._sample_input_label = hparams._sample_input_label
            del hparams._sample_input_label
        self.hparams.update(vars(hparams))
        self.save_hyperparameters(
            ignore=[key for key in self.hparams.keys() if key[0] == "_"]
        )
        kwargs = {
            k: self.hparams[k]
            for k in [
                "in_chans",
                "num_classes",
                "global_pool",
                "qkv_bias",
                "qk_norm",
                "init_values",
                "class_token",
                "no_embed_class",
                "reg_tokens",
                "pre_norm",
                "fc_norm",
                "dynamic_img_size",
                "dynamic_img_pad",
                "drop_rate",
                "pos_drop_rate",
                "patch_drop_rate",
                "proj_drop_rate",
                "attn_drop_rate",
                "drop_path_rate",
                "weight_init",
                "fix_init",
            ]
        }
        if "transit" in self.hparams.model_name:
            kwargs.update(
                {
                    k: self.hparams.get(k, None)
                    for k in [
                        "iterations",
                        "n_deq_layers",
                        "depth",
                        "block_type",
                        "z_init_type",
                        "norm_type",
                        "prefix_filter_out",
                        "filter_out",
                        "jac_reg",
                        "jac_loss_weight",
                        "log_sradius",
                        "stochastic_depth_sigma",
                        "stability_reg",
                        "stability_reg_weight",
                        "update_rate",
                    ]
                }
            )
            kwargs["logger"] = self.log

        if "moe" in self.hparams.model_name:
            kwargs.update(
                {
                    k: self.hparams[k]
                    for k in [
                        "num_experts",
                        "gating_top_n",
                        "threshold_train",
                        "threshold_eval",
                        "capacity_factor_train",
                        "capacity_factor_eval",
                        "balance_loss_coef",
                        "router_z_loss_coef",
                    ]
                }
            )

        self.model = create_model(model_name=self.hparams.model_name, **kwargs)
        if hparams.distill:
            self.teacher = create_model(
                model_name=hparams.teacher_model,
                num_classes=hparams.num_classes,
                pretrained=True,
            )
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher.eval()

        self.criterion = get_criterion(hparams)
        if hparams.distill:
            self.distill_criterion = torch.nn.KLDivLoss()
        # CutMix and MixUp
        if hparams.cutmix:
            self.cutmix = CutMix(beta=hparams.cutmix_beta)
        if hparams.mixup:
            self.mixup = MixUp(alpha=hparams.mixup_alpha)

    def log_time(func):
        """
        A decorator to measure the time of a function and log it.
        """

        def wrapper(self, *args, **kwargs):
            start = time.time()
            result = func(self, *args, **kwargs)
            end = time.time()
            try:
                self.log(f"{func.__name__}_time", end - start)
            except MisconfigurationException:
                pass
            return result

        return wrapper

    @log_time
    def forward(self, x):
        return self.model(x)

    def calculate_loss(self, out, label, img, **kwargs):
        loss = self.criterion(out, label)
        if kwargs.get("aux_loss", 0) != 0:
            loss += kwargs["aux_loss"]
        if kwargs.get("jac_loss", 0) != 0:
            loss += kwargs["jac_loss"]
        if kwargs.get("stability_loss", 0) != 0:
            loss += kwargs["stability_loss"]

        # distillation loss
        if self.hparams.distill:
            with torch.no_grad():
                self.teacher.eval()
                teacher_out = self.teacher(img)
            alpha = self.hparams.distill_alpha
            if kwargs.get("distill_token") is not None:
                out = kwargs["distill_token"]
            if self.hparams.distill_type == "hard":
                loss = 0.5 * loss + 0.5 * self.criterion(out, teacher_out.argmax(-1))
            elif self.hparams.distill_type == "soft":
                tau = self.hparams.distill_temperature
                loss = (1 - alpha) * loss + alpha * tau**2 * self.distill_criterion(
                    torch.nn.functional.softmax(out / tau, dim=1),
                    torch.nn.functional.softmax(teacher_out / tau, dim=1),
                )
            else:
                raise ValueError(
                    f"Distillation type {self.hparams.distill_type} not implemented."
                )

        return loss

    def configure_optimizers(self):
        """
        Example:
            param_group_fn = lambda model: [
                {"params": group_1, "lr":group_1_lr},
                {"params": group_2},
            ]
        """
        param_group_fn = None
        self.optimizer = create_optimizer_v2(
            self, param_group_fn=param_group_fn, **optimizer_kwargs(cfg=self.hparams)
        )

        self.scheduler, self.hparams.num_epochs = create_scheduler(self.hparams, self.optimizer)
        # save new num_epochs
        self.save_hyperparameters({"num_epochs": self.hparams.num_epochs})

        if self.scheduler is None:
            return self.optimizer
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "monitor": "train_loss",
        }

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.trainer.current_epoch)

    def on_fit_start(self):
        self.print_model_summary()

    def on_train_start(self):
        # wandb watch model
        if isinstance(self.logger, pl.loggers.WandbLogger):
            log = {
                (True, False): "gradients",
                (True, True): "all",
                (False, True): "parameters",
                (False, False): None,
            }[(self.hparams.log_gradients, self.hparams.log_weights)]
            print(f"[INFO] WandB watch log: {log}")
            self.logger.watch(
                self.model,
                log=log,
            )

        # Number of parameters:
        self.log(
            "trainable_params",
            float(sum(p.numel() for p in self.model.parameters() if p.requires_grad)),
        )
        self.log("total_params", float(sum(p.numel() for p in self.model.parameters())))

        # Log tags
        if hasattr(self.logger.experiment, "add_tags"):
            if self.hparams.tags:
                tags = self.hparams.tags.split(",")
                self.logger.experiment.add_tags(
                    [tag.strip() for tag in tags if tag.strip()]
                )
        if self.hparams.distill:
            if self.hparams.finetune_teacher:
                print("[INFO] Finetuning teacher model.")
                self.teacher.train()

                # print("[INFO] All layers are frozen except the last one.")
                # # freeze all layers except the last one
                # for param in self.teacher.parameters():
                #     param.requires_grad = False
                # self.teacher.head.requires_grad = True

                optimizer = torch.optim.Adam(self.teacher.head.parameters(), lr=1e-3)
                criterion = torch.nn.CrossEntropyLoss()
                for epoch in range(self.hparams.finetune_teacher_epochs):
                    corrects = 0
                    for img, label in tqdm(self.trainer.train_dataloader):
                        img, label = img.to(self.device), label.to(self.device)
                        out = self.teacher(img)
                        loss = criterion(out, label)
                        corrects += torch.eq(out.argmax(-1), label).sum().item()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    print(
                        f"[INFO] Teacher finetune epoch: {epoch+1}/{self.hparams.finetune_teacher_epochs}"
                        + f" Accuracy: {corrects/len(self.trainer.train_dataloader.dataset)}"
                    )
                self.teacher.eval()

            else:
                print("[INFO] Checking the accuracy of the teacher model.")
                self.teacher.eval()
                corrects = 0
                with torch.no_grad():
                    for img, label in tqdm(self.trainer.val_dataloaders):
                        img, label = img.to(self.device), label.to(self.device)
                        out = self.teacher(img)
                        pred = out.argmax(-1)
                        corrects += torch.eq(pred, label).sum().item()
                print(
                    f"[INFO] Teacher accuracy: {corrects/len(self.trainer.val_dataloaders.dataset)}"
                )

    def _step(self, img, label):
        if self.hparams.trajectory_loss_steps > 0:
            assert self.hparams.n_deq_layers == 1
            if self.hparams.cutmix or self.hparams.mixup:
                img, label, rand_label, lambda_ = self.cutmix_mixup(img, label)
                assert (
                    not self.hparams.use_distill_token
                ), "Trajectory loss is not compatible with distillation token."
                out_steps = self.model._intermediate_layers(
                    img,
                    n=list(
                        range(
                            self.hparams.iterations - self.hparams.trajectory_loss_steps,  self.hparams.iterations
                        )
                    ),
                )[0]

                loss_kwargs = {}
                for key in ["aux_loss", "jac_loss", "stability_loss"]:
                    if hasattr(self.model, key):
                        val = getattr(self.model, key)
                        if val != 0:
                            loss_kwargs[key] = val
                            self.log(key, val)

                loss = 0
                for out_step in out_steps:
                    out = self.model.forward_head(self.model.norm(out_step))
                    loss += (
                        self.calculate_loss(
                            out,
                            label,
                            img,
                            **loss_kwargs,
                        )
                        * lambda_
                    ) + (
                        self.calculate_loss(
                            out,
                            rand_label,
                            img,
                            **loss_kwargs,
                        )
                        * (1.0 - lambda_)
                    )
                loss /= self.hparams.trajectory_loss_steps
            else:
                raise NotImplementedError(
                    "trajectory_loss_steps is not implemented for no cutmix or mixup."
                )
            
            return out, loss

        if self.hparams.cutmix or self.hparams.mixup:
            img, label, rand_label, lambda_ = self.cutmix_mixup(img, label)
            if self.hparams.use_distill_token:
                out, distill_token = self(img)
            else:
                out = self(img)
                distill_token = None

            loss_kwargs = {"distill_token": distill_token}
            for key in ["aux_loss", "jac_loss", "stability_loss"]:
                if hasattr(self.model, key):
                    val = getattr(self.model, key)
                    if val != 0:
                        loss_kwargs[key] = val
                        self.log(key, val)

            loss = (
                self.calculate_loss(
                    out,
                    label,
                    img,
                    **loss_kwargs,
                )
                * lambda_
            ) + (
                self.calculate_loss(
                    out,
                    rand_label,
                    img,
                    **loss_kwargs,
                )
                * (1.0 - lambda_)
            )
        else:
            if (
                self.hparams.use_distill_token
                or hasattr(self.model, "aux_loss")
                or hasattr(self.model, "jac_loss")
            ):
                raise NotImplementedError()
            out = self(img)
            loss = self.calculate_loss(out, label, img)

        return out, loss

    def on_train_epoch_start(self):
        if self.hparams.use_distill_token:
            self.model.set_distilled_training(True)

    def training_step(self, batch, batch_idx):
        # TODO: log input images of the first batch (only once)
        img, label = batch
        out, loss = self._step(img, label)

        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return {
            "loss": loss,
            "acc": acc,
        }

    def on_train_epoch_end(self):
        if self.hparams.use_distill_token:
            self.model.set_distilled_training(False)
        # log learning rate
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.log(f"lr_{i}", param_group["lr"], on_epoch=True)
        # check if there is any nan value in model parameters
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                raise ValueError(f"[ERROR] {name} has nan value. Training stopped.")
        # log output histogram of each layer
        if self.hparams.log_layer_outputs:
            self.log_layer_outputs()

        if self.current_epoch == self.hparams.num_epochs:
            self.trainer.should_stop = True

    def optimizer_step(self, *args, **kwargs):
        """
        Add weight normalization, etc here.
        """
        super().optimizer_step(*args, **kwargs)

        if hasattr(self.model, "norm_weights_"):
            self.model.norm_weights_()

    def on_train_batch_end(self, out, batch, batch_idx):
        if batch_idx == self.trainer.num_training_batches - 1:  # only on last batch
            pass

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        img, label = batch
        if hasattr(self.model, "pre_z"):
            self.model.pre_z = None
        out = self(img)
        loss = self.calculate_loss(out, label, img)
        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return {
            "loss": loss,
            "acc": acc,
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def log_free_mem(self):
        free_memory, total_memory = torch.cuda.mem_get_info()
        self.log("free_memory", free_memory)

    def on_train_end(self):
        if hasattr(self.logger.experiment, "end"):
            self.logger.experiment.end()

    def print_model_summary(self):
        print(self.model)
        summary = pl.utilities.model_summary.ModelSummary(
            self, max_depth=self.hparams.model_summary_depth
        )
        if hasattr(self.logger.experiment, "log_asset_data"):
            self.logger.experiment.log_asset_data(
                str(summary), file_name="model_summary.txt"
            )
        print(summary)
        #### use torchsummary instead:
        # from torchsummary import summary
        # summary(self.model, self._sample_input_data.shape[1:], device="cuda")

    def cutmix_mixup(self, img, label):
        if self.hparams.cutmix and self.hparams.mixup:
            action = (
                "mixup" if torch.rand(1).item() <= self.hparams.mixup_prob else "cutmix"
            )
        elif self.hparams.cutmix:
            action = "cutmix"
        elif self.hparams.mixup:
            action = "mixup"
        else:
            return (
                img,
                label,
                torch.zeros_like(label),
                1.0,
            )
        if action == "cutmix":
            img, label, rand_label, lambda_ = self.cutmix((img, label))
        elif action == "mixup":
            img, label, rand_label, lambda_ = self.mixup((img, label))

        return img, label, rand_label, lambda_

    def log_layer_outputs(self):
        # log the output of each layer
        layer_outputs = get_layer_outputs(self.model, self._sample_input_data)
        for name, output in layer_outputs.items():
            self.log_histogram(output, name + ".output", self.global_step)

    def log_histogram(self, tensor, name, step):
        """
        Log a histogram of a tensor.
        """
        # comet logger
        if isinstance(self.logger, pl.loggers.comet.CometLogger):
            try:
                self.logger.experiment.log_histogram_3d(
                    tensor.detach().cpu(),
                    name=name,
                    step=step,
                )
            except IndexError:
                # Values closer than 1e-20 to zerro will lead to index error
                positive_output = tensor[tensor > 0]
                pos_min = (
                    positive_output.min().item()
                    if positive_output.numel() > 0
                    else float("inf")
                )
                negative_output = tensor[tensor < 0]
                neg_min = (
                    abs(negative_output.max().item())
                    if negative_output.numel() > 0
                    else float("inf")
                )
                self.logger.experiment.log_histogram_3d(
                    tensor.detach().cpu(),
                    name=name,
                    step=step,
                    start=min(pos_min, neg_min),
                )
            except Exception as e:
                raise e
        # wandb logger
        elif isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log(
                {name: wandb.Histogram(tensor.detach().cpu().numpy())},
                step=step,
            )
        else:
            print(
                f"[INFO] histogram '{name}' cannot be logged. Logger not supported. Logger: {self.logger}"
            )
