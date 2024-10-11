import time
import pytorch_lightning as pl
import torch
import wandb
from tqdm import tqdm

from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.models import create_model

from RiT.models import *
from RiT.utils import get_criterion, get_scheduler, get_layer_outputs
from RiT.augmentation import CutMix, MixUp

from RiT.models.repeat_transformer import RiTHalt, SimpleRiT2
import matplotlib.pyplot as plt


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
        if self.hparams.model_name.startswith("transit"):
            # add Transit kwargs
            kwargs.update(
                {
                    k: self.hparams[k]
                    for k in [
                        "depth",
                        "block_type",
                        "z_init_type",
                        "f_solver",
                        "b_solver",
                        "no_stat",
                        "f_max_iter",
                        "b_max_iter",
                        "f_tol",
                        "b_tol",
                        "f_stop_mode",
                        "b_stop_mode",
                        "eval_factor",
                        "eval_f_max_iter",
                        "ift",
                        "hook_ift",
                        "grad",
                        "tau",
                        "sup_gap",
                        "sup_loc",
                        "n_states",
                        "indexing",
                        "norm_type",
                        "prefix_filter_out",
                        "filter_out",
                    ]
                }
            )
            kwargs["logger"] = self.log

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
            self.log(f"{func.__name__}_time", end - start)
            return result

        return wrapper

    @log_time
    def forward(self, x):
        return self.model(x)

    def calculate_loss(self, out, label, img, **kwargs):
        if self.hparams.criterion in ["fwce", "wce"]:
            loss = self.criterion(out, label, kwargs["confidences"])
        loss = self.criterion(out, label)

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

        self.scheduler = get_scheduler(self.optimizer, self.hparams)

        if self.scheduler is None:
            return self.optimizer
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "monitor": "train_loss",
        }

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
        if self.hparams.cutmix or self.hparams.mixup:
            img, label, rand_label, lambda_ = self.cutmix_mixup(img, label)
            if self.hparams.use_distill_token:
                out, distill_token = self(img)
            else:
                out = self(img)
                distill_token = None
            if type(out) == dict:
                if "confidences" in out:
                    confidences = out["confidences"]
                    out = out["logits"]
                else:
                    if self.hparams.criterion in ["fwce", "wce"]:
                        N = 5
                        confidences = None
                        # out = out["logits"][-N:]
                        out = out["logits"]
                    else:
                        confidences = None
                        out = out["logits"][-1]
            else:
                confidences = None
            loss = (
                self.calculate_loss(
                    out,
                    label,
                    img,
                    distill_token=distill_token,
                    confidences=confidences,
                )
                * lambda_
            ) + (
                self.calculate_loss(
                    out,
                    rand_label,
                    img,
                    distill_token=distill_token,
                    confidences=confidences,
                )
                * (1.0 - lambda_)
            )
        else:
            if type(out) == dict:
                if "confidences" in out:
                    confidences = out["confidences"]
                    out = out["logits"]
                else:
                    if self.hparams.criterion in ["fwce", "wce"]:
                        N = 5
                        confidences = None
                        # out = out["logits"][-N:]
                        out = out["logits"]
                    else:
                        confidences = None
                        out = out["logits"][-1]
            else:
                confidences = None
            loss = self.calculate_loss(
                out, label, img, distill_token=distill_token, confidences=confidences
            )

            if out.dim() == 3:
                if confidences is not None:
                    out = out[confidences.argmax(0)]
                else:
                    out = out[-1]

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

        # DELETE LATER
        if hasattr(self.model, "halt_noise_scale"):
            try:
                if self.model.halt_noise_scale != 0:
                    self.model.halt_noise_scale = (5 - 1) * self.current_epoch / (
                        self.trainer.max_epochs
                    ) + 1
                self.log("halt_noise_scale", self.model.halt_noise_scale)

                with torch.no_grad():
                    # res = self.model.inference(self._sample_input_data,
                    #                         repeats= self.model.repeats * 2,
                    #                         halt="classify",
                    #                         halt_threshold=self.model.halt_threshold,
                    #                         ema_alpha=0.5,
                    #                         halt_noise_scale=self.model.halt_noise_scale,
                    #                         )
                    # block_halt = torch.stack(res['block_halt'])[...,0]
                    # plt.figure(figsize=(10, 6))
                    # plt.plot(block_halt.cpu().numpy())
                    # plt.title(f"Block Halt, epoch: {self.current_epoch}")
                    # plt.savefig("block_halt.png")
                    # plt.clf()
                    # # log to wandb
                    # if isinstance(self.logger, pl.loggers.WandbLogger):
                    #     self.logger.experiment.log(
                    #         {"Block Halt": wandb.Image("block_halt.png")},
                    #         step=self.global_step,
                    #     )
                    repeats = self.model.repeats
                    self.model.repeats = repeats * 2
                    res = self(self._sample_input_data)
                    outputs = res["logits"]
                    block_outputs = res["block_outputs"]
                    # convergence
                    mse = []
                    for i in range(outputs.shape[0] - 1):
                        mse.append(F.mse_loss(outputs[i], outputs[i + 1]).cpu())

                    plt.figure(figsize=(8, 10))
                    plt.subplot(2, 1, 1)
                    plt.plot(torch.arange(1, len(mse) + 1), mse)
                    plt.title(r"Classifier outputs convergence")
                    plt.xlabel("Iterations")
                    plt.ylabel(r"$(x_{i+1} - x_{i})^2$")
                    mse = []
                    for i in range(block_outputs.shape[0] - 1):
                        mse.append(
                            F.mse_loss(block_outputs[i], block_outputs[i + 1]).cpu()
                        )

                    plt.subplot(2, 1, 2)
                    plt.plot(torch.arange(1, len(mse) + 1), mse)
                    plt.title("Blocks outputs convergence")
                    plt.xlabel("Iterations")
                    plt.ylabel(r"$(x_{i+1} - x_{i})^2$")
                    plt.savefig("outputs_convergence.png")
                    plt.clf()
                    # log to wandb
                    if isinstance(self.logger, pl.loggers.WandbLogger):
                        self.logger.experiment.log(
                            {
                                "Outputs Convergence": wandb.Image(
                                    "outputs_convergence.png"
                                )
                            },
                            step=self.global_step,
                        )

                    # Performance:
                    loss = []
                    accuracy = []
                    mse = []
                    for out in outputs:
                        loss.append(
                            F.cross_entropy(out, self._sample_input_label).cpu()
                        )
                        accuracy.append(
                            (out.argmax(-1) == self._sample_input_label)
                            .float()
                            .mean()
                            .cpu()
                        )

                    plt.figure(figsize=(8, 8))
                    plt.subplot(2, 1, 1)
                    plt.plot(loss)
                    plt.title("Performance")
                    plt.xlabel("Iterations")
                    plt.ylabel("Cross Entropy Loss")
                    plt.subplot(2, 1, 2)
                    plt.plot(accuracy)
                    plt.xlabel("Iterations")
                    plt.ylabel("Accuracy")
                    plt.savefig("performance.png")
                    plt.clf()
                    # log to wandb
                    if isinstance(self.logger, pl.loggers.WandbLogger):
                        self.logger.experiment.log(
                            {"Performance": wandb.Image("performance.png")},
                            step=self.global_step,
                        )

                    self.model.repeats = repeats

                # if isinstance(self.model, SimpleRiT2):
                #     with torch.no_grad():
                #         out = self.model(self._sample_input_data)
                #         plt.figure(figsize=(10, 6))
                #         plt.plot(out["confidences"].cpu().numpy())
                #         plt.title(f"Confidences, epoch: {self.current_epoch}")
                #         plt.savefig("confidences.png")
                #         plt.clf()
                #         # log to wandb
                #         if isinstance(self.logger, pl.loggers.WandbLogger):
                #             self.logger.experiment.log(
                #                 {"Confidences": wandb.Image("confidences.png")},
                #                 step=self.global_step,
                #             )
                #             self.logger.experiment.log(
                #                 {"Average Confidence": out["confidences"].argmax(0).float().mean().item()},
                #                 step=self.global_step,
                #                 )
            except Exception as e:
                print(f"[ERROR] {e}")

    def optimizer_step(self, *args, **kwargs):
        """
        Add weight normalization, etc here.
        """
        super().optimizer_step(*args, **kwargs)

    def on_train_batch_end(self, out, batch, batch_idx):
        if batch_idx == self.trainer.num_training_batches - 1:  # only on last batch
            pass

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        if type(out) == dict:
            if "confidences" in out:
                confidences = out["confidences"]
                out = out["logits"]
            else:
                if self.hparams.criterion in ["fwce", "wce"]:
                    N = 5
                    confidences = None
                    # out = out["logits"][-N:]
                    out = out["logits"]
                else:
                    confidences = None
                    out = out["logits"][-1]
        else:
            confidences = None
        loss = self.calculate_loss(out, label, img, confidences=confidences)
        if out.dim() == 3:
            if confidences is not None:
                out = out[confidences.argmax(0)]
            else:
                out = out[-1]
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
