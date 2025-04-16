import os
import time
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm import tqdm


from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.models import create_model

from RiT.models import *
from RiT.utils import get_criterion, get_layer_outputs
from RiT.augmentation import CutMix, MixUp

from schedulefree import ScheduleFreeWrapperReference
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
                "depth",
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
                "norm_layer",
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
                        "update_rate",  # old version
                        "injection",
                        "inject_input",  # old version
                        "use_head_vit",
                        "phantom_grad",
                        "phantom_grad_steps",
                        "phantom_grad_update_rate",
                        "convergence_threshold",
                        "stable_skip",
                        "n_pre_layers",
                        "n_post_layers",
                        "expand_tokens",
                        "expand_tokens_keep_input",
                    ]
                }
            )
            kwargs["logger"] = self.log
        if "cat_vit" in self.hparams.model_name:
            kwargs.update(
                {
                    k: self.hparams[k]
                    for k in [
                        "use_v",
                    ]
                }
            )

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
        if "wtvit" in self.hparams.model_name:
            kwargs.update(
                {
                    k: self.hparams[k]
                    for k in [
                        "n_pre_layers",
                        "n_post_layers",
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

        if hparams.weight_distill:
            self.teacher = create_model(
                model_name=hparams.teacher_model,
                num_classes=hparams.num_classes,
            )
            model_state_dict = torch.load(hparams.teacher_model_path)["state_dict"]
            state_dict = {
                k.replace("model.", "", 1): v for k, v in model_state_dict.items()
            }
            state_dict = {
                k.replace("teacher.", "", 1): v for k, v in state_dict.items()
            }
            self.teacher.load_state_dict(state_dict, strict=False)
            # distill teacher weights to the model
            self.model.cls_token = self.teacher.cls_token
            self.model.patch_embed = self.teacher.patch_embed
            self.model.pos_drop = self.teacher.pos_drop
            self.model.patch_drop = self.teacher.patch_drop
            self.model.pos_embed = self.teacher.pos_embed
            self.model.norm_pre = self.teacher.norm_pre
            self.model.norm = self.teacher.norm
            self.model.head = self.teacher.head
            self.model.head_drop = self.teacher.head_drop
            self.model.fc_norm = self.teacher.fc_norm
            if len(self.model.pre_layers) > 0:
                for i in range(len(self.model.pre_layers)):
                    self.model.pre_layers[i] = self.teacher.blocks[i]
            if len(self.model.post_layers) > 0:
                for i in range(len(self.model.post_layers)):
                    self.model.post_layers[i] = self.teacher.blocks[-i - 1]

            assert (
                len(self.model.deq_layers) == 1 and len(self.model.deq_layers[0]) == 1
            )

            def get_nested_attribute(module, name):
                attributes = name.split(".")
                for attr in attributes:
                    module = getattr(module, attr)
                return module

            if hparams.weight_distill_type == "avg":
                for name, p in self.model.deq_layers[0][0].named_parameters():
                    blocks = torch.stack(
                        [
                            get_nested_attribute(block, name)
                            for block in self.teacher.blocks[
                                len(self.model.pre_layers) : (
                                    -len(self.model.post_layers)
                                    if len(self.model.post_layers) > 0
                                    else None
                                )
                            ]
                        ]
                    )
                    p.data = torch.mean(blocks, dim=0)

            elif hparams.weight_distill_type == "weighted":
                for name, p in self.model.deq_layers[0][0].named_parameters():
                    blocks = torch.stack(
                        [
                            get_nested_attribute(block, name)
                            for block in self.teacher.blocks[
                                len(self.model.pre_layers) : (
                                    -len(self.model.post_layers)
                                    if len(self.model.post_layers) > 0
                                    else None
                                )
                            ]
                        ]
                    )
                    weights = torch.tensor(
                        [1 / (2 ** (i + 1)) for i in range(len(blocks))]
                    )
                    p.data = torch.sum(
                        blocks * weights.view(-1, *(1,) * (blocks.ndim - 1)), dim=0
                    )
            elif hparams.weight_distill_type == "svd":
                for name, p in self.model.deq_layers[0][0].named_parameters():
                    blocks = torch.stack(
                        [
                            get_nested_attribute(block, name)
                            for block in self.teacher.blocks[
                                len(self.model.pre_layers) : (
                                    -len(self.model.post_layers)
                                    if len(self.model.post_layers) > 0
                                    else None
                                )
                            ]
                        ]
                    )
                    if blocks.ndim == 2:
                        p.data = torch.mean(blocks, dim=0)
                    else:
                        U_list, S_list, V_list = [], [], []
                        for i in range(len(blocks)):
                            W = blocks[i]  # Weight matrix of layer i
                            U, S, Vh = torch.linalg.svd(
                                W, full_matrices=False
                            )  # Compute SVD
                            U_list.append(U)
                            S_list.append(S)
                            V_list.append(Vh.T)

                        # Step 2: Aggregate the singular values
                        S_aggregate = torch.stack(S_list).mean(dim=0)  # (min(D1, D2))
                        # Step 3: Align and aggregate the singular vectors
                        # Align U and V by averaging them directly (optional: apply Procrustes alignment)
                        U_aggregate = torch.stack(U_list).mean(dim=0)  # (D1, D1)
                        V_aggregate = torch.stack(V_list).mean(dim=0)  # (D2, D2)
                        # Step 4: Reconstruct the aggregated weight matrix
                        # Use the aggregated singular values and vectors to reconstruct the weight matrix
                        W_reusable = (
                            U_aggregate @ torch.diag(S_aggregate) @ V_aggregate.T
                        )

                        p.data = W_reusable

            elif hparams.weight_distill_type == "procrustes_alignment":

                def procrustes_alignment(stack):
                    """
                    Aligns matrices in `stack` using Procrustes analysis.
                    Args:
                        stack: Tensor of shape (N, D, D) representing a set of square matrices.
                    Returns:
                        Aligned tensor of shape (N, D, D).
                    """
                    reference = stack[0]  # Use the first matrix as the reference
                    aligned_stack = []

                    for mat in stack:
                        # Compute the cross-covariance matrix
                        M = reference.T @ mat
                        # Perform SVD on M
                        U, _, Vh = torch.linalg.svd(M)
                        # Compute the optimal alignment
                        R = U @ Vh
                        aligned_stack.append(mat @ R)  # Align mat to the reference

                    return torch.stack(aligned_stack)

                for name, p in self.model.deq_layers[0][0].named_parameters():
                    blocks = torch.stack(
                        [
                            get_nested_attribute(block, name)
                            for block in self.teacher.blocks[
                                len(self.model.pre_layers) : (
                                    -len(self.model.post_layers)
                                    if len(self.model.post_layers) > 0
                                    else None
                                )
                            ]
                        ]
                    )
                    if blocks.ndim == 2:
                        p.data = torch.mean(blocks, dim=0)
                    else:
                        k = self.hparams.svd_k  # Number of singular vectors to keep
                        # Step 1: Perform SVD on each layer's weight matrix
                        U_list, S_list, V_list = [], [], []
                        for i in range(len(blocks)):
                            W = blocks[i]  # Weight matrix of layer i
                            U, S, Vh = torch.linalg.svd(
                                W, full_matrices=False
                            )  # Compute SVD
                            U_list.append(
                                U[:, :k]
                            )  # Keep the top-k left singular vectors
                            S_list.append(S[:k])  # Keep the top-k singular values
                            V_list.append(
                                Vh[:k, :].T
                            )  # Keep the top-k right singular vectors

                        # Convert lists to tensors for easier processing
                        U_stack = torch.stack(U_list)  # Shape: (N, D1, D1)
                        V_stack = torch.stack(V_list)  # Shape: (N, D2, D2)
                        S_stack = torch.stack(S_list)  # Shape: (N, min(D1, D2))

                        # Align U and V matrices
                        U_aligned = procrustes_alignment(U_stack)  # Shape: (N, D1, D1)
                        V_aligned = procrustes_alignment(V_stack)  # Shape: (N, D2, D2)

                        # Step 3: Aggregate the singular values and aligned singular vectors
                        S_aggregate = S_stack.mean(dim=0)  # (min(D1, D2))
                        U_aggregate = U_aligned.mean(dim=0)  # (D1, D1)
                        V_aggregate = V_aligned.mean(dim=0)  # (D2, D2)

                        # Step 4: Reconstruct the aggregated weight matrix
                        W_reusable = (
                            U_aggregate @ torch.diag(S_aggregate) @ V_aggregate.T
                        )

                        p.data = W_reusable

            else:
                raise ValueError(
                    f"Weight distillation type {hparams.weight_distill_type} not implemented."
                )

            del self.teacher

        if hparams.expand_weights:
            teacher_kwargs = kwargs.copy()
            if "transit" in self.hparams.teacher_model:
                teacher_kwargs.update(
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
                            "update_rate",  # old version
                            "injection",
                            "inject_input",  # old version
                            "use_head_vit",
                            "phantom_grad",
                            "phantom_grad_steps",
                            "phantom_grad_update_rate",
                            "convergence_threshold",
                            "stable_skip",
                            "n_pre_layers",
                            "n_post_layers",
                            "expand_tokens",
                            "expand_tokens_keep_input",
                        ]
                    }
                )
            self.teacher = create_model(
                model_name=hparams.teacher_model,
                **teacher_kwargs,
            )
            model_state_dict = torch.load(hparams.teacher_model_path)["state_dict"]
            state_dict = {
                k.replace("model.", "", 1): v for k, v in model_state_dict.items()
            }
            self.teacher.load_state_dict(state_dict)
            # distill teacher weights to the model
            self.model.cls_token = self.teacher.cls_token
            self.model.patch_embed = self.teacher.patch_embed
            self.model.pos_drop = self.teacher.pos_drop
            self.model.patch_drop = self.teacher.patch_drop
            self.model.pos_embed = self.teacher.pos_embed
            self.model.norm_pre = self.teacher.norm_pre
            self.model.norm = self.teacher.norm
            self.model.head = self.teacher.head
            self.model.head_drop = self.teacher.head_drop
            self.model.fc_norm = self.teacher.fc_norm
            if len(self.teacher.pre_layers) > 0:
                for i in range(len(self.teacher.pre_layers)):
                    layer = self.teacher.pre_layers[i]
                    self.model.blocks[i].norm1 = layer.norm1
                    self.model.blocks[i].norm2 = layer.norm2
                    self.model.blocks[i].attn.qkv = layer.attn.qkv
                    self.model.blocks[i].attn.proj = layer.attn.proj
                    self.model.blocks[i].attn.q_norm = layer.attn.q_norm
                    self.model.blocks[i].attn.k_norm = layer.attn.k_norm
                    self.model.blocks[i].attn.attn_drop = layer.attn.attn_drop
                    self.model.blocks[i].attn.proj_drop = layer.attn.proj_drop
                    self.model.blocks[i].ls1 = layer.ls1
                    self.model.blocks[i].ls2 = layer.ls2
                    self.model.blocks[i].drop_path1 = layer.drop_path1
                    self.model.blocks[i].drop_path2 = layer.drop_path2
                    self.model.blocks[i].mlp.fc1 = layer.mlp.fc1
                    self.model.blocks[i].mlp.fc2 = layer.mlp.fc2
                    self.model.blocks[i].mlp.drop1 = layer.mlp.drop1
                    self.model.blocks[i].mlp.drop2 = layer.mlp.drop2
                    self.model.blocks[i].mlp.act = layer.mlp.act
                    self.model.blocks[i].mlp.norm = layer.mlp.norm
            if len(self.teacher.post_layers) > 0:
                for i in range(len(self.teacher.post_layers)):
                    layer = self.teacher.post_layers[i]
                    self.model.blocks[-i - 1].norm1 = layer.norm1
                    self.model.blocks[-i - 1].norm2 = layer.norm2
                    self.model.blocks[-i - 1].attn.qkv = layer.attn.qkv
                    self.model.blocks[-i - 1].attn.proj = layer.attn.proj
                    self.model.blocks[-i - 1].attn.q_norm = layer.attn.q_norm
                    self.model.blocks[-i - 1].attn.k_norm = layer.attn.k_norm
                    self.model.blocks[-i - 1].attn.attn_drop = layer.attn.attn_drop
                    self.model.blocks[-i - 1].attn.proj_drop = layer.attn.proj_drop
                    self.model.blocks[-i - 1].ls1 = layer.ls1
                    self.model.blocks[-i - 1].ls2 = layer.ls2
                    self.model.blocks[-i - 1].drop_path1 = layer.drop_path1
                    self.model.blocks[-i - 1].drop_path2 = layer.drop_path2
                    self.model.blocks[-i - 1].mlp.fc1 = layer.mlp.fc1
                    self.model.blocks[-i - 1].mlp.fc2 = layer.mlp.fc2
                    self.model.blocks[-i - 1].mlp.drop1 = layer.mlp.drop1
                    self.model.blocks[-i - 1].mlp.drop2 = layer.mlp.drop2
                    self.model.blocks[-i - 1].mlp.act = layer.mlp.act
                    self.model.blocks[-i - 1].mlp.norm = layer.mlp.norm
            assert (
                len(self.teacher.deq_layers) == 1
                and len(self.teacher.deq_layers[0]) == 1
            )
            main_layer = self.teacher.deq_layers[0][0]

            for i in range(
                len(self.teacher.pre_layers),
                len(self.model.blocks) - len(self.teacher.post_layers),
            ):
                self.model.blocks[i].norm1 = main_layer.norm1
                self.model.blocks[i].norm2 = main_layer.norm2
                self.model.blocks[i].attn.qkv = main_layer.attn.qkv
                self.model.blocks[i].attn.proj = main_layer.attn.proj
                self.model.blocks[i].attn.q_norm = main_layer.attn.q_norm
                self.model.blocks[i].attn.k_norm = main_layer.attn.k_norm
                self.model.blocks[i].attn.attn_drop = main_layer.attn.attn_drop
                self.model.blocks[i].attn.proj_drop = main_layer.attn.proj_drop
                self.model.blocks[i].ls1 = main_layer.ls1
                self.model.blocks[i].ls2 = main_layer.ls2
                self.model.blocks[i].drop_path1 = main_layer.drop_path1
                self.model.blocks[i].drop_path2 = main_layer.drop_path2
                self.model.blocks[i].mlp.fc1 = main_layer.mlp.fc1
                self.model.blocks[i].mlp.fc2 = main_layer.mlp.fc2
                self.model.blocks[i].mlp.drop1 = main_layer.mlp.drop1
                self.model.blocks[i].mlp.drop2 = main_layer.mlp.drop2
                self.model.blocks[i].mlp.act = main_layer.mlp.act
                self.model.blocks[i].mlp.norm = main_layer.mlp.norm

            del self.teacher

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

    def calculate_loss(self, out, label, img, reduction="mean", **kwargs):
        self.criterion.reduction = reduction
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

        if self.hparams.schedule_free:
            assert self.hparams.sched in [
                "none",
                None,
            ], "Schedule-free is called but scheduler is not none."

        if self.hparams.lr_cycle_steps > 0:
            epochs = self.hparams.epochs
            self.hparams.epochs = self.hparams.lr_cycle_steps 
            self.scheduler, _ = create_scheduler(
                self.hparams, self.optimizer
            )
            self.hparams.epochs = epochs
        else:
            self.scheduler, self.hparams.epochs = create_scheduler(
                self.hparams, self.optimizer
            )
        # save new epochs
        self.save_hyperparameters({"epochs": self.hparams.epochs})

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
        # schedule-free
        if self.hparams.schedule_free:
            self.optimizer = ScheduleFreeWrapperReference(
                self.optimizer,
                momentum=self.hparams.momentum,
                weight_decay_at_y=self.hparams.decay_rate,
            )

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
            if self.hparams.cutmix or self.hparams.mixup:
                img, label, rand_label, lambda_ = self.cutmix_mixup(img, label)
                assert (
                    not self.hparams.use_distill_token
                ), "Trajectory loss is not compatible with distillation token."
                if self.hparams.stochastic_depth_sigma > 0:
                    steps = (
                        (
                            self.hparams.iterations
                            + torch.randn(1) * self.hparams.stochastic_depth_sigma
                        )
                        .clamp(min=1)
                        .int()
                        .item()
                    )
                else:
                    steps = self.hparams.iterations
                out_steps = self.model._intermediate_layers(
                    img,
                    n=list(
                        range(
                            max(0, steps - self.hparams.trajectory_loss_steps),
                            steps,
                        )
                    ),
                    max_iter=steps,
                )

                loss_kwargs = {}
                for key in ["aux_loss", "jac_loss", "stability_loss"]:
                    if hasattr(self.model, key):
                        val = getattr(self.model, key)
                        if val != 0:
                            loss_kwargs[key] = val
                            self.log(key, val)

                loss = 0
                loss_weight = 0 if self.hparams.incremental_trajectory_loss else 1
                for out_step in out_steps:
                    out = self.model.forward_head(self.model.norm(self.model.post_layers(out_step)))
                    loss += (
                        (
                            self.calculate_loss(
                                out,
                                label,
                                img,
                                **loss_kwargs,
                            )
                            * lambda_
                        )
                        + (
                            self.calculate_loss(
                                out,
                                rand_label,
                                img,
                                **loss_kwargs,
                            )
                            * (1.0 - lambda_)
                        )
                    ) * loss_weight
                    if self.hparams.incremental_trajectory_loss:
                        loss_weight += 1 / self.hparams.trajectory_loss_steps
                loss /= min(self.hparams.trajectory_loss_steps, steps) / 2
            else:
                raise NotImplementedError(
                    "trajectory_loss_steps is not implemented for no cutmix or mixup."
                )

            return out, loss

        if self.hparams.convergence_loss_threshold > 0:
            assert self.hparams.n_deq_layers == 1
            if self.hparams.cutmix or self.hparams.mixup:
                img, label, rand_label, lambda_ = self.cutmix_mixup(img, label)
                assert (
                    not self.hparams.use_distill_token
                ), "Trajectory loss is not compatible with distillation token."
                out_steps = self.model.post_layers(
                    self.model._intermediate_layers(
                        img,
                        n=list(range(0, self.hparams.iterations)),
                    )[0]
                )

                loss_kwargs = {}
                for key in ["aux_loss", "jac_loss", "stability_loss"]:
                    if hasattr(self.model, key):
                        val = getattr(self.model, key)
                        if val != 0:
                            loss_kwargs[key] = val
                            self.log(key, val)

                loss = 0
                loss_weight = 1
                # converged_mask = torch.zeros(out_steps.shape[1], device=out_steps.device, dtype=torch.bool)
                converged_mask = torch.zeros(out_steps.shape[1]).to(out_steps)
                for i in range(1, len(out_steps)):
                    step_loss = 0
                    out_step = out_steps[i]
                    rel_diff = torch.norm(
                        out_step - out_steps[i - 1],
                        p=2,
                        dim=list(range(1, out_step.ndim)),
                    ) / torch.norm(
                        out_steps[i - 1],
                        p=2,
                        dim=list(range(1, out_step.ndim)),
                    )

                    out = self.model.forward_head(self.model.norm(out_step))

                    step_loss = (
                        self.calculate_loss(
                            out,
                            label,
                            img,
                            reduction="none",
                            **loss_kwargs,
                        )
                        * lambda_
                    ) + (
                        self.calculate_loss(
                            out,
                            rand_label,
                            img,
                            reduction="none",
                            **loss_kwargs,
                        )
                        * (1.0 - lambda_)
                    )
                    # loss += (step_loss * ~converged_mask).mean() * loss_weight
                    loss += (step_loss * (1 - converged_mask)).mean() * loss_weight
                    loss_weight = loss_weight + 0.1

                    # converged_mask = converged_mask + (
                    #     rel_diff < self.hparams.convergence_loss_threshold
                    # )
                    converged_mask = converged_mask + (
                        1 - converged_mask
                    ) * torch.nn.functional.sigmoid(
                        (self.hparams.convergence_loss_threshold - rel_diff) * 10
                    )

                # loss /= self.hparams.iterations
                self.log("converged_ratio", converged_mask.float().mean())
            else:
                raise NotImplementedError(
                    "trajectory_loss_steps is not implemented for no cutmix or mixup."
                )

            return out, loss

        if self.hparams.incremental_iterations:
            step = int(
                self.hparams.incremental_iterations_min
                + (
                    self.hparams.incremental_iterations_max
                    - self.hparams.incremental_iterations_min
                )
                * self.trainer.current_epoch
                / self.hparams.epochs
            )
            if self.model.iterations != step:
                self.model.iterations = step
                print(f"[INFO] Iterations changed to {step}.")
                self.log("iterations", step, on_epoch=True)

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
        if hasattr(self.optimizer, "train"):  # schedule-free
            self.optimizer.train()
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
        if (self.hparams.weight_tie_cycle > 0) and (self.current_epoch>0) and (self.current_epoch % self.hparams.weight_tie_cycle == 0):
            if self.model.weight_tie:
                print(
                    f"\n\n[INFO] Relaxing weights at epoch {self.current_epoch}."
                )
                self.model.relax_weights()
            else:
                print(
                    f"\n\n[INFO] Tying weights at epoch {self.current_epoch}."
                )
                self.model.tie_weights()

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

        if self.current_epoch == self.hparams.epochs:
            self.trainer.should_stop = True

        if self.hparams.log_iterations_conv:
            steps = self.hparams.iterations * 2
            loss = torch.zeros(steps, device=self.device)
            accuracy = torch.zeros(steps, device=self.device)
            test_dl = self.trainer.val_dataloaders
            if isinstance(test_dl, list):
                test_dl = test_dl[0]
            with torch.no_grad():
                for i, (data, label) in enumerate(test_dl):
                    data, label = data.to(self.device), label.to(self.device)
                    outputs = self.model._intermediate_layers(
                        data,
                        n=list(range(steps)),
                        max_iter=steps,
                    )  # [steps, batch, tokens, dim]

                    preds = []
                    for out in outputs:
                        pred = self.model.forward_head(
                            self.model.norm(self.model.post_layers(out))
                        )
                        preds.append(pred)
                    preds = torch.stack(preds).detach()

                    batch_loss = []
                    batch_accuracy = []
                    for pred in preds:
                        batch_loss.append(F.cross_entropy(pred, label))
                        batch_accuracy.append((pred.argmax(-1) == label).float().mean())
                    loss += torch.stack(batch_loss)
                    accuracy += torch.stack(batch_accuracy)

                    if i == 5:
                        break
            loss /= i + 1
            accuracy /= i + 1

            loss = loss.cpu().numpy()
            accuracy = accuracy.cpu().numpy()

            plt.figure(figsize=(6, 8))
            plt.subplot(3, 1, 1)
            x_axis = torch.arange(1, len(loss) + 1)
            plt.plot(x_axis, loss)
            plt.axvline(
                x=self.hparams.iterations, color="red", linestyle="--", alpha=0.5
            )
            plt.title(f"Performance - epoch {self.current_epoch}")
            plt.xlabel("Iterations")
            plt.ylabel("Cross Entropy Loss")
            # plt.yscale("log")
            plt.subplot(3, 1, 2)
            plt.plot(x_axis, accuracy)
            plt.axvline(
                x=self.hparams.iterations, color="red", linestyle="--", alpha=0.5
            )
            plt.xlabel("Iterations")
            plt.ylabel("Accuracy")
            # plt.yscale("log")

            conv = lambda x: np.linalg.norm(
                (x[1:] - x[:-1]).reshape((x.shape[0] - 1, -1)), axis=1
            ) / np.linalg.norm(x[1:].reshape((x.shape[0] - 1, -1)), axis=1)
            plt.subplot(3, 1, 3)
            plt.plot(torch.arange(1, len(outputs)), conv(outputs.cpu()))
            plt.axvline(
                x=self.hparams.iterations, color="red", linestyle="--", alpha=0.5
            )
            plt.title(r"Block outputs convergence")
            plt.xlabel("Iterations")
            plt.ylabel(r"$(x_{i+1} - x_{i})^2/x_{i+1}^2$")
            plt.yscale("log")

            path = os.path.join(
                self.logger.save_dir,
                f"iterations_convergence_{self.current_epoch}.png",
            )
            plt.savefig(
                path,
                # dpi=300,
            )
            # save to wandb logger using log_image
            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log(
                    {
                        "iterations_convergence": wandb.Image(path),
                    },
                    step=self.global_step,
                )
            plt.close()

    def optimizer_step(self, *args, **kwargs):
        """
        Add weight normalization, etc here.
        """
        super().optimizer_step(*args, **kwargs)

        # if hasattr(self.model, "norm_weights_"):
        #     self.model.norm_weights_()

    def on_train_batch_end(self, out, batch, batch_idx):
        if batch_idx == self.trainer.num_training_batches - 1:  # only on last batch
            pass

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()
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
