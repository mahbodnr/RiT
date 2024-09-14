# %%
import argparse
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from RiT.network import Net
from RiT.datasets import get_dataloader
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("dark")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
"""
parser.add_argument(
    "--example-arg",
    type=str,
)
"""
# %%

model_path= r"model_checkpoints/srit_d1_small_patch16_224_tiny-imagenet_hzktt_20240911120716.ckpt" # no noise
# model_path= r"model_checkpoints/srit_d1_small_patch16_224_tiny-imagenet_bhufm_20240911113004.ckpt" # noise
model_path= r"model_checkpoints/srit_d1_small_patch16_224_tiny-imagenet_lvgdf_20240913160217.ckpt" # cls
model_path= r"model_checkpoints/srit_d1_small_patch16_224_tiny-imagenet_nuvqe_20240913170438.ckpt" #cls noise
state = torch.load(model_path)
trainer = pl.Trainer()
args = argparse.Namespace(**state["hyper_parameters"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch set default dtype
if args.default_dtype == "float64":
    torch.set_default_dtype(torch.float64)
elif args.default_dtype == "float32":
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision(args.matmul_precision)

# Load the data
train_dl, test_dl = get_dataloader(args)
data = next(iter(test_dl))[0].to(device)
# Load the model
net = Net(args).to(device)
# del state["state_dict"]["model.timesteps"]
net.load_state_dict(state["state_dict"])
net.eval()
# %%
with torch.no_grad():
    img, label = next(iter(test_dl))
    img, label = img.to(device), label.to(device)
    res = net.model.inference(img,
                            repeats= 12,
                            halt="classify",
                            halt_threshold=0.9,
                            ema_alpha=0.5,
                            halt_noise_scale=0,
                            )
    halt_probs_hist = res["halt_probs"]
    halted = res["halted"]
    block_halt = res['block_halt']

    plt.subplots(3, 1, figsize=(20, 30))
    plt.subplot(3, 1, 1)
    # hist = torch.stack(halt_probs_hist)[...,0].mean(-1)
    hist = torch.stack(halt_probs_hist)[...,0,0]
    plt.plot(hist.cpu().numpy())
    plt.title("Halt Probabilities")
    plt.subplot(3, 1, 2)
    # hist = torch.stack(halted)[...,0].to(dtype=torch.float).mean(-1)
    hist = torch.stack(halted)[...,0,0]#.sum(-1)
    plt.plot(hist.cpu().numpy())
    plt.title("Halted")
    plt.subplot(3, 1, 3)
    hist = torch.stack(block_halt)[...,0,0]
    plt.plot(hist.cpu().numpy())
    plt.title("Block Halt")
    # plt.show()
    plt.savefig("hist.png")
    plt.clf()

# %%
with torch.no_grad():
    img, label = next(iter(test_dl))
    img, label = img.to(device), label.to(device)
    res = net.model.inference(img,
                            repeats= 12,
                            halt="classify",
                            halt_threshold=0.9,
                            ema_alpha=0.5,
                            halt_noise_scale=0,
                            )
    block_halt = torch.stack(res['block_halt'])[...,0]
    plt.figure(figsize=(20, 10))
    plt.plot(block_halt.cpu().numpy())
    plt.title("Block Halt")
    plt.savefig("block_halt_t.png")
    plt.clf()
# %%
with torch.no_grad():
    correct_preds_all = 0
    correct_preds_halt = 0
    count = 0
    for img, label in test_dl:
        img, label = next(iter(test_dl))
        img, label = img.to(device), label.to(device)
        res = net.model.inference(img,
                                repeats= 12,
                                halt="classify",
                                halt_threshold=0.9,
                                ema_alpha=0.5,
                                halt_noise_scale=0,
                                )
        output = res["logits"]
        block_halt = torch.stack(res['block_halt'])[...,0]
        
        correct_preds_all += (output.argmax(-1) == label).sum().item()
        count += len(label)
        layer_outputs = torch.stack(res["layer_outputs"]).squeeze(0)
        halted_outputs = []
        for i, l in enumerate((block_halt<0.99).sum(0).int()):
            halted_outputs.append(layer_outputs[l, i])
        halted_outputs = torch.stack(halted_outputs)
        correct_preds_halt += (halted_outputs.argmax(-1) == label).sum().item()
    print(f"Accuracy all: {correct_preds_all/count}")
    print(f"Accuracy halt: {correct_preds_halt/count}")

# %%
model_paths = ["logs/RiT/saved_checkpoints/srit_d1_small_patch16_224_tiny-imagenet_bhufm_20240911113004-epoch=011-val_loss=3.77.ckpt",
    "logs/RiT/saved_checkpoints/srit_d1_small_patch16_224_tiny-imagenet_bhufm_20240911113004-epoch=025-val_loss=3.17.ckpt",
    "logs/RiT/saved_checkpoints/srit_d1_small_patch16_224_tiny-imagenet_bhufm_20240911113004-epoch=147-val_loss=2.60.ckpt",
    "logs/RiT/cx08vj6e/checkpoints/srit_d1_small_patch16_224_tiny-imagenet_bhufm_20240911113004-epoch=289-val_loss=2.59.ckpt"
]
for i in range(len(model_paths)):
    model_path = model_paths[i]
    state = torch.load(model_path)
    args = argparse.Namespace(**state["hyper_parameters"])
    net = Net(args).to(device)
    net.load_state_dict(state["state_dict"])

    with torch.no_grad():
        img, label = next(iter(test_dl))
        img, label = img.to(device), label.to(device)
        res = net.model.inference(img,
                                repeats= 50,
                                halt="ema",
                                halt_threshold=0.9,
                                ema_alpha=0.5,
                                halt_noise_alpha=0,
                                )
        halt_probs_hist = res["halt_probs"]
    p = torch.stack(halt_probs_hist)[...,0,0].cpu()

    p_mean = p.mean(-1)
    p_std = p.std(-1)
    plt.plot(p_mean.numpy(), label=f"Epoch {state['epoch']}")
    max_lim = (p_mean + p_std).numpy()
    max_lim[max_lim > 1] = 1
    plt.fill_between(range(len(p_mean)), F.relu(p_mean - p_std).numpy(), max_lim, alpha=0.3)
    plt.title("Halt Probabilities")
    plt.xlabel("Iterations")
    plt.ylabel("Halt Probability")
    plt.legend()
plt.show()

# %%
for i in range(p.shape[1]):
    line = p[:, i]
    # trim line from start to thr points that it hits the threshold
    thr = 0.9
    if (line > thr).any():
        line = line[:torch.where(line > thr)[0][0]+1]
        plt.plot(len(line)-1, 1, "x", color="red", zorder=2, markersize=5)
    plt.plot(line.cpu().numpy(), alpha=0.5,  linewidth=0.5, zorder=1, linestyle="-")
# add a horizontal line at the threshold
plt.axhline(y=thr, color="red", linestyle="--", zorder=0)
plt.title("Halt Probabilities")
plt.xlabel("Iterations")
plt.ylabel("Halt Probability")
plt.show()

# %%
with torch.no_grad():
    repeats = 100
    for img, label in test_dl:
        img, label = img.to(device), label.to(device)
        res = net.model.inference(img,
                                repeats= repeats,
                                halt="ema",
                                halt_threshold=0.9,
                                ema_alpha=0.5,
                                halt_noise_alpha=0,
                                )
        halt_probs_hist = torch.stack(res["halt_probs"])[...,0,0]
        halted = torch.stack(res["halted"])[...,0,0]
        block_halt = torch.stack(res['block_halt'])[...,0,0]
        logits= res["logits"]

        iterations = repeats - halted.sum(0)
        loss = F.cross_entropy(logits, label, reduction="none")
        correct_preds = torch.where(label == logits.argmax(-1))[0]

        plt.scatter(iterations[correct_preds].cpu().numpy(), loss[correct_preds].cpu().numpy(), color="blue", label="Correct Prediction", alpha=0.5)
        plt.scatter(iterations[~correct_preds].cpu().numpy(), loss[~correct_preds].cpu().numpy(), color="red", label="Incorrect Prediction", alpha=0.5)
    plt.title("Iterations vs Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    

#%%
trainer = pl.Trainer()
# net = Net(args).to(device)
# net.load_state_dict(state["state_dict"])
default_repeats = net.model.repeats
with torch.no_grad():
    for repeats in [10, 20, 50, 100]:
        print(f"Repeats: {repeats}")
        net.model.repeats = repeats
        trainer.test(net, test_dl)
    net.model.repeats = default_repeats
# %%
