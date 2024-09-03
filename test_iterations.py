import argparse
import torch
import pytorch_lightning as pl
from RiT.network import Net
from RiT.utils import get_dataloader
from torch.nn import functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
"""
parser.add_argument(
    "--example-arg",
    type=str,
)
"""
model_path= r"model_checkpoints/rith_d3_cifar_cifar10_lezzr_20240825140905.ckpt"
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
net.load_state_dict(state["state_dict"])
net.eval()


net.model.halt_threshold = 1
img, label = next(iter(test_dl))
img, label = img.to(device), label.to(device)
net.validation_step((img, label), 0)
net.model.halt_probs_hist
hist = torch.stack(net.model.halt_probs_hist)[:,:,0,0]
plt.plot(hist)
plt.savefig("hist.png")

for threshold in [0.4, 0.5, 0.9, 0.999, 1, None]:
    net.model.halt_threshold = threshold
    corrects = 0
    total = 0
    for i, batch in enumerate(test_dl):
        img, label = batch
        img, label = img.to(device), label.to(device)
        res = net.validation_step((img, label), i)
        acc = res["acc"]
        corrects += acc * img.size(0)
        total += img.size(0)
    test_acc = corrects / total

    print(f"Threshold: {threshold}, Iterations: {net.model.iterations.mean().item()}, Test Accuracy: {test_acc.item()}")
    

# mse = []
# for output in range(1, len(net.model.block_output)):
#     mse.append((net.model.block_output[output] - net.model.block_output[output-1]).norm().item())

# plt.subplot(1, 4, 1)
# plt.plot(mse, label="mse")

# dot = []
# for output in range(1, len(net.model.block_output)):
#     dot.append((net.model.block_output[output] * net.model.block_output[output-1]).sum().item())
# plt.subplot(1, 4, 2)
# plt.plot(dot, label="dot")

# cosine = []
# for output in range(1, len(net.model.block_output)):
#     cosine.append((net.model.block_output[output] * net.model.block_output[output-1]).sum().item() / (net.model.block_output[output].norm().item() * net.model.block_output[output-1].norm().item()))
# plt.subplot(1, 4, 3)
# plt.plot(cosine, label="cosine")

# kl = []
# for output in range(1, len(net.model.block_output)):
#     kl.append(F.kl_div(F.log_softmax(net.model.block_output[output], dim=1), F.softmax(net.model.block_output[output-1], dim=1)).item())
# plt.subplot(1, 4, 4)
# plt.plot(kl, label="kl")

# plt.legend()
# plt.show()
