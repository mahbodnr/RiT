import argparse
import torch
import pytorch_lightning as pl
from RiT.network import Net
from RiT.datasets import get_dataloader
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

model_path= r"model_checkpoints/rith_d1_small_patch16_224_tiny-imagenet_pbtqq_20240905194647.ckpt"
state = torch.load(model_path)
trainer = pl.Trainer()
args = argparse.Namespace(**state["hyper_parameters"])
device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_gpu) else "cpu")

# torch set default dtype
if args.default_dtype == "float64":
    torch.set_default_dtype(torch.float64)
elif args.default_dtype == "float32":
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision(args.matmul_precision)

# Load the data
train_dl, test_dl = get_dataloader(args)
# Load the model
net = Net(args).to(device)
net.load_state_dict(state["state_dict"])
net.eval()

print("Train data:")
trainer.test(net, train_dl)
print("Test data:")
trainer.test(net, test_dl)
