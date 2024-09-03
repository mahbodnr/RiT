import argparse
import torch
import pytorch_lightning as pl
from RiT.network import Net
from RiT.datasets import get_dataloader
from RiT.training import train

parser = argparse.ArgumentParser()
"""
parser.add_argument(
    "--example-arg",
    type=str,
)
"""
model_path= r"PATH/TO/MODEL"
state = torch.load(model_path)
trainer = pl.Trainer()
args = argparse.Namespace(**state["hyper_parameters"])

# torch set default dtype
if args.default_dtype == "float64":
    torch.set_default_dtype(torch.float64)
elif args.default_dtype == "float32":
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision(args.matmul_precision)

# Load the model
net = Net(args)
net.load_state_dict(state["state_dict"])
net.train()

# Load the data
train_dl, test_dl = get_dataloader(args)
args._sample_input_data = next(iter(train_dl))[0][0:10].to(
    "cuda" if args.gpus else "cpu"
)

if __name__ == "__main__":
    train(args)