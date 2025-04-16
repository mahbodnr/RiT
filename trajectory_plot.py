import torch
from torch.nn import functional as F
from RiT.network import Net
from RiT.datasets import get_dataloader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse

sns.set_style("dark")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


models = {
    # imagenet:
    "Normal": r"model_checkpoints/transit_tiny_patch16_224_imagenet_qodex_20250401194517.ckpt",
    "W. Incr. Loss": r"model_checkpoints/transit_tiny_patch16_224_imagenet_vgbny_20250407125740.ckpt",
}

fig = plt.figure(figsize=(12, 14))

for i, (model_name, model_path) in enumerate(models.items()):
    print(f"Loading {model_name}")
    state = torch.load(model_path)
    args = argparse.Namespace(**state["hyper_parameters"])
    args.pin_memory = False

    # torch set default dtype
    if args.default_dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif args.default_dtype == "float32":
        torch.set_default_dtype(torch.float32)
        torch.set_float32_matmul_precision(args.matmul_precision)

    if i == 0:
        print(f"Loading the dataset: {args.dataset}")
        # Load the data
        args.eval_batch_size = 512
        train_dl, test_dl = get_dataloader(args)
        data, label = next(iter(test_dl))
        data, label = data.to(device), label.to(device)

    # Load the model
    net = Net(args).to(device)
    net.load_state_dict(
        state["state_dict"],
        strict=False,
    )
    net = net.to(device)
    net.eval()
    steps = 50
    loss = torch.zeros(steps, device=device)
    accuracy = torch.zeros(steps, device=device)
    with torch.no_grad():
        for data, label in test_dl:
            data, label = data.to(device), label.to(device)
            outputs = net.model._intermediate_layers(
                data,
                n=list(range(steps)),
                max_iter=steps,
            )  # [steps, batch, tokens, dim]

            preds = []
            for out in outputs:
                pred = net.model.forward_head(net.model.norm(net.model.post_layers(out)))
                preds.append(pred)
            preds = torch.stack(preds).detach()

            batch_loss = []
            batch_accuracy = []
            for pred in preds:
                batch_loss.append(F.cross_entropy(pred, label))
                batch_accuracy.append((pred.argmax(-1) == label).float().mean())
            loss += torch.stack(batch_loss)
            accuracy += torch.stack(batch_accuracy)
    loss /= len(test_dl)
    accuracy /= len(test_dl)

    loss = loss.cpu().numpy()
    accuracy = accuracy.cpu().numpy()

    plt.subplot(3, 1, 1)
    x_axis = torch.arange(1, len(loss) + 1)
    plt.plot(x_axis, loss, label=model_name)
    if i == 0:
        plt.axvline(x=12, color="red", linestyle="--", alpha=0.5)
    plt.title("Performance")
    plt.xlabel("Iterations")
    plt.ylabel("Cross Entropy Loss")
    # plt.yscale("log")
    plt.subplot(3, 1, 2)
    plt.plot(x_axis, accuracy, label=model_name)
    if i == 0:
        plt.axvline(x=12, color="red", linestyle="--", alpha=0.5)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    # plt.yscale("log")

    conv = lambda x: np.linalg.norm(
        (x[1:] - x[:-1]).reshape((x.shape[0] - 1, -1)), axis=1
    ) / np.linalg.norm(x[1:].reshape((x.shape[0] - 1, -1)), axis=1)
    plt.subplot(3, 1, 3)
    plt.plot(torch.arange(1, len(outputs)), conv(outputs.cpu()), label=model_name)
    if i == 0:
        plt.axvline(x=12, color="red", linestyle="--", alpha=0.5)
    plt.title(r"Block outputs convergence")
    plt.xlabel("Iterations")
    plt.ylabel(r"$(x_{i+1} - x_{i})^2/x_{i+1}^2$")
    plt.yscale("log")

# fig.subplots_adjust(hspace=1)
plt.legend()
plt.savefig("performance.png")
print("Saved performance.png")
# plt.show()
