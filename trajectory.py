# %%
import argparse
import torch
from torch.nn import functional as F
from RiT.network import Net
from RiT.datasets import get_dataloader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torchdeq import get_deq


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

# transait
model_path = (
    # r"model_checkpoints/transit_tiny_patch16_224_imagenet_vdeyf_20241017131041.ckpt"
    # r"model_checkpoints/ntransit_tiny_patch16_224_imagenet_ehzxr_20241025181909.ckpt"
    # r"model_checkpoints/ntransit_small_patch16_224_imagenet_gnbod_20241028104816.ckpt"
    # r"model_checkpoints/ntransit_tiny_patch16_224_tiny-imagenet_ilibt_20241111153746.ckpt"
    # r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_misoc_20241112194558.ckpt"
    # r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_edttv_20241114174424.ckpt"
    # r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_lypvp_20241115142250.ckpt" # pre-sd
    r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_tqflk_20241115142246.ckpt" #zero-sd
    # r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_iioew_20241116060215.ckpt"
)

state = torch.load(model_path)
args = argparse.Namespace(**state["hyper_parameters"])

# torch set default dtype
if args.default_dtype == "float64":
    torch.set_default_dtype(torch.float64)
elif args.default_dtype == "float32":
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision(args.matmul_precision)
#%%
# Load the data
args.eval_batch_size = 256
train_dl, test_dl = get_dataloader(args)
data, label = next(iter(test_dl))
data, label = data.to(device), label.to(device)

#%%
# Load the model
net = Net(args).to(device)
net.load_state_dict(
    state["state_dict"],
    strict=False,
)
net.eval()

# %%
if "indexing" in state["hyper_parameters"]:
    del state["hyper_parameters"]["indexing"]
# %%
# import pytorch_lightning as pl
# state["hyper_parameters"]["eval_f_max_iter"] = 12
# net.model.deq = get_deq(**state["hyper_parameters"])
# torch.set_float32_matmul_precision('medium')
# trainer = pl.Trainer(accelerator="auto")
# with torch.no_grad():
#     trainer.test(net, test_dl)

# %%
# data, label = data.to("cpu"), label.to("cpu")
# net.to("cpu")
# %%

# steps = args.grad
net = net.to(device)
steps = 50
with torch.no_grad():
    output = net.model._intermediate_layers(
        data,
        n=list(range(1, steps+1)),
        max_iter = steps,
    )
outputs = torch.stack(output[0]) # [steps, batch, tokens, dim]
print(outputs.shape)
preds = []
for out in outputs:
    pred = net.model.forward_head(net.model.norm(out))
    preds.append(pred)
preds = torch.stack(preds).detach()
print(preds.shape)

# %% convergence
conv = lambda x: np.linalg.norm((x[1:] - x[:-1]).reshape((x.shape[0]-1, -1)), axis=1)

plt.figure(figsize=(8, 10))
plt.subplot(2, 1, 1)
plt.plot(torch.arange(1, len(outputs)), conv(outputs.cpu()))
plt.title(r"Block outputs convergence")
plt.xlabel("Iterations")
plt.ylabel(r"$(x_{i+1} - x_{i})^2$")
# plt.yscale("log")
plt.subplot(2, 1, 2)
plt.plot(torch.arange(1, len(preds)), conv(preds.cpu()))
plt.title(r"Predictions convergence")
plt.xlabel("Iterations")
plt.ylabel(r"$(x_{i+1} - x_{i})^2$")
# plt.yscale("log")
plt.show()


# %% Performance:
loss = []
accuracy = []
mse = []
with torch.no_grad():
    for pred in preds:
        loss.append(F.cross_entropy(pred, label).cpu())
        accuracy.append((pred.argmax(-1) == label).float().mean().cpu())

x_axis = torch.arange(1, len(loss) + 1)
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(x_axis, loss)
plt.axvline(x=12, color="red", linestyle="--", alpha=0.5)
plt.title("Performance")
plt.xlabel("Iterations")
plt.ylabel("Cross Entropy Loss")
plt.subplot(2, 1, 2)
plt.plot(x_axis, accuracy)
plt.axvline(x=12, color="red", linestyle="--", alpha=0.5)
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.show()
# %%
# make a gif showing the change of the output distributions over time
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# selected_output = outputs.view(outputs.shape[0], -1)
# selected_output = outputs.mean(1)

fig, ax = plt.subplots()

def update(i):
    ax.clear()
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    preds = net.model.forward_head(net.model.norm(outputs[i])).detach().cpu()
    ax.hist(
        F.softmax(preds.view(-1), dim=-1).numpy(),
        bins=200,
        density=False,
        alpha=0.85,
        # color="blue",
    )
    ax.set_title(f"Iteration: {i}")
    return ax


ani = animation.FuncAnimation(
    fig, update, frames=range(outputs.shape[0]), repeat=False
)
# save
writer = PillowWriter(fps=5)
ani.save("output_distributions.gif", writer=writer)
plt.show()

# %%
# make a gif showing the change of the output distributions over time
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

fig, ax = plt.subplots()

preds = net.model.forward_head(net.model.norm(outputs)).detach().cpu()

def update(i):
    ax.clear()
    iteration_output = outputs[i]
    iteration_preds =  net.model.forward_head(net.model.norm(iteration_output)).detach().cpu()
    res = torch.zeros(iteration_preds.shape[-1])
    for p in iteration_preds:
        sorted_output, indicies = torch.sort(
            F.softmax(p), descending=False
        )
        res += sorted_output
    res /= preds.shape[1]
    ax.plot(np.arange(len(res)), np.log(res.cpu().numpy()), alpha=0.5)

    # sorted_output , indicies = torch.sort(F.softmax(selected_output[i]), descending=False)
    # # sorted_output , indicies = torch.sort((selected_output[i]), descending=False)
    # ax.plot(np.arange(len(sorted_output)), np.log(sorted_output.cpu().numpy()), alpha=0.5, color="blue")
    ax.set_title(f"Iteration: {i}")
    return ax


ani = animation.FuncAnimation(
    fig, update, frames=range(selected_output.shape[0]), repeat=False
)
# save
writer = PillowWriter(fps=5)
ani.save("sorted_output_distributions.gif", writer=writer)
plt.show()

# %%
import umap
from tqdm import tqdm
token = "all"
dots = []
with torch.no_grad():
    for smaple in tqdm(range(outputs.shape[1])):
        reducer = umap.UMAP(n_components=3)
        if token == "cls":
            out = outputs[:, smaple, 0, :]
        elif token == "all":
            out = outputs[:, smaple].view(outputs.shape[0], -1)
        dots.append(reducer.fit_transform(out.cpu().numpy()))
        if smaple == 16:
            break
dots = np.stack(dots)
dots.shape
# %%
# batch_idx = 3
# tr = dots[:,batch_idx]
N = 12
fig = plt.figure(figsize=(15, 15))
# ax = fig.add_subplot(111, projection='3d')
for i in range(16):
    # for i in torch.where(label == 10)[0]:
    ax = fig.add_subplot(4, 4, i + 1, projection="3d")
    tr = dots[i][:]
    ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], alpha=0.5)
    ax.scatter(
        tr[:, 0], tr[:, 1], tr[:, 2], c=np.arange(tr.shape[0]), s=50, cmap="viridis"
    )
    ax.scatter(tr[N, 0], tr[N, 1], tr[N, 2], c="red", s=150)
    ax.scatter(tr[0, 0], tr[0, 1], tr[0, 2], c="black", s=150, label="start")
ax.set_title(f"Trajectory")
plt.show()

# %%
mse = []
cls_outputs = block_outputs[:, :, 0, :]
for i in range(cls_outputs.shape[0] - 1):
    mse.append(F.mse_loss(cls_outputs[i], cls_outputs[i + 1], reduce=False).cpu())

plt.plot(mse)
plt.show()
# %%
mse = []
cls_outputs = block_outputs[:, :, 0, :]
for i in range(cls_outputs.shape[0] - 1):
    mse.append(
        F.mse_loss(cls_outputs[i], cls_outputs[i + 1], reduce=False).mean(-1).cpu()
    )

plt.plot(torch.stack(mse), alpha=0.5)
plt.show()

# %% histogram:
plt.hist(abs(o[-1] - o[-2]).mean(0), bins = 100, range=(0, 0.2))
# %% dimnsion
def first_derivative_torch(x, y):
    dx = torch.diff(x)
    dy = torch.diff(y)
    derivative = dy / dx
    return torch.cat((derivative[:1], derivative))


# outputs: [steps, batch, classes]
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
tr = outputs[100:, 0, :]
res = torch.zeros(tr.shape[0])
for p in tr:
    dists = (tr - p.unsqueeze(0)).norm(dim=-1)
    sorted_dists, indicies = torch.sort(dists)
    res += sorted_dists
res /= tr.shape[0]
plt.plot(
    torch.log(res),
    torch.log(torch.arange(1, len(res) + 1)),
)
plt.xlabel("log($\epsilon$)")
plt.ylabel("log(N)")

d = first_derivative_torch(torch.log(res), torch.log(torch.arange(1, len(res) + 1)))
plt.subplot(2, 1, 2)
plt.plot(torch.log(res), d)
plt.xlabel("log($\epsilon$)")
plt.ylabel("derivative")
plt.show()


# %%
x = net.model.embed_input(data[0:1])
repeats = 2
jacobian = []
for t in range(repeats):
    for block in net.model.blocks:
        x = x.detach()
        x.requires_grad = True
        x = block(x)
        y = net.model.classify(x)
        grad = autograd.grad(y.sum(), x, retain_graph=True, create_graph=True)[0]
        jacobian.append(grad.view(-1))
jacobian = torch.stack(jacobian, dim=0)
# %%
x = net.model.embed_input(data[0:1])
repeats = 1
jacobian = []
for t in range(repeats):
    for block in net.model.blocks:
        j = torch.autograd.functional.jacobian(block, x)
        jacobian.append(j)
        x = block(x)
jacobian = torch.stack(jacobian, dim=0)


# %%
J = jacobian[0]
eigenvalues = torch.linalg.eigvals(
    J.view(J.shape[1] * J.shape[2], J.shape[1] * J.shape[2])
)

# %%
device = "cpu"
net = net.to(device)
data = data.to(device)


# %%
def power_iteration(matvec_func, v, num_iterations=100, tolerance=1e-6):
    for _ in range(num_iterations):
        v_new = matvec_func(v)
        v_new_norm = torch.norm(v_new)
        if v_new_norm == 0:
            return torch.zeros_like(v), 0.0
        v_new /= v_new_norm
        if torch.allclose(v, v_new, rtol=tolerance):
            break
        v = v_new

    eigenvalue = torch.dot(matvec_func(v).view(-1), v.view(-1))
    return v, eigenvalue


def estimate_largest_eigenvalue(block, x):
    def matvec_func(v):
        with torch.enable_grad():
            x_detached = x.detach().requires_grad_()
            y = block(x_detached)
            v_reshaped = v.view_as(y)
            grad = torch.autograd.grad(y, x_detached, v_reshaped, retain_graph=True)[0]
        return grad.view(-1)

    v = torch.randn_like(x).view(-1)
    v /= torch.norm(v)

    _, eigenvalue = power_iteration(matvec_func, v)
    return eigenvalue.item()


x = net.model.embed_input(data[5:6])
repeats = 150
eigenvalues = []
with torch.no_grad():
    for t in range(repeats):
        for block in net.model.blocks:

            def block_norm(x):
                x = block(x)
                x = (x - x.mean(dim=(-1, -2), keepdim=True)) / x.std(
                    dim=(-1, -2), keepdim=True
                )
                return x

            eigenvalue = estimate_largest_eigenvalue(block_norm, x)
            eigenvalues.append(eigenvalue)
            x = block_norm(x)
            # x = block(x)
            # x = (x - x.mean(dim=(-1,-2), keepdim=True)) / x.std(dim=(-1,-2), keepdim=True)


plt.plot(eigenvalues)
plt.axhline(y=1, color="red", linestyle="--")
plt.show()
