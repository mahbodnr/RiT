# %%
import argparse
import torch
from torch.nn import functional as F
from RiT.network import Net
from RiT.datasets import get_dataloader
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
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
# small
# srit2 12
model_path= r"model_checkpoints/srit2_d1_small_patch16_224_tiny-imagenet_reujd_20240916165119.ckpt"
# srit2 12 sd
# model_path= r"model_checkpoints/srit2_d1_small_patch16_224_tiny-imagenet_kxwht_20240917191143.ckpt"
# srit halt none 12
# model_path= r"model_checkpoints/srit_d1_small_patch16_224_tiny-imagenet_yzbck_20240916141955.ckpt"
# es fwce
# model_path= r"model_checkpoints/srit2_d1_small_patch16_224_tiny-imagenet_icjnd_20240918160338.ckpt"
# lambda 12
model_path = r"model_checkpoints/srit2_d1_small_patch16_224_tiny-imagenet_slktl_20240919182541.ckpt"

# tiny
# srit2 12
# model_path= r"model_checkpoints/srit2_d1_tiny_patch16_224_tiny-imagenet_boutp_20240918135924.ckpt"
# srit2 50
# model_path= r"model_checkpoints/srit2_d1_tiny_patch16_224_tiny-imagenet_wulxe_20240917182023.ckpt"
# 50 es fwce
# model_path= r"model_checkpoints/srit2_d1_tiny_patch16_224_tiny-imagenet_rqzqd_20240918150430.ckpt"
# z-stand 12
model_path = r"model_checkpoints/srit2_d1_tiny_patch16_224_tiny-imagenet_yhcsi_20240919124007.ckpt"
# lambda 12 => not fixed
# model_path = r"model_checkpoints/srit2_d1_tiny_patch16_224_tiny-imagenet_wytgs_20240919164047.ckpt"
# lambda 12 es => fixed
# model_path = r"model_checkpoints/srit2_d1_tiny_patch16_224_tiny-imagenet_jbyly_20240919162050.ckpt"
# lambda 12 sd => not fixed but keeps the accuracy to some extend
# model_path = r"model_checkpoints/srit2_d1_tiny_patch16_224_tiny-imagenet_tiqza_20240919163742.ckpt"

# tmp
model_path = r"logs/RiT/32zvbfxb/checkpoints/srit_d1_tiny_patch16_224_tiny-imagenet_dvvtv_20240921095027-epoch=215-val_loss=2.61.ckpt"

state = torch.load(model_path)
args = argparse.Namespace(**state["hyper_parameters"])
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
# %%
# import pytorch_lightning as pl
# trainer = pl.Trainer(accelerator="auto")
# net.model.repeats = 12
# trainer.test(net, test_dl)
# # %%
# with torch.no_grad():
#     for img, label in test_dl:
#         img, label = img.to(device), label.to(device)
#         out = net(img)
#         if type(out) == dict:
#             confidences = out["confidences"]
#             out = out["logits"]
#         else:
#             confidences = None
#         loss = net.calculate_loss(out, label, confidences=confidences)
#         if out.dim() == 3:
#             out = out[confidences.argmax(0)]
#         acc = torch.eq(out.argmax(-1), label).float().mean()
#         print(F"Loss: {loss}, Accuracy: {acc}")

# %% SRiT2
with torch.no_grad():
    net.model.stochastic_depth = False
    net.model.repeats = 50

    img, label = next(iter(test_dl))
    img, label = img.to(device), label.to(device)
    res = net(img)
    outputs = res["logits"]
    block_outputs = res["block_outputs"]
    print(F"outputs: {outputs.shape}, block_outputs: {block_outputs.shape}")
# %% SRiT
with torch.no_grad():
    img, label = next(iter(test_dl))
    img, label = img.to(device), label.to(device)
    res = net.model.inference(img,
                            repeats= 150,
                            halt=None,
                            halt_threshold=1,
                            ema_alpha=0.5,
                            )
    block_outputs = torch.stack(res["block_outputs"])
    outputs = []
    for block_output in block_outputs:
        outputs.append(net.model.classify(block_output))
    outputs = torch.stack(outputs)
    print(F"outputs: {outputs.shape}, block_outputs: {block_outputs.shape}")

# %% convergence
mse = []
for i in range(outputs.shape[0] -1):
    mse.append(F.mse_loss(outputs[i], outputs[i+1]).cpu())

plt.figure(figsize=(8, 10))
plt.subplot(2, 1, 1)
plt.plot(torch.arange(1, len(mse)+1), mse)
plt.title(r"Classifier outputs convergence")
plt.xlabel("Iterations")
plt.ylabel(r"$(x_{i+1} - x_{i})^2$")
mse = []
for i in range(block_outputs.shape[0] -1):
    mse.append(F.mse_loss(block_outputs[i], block_outputs[i+1]).cpu())

plt.subplot(2, 1, 2)
plt.plot(torch.arange(1, len(mse)+1), mse)
plt.title("Blocks outputs convergence")
plt.xlabel("Iterations")
plt.ylabel(r"$(x_{i+1} - x_{i})^2$")
plt.show()
# %% block outputs all
mse = []
for i in range(block_outputs.shape[0] -1):
    mse.append(F.mse_loss(block_outputs[i], block_outputs[i+1], reduce=False).mean((-1, -2)).cpu())


plt.plot(torch.arange(1, len(mse)+1), torch.stack(mse), alpha=0.5)
plt.title("Blocks outputs convergence")
plt.xlabel("Iterations")
plt.ylabel(r"$(x_{i+1} - x_{i})^2$")
plt.show()

# %% Performance:
loss = []
accuracy = []
mse = []
for out in outputs:
    loss.append(F.cross_entropy(out, label).cpu())
    accuracy.append((out.argmax(-1) == label).float().mean().cpu())

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
plt.show()
# %%
# make a gif showing the change of the output distributions over time
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# selected_output = outputs.view(outputs.shape[0], -1)
selected_output = outputs[:, 0, :]
# selected_output = outputs.mean(1)

fig, ax = plt.subplots()
def update(i):
    ax.clear()
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    ax.hist(F.softmax(selected_output[i].cpu(), dim=-1).numpy(), bins=200, density=True, alpha=0.5, color="blue")
    ax.set_title(f"Iteration: {i}")
    return ax

ani = animation.FuncAnimation(fig, update, frames=range(selected_output.shape[0]), repeat=False)
# save
writer = PillowWriter(fps=5)
ani.save("output_distributions.gif", writer=writer)
plt.show()

# %%
# make a gif showing the change of the output distributions over time
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# selected_output = outputs.view(outputs.shape[0], -1)
selected_output = outputs[:, 0, :]
# selected_output = outputs.mean(1)

fig, ax = plt.subplots()
def update(i):
    ax.clear()
    # sort the outputs
    # ax.set_ylim(0, 1)
    res = torch.zeros(outputs.shape[2])
    for j in range(outputs.shape[1]):
        selected_output = outputs[:, j, :].cpu()
        sorted_output , indicies = torch.sort(F.softmax(selected_output[i]), descending=False)
        res += sorted_output
    res /= outputs.shape[1]
    ax.plot(np.arange(len(res)), np.log(res.cpu().numpy()), alpha=0.5, color="blue")

    # sorted_output , indicies = torch.sort(F.softmax(selected_output[i]), descending=False)
    # # sorted_output , indicies = torch.sort((selected_output[i]), descending=False)
    # ax.plot(np.arange(len(sorted_output)), np.log(sorted_output.cpu().numpy()), alpha=0.5, color="blue")
    ax.set_title(f"Iteration: {i}")
    return ax

ani = animation.FuncAnimation(fig, update, frames=range(selected_output.shape[0]), repeat=False)
# save
writer = PillowWriter(fps=5)
ani.save("sorted_output_distributions.gif", writer=writer)
plt.show()

# %% 
import umap
from tqdm import tqdm
dots = []
for smaple in tqdm(range(outputs.shape[1])):
    reducer = umap.UMAP(n_components=3)
    dots.append(reducer.fit_transform(
        outputs[:, smaple, :].cpu().numpy()
    ))
    if smaple == 20:
        break
dots = np.stack(dots)
dots.shape
# %%
# batch_idx = 3
# tr = dots[:,batch_idx]
fig = plt.figure(figsize=(15, 15))
# ax = fig.add_subplot(111, projection='3d')
for i in range(16):
# for i in torch.where(label == 10)[0]:
    ax = fig.add_subplot(4, 4, i+1, projection='3d')
    tr = dots[i][:]
    ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], alpha=0.5)
    ax.scatter(tr[:, 0], tr[:, 1], tr[:, 2], c=np.arange(tr.shape[0]), s=50, cmap="viridis")
    N = 49
    ax.scatter(tr[N, 0], tr[N, 1], tr[N, 2], c="red", s=150)
    ax.scatter(tr[0, 0], tr[0, 1], tr[0, 2], c="black", s=150, label="start")
ax.set_title(f"Trajectory")
plt.show()

# %%
mse = []
cls_outputs = block_outputs[:, :, 0, :]
for i in range(cls_outputs.shape[0] -1):
    mse.append(F.mse_loss(cls_outputs[i], cls_outputs[i+1], reduce=False).cpu())

plt.plot(mse)
plt.show()
# %%
mse = []
cls_outputs = block_outputs[:, :, 0, :]
for i in range(cls_outputs.shape[0] -1):
    mse.append(F.mse_loss(cls_outputs[i], cls_outputs[i+1], reduce=False).mean(-1).cpu())

plt.plot(torch.stack(mse), alpha=0.5)
plt.show()
# %% dimnsion
def first_derivative_torch(x, y):
    dx = torch.diff(x)
    dy = torch.diff(y)
    derivative = dy / dx
    return torch.cat((derivative[:1], derivative))
# outputs: [steps, batch, classes]
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
tr = outputs[100:,0,:]
res = torch.zeros(tr.shape[0])
for p in tr:
    dists = (tr - p.unsqueeze(0)).norm(dim=-1)
    sorted_dists, indicies = torch.sort(dists)
    res += sorted_dists
res /= tr.shape[0]
plt.plot(torch.log(res), torch.log(torch.arange(1, len(res)+1)), )
plt.xlabel("log($\epsilon$)")
plt.ylabel("log(N)")

d = first_derivative_torch(torch.log(res), torch.log(torch.arange(1, len(res)+1)))
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
eigenvalues = torch.linalg.eigvals(J.view(J.shape[1] * J.shape[2], J.shape[1] * J.shape[2]))

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
                x = (x - x.mean(dim=(-1,-2), keepdim=True)) / x.std(dim=(-1,-2), keepdim=True)
                return x
            eigenvalue = estimate_largest_eigenvalue(block_norm, x)
            eigenvalues.append(eigenvalue)
            x = block_norm(x)
            # x = block(x)
            # x = (x - x.mean(dim=(-1,-2), keepdim=True)) / x.std(dim=(-1,-2), keepdim=True)


plt.plot(eigenvalues)
plt.axhline(y=1, color="red", linestyle="--")
plt.show()

