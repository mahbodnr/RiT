import torch
from torch.nn import functional as F
from RiT.network import Net
from RiT.datasets import get_dataloader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse

sns.set_style("dark")
# sns.set_palette("Set2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


models = {
    # prenorm
    # "Transit (12 iters)": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_dqedp_20241126231131.ckpt", # inj
    # "Transit (12 iters)": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_blhuc_20241128170925.ckpt",
    # "Transit (6 iters)": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_gxjak_20241202210127.ckpt",
    # "Transit (18 iters)": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_hxoam_20241202215314.ckpt",
    

    # " + Injection": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_fcbiw_20241202220836.ckpt",
    # "NormalizedTransit": r"model_checkpoints/ntransit_tiny_patch16_224_tiny-imagenet_ldhuu_20241202102231.ckpt",
    # "N + Injection": r"model_checkpoints/ntransit_tiny_patch16_224_tiny-imagenet_cbuns_20241203181929.ckpt",
    # "Trajectory Loss(TL) 12": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_xvzkd_20241202153317.ckpt",
    # "TL 12 + Inj": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_moyxr_20241128103358.ckpt",
    # "TL 10 + Inj": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_becei_20241202170210.ckpt",
    
    # "Stochastic Depth ($\sigma=1$)": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_gkvqx_20241201095838.ckpt",
    # "Stochastic Depth ($\sigma=2$)": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_pyskk_20241201171805.ckpt",
    # "Stochastic Depth ($\sigma=5$)": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_odeta_20241130111344.ckpt",
    # "NormalizedTransit": r"model_checkpoints/ntransit_tiny_patch16_224_tiny-imagenet_imvpt_20241128170914.ckpt", #(v1)
    # "NormalizedTransit": r"model_checkpoints/ntransit_tiny_patch16_224_tiny-imagenet_rawyt_20241128172953.ckpt", #(v2)
    # "+Injection": r"model_checkpoints/ntransit_tiny_patch16_224_tiny-imagenet_decby_20241129113333.ckpt",
    
    # "Gradual TL (GTL) 12": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_rjtvo_20241201095510.ckpt",
    # "GTL 12 + Inl": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_vadex_20241202154012.ckpt",
    # "ViT Classifier (VC)": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_ctpvo_20241202112137.ckpt",
    # "VC TL 12": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_ejccf_20241202113419.ckpt",
    # "VC GTL 12": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_uoozh_20241202113001.ckpt",
    # "VC TL 12 + Inj": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_sginz_20241202154712.ckpt",
    # "VC GTL 12 + Inj": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_nrbjb_20241202154713.ckpt",

    # "VC GTL 6 + Inj": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_cuiwr_20241202223559.ckpt",
    # "VC GTL 9 + Inj": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_inzor_20241202225640.ckpt",
    # "VC GTL 18 + Inj": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_ukrmx_20241203000338.ckpt",
    # "VC GTL 24 + Inj": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_xzkme_20241202221427.ckpt",

    # "Block Inj": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_rjnwk_20241203114246.ckpt",
    # "Block Inj + VC": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_tjtep_20241203141544.ckpt",

    # " + Trajectory Loss": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_xvzkd_20241202153317.ckpt",
    # " + Inj": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_faska_20241202154023.ckpt",
    # " + ViT on Top": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_sginz_20241202154712.ckpt", 
    # " + Gradual loss": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_nrbjb_20241202154713.ckpt", 

    # "+ Injection ViT": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_rjnwk_20241203114246.ckpt",
    # "+ Injection on Top": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_tjtep_20241203141544.ckpt",
    # "Block Inj depth=3, iter=4": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_xrjmg_20241203162618.ckpt",
    # "phantom grad 12" : r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_yvdqb_20241204135355.ckpt",
    # "phantom grad 40" : r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_jilfl_20241204142434.ckpt",
    # "+ Transformer injection": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_rjnwk_20241203114246.ckpt",
    # "+ Linear + nom injection": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_anckp_20241205111231.ckpt",

    # ImageNet:
    # "Tiny": r"model_checkpoints/transit_tiny_patch16_224_imagenet_rqxtq_20241207194052.ckpt",
    # "+ ViT classifier": r"model_checkpoints/transit_tiny_patch16_224_imagenet_wcwgq_20241207195701.ckpt",
    # "Small": r"model_checkpoints/transit_small_patch16_224_imagenet_iyfse_20241207195701.ckpt",
    # "Base": r"model_checkpoints/transit_base_patch16_224_imagenet_rugpq_20241211110712.ckpt",
    #Tiny imagenet: 
    # "Transit 4x3": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_jyppv_20241219172741.ckpt",
    # "Transit 12x3": r"model_checkpoints/.ckpt",
    # "rTransit 4x3": r"model_checkpoints/rtransit_tiny_patch16_224_tiny-imagenet_abcqx_20241218160959.ckpt",
    # "rTransit 12x3": r"model_checkpoints/rtransit_tiny_patch16_224_tiny-imagenet_zfzum_20241218143507.ckpt",
    # Imagenet
    # "12 x 3": r"model_checkpoints/transit_tiny_patch16_224_imagenet_buvxm_20241217155827.ckpt",
    # "3  x 4": r"model_checkpoints/transit_tiny_patch16_224_imagenet_cubip_20241217152746.ckpt",

    # "2 layers linear": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_cxzee_20241220122634.ckpt",
    # "2 layers block": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_jyfpa_20241220125035.ckpt",
    # "3 layers": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_jaxwj_20250106131452.ckpt",
    # "3 layers no head": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_prjqo_20250106131829.ckpt",

    # No injection
    "1 pre layer": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_fwvti_20250109165446.ckpt",
    "drop path": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_aptpv_20250109144643.ckpt",
    "init values": r"model_checkpoints/transit_tiny_patch16_224_tiny-imagenet_kykyl_20250109110554.ckpt",

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
                pred = net.model.forward_head(net.model.norm(out))
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
