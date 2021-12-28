from loguru import logger
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from os.path import exists

def log_param(param):
    for key, value in param.items():
        logger.info(f"{key} : {value}")


def find_latent_space_and_show(model, DataLoader, data_root, num_show_images):
    attr_list = ['Bald', 'Bangs', 'Eyeglasses', 'Male', 'Mustache', 'Pale_Skin', 'Smiling', 'Black_Hair', 'Blond_Hair',
                 'Wearing_Lipstick']
    attr_table = pd.read_csv(f"{data_root}list_attr_celeba.csv", header=0, encoding='utf8')

    latent_embedding = None
    device = model.device
    print(device)
    with torch.no_grad():
        for X, _ in tqdm(DataLoader):
            if exists("latent_embedding.pt"):
                latent_embedding = torch.load("latent_embedding.pt").to(device)
                break

            X = X.to(device)
            z_mean, _ = model.encoder(X)  # batchsize X z_dim
            if latent_embedding is None:
                latent_embedding = z_mean
            else:
                latent_embedding = torch.cat([latent_embedding, z_mean], dim=0)

    torch.save(latent_embedding.cpu(), "latent_embedding.pt")
    # latent_embedding = torch.stack(latent_embedding, dim = 0) # number of images X z_dim

    attr_latent_axis = {}

    img = X[:num_show_images, :].to(device)
    img_latent_mean, img_latent_logvar = model.encoder(img)
    img_latent = model.sample_with_reparameterization(img_latent_mean, img_latent_logvar)
    img_recon = model.decoder(img_latent)
    img = torch.cat([img.unsqueeze(1), img_recon.unsqueeze(1)], dim=1)

    for attr in attr_list:
        attr_series = attr_table[attr]
        attr_series = torch.tensor(attr_series.values.astype(np.int32)).to(device)
        mask_false = attr_series < 0
        mask_true = attr_series > 0

        latent_mean_true = latent_embedding[mask_true].mean(dim=0).reshape(-1)
        latent_mean_false = latent_embedding[mask_false].mean(dim=0).reshape(-1)

        latent_mean_diff = latent_mean_true - latent_mean_false
        latent_axis = torch.argmax(torch.abs(latent_mean_diff)).cpu().item()
        sign = 1 if latent_mean_diff[latent_axis] > 0 else -1
        attr_latent_axis[attr] = (latent_axis, sign)

        attr_changed_latent = img_latent.clone()
        attr_changed_latent += (sign * 100)
        print(latent_axis, sign, (attr_changed_latent[:, latent_axis] - img_latent[:, latent_axis]).sum())
        # print(f"raw : {img_latent}, changed : {attr_changed_latent}")
        changed_img = model.decoder(attr_changed_latent).unsqueeze(dim=1)
        img = torch.cat([img, changed_img], dim=1)
        print(img.shape, (changed_img - img_recon.unsqueeze(1)).sum())

    img = torch.permute(img, (0, 1, 3, 4, 2)).cpu().detach().numpy()
    attr_list = ['raw', 'recon'] + attr_list
    N, K = num_show_images, len(attr_list)
    plt.rcParams['figure.figsize'] = (8 * N, 6 * K)
    fig, axs = plt.subplots(N, K)

    for i in range(N):
        for j, attr in enumerate(attr_list):
            axs[i, j].imshow((img[i, j, :, :, :] + 1) / 2)
            if i == N - 1:
                axs[i, j].set_xlabel(f"{attr}", rotation=45, fontdict={'size' : 100})

    # plt.imshow(npimg)
    plt.title("Change the latent space with labels")
    plt.savefig(f"./result/change_latent.png", dpi=300)
    plt.clf()
