from loguru import logger
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def log_param(param):
    for key, value in param.items():
        logger.info(f"{key} : {value}")


def find_latent_space_and_show(model, DataLoader,  data_root, num_show_images):
    attr_list = ['Bald', 'Bangs', 'Eyeglasses', 'Male', 'Mustache', 'Pale_Skin', 'Smiling', 'Black_Hair', 'Blond_Hair', 'Wearing_Lipstick']
    attr_table = pd.read_csv(f"{data_root}list_attr_celeba.csv", header = 0, encoding='utf8')

    latent_embedding = []
    device = model.device
    with torch.no_grad():
        for X, _ in DataLoader:
            X = X.to(device)
            z_mean, _ = model.encoder(X) # batchsize X z_dim
            latent_embedding.append(z_mean)

    latent_embedding = torch.stack(latent_embedding, dim = 0) # number of images X z_dim

    attr_latent_axis = {}

    img = X[:num_show_images, :].unsqueeze(dim = 1)
    for attr in attr_list:
        attr_series = attr_table[attr]
        attr_series = torch.tensor(attr_series.values.astype(np.int32)).to(device)
        mask_false = attr_series < 0
        mask_true = attr_series > 0

        latent_mean_true = latent_embedding[mask_true].mean(dim = 0).reshape(-1)
        latent_mean_false = latent_embedding[mask_false].mean(dim=0).reshape(-1)

        latent_mean_diff = latent_mean_true - latent_mean_false
        latent_axis = torch.argmax(torch.abs(latent_mean_diff)).cpu().items()
        sign = 1 if latent_mean_diff[latent_axis] > 0 else -1
        attr_latent_axis[attr] = (latent_axis, sign)

        latent_changed = z_mean[:, latent_axis] + (sign * 3)
        changed_img = model.decoder(latent_changed).unsqueeze(dim = 1)
        img = torch.cat([img,changed_img], dim = 1)


    attr_list = ['raw', 'recon'] + attr_list
    N, K = num_show_images, len(attr_list)
    plt.rcParams['figure.figsize'] = (8*N, 6*K)
    fig, axs = plt.figure(N, K)

    for i in range(N):
        for j, attr in enumerate(attr_list):
            axs[i, j].plot(img[i,j,:,:,:])
            if i == N-1:
                axs[i, j].set_xlabel(f"{attr}", rotation=45)

    # plt.imshow(npimg)
    plt.savefig(f"./result/change_latent.png", dpi = 300)
    plt.clf()








