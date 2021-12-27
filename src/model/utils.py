import torch
import numpy as np
import matplotlib.pyplot as plt


def show_and_save(file_name, img): # N X 3 X 128 X 128
    npimg = np.transpose(img.detach().numpy(), (0, 2, 3, 1))

    N = int(img.shape[0] ** 0.5)
    plt.rcParams['figure.figsize'] = (8*N, 6*N)
    fig, axs = plt.figure(N, N)

    for i in range(N):
        for j in range(N):
            idx = i * N + j
            axs[i, j].plot(img[idx,:,:,:])

    # plt.imshow(npimg)
    plt.savefig(f"./result/{file_name}.png", dpi = 300)
    plt.clf()


def save_model(epoch, model, device):
    torch.save(model.decoder.cpu().state_dict(), './result/parameters/VAE_GAN_decoder_%d.pth' % epoch)
    torch.save(model.encoder.cpu().state_dict(), './result/parameters/VAE_GAN_encoder_%d.pth' % epoch)
    torch.save(model.discriminator.cpu().state_dict(), './result/parameters/VAE_GAN_discriminator_%d.pth' % epoch)
    model.decoder.to(device)
    model.encoder.to(device)
    model.discriminator.to(device)


def load_model(epoch, model, device):
    #  restore models
    model.decoder.load_state_dict(torch.load('./result/parameters/VAE_GAN_decoder_%d.pth' % epoch))
    model.decoder.to(device)
    model.encoder.load_state_dict(torch.load('./result/parameters/VAE_GAN_encoder_%d.pth' % epoch))
    model.encoder.to(device)
    model.discriminator.load_state_dict(torch.load('./result/parameters/VAE_GAN_discriminator_%d.pth' % epoch))
    model.discriminator.to(device)
