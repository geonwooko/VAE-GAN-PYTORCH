from loguru import logger
import torch
import numpy as np
import matplotlib.pyplot as plt

def log_param(param):
    for key, value in param.items():
        logger.info(f"{key} : {value}")


def show_and_save(file_name, img):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./result/s.png" % file_name
    fig = plt.figure(dpi=300)
    fig.suptitle(file_name, fontsize=14, fontweight='bold')
    # plt.imshow(npimg)
    plt.imsave(f, npimg)


def save_model(epoch, encoder, decoder, discriminator):
    torch.save(decoder.cpu().state_dict(), './result/parameters/VAE_GAN_decoder_%d.pth' % epoch)
    torch.save(encoder.cpu().state_dict(), './result/parameters/VAE_GAN_encoder_%d.pth' % epoch)
    torch.save(discriminator.cpu().state_dict(), './result/parameters/VAE_GAN_discriminator_%d.pth' % epoch)
    decoder.cuda()
    encoder.cuda()
    discriminator.cuda()


def load_model(epoch, encoder, decoder, discriminator):
    #  restore models
    decoder.load_state_dict(torch.load('./result/parameters/VAE_GAN_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load('./result/parameters/VAE_GAN_encoder_%d.pth' % epoch))
    encoder.cuda()
    discriminator.load_state_dict(torch.load('./result/parameters/VAE_GAN_discriminator_%d.pth' % epoch))
    discriminator.cuda()