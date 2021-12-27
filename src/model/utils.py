import torch
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt


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
