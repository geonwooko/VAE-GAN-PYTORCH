import torch
from torchvision.utils import save_image
from scipy.linalg import sqrtm
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

# Reference to https://velog.io/@viriditass/GAN%EC%9D%80-%EC%95%8C%EA%B2%A0%EB%8A%94%EB%8D%B0-%EA%B7%B8%EB%9E%98%EC%84%9C-%EC%96%B4%EB%96%A4-GAN%EC%9D%B4-%EB%8D%94-%EC%A2%8B%EC%9D%80%EA%B1%B4%EB%8D%B0-How-to-evaluate-GAN
def calculate_fid(origin_image, new_image):
    origin_image = ((origin_image+1)/2).reshape(origin_image.shape[0], -1)
    new_image = ((new_image+1)/2).reshape(new_image.shape[0], -1)
    # calculate mean and covariance statistics
    mu1, sigma1 = origin_image.mean(dim = 0), torch.cov(origin_image).cpu().detach().numpy()
    mu2, sigma2 = new_image.mean(dim = 0), torch.cov(new_image).cpu().detach().numpy()
	# calculate sum squared difference between means
    ssdiff = torch.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
