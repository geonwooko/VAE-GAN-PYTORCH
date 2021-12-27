import torch
import torch.nn.functional as F
import numpy as np
import json
from os.path import exists

from .vae_gan import VAE_GAN
from .utils import save_model, load_model
from torchvision.utils import save_image
from tqdm import tqdm
from loguru import logger

class MyTrainer:
    def __init__(self, DL, hyperpm):
        self.device = f'cuda:{hyperpm["cudanum"]}' if torch.cuda.is_available() else 'cpu'
        self.DL = DL
        self.hyperpm = hyperpm
        self.gamma = hyperpm['gamma']
        self.beta = hyperpm['beta']

        self.model = VAE_GAN(hyperpm).to(self.device)
        # self.discriminator = Discriminator().to(self.device)

        self.loss_dict = {}
        self.loss_dict['KLD'] = []
        self.loss_dict['MSE'] = []
        self.loss_dict['GAN'] = []
        self.loss_dict['loss_encoder'] = []
        self.loss_dict['loss_decoder'] = []
        self.loss_dict['loss_discriminator'] = []

        self.encoder_optimizer = torch.optim.Adam(self.model.encoder.parameters(), lr = self.hyperpm['lr'])
        self.decoder_optimizer = torch.optim.Adam(self.model.decoder.parameters(), lr=self.hyperpm['lr'])

        self.discriminator_optimizer = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.hyperpm['lr'])

    def get_loss(self, mean, logvar, disc_X_real, disc_X_prior, sim_X_real, sim_X_recon):

        KLD =  - 0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        MSE = torch.mean((sim_X_real - sim_X_recon).pow(2))
        GAN = torch.mean(-torch.log(disc_X_real) -torch.log(1 - disc_X_prior))

        loss_encoder = self.beta * KLD + MSE
        loss_decoder = self.gamma * MSE - GAN
        loss_discriminator = GAN

        self.loss_dict['KLD'].append(KLD.item())
        self.loss_dict['MSE'].append(MSE.item())
        self.loss_dict['GAN'].append(GAN.item())
        self.loss_dict['loss_encoder'].append(loss_encoder.item())
        self.loss_dict['loss_decoder'].append(loss_decoder.item())
        self.loss_dict['loss_discriminator'].append(loss_discriminator.item())

        return KLD, MSE, GAN, loss_encoder, loss_decoder, loss_discriminator


    def train(self):
        with torch.autograd.set_detect_anomaly(True):

            epoch_pbar = tqdm(range(self.hyperpm['nepoch']), position = 0, leave=False, desc='epoch')
            for epoch in epoch_pbar:
                # Save file 존재할때 불러옴
                if exists(f'./result/parameters/VAE_GAN_decoder_{epoch}.pth'):
                    load_model(epoch, self.model, self.device)
                    continue

                losses = []
                batch_pbar = tqdm(self.DL, position = 1, leave =False, desc='batch')
                for batch_idx, (X, _) in enumerate(batch_pbar):

                    # Train discriminator
                    X = X.to(self.device)
                    mean, logvar, X_recon, X_prior, disc_X_real, disc_X_prior, sim_X_real, sim_X_recon = self.model(X)

                    # Discriminate BCE(NLL)
                    KLD, MSE, GAN, loss_encoder, loss_decoder, loss_discriminator = self.get_loss(mean, logvar, disc_X_real, disc_X_prior, sim_X_real, sim_X_recon)

                    self.model.zero_grad()
                    # encoder
                    loss_encoder.backward(retain_graph=True)  # someone likes to clamp the grad here: [p.grad.data.clamp_(-1,1) for p in net.encoder.parameters()]
                    self.model.decoder.zero_grad()
                    self.model.discriminator.zero_grad()

                    loss_decoder.backward(retain_graph=True)
                    self.model.discriminator.zero_grad()
                    loss_discriminator.backward()
                    self.encoder_optimizer.step()
                    self.decoder_optimizer.step()
                    self.discriminator_optimizer.step()


                    if (batch_idx % 100 == 0):
                        with open(f"./result/loss/{epoch}_{batch_idx}.json", "w", encoding='utf8') as f:
                            json.dump(self.loss_dict, f)

                    batch_pbar.write(
                        f'Epoch : {batch_idx}/{len(batch_pbar)} loss_encoder : {loss_encoder:.2f}    loss_decoder : {loss_decoder:.2f}   loss_discriminator : {loss_discriminator:.2f}')
                    batch_pbar.write(
                        f'KLD : {KLD}   MSE : {MSE} GAN : {GAN}'
                    )
                    batch_pbar.update()

                # Save the parameters
                save_model(epoch, self.model, self.device)

                # Show the result from random generation
                with torch.no_grad():
                    sample_prior = torch.randn(32, 128).to(self.device)
                    random_generated_images = self.model.decoder(sample_prior)
                    random_generated_images = (random_generated_images + 1) / 2
                    file_name = f"result_{epoch}"
                    save_image(random_generated_images, f"./result/{file_name}.png")

            return self.model



