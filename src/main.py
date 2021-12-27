from loguru import logger
from model.train import MyTrainer
from utils import log_param
from data import get_CelebA_DL
from torchvision import datasets, transforms
import fire
import torch

from torchvision.utils import save_image

def reconstruct_random_image(model):
    pass

def generate_random_image(model):
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else 'cpu'
        X_lim = torch.linspace(-2.5, 2.5, 8)
        Y_lim = torch.linspace(-2.5, 2.5, 8)
        grid_X, grid_Y = torch.meshgrid(X_lim, Y_lim, indexing = 'ij')

        z = torch.hstack([grid_X.reshape(-1, 1), grid_Y.reshape(-1, 1)]).to(device)
        sample = model.decoder(z)

        save_image(sample.view(64, 1, 28, 28), './sample3.png')

def run_model(hyperpm):
    CelebA_DL = get_CelebA_DL(hyperpm['datadir'])

    trainer = MyTrainer(CelebA_DL, hyperpm)
    model = trainer.train()
    
    generate_random_image(model)

def main(in_dim = 784,
         hidden_dim1 = 512,
         hidden_dim2 = 256,
         lr = 1e-3,
         nepoch = 20,
         dropout = 1e-1,
         gamma = 5,
         beta = 1,
         cudanum = 0,
         datadir = "./datasets/"):

    hyperpm = {}

    hyperpm['in_dim'] = in_dim
    hyperpm['hidden_dim1'] = hidden_dim1
    hyperpm['hidden_dim2'] = hidden_dim2
    hyperpm['lr'] = lr
    hyperpm['nepoch'] =nepoch
    hyperpm['cudanum'] = cudanum
    hyperpm['dropout_rate'] = dropout
    hyperpm['z_dim'] = 128
    hyperpm['gamma'] = gamma
    hyperpm['datadir'] = datadir
    hyperpm['beta'] = beta
    log_param(hyperpm)

    run_model(hyperpm)



if __name__ == '__main__':
    fire.Fire(main)