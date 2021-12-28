from model.train import MyTrainer
from utils import log_param, find_latent_space_and_show
from data import get_CelebA_DL
import fire

def run_model(hyperpm):
    data_root = hyperpm['datadir']
    CelebA_DL, CelebA_DS = get_CelebA_DL(data_root, hyperpm['cudanum'], hyperpm['batchsize'])

    trainer = MyTrainer(CelebA_DL, hyperpm)
    model = trainer.train()
    
    find_latent_space_and_show(model, CelebA_DL, data_root, hyperpm['num_show_images'])



def main(in_dim = 784,
         hidden_dim1 = 512,
         hidden_dim2 = 256,
         lr = 1e-3,
         nepoch = 20,
         dropout = 1e-1,
         gamma = 50,
         beta = 10,
         cudanum = 0,
         batchsize = 64,
         num_show_images = 10,
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
    hyperpm['batchsize'] = batchsize
    hyperpm['num_show_images'] = num_show_images
    log_param(hyperpm)

    run_model(hyperpm)



if __name__ == '__main__':
    fire.Fire(main)