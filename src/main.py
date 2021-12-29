from model.train import MyTrainer
from utils import log_param, find_latent_space_and_show
from data import get_CelebA_DL
import fire

def run_model(hyperpm):
    data_root = hyperpm['datadir']
    CelebA_DL = get_CelebA_DL(data_root, hyperpm['cudanum'], hyperpm['batchsize'], shuffle=True)

    trainer = MyTrainer(CelebA_DL, hyperpm)
    model = trainer.train()
    del CelebA_DL

    non_shuffle_CelebA_DL = get_CelebA_DL(data_root, hyperpm['cudanum'], hyperpm['batchsize'], shuffle=False)
    find_latent_space_and_show(model, non_shuffle_CelebA_DL, data_root, hyperpm['num_show_images'])



def main(lr = 1e-3,
         nepoch = 20,
         gamma = 15,
         beta = 5,
         cudanum = 0,
         batchsize = 64,
         early_stopping = None,
         num_show_images = 10,
         datadir = "./datasets/"):

    hyperpm = {}

    hyperpm['lr'] = lr
    hyperpm['nepoch'] =nepoch
    hyperpm['cudanum'] = cudanum
    hyperpm['z_dim'] = 128
    hyperpm['gamma'] = gamma
    hyperpm['datadir'] = datadir
    hyperpm['beta'] = beta
    hyperpm['early_stopping'] = early_stopping
    hyperpm['batchsize'] = batchsize
    hyperpm['num_show_images'] = num_show_images
    log_param(hyperpm)

    run_model(hyperpm)



if __name__ == '__main__':
    fire.Fire(main)