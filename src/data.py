from torchvision import datasets, transforms
import torch

def get_CelebA_DL(data_root, cudanum, batch_size):
    transform_list = [ # use aligned datasets
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ]

    device = f'cuda:{cudanum}' if torch.cuda.is_available() else 'cpu'
    transform_list = transforms.Compose(transform_list)
    CelebA_ds = datasets.ImageFolder(root=data_root, transform=transform_list)
    CelebA_dl = torch.utils.data.DataLoader(CelebA_ds, batch_size = batch_size, shuffle=True)
    return CelebA_dl

