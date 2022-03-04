# Dependencies: LARS, byol_pytorch
import argparse
import os
import torch
from byol_pytorch import BYOL
from byol_pytorch.byol_pytorch import RandomApply
from torchvision import models
from torchvision import transforms as T
import torchvision.transforms.functional as F
import medmnist.dataset
from medmnist.dataset import DermaMNIST, PathMNIST, RetinaMNIST

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='BYOL for RetinaMNIST')
parser.add_argument('data', choices=['derma', 'blood', 'pathology', 'retina'], help='Dataset Type within MedMNIST V2 for self-supervised pretraining')
parser.add_argument('--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

args = parser.parse_args()


model = models.__dict__[args.arch](pretrained=args.pretrained)

AUG = torch.nn.Sequential(
            T.RandomRotation(degrees=180, interpolation=F.InterpolationMode.LANCZOS),
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            # T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.RandomResizedCrop(size=(28, 28), scale=(0.8, 1.0)),
            # T.Normalize(
            #     mean=torch.tensor([0.485, 0.456, 0.406]),
            #     std=torch.tensor([0.229, 0.224, 0.225])),
        )

learner = BYOL(
    model,
    image_size = 28,
    hidden_layer = -2,
    augment_fn = AUG
)
opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
ds_ss_train = RetinaMNIST(split='train', download=True, root='./data/')
ds_ss_val = RetinaMNIST(split='val', download=True, root='./data/')
# if args.distributed:
#     train_sampler = torch.utils.data.distributed.DistributedSampler(ds_ss_train)
# else:
#     train_sampler = None
loader_ss_train = torch.utils.data.Dataloader(ds_ss_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
loader_ss_val = torch.utils.data.Dataloader(ds_ss_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

for _ in range(100):
    loss = learner(images)
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_moving_average() # update moving average of target encoder

# save your improved network
torch.save(model.state_dict(), './improved-net.pt')