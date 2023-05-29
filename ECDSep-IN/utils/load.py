import os
import torch
import numpy as np
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F

from utils import custom_datasets

from models import tinyimagenet_resnet
from models import imagenet_resnet

import sys
sys.path.append("..")
sys.path.append("../..")
import inflation

def device(gpu, tpu=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    use_cuda = torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


def MSELoss(output, target, reduction='mean'):
    num_classes = output.size(1)
    labels = F.one_hot(target, num_classes=num_classes)
    if reduction is 'mean':
        return torch.mean(torch.sum((output - labels)**2, dim=1))/2
    elif reduction is 'sum':
        return torch.sum((output - labels)**2)/2
    elif reduction is None:
        return ((output - labels)**2)/2
    else:
        raise ValueError(reduction + " is not valid")


def loss(name):
    losses = {
        "mse": MSELoss,
        "ce": torch.nn.CrossEntropyLoss()
    }
    return losses[name]


def dimension(dataset):
    if dataset == "mnist":
        input_shape, num_classes = (1, 28, 28), 10
    if dataset == "tiny-imagenet":
        input_shape, num_classes = (3, 64, 64), 200
    if dataset == "imagenet":
        input_shape, num_classes = (3, 224, 224), 1000
    return input_shape, num_classes


def get_transform(size, padding, mean, std, preprocess):
    transform = []
    if preprocess:
        transform.append(transforms.RandomCrop(size=size, padding=padding))
        transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean, std))
    return transforms.Compose(transform)


def dataloader(
    dataset, batch_size, train, workers, length=None, datadir="Data", tpu=False, 
    shuffle=True, data_augment=True,
):
    # Dataset
    if dataset == "mnist":
        mean, std = (0.1307,), (0.3081,)
        transform = get_transform(
            size=28, padding=0, mean=mean, std=std, preprocess=False
        )
        dataset = datasets.MNIST(
            datadir, train=train, download=True, transform=transform
        )
    if dataset == "tiny-imagenet":
        mean, std = (0.480, 0.448, 0.397), (0.276, 0.269, 0.282)
        transform = get_transform(
            size=64, padding=4, mean=mean, std=std, preprocess=train
        )
        dataset = custom_datasets.TINYIMAGENET(
            datadir, train=train, download=True, transform=transform
        )
    if dataset == "imagenet":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if train and data_augment:
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224), #, scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        folder = f"{datadir}/imagenet_raw/{'train' if train else 'val'}"
        dataset = datasets.ImageFolder(folder, transform=transform)

    # Dataloader
    shuffle = (train is True) and shuffle
    if length is not None:
        indices = torch.randperm(len(dataset))[:length]
        dataset = torch.utils.data.Subset(dataset, indices)

    sampler = None
    kwargs = {}
    if torch.cuda.is_available():
        kwargs = {"num_workers": workers, "pin_memory": True}
    elif tpu:
        import torch_xla.core.xla_model as xm

        kwargs = {"num_workers": workers}  
        if xm.xrt_world_size() > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=shuffle,
            )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False if sampler else shuffle,
        sampler=sampler,
        **kwargs,
    )

    return dataloader


def model(model_architecture, model_class):
    
    tinyimagenet_models = {
        "resnet18": tinyimagenet_resnet.resnet18,
            }
    imagenet_models = {
        "resnet18": imagenet_resnet.resnet18,
    }
    models = {
        "tinyimagenet": tinyimagenet_models,
        "imagenet": imagenet_models,
    }
    return models[model_class][model_architecture]


def optimizer(optimizer, args):
    optimizers = {
            "ECDSep": (inflation.ECDSep, {
            "eps1": args.eps1, 
            "eps2": args.eps2, 
            "F0": args.F0,
            "nu": args.nu, 
            "deltaEn": args.deltaEn,
            "consEn": args.consEn, 
            "weight_decay": args.wd,
            "eta": args.eta
            }),

        "sgd": (optim.SGD, {}),
        "momentum": (optim.SGD, {
            "momentum": args.momentum, 
            "dampening": args.dampening, 
            "nesterov": args.nesterov
            },),
        "adam": (optim.Adam, {
            "betas": (args.beta1, args.beta2),
            "eps": args.eps
            }),
        "adamw": (optim.AdamW, {
            "betas": (args.beta1, args.beta2),
            "eps": args.eps
            }),
        "rms": (optim.RMSprop, {}),
    }
    return optimizers[optimizer]