import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import torchvision
import models
import utils
import tabulate

sys.path.append("..")
sys.path.append("../..")

import inflation

from torch.optim.swa_utils import AveragedModel, update_bn, SWALR


parser = argparse.ArgumentParser(description='SWA training')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset name (default: CIFAR100)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=1, metavar='N', help='evaluation frequency (default: 1)')
parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')

parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
parser.add_argument('--swa_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swa_lr', type=float, default=0.05, metavar='LR', help='SWA LR (default: 0.05)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')
parser.add_argument('--swa_on_cpu', action='store_true', help='store swa model on cpu flag (default: off)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')


parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam", "ECDSep","adamw"],
        help="optimizer (default: sgd)",
    )


parser.add_argument(
    "--beta1", type=float, default=0.9, help="beta1 parameter (default: 0.9)"
)
parser.add_argument(
    "--beta2", type=float, default=0.999, help="beta 2 parameter (default: 0.999)"
)
parser.add_argument(
    "--eps", type=float, default=1e-8, help="epsilon parameter - (default: 1e-8)"
)

parser.add_argument(
    "--F0", type=float, default=0, help="F0 V parameter - (default: 0)"
)

parser.add_argument(
    "--deltaEn", type=float, default=0.0, help="deltaEn - (default: 0.0)"
)
parser.add_argument(
    "--consEn", type=bool, default=True, help="consEn (default: True)",
)
parser.add_argument(
    "--n_fixed_bounces", type=int, default=1, help="Number of fixed bounces (default: 1)",
)
parser.add_argument(
    "--eta", type=float, default=1.0, help="eta parameter for Ruthless - (default: 1.0)"
)
parser.add_argument(
    "--nu", type=float, default=0.00001, help="nu parameter for Generalized bounces (default: 0.00001)"
)


args = parser.parse_args()


### Initialization of optimizers

optimizers = {            

            "ECDSep": (inflation.ECDSep, {
            "F0": args.F0,
            "nu": args.nu, 
            "deltaEn": args.deltaEn,
            "consEn": args.consEn, 
            "weight_decay": args.wd,
            "eta": args.eta
            }),

        "sgd": (torch.optim.SGD, {
            "momentum": args.momentum
            },),
        "adam": (torch.optim.Adam, {
            "betas": (args.beta1, args.beta2),
            "eps": args.eps
            }),
        "adamw": (torch.optim.AdamW, {
            "betas": (args.beta1, args.beta2),
            "eps": args.eps
            }),
    }

#### Starting the evolution
print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

print('Loading dataset %s from %s' % (args.dataset, args.data_path))
ds = getattr(torchvision.datasets, args.dataset)
path = os.path.join(args.data_path, args.dataset.lower())
train_set = ds(path, train=True, download=True, transform=model_cfg.transform_train)
test_set = ds(path, train=False, download=True, transform=model_cfg.transform_test)
loaders = {
    'train': torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    ),
    'test': torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
}
num_classes = max(train_set.targets) + 1

print('Preparing model')
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
if len(args.gpu_ids) > 1:
    model = torch.nn.DataParallel(model, args.gpu_ids)
    print("Using gpu_ids {}".format(args.gpu_ids))
    device = torch.device(model.device_ids[0])
else:
    device = torch.device("cuda:0")
model.to(device)

if args.swa:
    print('SWA training')
    if args.swa_on_cpu:
        swa_model = AveragedModel(model, device=torch.device('cpu'))
    else:
        swa_model = AveragedModel(model)
else:
    print('Not averaging in this experiment')


def schedule(epoch):
    t = (epoch) / (args.swa_start if args.swa else args.epochs)
    lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor


criterion = F.cross_entropy

opt_class, opt_kwargs = optimizers[args.optimizer]
opt_kwargs.update({"lr": args.lr_init, "weight_decay": args.wd})
optimizer = opt_class(model.parameters(), **opt_kwargs)



schedulers = False

start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if args.swa:
        swa_state_dict = checkpoint['swa_state_dict']
        if swa_state_dict is not None:
            swa_model.load_state_dict(swa_state_dict)

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'best_te_acc', 'best_tr_loss', 'time']
if args.swa:
    columns = columns[:-1] + ['swa_te_loss', 'swa_te_acc', 'best_swa_te_acc'] + columns[-1:]
    swa_res = {'loss': None, 'accuracy': None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    state_dict=model.state_dict(),
    swa_state_dict=swa_model.state_dict() if args.swa else None,
    optimizer=optimizer.state_dict()
)


best_te_acc = 0.0
best_tr_loss = 1e20

if args.swa :
    best_swa_te_acc = 0.0

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer, device=device)
    if train_res['loss'] < best_tr_loss:
                best_tr_loss = train_res['loss']

    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
        test_res = utils.eval(loaders['test'], model, criterion, device=device)
        if test_res['accuracy'] > best_te_acc:
            best_te_acc = test_res['accuracy']
        
    else:
        test_res = {'loss': None, 'accuracy': None}

    lr = optimizer.param_groups[0]['lr']
    
    if schedulers:
        if args.swa and (epoch + 1) >= args.swa_start:
            swa_scheduler.step()
        else:
            scheduler.step()
    
    if args.swa and (epoch + 1) >= args.swa_start and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
        swa_model.update_parameters(model)

        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            update_bn(loaders['train'], swa_model, device=torch.device('cuda'))
            if args.swa_on_cpu:
                # moving swa_model to gpu for evaluation
                model = model.cpu()
                swa_model = swa_model.to(device)
            print("SWA eval")
            swa_res = utils.eval(loaders['test'], swa_model, criterion, device=device)
            if args.swa_on_cpu:
                model = model.to(device)
                swa_model = swa_model.cpu()

            if swa_res['accuracy'] > best_swa_te_acc:
                best_swa_te_acc = swa_res['accuracy']
            
        else:
            swa_res = {'loss': None, 'accuracy': None}

    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            state_dict=model.state_dict(),
            swa_state_dict=swa_model.state_dict() if args.swa else None,
            optimizer=optimizer.state_dict()
        )

    time_ep = time.time() - time_ep
    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'], best_te_acc, best_tr_loss, time_ep]
    if args.swa:
        values = values[:-1] + [swa_res['loss'], swa_res['accuracy'], best_swa_te_acc] + values[-1:]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)


if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        state_dict=model.state_dict(),
        swa_state_dict=swa_model.state_dict() if args.swa else None,
        optimizer=optimizer.state_dict()
    )

print("Best SWA test accuracy: ", best_swa_te_acc)

