import torch
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


class MultiStepLR_MOM(_LRScheduler):
    """Decays the learning rate and momentum of each parameter group by lr_gamma or mom_gamma once the
    number of epoch reaches one of their respective milestones."""

    def __init__(self, optimizer, lr_milestones, mom_milestones, 
                 lr_gamma=0.1, mom_gamma=0.1, last_epoch=-1, verbose=False):
        self.lr_milestones = Counter(lr_milestones)
        self.mom_milestones = Counter(mom_milestones)
        self.lr_gamma = lr_gamma
        self.mom_gamma = mom_gamma
        self.verbose = verbose
        self.use_beta1 = ('momentum' not in optimizer.defaults) and ('betas' in optimizer.defaults)
        super(MultiStepLR_MOM, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        # Update momentum
        if self.last_epoch in self.mom_milestones:
            for group in self.optimizer.param_groups:
                if self.use_beta1:
                    beta1, beta2 = group['betas']
                    computed_mom = 1 - (1 - beta1) / (self.mom_gamma ** self.mom_milestones[self.last_epoch])
                    computed_mom = np.clip(computed_mom, -1, 1)
                    group['betas'] = (computed_mom, beta2)
                else:
                    computed_mom = 1 - (1 - group['momentum']) / (self.mom_gamma ** self.mom_milestones[self.last_epoch])
                    computed_mom = np.clip(computed_mom, -1, 1)
                    group['momentum'] = computed_mom
            if self.verbose:
                print('Adjusting momentum rate to {:.4e}.'.format(computed_mom))
        
        # Update lr
        if self.last_epoch not in self.lr_milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            if self.verbose:
                computed_lr = self.optimizer.param_groups[0]['lr'] * self.lr_gamma ** self.lr_milestones[self.last_epoch]
                print('Adjusting learning rate to {:.4e}.'.format(computed_lr))
            return [group['lr'] * self.lr_gamma ** self.lr_milestones[self.last_epoch]
                    for group in self.optimizer.param_groups]

