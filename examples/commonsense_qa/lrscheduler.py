from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR
from ignite.handlers import create_lr_scheduler_with_warmup

def build_lrscheduler(optimizer, trainer_cfg:DictConfig,steps_per_epoch:int):
    if trainer_cfg.lrscheduler == "cosine":
        lrscheduler = CosineAnnealingLR(
            optimizer, T_max=(trainer_cfg.epochs-trainer_cfg.warmup_epochs)*steps_per_epoch, 
            eta_min=trainer_cfg.warmup_start # set annealing end lr to warmup start lr
        )
        lrscheduler = create_lr_scheduler_with_warmup(
            lrscheduler,warmup_start_value=trainer_cfg.warmup_start,
            warmup_duration=trainer_cfg.warmup_epochs*steps_per_epoch
        ) 
    else:
        raise NotImplementedError
    return lrscheduler
