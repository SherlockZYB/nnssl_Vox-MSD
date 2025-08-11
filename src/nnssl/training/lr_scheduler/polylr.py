from torch.optim.lr_scheduler import _LRScheduler
import math


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
            
class CosineAnnealingLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,                # 绑定的优化器
        initial_lr: float,       # 初始学习率（最大值）
        max_steps: int,           # 总训练步数（一个完整余弦周期）
        eta_min: float = 1e-5,       # 学习率最小值（默认0）
        current_step: int = None  # 当前步数（用于恢复训练）
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.eta_min = eta_min
        self.max_steps = max_steps
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        # 余弦退火公式 [1,2,7](@ref)
        cos_factor = (1 + math.cos(math.pi * current_step / self.max_steps))
        new_lr = self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * cos_factor
        
        # 更新优化器中所有参数组的学习率
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr


class ContinuedPolyLRSchedulerWithWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer,
        start_epoch: int,
        initial_lr,
        warmup_lr,
        warmup_epochs,
        total_epochs,
        final_lr,
        last_epoch=-1,
        exponent: float = 0.9,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.warmup_lr = warmup_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.start_epoch: int = start_epoch
        self.final_lr = final_lr
        self.exponent = exponent
        self.ctr = 0
        super(ContinuedPolyLRSchedulerWithWarmup, self).__init__(optimizer, last_epoch)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        if current_step < (self.warmup_epochs + self.start_epoch):
            new_lr = self.warmup_lr + self.initial_lr * (
                (max(0, current_step - self.start_epoch)) / (self.warmup_epochs)
            )
        else:
            decay_steps = self.total_epochs - self.start_epoch - self.warmup_epochs
            adjusted_step = current_step - self.start_epoch - self.warmup_epochs
            new_lr = self.final_lr + (self.initial_lr - self.final_lr) * (
                (1 - (adjusted_step / decay_steps)) ** self.exponent
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
