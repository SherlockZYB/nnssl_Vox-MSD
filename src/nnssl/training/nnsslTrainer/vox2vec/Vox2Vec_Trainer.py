import os
import random
from itertools import combinations

import numpy as np
import torch
from math import prod

from batchgenerators.utilities.file_and_folder_operations import join
from torch import nn
from typing import *
from tqdm import tqdm
from typing_extensions import override

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans, DynamicArchitecturePlans
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.architectures.pclrv2_architecture import PCRLv2Architecture
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.dataloading.pcrlv2_transform import PCLRv2Transform, Shape3D
from nnssl.ssl_data.dataloading.swin_unetr_transform import SwinUNETRTransform
from nnssl.training.loss.pcrlv2_loss import PCRLv2Loss

from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper
from torch import autocast
from nnssl.utilities.helpers import dummy_context

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import save_json

def select_from_pyramid(
            feature_pyramid: Sequence[torch.Tensor],
            indices: torch.Tensor,
    ) -> torch.Tensor:
        """Select features from feature pyramid by their indices w.r.t. base feature map.

        Args:
            feature_pyramid (Sequence[torch.Tensor]): Sequence of tensors of shapes ``(c_i, h_i, w_i, d_i)``.
            indices (torch.Tensor): tensor of shape ``(n, 3)``

        Returns:
            torch.Tensor: tensor of shape ``(n, \sum_i c_i)``
        """
        feature = []
        # for i,x in enumerate(feature_pyramid):
        #     x = x.moveaxis(0, -1) # 通道转到最后一维，因为每一个点，都会影响到所有的通道
        #     feature_indices = (torch.div(indices, 2 ** i, rounding_mode='trunc')).unbind(1)
        #     scale_feature = x[feature_indices[1:4]] # 同理，不对针对通道进行选取
        #     feature.append(scale_feature)
        # 把以上循环化为一条代码
        return torch.cat(
            [x.moveaxis(0, -1)[(torch.div(indices, 2 ** i, rounding_mode='trunc')).unbind(1)[1:4]] for i, x in enumerate(feature_pyramid)]
            , dim=1)

def sum_pyramid_channels(base_channels: int, num_scales: int):
    return sum(base_channels * 2 ** i for i in range(num_scales))

class Lambda(nn.Module):
    def __init__(self, func, **kwargs):
        super().__init__()

        self.func = func
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs, **self.kwargs)
    
class Vox2VecTrainer(AbstractBaseTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
        # Our network has 5 downsampling stages, we need a minimum size of 2⁵=32 per axis. Although we could set
        # the global_input_size to our standard (160, 160, 160), the orig config *never* upsamples their
        # global crops. We try to scale the config while keeping it reasonable and close to the original.
        global_patch_sizes: tuple[Shape3D] = (
            (96, 96, 96),
            (128, 128, 96),
            (128, 128, 128),
            (160, 160, 128),
            (160, 160, 160),
        ),
        global_input_size: Shape3D = (160, 160, 160),
        local_patch_sizes: tuple[Shape3D] = ((32, 32, 32), (64, 64, 32), (64, 64, 64)),
        local_input_size: Shape3D = (64, 64, 64),
        # orig config
        # global_patch_sizes: tuple[Shape3D] = ((64, 64, 32), (96, 96, 64), (96, 96, 96), (112, 112, 64)),
        # global_input_size: Shape3D = (64, 64, 32),
        # local_patch_sizes: tuple[Shape3D] = ((8, 8, 8), (16, 16, 16), (32, 32, 16), (32, 32, 32)),
        # local_input_size: Shape3D = (16, 16, 16),
        num_locals: int = 6,
        num_mid_stages: int = 4,
        min_IoU: float = 0.3,
    ):
        # We want the dataloader to give us a patch_size big enough, to accommodate for the largest patch size
        # for each axis, while making it possible to have different overlapping volumes and still allow the anatomy
        # to be a large part of the patch
        plan.configurations[configuration_name].patch_size = (180, 180, 180)

        super().__init__(plan, configuration_name, fold, pretrain_json, device)

        self.global_patch_sizes = global_patch_sizes
        self.global_input_size = global_input_size
        self.local_patch_sizes = local_patch_sizes
        self.local_input_size = local_input_size
        self.num_locals = num_locals
        self.num_mid_stages = num_mid_stages
        self.min_IoU = min_IoU

    
    @staticmethod
    def _vox_to_vec(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        feature_pyramid = self.net(patches)
        # 针对一组patches，访问这一组里的所有patch，并获得该组的特征
        return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])

    @override
    def build_architecture_and_adaptation_plan(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int
    ) -> nn.Module:
        architecture = get_network_by_name(
            config_plan,
            "ResEncL",
            num_input_channels,
            num_output_channels,
            deep_supervision=True
        )
        # --------------------- Build associated adaptation plan --------------------- #
        arch_plans = ArchitecturePlans(arch_class_name="ResEncL")
        adapt_plan = AdaptationPlan(
            architecture_plans=arch_plans,
            pretrain_plan=self.plan,
            pretrain_num_input_channels=num_input_channels,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            key_to_encoder="encoder.stages",
            key_to_stem="encoder.stem",
            keys_to_in_proj=("encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0"),
        )
        save_json(adapt_plan.serialize(), self.adaptation_json_plan)
        
        
        embed_dim = sum_pyramid_channels(base_channels=32,num_scales=4)
        proj_dim = 128
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
            # 在proj_head的最后添加了l2 normalization
            Lambda(F.normalize)
        )
            
        return architecture, adapt_plan
    
    @override
    def build_loss(self):
        return None
    
    @override
    def get_dataloaders(self):

        tr_transforms = self.get_training_transforms()
        val_transforms = self.get_validation_transforms()

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size=self.config_plan.patch_size)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(
                self.num_iterations_per_epoch,
                data_loader=dl_tr,
                transform=tr_transforms,
                num_processes=allowed_num_processes,
                num_cached=6,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
            mt_gen_val = LimitedLenWrapper(
                self.num_val_iterations_per_epoch,
                data_loader=dl_val,
                transform=val_transforms,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=3,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
        return mt_gen_train, mt_gen_val

    @override
    def train_step(self, batch: dict) -> dict:

        aug_global_crops_A = batch[
            "aug_global_crops_A"
        ]  # [B,              C, X_global_input_size, Y_global_input_size, Z_global_input_size]
        global_crops_A = batch[
            "global_crops_A"
        ]  # [B,              C, X_global_input_size, Y_global_input_size, Z_global_input_size]
        aug_global_crops_B = batch[
            "aug_global_crops_B"
        ]  # [B,              C, X_global_input_size, Y_global_input_size, Z_global_input_size]
        aug_local_crops = batch[
            "aug_local_crops"
        ]  # [(B*num_locals), C, X_local_input_size,  Y_local_input_size,  Z_local_input_size ]

        aug_global_crops_A = aug_global_crops_A.to(self.device, non_blocking=True)
        global_crops_A = global_crops_A.to(self.device, non_blocking=True)
        aug_global_crops_B = aug_global_crops_B.to(self.device, non_blocking=True)
        aug_local_crops = aug_local_crops.to(self.device, non_blocking=True)
        
        voxels_A=[]
        voxels_B=[]

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            embeds_1 = self.proj_head(self._vox_to_vec(aug_global_crops_A, voxels_A))
            embeds_2 = self.proj_head(self._vox_to_vec(aug_global_crops_B, voxels_B))

            # 最后利用得到的embedings进行对比学习的计算
            temp=self.loss_temp
            logits_11 = torch.matmul(embeds_1, embeds_1.T) / temp
            logits_11.fill_diagonal_(float('-inf'))
            logits_12 = torch.matmul(embeds_1, embeds_2.T) / temp
            logits_22 = torch.matmul(embeds_2, embeds_2.T) / temp
            logits_22.fill_diagonal_(float('-inf'))
            loss_1 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_11, logits_12], dim=1), dim=1))
            loss_2 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_12.T, logits_22], dim=1), dim=1))
            l = (loss_1 + loss_2) / 2
            

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return (
            {"loss": l.detach().cpu().numpy()},
        )
        # return {"loss": np.array(0)}

    def run_training(self):
        try:
            self.on_train_start()

            for epoch in range(self.current_epoch, self.num_epochs):
                self.on_epoch_start()

                self.on_train_epoch_start()
                train_outputs = []


                for batch_id in tqdm(
                    range(self.num_iterations_per_epoch),
                    desc=f"Epoch {epoch}",
                    disable=True if (("LSF_JOBID" in os.environ) or ("SLURM_JOB_ID" in os.environ)) else False,
                ):
                    l = self.train_step(next(self.dataloader_train))
                    train_outputs.append(l)

                    # train_outputs.append(self.train_step(next(self.dataloader_train)))

                self.on_train_epoch_end(train_outputs)

                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs = []
                    for batch_id in range(self.num_val_iterations_per_epoch):
                        val_outputs.append(self.validation_step(next(self.dataloader_val)))
                        # val_outputs.append(self.validation_step(next(self.dataloader_val)))
                    self.on_validation_epoch_end(val_outputs)

                if self.exit_training_flag:
                    # This is a signal that we need to resubmit, so we break the loop and exit gracefully
                    print("Finished last epoch before restart.")
                    self.print_to_log_file("Finished last epoch before restart.")
                    raise KeyboardInterrupt

                self.on_epoch_end()

            self.on_train_end()
        except KeyboardInterrupt:
            print("Keyboard interrupt.")
            self.print_to_log_file("Keyboard interrupt. Exiting gracefully.")
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))
            raise KeyboardInterrupt

    @override
    def validation_step(self, batch: dict) -> dict:
        aug_global_crops_A = batch[
            "aug_global_crops_A"
        ]  # [B,              C, X_global_input_size, Y_global_input_size, Z_global_input_size]
        global_crops_A = batch[
            "global_crops_A"
        ]  # [B,              C, X_global_input_size, Y_global_input_size, Z_global_input_size]
        aug_global_crops_B = batch[
            "aug_global_crops_B"
        ]  # [B,              C, X_global_input_size, Y_global_input_size, Z_global_input_size]
        aug_local_crops = batch[
            "aug_local_crops"
        ]  # [(B*num_locals), C, X_local_input_size,  Y_local_input_size,  Z_local_input_size ]

        aug_global_crops_A = aug_global_crops_A.to(self.device, non_blocking=True)
        global_crops_A = global_crops_A.to(self.device, non_blocking=True)
        aug_global_crops_B = aug_global_crops_B.to(self.device, non_blocking=True)
        aug_local_crops = aug_local_crops.to(self.device, non_blocking=True)

        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                reconstructions_A, embeddings_A, mid_reconstructions_A = self.network(aug_global_crops_A)
                embeddings_B = self.network(aug_global_crops_B, embeddings_only=True)
                local_embeddings = self.network(aug_local_crops, embeddings_only=True)
                rec_l, mid_rec_l, g_sim_l, l_sim_l = self.loss(
                    reconstructions_A,
                    mid_reconstructions_A,
                    global_crops_A,
                    embeddings_A,
                    embeddings_B,
                    local_embeddings,
                )
                l = rec_l + mid_rec_l + g_sim_l + l_sim_l
        return {"loss": l.detach().cpu().numpy()}

    def get_training_transforms(self) -> AbstractTransform:
        tr_transforms = Compose(
            [
                PCLRv2Transform(
                    self.global_patch_sizes,
                    self.global_input_size,
                    self.local_patch_sizes,
                    self.local_input_size,
                    self.num_locals,
                    self.min_IoU,
                ),
                NumpyToTensor(
                    keys=["aug_global_crops_A", "global_crops_A", "aug_global_crops_B", "aug_local_crops"],
                    cast_to="float",
                ),
            ]
        )
        return tr_transforms

    def get_validation_transforms(self) -> AbstractTransform:
        return self.get_training_transforms()

