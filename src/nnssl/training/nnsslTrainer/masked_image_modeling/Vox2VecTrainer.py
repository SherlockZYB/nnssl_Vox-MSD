import os
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
from deprecated import deprecated
from typing_extensions import override
from dataclasses import asdict


import torch
from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans, DynamicArchitecturePlans
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.architectures.get_network_from_plan import get_network_from_plans
from nnssl.data.nnsslFilter.iqs_filter import OpenMindIQSFilter
from nnssl.data.nnsslFilter.modality_filter import ModalityFilter
from nnssl.data.raw_dataset import Collection
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.configure_basic_dummyDA import configure_rotation_dummyDA_mirroring_and_inital_patch_size
from nnssl.ssl_data.data_augmentation.transforms_for_dummy_2d import Convert2DTo3DTransform, Convert3DTo2DTransform
from nnssl.ssl_data.dataloading.data_loader_3d import nnsslIndexableCenterCropDataLoader3D
from nnssl.ssl_data.dataloading.indexable_dataloader import IndexableSingleThreadedAugmenter
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper

from nnssl.training.loss.mse_loss import MAEMSELoss, LossMaskMSELoss
from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from torch import nn
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from torch import autocast
from nnssl.utilities.helpers import dummy_context
from torch.nn.parallel import DistributedDataParallel as DDP
from batchgenerators.utilities.file_and_folder_operations import join
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import save_json
from nnssl.architectures.dinomae_architecture import DinoMAEArchitecture
from nnssl.training.lr_scheduler.polylr import PolyLRScheduler, CosineAnnealingLRScheduler
from nnssl.training.loss.dino_loss import DINOLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from nnssl.ssl_data.dataloading.vox2vec_transform_ import Vox2VecTransform, Shape3D
import torch.nn.functional as F
import torch.multiprocessing as mp

from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA
import numpy as np
import copy
from typing import *

class Lambda(nn.Module):
    def __init__(self, func, **kwargs):
        super().__init__()

        self.func = func
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs, **self.kwargs)
    
def sum_pyramid_channels(base_channels: int, num_scales: int):
    return sum(base_channels * 2 ** i for i in range(num_scales))

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
        for i,x in enumerate(feature_pyramid):
            x = x.moveaxis(0, -1) # 通道转到最后一维，因为每一个点，都会影响到所有的通道
            feature_indices = (torch.div(indices, 2 ** i, rounding_mode='trunc')).int()
            scale_feature = x[feature_indices[0],feature_indices[1],feature_indices[2]] # 同理，不对针对通道进行选取
            feature.append(scale_feature)
        # 把以上循环化为一条代码
        return torch.cat(feature).unsqueeze(0)
        # return torch.cat(
        #     [x.moveaxis(0, -1)[(torch.div(indices, 2 ** i, rounding_mode='trunc')).unbind(1)[1:4]] for i, x in enumerate(feature_pyramid)]
            # , dim=1)
        
        
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0.):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

# 单纯用MAE可能学习全局信息的能力较差，而MAE确实又被证明学习局部信息的能力很强，因此考虑在MAE本身的基础上，加上DINO（DINO对batch size要求更低）
def create_blocky_mask(tensor_size, block_size, sparsity_factor=0.75, rng_seed: None | int = None) -> torch.Tensor:
    """
    Create the smallest binary mask for the encoder by choosing a percentage of pixels at that resolution..

    :param tensor_size: Tuple of the dimensions of the tensor (height, width, depth).
    :param block_size: Size of the block to be masked (set to 0) in the smaller mask.
    :return: A binary mask tensor.
    """
    # Calculate the size of the smaller mask
    small_mask_size = tuple(size // block_size for size in tensor_size)

    # Create the smaller mask
    flat_mask = torch.ones(np.prod(small_mask_size))
    n_masked = int(sparsity_factor * flat_mask.shape[0])
    if rng_seed is None:
        mask_indices = torch.randperm(flat_mask.shape[0])[:n_masked]
    else:
        gen = torch.Generator.manual_seed(rng_seed)
        mask_indices = torch.randperm(flat_mask.shape[0], generator=gen)[:n_masked]
    flat_mask[mask_indices] = 0
    small_mask = torch.reshape(flat_mask, small_mask_size)
    return small_mask


class Vox2VecTrainer(AbstractBaseTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        # plan.configurations[configuration_name].batch_size = 1
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.config_plan.patch_size = (160, 160, 160)
        self.mask_percentage: float = 0.75

        self.im_output_folder = os.path.join(self.output_folder, "img_log")
        os.makedirs(self.im_output_folder, exist_ok=True)
        self.save_imgs_every_n_epochs = 200
        self.dino_loss = DINOLoss(
                512,
                ncrops=2,  # total number of crops = 2 global crops + local_crops_number
                warmup_teacher_temp=0.04,
                teacher_temp=0.07,
                warmup_teacher_temp_epochs=20,
                nepochs=1000,
            ).cuda()
        self.momentum_schedule = cosine_scheduler(0.996, 1, 1000, 250)
        self.features_per_stage = [32, 64, 128, 256, 320, 320]

    def initialize(self):
        # self.recon_dataloader = self.get_qual_recon_dataloader()
        self.initial_lr = 1e-3 # ！！！这里为了dinomae的正常训练，调小了学习率_zff
        super(Vox2VecTrainer, self).initialize()
        self.ema_network, _ = self.build_architecture_and_adaptation_plan(
                self.config_plan, self.num_input_channels, self.num_output_channels
            )
        self.ema_network.to(self.device)
        embed_dim = 1120
        proj_dim = 512
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
        self.proj_head.to(self.device)
        # self.num_iterations_per_epoch = 1
        

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return MAEMSELoss()
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )
        lr_scheduler = CosineAnnealingLRScheduler(optimizer, self.initial_lr, self.num_epochs, 1e-5)
        return optimizer, lr_scheduler

    @override
    def build_architecture_and_adaptation_plan(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int
    ) -> nn.Module:
        # ---------------------------- Create architecture --------------------------- #
        arch = get_network_by_name(
            config_plan,
            "ResEncL",
            num_input_channels,
            num_output_channels,
        )
        # --------------------- Build associated adaptation plan --------------------- #
        architecture = DinoMAEArchitecture(arch, arch.encoder.output_channels)
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
        return architecture, adapt_plan

    def get_dataloaders(self):
        """
        Dataloader creation is very different depending on the use-case of training.
        This method has to be implemneted for other use-cases aside from MAE more specifically."""
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.config_plan.patch_size
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)
        if do_dummy_2d_data_aug:
            self.print_to_log_file("Using dummy 2D data augmentation")

        # ------------------------ Training data augmentations ----------------------- #
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            mirror_axes,
            do_dummy_2d_data_aug,
            order_resampling_data=3,
            order_resampling_seg=1,
            use_mask_for_norm=self.config_plan.use_mask_for_norm,
        )

        # ----------------------- Validation data augmentations ---------------------- #
        val_transforms = self.get_validation_transforms()

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size)

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
                wait_time=0.02
            )
        return mt_gen_train, mt_gen_val

    def _vox_to_vec(self, feature_pyramid: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        # 这里默认batch size为1，所以用x[0]。不然就得改，给voxels加上batch维度，有点麻烦，现在怎么简单怎么来
        voxel_feature = torch.cat([select_from_pyramid([x[0] for x in feature_pyramid], v) for j, v in enumerate(voxels)])
        return voxel_feature
    
    def train_step(self, batch: dict) -> dict:
        data_s = batch["aug_patch_A"]
        voxel_s = batch["voxel_A"]
        data_t = batch["aug_patch_B"]
        voxel_t = batch["voxel_B"]
        
        
        data_s = data_s.to(self.device, non_blocking=True)
        data_t = data_t.to(self.device, non_blocking=True)
        
        
        self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            feat_s, output_s = self.network(data_s)
            feat_t, output_t = self.ema_network(data_t)
            
            s_concat_feature = self._vox_to_vec(feat_s, voxel_s)
            t_concat_feature = self._vox_to_vec(feat_t, voxel_t) 
            proj_feat_s = self.proj_head(s_concat_feature)
            proj_feat_t = self.proj_head(t_concat_feature)
            
            dino_loss = self.dino_loss(proj_feat_s, proj_feat_t, self.current_epoch)
            # rec_loss = self.loss(output_s, label_s, mask_s)
            # del data
            l = dino_loss
        
        # for name, param in self.network.named_parameters():
        #         if param.requires_grad and param.grad is not None:
        #             has_nan = torch.isnan(param.grad).any()
        #             has_inf = torch.isinf(param.grad).any()
        #             if has_nan or has_inf:
        #                 print(f"参数 '{name}' 的梯度包含无效值:")
        #                 print(f"  - NaN: {has_nan.item()}")
        #                 print(f"  - Inf: {has_inf.item()}")
                        
        #                 # 深入追踪到该参数的计算图
        #                 print("相关模块:")
        #                 for n, mod in self.network.named_modules():
        #                     if name.startswith(n):
        #                         print(f"  - {mod.__class__.__name__}[{n}]")
                                
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
        # 对teacher进行EMA更新
        with torch.no_grad():
            m = self.momentum_schedule[self.current_epoch]  # momentum parameter
            for param_q, param_k in zip(self.network.parameters(), self.ema_network.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                    
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data_s = batch["aug_patch_A"]
        voxel_s = batch["voxel_A"]
        data_t = batch["aug_patch_B"]
        voxel_t = batch["voxel_B"]
        
        
        data_s = data_s.to(self.device, non_blocking=True)
        data_t = data_t.to(self.device, non_blocking=True)
        
        
        self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            feat_s, output_s = self.network(data_s)
            feat_t, output_t = self.ema_network(data_t)
            
            s_concat_feature = self._vox_to_vec(feat_s, voxel_s)
            t_concat_feature = self._vox_to_vec(feat_t, voxel_t) 
            proj_feat_s = self.proj_head(s_concat_feature)
            proj_feat_t = self.proj_head(t_concat_feature)
            
            dino_loss = self.dino_loss(proj_feat_s, proj_feat_t, self.current_epoch)
            # rec_loss = self.loss(output_s, label_s, mask_s)
            # del data
            l = dino_loss
            
        return {"loss": l.detach().cpu().numpy()}

    @deprecated
    @staticmethod
    def rescale_images(
        img_arr: torch.Tensor, recon_arr: torch.Tensor, full_img_min: float, full_img_max: float
    ) -> np.ndarray:
        img_arr = (img_arr - full_img_min) / (full_img_max - full_img_min)
        rec_arr = (recon_arr - full_img_min) / (full_img_max - full_img_min)
        return img_arr, rec_arr

    def log_img_volume(
        self, img: np.ndarray | torch.Tensor, meta_info: dict, filename: str, dtype: np.dtype = np.float32
    ):
        """Logs a 3D numpy array given the meta info to output folder with filename for visual inspection"""
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        img = img.squeeze().astype(dtype)
        sitk_img: sitk.Image = sitk.GetImageFromArray(img)
        sitk_img.SetSpacing(meta_info["sitk_stuff"]["spacing"])
        sitk_img.SetOrigin(meta_info["sitk_stuff"]["origin"])
        sitk_img.SetDirection(meta_info["sitk_stuff"]["direction"])
        sitk.WriteImage(sitk_img, os.path.join(self.im_output_folder, filename))

    def get_qual_recon_dataloader(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be

        # ----------------------- Validation data augmentations ---------------------- #
        val_transforms = self.get_validation_transforms()
        dl_val = self.get_centercrop_val_dataloader()

        mt_gen_val = IndexableSingleThreadedAugmenter(dl_val, val_transforms)
        return mt_gen_val

    def get_centercrop_val_dataloader(self):
        """Returns a centercropped dataloader."""
        _, dataset_val = self.get_tr_and_val_datasets()

        dl_val = nnsslIndexableCenterCropDataLoader3D(
            dataset_val,
            1,
            self.config_plan.patch_size,
            self.config_plan.patch_size,
            sampling_probabilities=None,
            pad_sides=None,
            max_samples=25,
        )
        return dl_val

    def run_training(self):
        try:
            # torch.autograd.set_detect_anomaly(True)
            self.on_train_start()
            
            # 在这里网络才加载好了权重，因此在这里进行ema_network的创造
            self.ema_network.load_state_dict(self.network.state_dict())
            print("ema_network is copied from network")
            # there is no backpropagation through the teacher, so no need for gradients
            for p in self.ema_network.parameters():
                p.requires_grad = False
                
            for epoch in range(self.current_epoch, self.num_epochs):
                self.on_epoch_start()

                self.on_train_epoch_start()
                train_outputs = []
                for batch_id in tqdm(
                    range(self.num_iterations_per_epoch),
                    desc=f"Epoch {epoch}",
                    disable=True if (("LSF_JOBID" in os.environ) or ("SLURM_JOB_ID" in os.environ)) else False,
                ):
                    train_outputs.append(self.train_step(next(self.dataloader_train)))
                self.on_train_epoch_end(train_outputs)

                with torch.no_grad():
                    self.on_validation_epoch_start()
                    # val_outputs = []
                    # for batch_id in range(self.num_val_iterations_per_epoch):
                    #     val_batch = next(self.dataloader_val)
                    #     val_outputs.append(self.validation_step(val_batch))
                    self.on_validation_epoch_end(train_outputs)

                self.on_epoch_end()
                if self.exit_training_flag:
                    print("Finished last epoch before restart.")
                    self.print_to_log_file("Finished last epoch before restart.")
                    raise KeyboardInterrupt

            self.on_train_end()
        except KeyboardInterrupt:
            self.print_to_log_file("Keyboard interrupt. Exiting gracefully.")
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))
            raise KeyboardInterrupt

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: dict,
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        order_resampling_data: int = 3,
        order_resampling_seg: int = 1,
        border_val_seg: int = -1,
        use_mask_for_norm: List[bool] = None,
    ) -> AbstractTransform:
        tr_transforms = Compose(
            [
                Vox2VecTransform(
                    patch_size= (160, 160, 160),
                    min_IoU=0.3
                ),
                NumpyToTensor(
                    keys=["aug_patch_A", "aug_patch_B", "patch_A", "patch_B","voxel_A","voxel_B"],
                    cast_to="float",
                ),
            ]
        )
        return tr_transforms

    @staticmethod
    def get_validation_transforms() -> AbstractTransform:
        val_transforms = Compose(
            [
                Vox2VecTransform(
                    patch_size= (160, 160, 160),
                    min_IoU=0.3
                ),
                NumpyToTensor(
                    keys=["aug_patch_A", "aug_patch_B", "patch_A", "patch_B","voxel_A","voxel_B","mask_A","mask_B"],
                    cast_to="float",
                ),
            ]
        )
        return val_transforms