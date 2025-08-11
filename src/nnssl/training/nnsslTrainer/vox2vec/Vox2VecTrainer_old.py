import os
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
from deprecated import deprecated
from typing_extensions import override
from dataclasses import asdict
from typing import *

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
from nnssl.ssl_data.dataloading.vox2vec_transform import Vox2VecTransform, Shape3D
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from torch import autocast
from nnssl.utilities.helpers import dummy_context
from torch.nn.parallel import DistributedDataParallel as DDP
from batchgenerators.utilities.file_and_folder_operations import join
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import save_json

from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA
import numpy as np


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

class Vox2VecTrainer_old(AbstractBaseTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        # plan.configurations[configuration_name].batch_size = 1
        plan.configurations[configuration_name].patch_size = (180, 180, 180)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)

        self.im_output_folder = os.path.join(self.output_folder, "img_log")
        os.makedirs(self.im_output_folder, exist_ok=True)
        self.save_imgs_every_n_epochs = 200

    def initialize(self):
        # self.recon_dataloader = self.get_qual_recon_dataloader()
        super(Vox2VecTrainer_old, self).initialize()

    @staticmethod
    def _vox_to_vec(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        feature_pyramid = self.net(patches)
        # 针对一组patches，访问这一组里的所有patch，并获得该组的特征
        return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return MAEMSELoss()

    @override
    def build_architecture_and_adaptation_plan(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int
    ) -> nn.Module:
        # ---------------------------- Create architecture --------------------------- #
        architecture = get_network_by_name(
            config_plan,
            "ResEncL",
            num_input_channels,
            num_output_channels,
            deep_supervision = True
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
            mirror_axes
        ) = configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)
        if do_dummy_2d_data_aug:
            self.print_to_log_file("Using dummy 2D data augmentation")

        # ------------------------ Training data augmentations ----------------------- #
        tr_transforms = self.get_training_transforms()

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
                wait_time=0.02,
            )
        return mt_gen_train, mt_gen_val

    def train_step(self, batch: dict) -> dict:
        patch_A = batch["patch_A"]
        aug_patch_A = batch["aug_patch_A"]
        aug_patch_B = batch["aug_patch_B"]
        voxels_A = batch["voxels_A"]
        voxels_B = batch["voxels_B"]
        patch_A = patch_A.to(self.device, non_blocking=True)
        aug_patch_A = aug_patch_A.to(self.device, non_blocking=True)
        aug_patch_B = aug_patch_B.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            embeds_1 = self.proj_head(self._vox_to_vec(aug_patch_A, voxels_A))
            embeds_2 = self.proj_head(self._vox_to_vec(aug_patch_B, voxels_B))
            # del data
            temp=0.1
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
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        patch_A = batch["patch_A"]
        aug_patch_A = batch["aug_patch_A"]
        aug_patch_B = batch["aug_patch_B"]
        patch_A = patch_A.to(self.device, non_blocking=True)
        aug_patch_A = aug_patch_A.to(self.device, non_blocking=True)
        aug_patch_B = aug_patch_B.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            embeds_1 = self.proj_head(self._vox_to_vec(aug_patch_A, voxels_A))
            embeds_2 = self.proj_head(self._vox_to_vec(aug_patch_B, voxels_B))
            # del data
            temp=0.1
            logits_11 = torch.matmul(embeds_1, embeds_1.T) / temp
            logits_11.fill_diagonal_(float('-inf'))
            logits_12 = torch.matmul(embeds_1, embeds_2.T) / temp
            logits_22 = torch.matmul(embeds_2, embeds_2.T) / temp
            logits_22.fill_diagonal_(float('-inf'))
            loss_1 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_11, logits_12], dim=1), dim=1))
            loss_2 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_12.T, logits_22], dim=1), dim=1))
            l = (loss_1 + loss_2) / 2

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
                    train_outputs.append(self.train_step(next(self.dataloader_train)))
                self.on_train_epoch_end(train_outputs)

                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs = []
                    for batch_id in range(self.num_val_iterations_per_epoch):
                        val_batch = next(self.dataloader_val)
                        val_outputs.append(self.validation_step(val_batch))
                    self.on_validation_epoch_end(val_outputs)

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

    def get_training_transforms(
        self,
        patch_size: tuple[Shape3D] = ((160, 160, 160)),
        min_IoU: float = 0.3,
    ) -> AbstractTransform:
        tr_transforms = []
        tr_transforms = Compose(
            [
                Vox2VecTransform(
                    patch_size,
                    min_IoU,
                ),
                NumpyToTensor(
                    keys=["patch_A", "aug_patch_B", "aug_patch_A"],
                    cast_to="float",
                ),
            ]
        )
        return tr_transforms

    def get_validation_transforms(self) -> AbstractTransform:
        return self.get_training_transforms()
