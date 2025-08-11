import torch
from torch import nn
from torch._dynamo import OptimizedModule
from typing import *

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.architectures.evaMAE_module_VoxMSD import EvaMAE
from torch import autocast
from nnssl.utilities.helpers import dummy_context
from nnssl.experiment_planning.experiment_planners.plan import Plan
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from nnssl.ssl_data.dataloading.vox2vec_transform_eva import Vox2VecTransform, Shape3D
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer
from nnssl.training.lr_scheduler.warmup import Lin_incr_LRScheduler, PolyLRScheduler_offset
from torch.nn.parallel import DistributedDataParallel as DDP
from nnssl.utilities.helpers import empty_cache
from batchgenerators.utilities.file_and_folder_operations import save_json
from nnssl.training.loss.dino_loss import DINOLoss
import numpy as np
import torch.nn.functional as F
import os
from tqdm import tqdm

class Lambda(nn.Module):
    def __init__(self, func, **kwargs):
        super().__init__()

        self.func = func
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs, **self.kwargs)
    
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

class VoxMSDEvaTrainer(BaseMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):

        super(VoxMSDEvaTrainer, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )
        # Fix the input patch size
        self.config_plan.patch_size = (160, 160, 160)

        ###settings taken from fabi
        self.drop_path_rate = 0.2
        self.attention_drop_rate = 0
        self.grad_clip = 1
        self.initial_lr = 2e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.training_stage = None

        # This represents Primus-M
        self.vit_patch_size = (8, 8, 8)
        self.embed_dim = 864
        self.encoder_eva_depth = 16
        self.encoder_eva_numheads = 12
        # ---
        self.decoder_eva_depth = 2
        self.decoder_eva_numheads = 12
        self.init_value = 0.1
        self.scale_attn_inner = True
        
        self.initial_lr = 3e-4
        self.num_epochs = 500
        
        self.dino_loss = DINOLoss(
                512,
                ncrops=2,  # total number of crops = 2 global crops + local_crops_number
                warmup_teacher_temp=0.04,
                teacher_temp=0.07,
                warmup_teacher_temp_epochs=20,
                nepochs=self.num_epochs,
            ).cuda()
        self.momentum_schedule = cosine_scheduler(0.996, 1, 1000, 250)
        self.features_per_stage = [32, 64, 128, 256, 320, 320]

    def initialize(self):
        # self.recon_dataloader = self.get_qual_recon_dataloader()
        super(VoxMSDEvaTrainer, self).initialize()
        self.ema_network, _ = self.build_architecture_and_adaptation_plan(
                self.config_plan, self.num_input_channels, self.num_output_channels
            )
        self.ema_network.to(self.device)
        embed_dim = 8*864
        proj_dim = 512
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
            Lambda(F.normalize)
        )
        self.proj_head.to(self.device)
        # self.num_iterations_per_epoch = 1
    
    def configure_optimizers(self, stage: str = "warmup_all"):
        assert stage in ["warmup_all", "train"]

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
        else:
            params = self.network.parameters()

        if stage == "warmup_all":
            self.print_to_log_file("train whole net, warmup")
            optimizer = torch.optim.AdamW(
                params, self.initial_lr, weight_decay=self.weight_decay, amsgrad=False, betas=(0.9, 0.98), fused=True
            )
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net)
            self.print_to_log_file(f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}")
        else:
            self.print_to_log_file("train whole net, default schedule")
            if self.training_stage == "warmup_all":
                # we can keep the existing optimizer and don't need to create a new one. This will allow us to keep
                # the accumulated momentum terms which already point in a useful driection
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.AdamW(
                    params,
                    self.initial_lr,
                    weight_decay=self.weight_decay,
                    amsgrad=False,
                    betas=(0.9, 0.98),
                    fused=True,
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net
            )
            self.print_to_log_file(f"Initialized train optimizer and lr_scheduler at epoch {self.current_epoch}")
        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_all")
        elif self.current_epoch == self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("train")

        super().on_train_epoch_start()

    @staticmethod
    def create_mask(
        keep_indices: torch.Tensor, image_size: Tuple[int, int, int], patch_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Create a mask tensor (1 for unmasked, 0 for masked) based on keep_indices.

        Args:
            keep_indices (torch.Tensor): Tensor of shape (B, num_kept_patches) indicating retained patches.
            image_size (Tuple[int, int, int]): Size of the full image (D, H, W).
            patch_size (Tuple[int, int, int]): Size of each patch (D_patch, H_patch, W_patch).

        Returns:
            torch.Tensor: Mask tensor of shape (B, 1, D, H, W) with 1 for unmasked and 0 for masked.
        """
        B, num_kept_patches = keep_indices.shape
        D, H, W = image_size
        D_patch, H_patch, W_patch = patch_size

        # Calculate the number of patches along each dimension
        num_patches_d = D // D_patch
        num_patches_h = H // H_patch
        num_patches_w = W // W_patch
        num_patches = num_patches_d * num_patches_h * num_patches_w

        # Create a flat mask of 0s with shape (B, num_patches)
        flat_mask = torch.zeros(B, num_patches, device=keep_indices.device)

        # Set retained patches to 1
        flat_mask.scatter_(1, keep_indices, 1)

        # Reshape to patch grid and expand to full image size
        mask = flat_mask.view(B, num_patches_d, num_patches_h, num_patches_w)
        mask = (
            mask.repeat_interleave(D_patch, dim=1).repeat_interleave(H_patch, dim=2).repeat_interleave(W_patch, dim=3)
        )
        mask = mask.unsqueeze(1)  # Add channel dimension (B, 1, D, H, W)
        return mask

    @override
    def build_architecture_and_adaptation_plan(
        self, config_plan, num_input_channels, num_output_channels
    ) -> nn.Module:
        network = EvaMAE(
            input_channels=1,
            embed_dim=self.embed_dim,
            patch_embed_size=self.vit_patch_size,
            output_channels=1,
            input_shape=tuple(self.config_plan.patch_size),
            encoder_eva_depth=self.encoder_eva_depth,
            encoder_eva_numheads=self.encoder_eva_numheads,
            decoder_eva_depth=self.decoder_eva_depth,
            decoder_eva_numheads=self.decoder_eva_numheads,
            patch_drop_rate=self.mask_percentage,
            drop_path_rate=self.drop_path_rate,
            attn_drop_rate=self.attention_drop_rate,
            init_values=self.init_value,
            scale_attn_inner=self.scale_attn_inner,
        )

        adapt_plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans("PrimusM"),
            pretrain_plan=self.plan,
            pretrain_num_input_channels=1,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            key_to_encoder="eva",
            key_to_stem="down_projection",
            keys_to_in_proj=("down_projection.proj",),
            key_to_lpe="eva.pos_embed",
        )
        save_json(adapt_plan.serialize(), self.adaptation_json_plan)
        return network, adapt_plan

    def on_validation_epoch_start(self):
        # Make sure the masking is still on.
        #   If set to eval token_dropout will be turned off
        # self.network.eval()
        pass
    
    def _vox_to_vec(self, feature_pyramid: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        pass
    
    def coord_to_token_idx(self, voxel_coords, keep_indices):
        """
        voxel_coords: (50, 3)  [d, h, w]
        keep_indices: (1, 2000)  保存的token索引
        """

        grid_coords = voxel_coords // 8  # (50, 3) 值域[0,19]
        

        flat_indices = grid_coords[:, 0] * (20 * 20) + grid_coords[:, 1] * 20 + grid_coords[:, 2]  # (50,)
        

        keep_indices = keep_indices.squeeze(0)  # (2000,)
        token_idxs = []
        for idx in flat_indices:
            pos = (keep_indices == idx).nonzero(as_tuple=True)[0]
            if pos.numel() > 0:
                token_idxs.append(pos.item())
            else:
                token_idxs.append(-1)  # 标记无效坐标
        
        return torch.tensor(token_idxs)  # (50,)
    
    def extract_token_features(self, feat_list, token_idxs):
        feats = []
        num_idxs = token_idxs.shape[0]
        if num_idxs==0:
            token_idxs = torch.zeros(2,dtype=int)
            num_idxs = 2
        for i in range(len(feat_list)):
            layer_feat = feat_list[i][0, token_idxs]  # (50, 864)
            feats.append(layer_feat)
        return torch.stack(feats).permute(1, 0, 2).reshape(num_idxs,8*864)# (16, 50, 864)

        # return feat_list.permute(1, 0, 2).reshape(50,864)
    
    def train_step(self, batch: dict) -> dict:
        data_s = batch["aug_patch_A"]
        label_s = batch["patch_A"]
        voxel_s = batch["voxel_A"]
        data_t = batch["aug_patch_B"]
        voxel_t = batch["voxel_B"]
        
        
        data_s = data_s.to(self.device, non_blocking=True)
        label_s = label_s.to(self.device, non_blocking=True)
        data_t = data_t.to(self.device, non_blocking=True)
        voxel_s = torch.tensor(voxel_s).to(self.device, non_blocking=True)
        voxel_t = torch.tensor(voxel_t).to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # Autocast for CUDA device
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            # Forward pass with PatchDropout
            feat_s, output_s, keep_indices_s = self.network(data_s)
            feat_t, output_t, keep_indices_t = self.ema_network(data_t)
            mask_s = self.create_mask(keep_indices_s, self.config_plan.patch_size, self.vit_patch_size)
            mask_t = self.create_mask(keep_indices_t, self.config_plan.patch_size, self.vit_patch_size)
            
            valid_s = mask_s[0, 0, voxel_s[:,0], voxel_s[:,1], voxel_s[:,2]] == 1  # (n,)
            valid_t = mask_t[0, 0, voxel_t[:,0], voxel_t[:,1], voxel_t[:,2]] == 1
            valid_mask = valid_s & valid_t  
            valid_voxel_s = voxel_s[valid_mask]  
            valid_voxel_t = voxel_t[valid_mask]
            num_samples = min(50, len(valid_voxel_s))
            indices = torch.randperm(len(valid_voxel_s))[:num_samples]
            selected_voxel_s = valid_voxel_s[indices]  # (50, 3)
            selected_voxel_t = valid_voxel_t[indices]
            token_idx_s = self.coord_to_token_idx(selected_voxel_s, keep_indices_s)
            token_idx_t = self.coord_to_token_idx(selected_voxel_t, keep_indices_t)

            toekn_feats_s = self.extract_token_features(feat_s, token_idx_s)
            toekn_feats_t = self.extract_token_features(feat_t, token_idx_t)

            proj_feat_s = self.proj_head(toekn_feats_s)
            proj_feat_t = self.proj_head(toekn_feats_t)
            
            # Calculate loss considering kept patches
            rec_loss = self.loss(output_s, data_s, mask_s)
            dino_loss = self.dino_loss(proj_feat_s, proj_feat_t, self.current_epoch)
            if(torch.isnan(dino_loss)):
                dino_loss = 4.0
                # rec_loss = 0.5
            l = rec_loss + 0.1 * dino_loss

        # Backward pass and optimization
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.optimizer.step()

        with torch.no_grad():
            m = self.momentum_schedule[self.current_epoch]  # momentum parameter
            for param_q, param_k in zip(self.network.parameters(), self.ema_network.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        data = data.to(self.device, non_blocking=True)

        # Autocast for CUDA device
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            # Forward pass with PatchDropout
            output, keep_indices = self.network(data)
            mask = self.create_mask(keep_indices, self.config_plan.patch_size, self.vit_patch_size)
            # Calculate loss considering kept patches
            l = self.loss(output, data, mask)

        return {"loss": l.detach().cpu().numpy()}

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint["network_weights"].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith("module."):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint["init_args"]

        self.current_epoch = checkpoint["current_epoch"]
        min_epoch = self.logger.load_checkpoint(checkpoint["logging"])
        # Apparently the val log is not written correctly when we currently save the checkpoint.
        self.current_epoch = min_epoch
        self._best_ema = checkpoint["_best_ema"]

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)

        # it's fine to do this every time we load because configure_optimizers will be a no-op if the correct optimizer
        # and lr scheduler are already set up
        if self.current_epoch < self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_all")
        else:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("train")

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None:
            if checkpoint["grad_scaler_state"] is not None:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])
                
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
                    keys=["aug_patch_A", "aug_patch_B", "patch_A", "patch_B"],
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
                    keys=["aug_patch_A", "aug_patch_B", "patch_A", "patch_B"],
                    cast_to="float",
                ),
                NumpyToTensor(
                    keys=["voxel_A", "voxel_B"],
                    cast_to="float",
                ),
            ]
        )
        return val_transforms
    
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


