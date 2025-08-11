import gc
from itertools import combinations, combinations_with_replacement
from math import prod, ceil
from random import choice

import torch
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
from numpy.random import randint
from typing import TypeAlias
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
import numpy as np
import torch.nn.functional as F
import torchio

BoundingBox3D: TypeAlias = tuple[int, int, int, int, int, int]  # x_start, y_start, z_start, xs, ys, zs
Shape3D: TypeAlias = tuple[int, int, int]   # x_size, y_size, z_size

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

class RandomSwap(AbstractTransform):

    def __init__(self, patch_size: tuple[int, int, int], num_swaps: int, data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key
        self.d, self.h, self.w = patch_size
        self.num_swaps = num_swaps

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)

        patch_d, patch_h, patch_w = self.d, self.h, self.w
        _, _, depth, height, width = data.shape
        aug_data = data.copy()

        for b in range(len(data)):
            for _ in range(self.num_swaps):
                d1, h1, w1 = np.random.randint(0, depth - patch_d), np.random.randint(0, height - patch_h), np.random.randint(0, width - patch_w)
                d2, h2, w2 = np.random.randint(0, depth - patch_d), np.random.randint(0, height - patch_h), np.random.randint(0, width - patch_w)

                patch1 = aug_data[b, :, d1:d1 + patch_d, h1:h1 + patch_h, w1:w1 + patch_w].copy()
                patch2 = aug_data[b, :, d2:d2 + patch_d, h2:h2 + patch_h, w2:w2 + patch_w].copy()

                aug_data[b, :, d1:d1 + patch_d, h1:h1 + patch_h, w1:w1 + patch_w] = patch2
                aug_data[b, :, d2:d2 + patch_d, h2:h2 + patch_h, w2:w2 + patch_w] = patch1

        data_dict[self.data_key] = aug_data
        return data_dict


class Vox2VecTransform(AbstractTransform):
    def __init__(
            self,
            patch_size: tuple[Shape3D],
            min_IoU: float,
            max_num_voxels: int,
        ):
        self.patch_size = patch_size
        self.min_IoU = min_IoU
        self.device = "cuda"
        self.max_num_voxels = max_num_voxels

        rotation_for_DA = {
                "x": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
                "y": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
                "z": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            }
        # order_resampling_data=3,
        # order_resampling_seg=1,
        
        self.global_spatial_transforms = Compose([
            MirrorTransform(axes=(0, 1, 2)),
            SpatialTransform(
                self.patch_size,
                patch_center_dist_from_border=None,
                do_elastic_deform=False,
                alpha=(0, 0),
                sigma=(0, 0),
                do_rotation=True,
                angle_x=rotation_for_DA["x"],
                angle_y=rotation_for_DA["y"],
                angle_z=rotation_for_DA["z"],
                p_rot_per_axis=1,  # todo experiment with this
                do_scale=True,
                scale=(0.7, 1.4),
                # border_mode_data="constant",
                # border_cval_data=0,
                # order_data=order_resampling_data,
                # ToDo: Why do we even do scale transforms and do specifically preprocess data? This largely makes no sense, right?
                # border_mode_seg="constant",
                # border_cval_seg=-1,
                # order_seg=order_resampling_seg,
                random_crop=False,  # random cropping is part of our dataloaders
                p_el_per_sample=0,
                p_scale_per_sample=0.2,
                p_rot_per_sample=0.2,
                independent_scale_for_each_axis=False,  # todo experiment with this
            )
        ])


    def __call__(self, **data_dict):
        imgs = data_dict.get('data')
        if imgs is None:
            raise ValueError(f"No data found for key {self.data_key}")
        patch_A, patch_B, voxels_A, voxels_B = self.get_global_and_local_crops(imgs)

        aug_patch_A = self.global_spatial_transforms(data=patch_A)["data"]
        aug_patch_B = self.global_spatial_transforms(data=patch_B)["data"]

        valid = np.zeros(1)
        valid_indices = np.zeros(1)
        wait_n=0
        while(len(valid_indices) < self.max_num_voxels):
            wait_n += 1
            if(wait_n==10):
                voxels_A = np.zeros(shape=(2,3))
                voxels_B = np.zeros(shape=(2,3))
                valid_indices=[0,1]
                break
            # We use the self.batch_size as it is not identical with the plan batch_size in ddp cases.
            mask_s = self.mask_creation(1, self.patch_size, 0.75)
            mask_t = self.mask_creation(1, self.patch_size, 0.75)
            # Make the mask the same size as the data
            rep_D, rep_H, rep_W = (
                aug_patch_A.shape[2] // mask_s.shape[2],
                aug_patch_A.shape[3] // mask_s.shape[3],
                aug_patch_A.shape[4] // mask_s.shape[4],
            )
            mask_s = mask_s.repeat_interleave(rep_D, dim=2).repeat_interleave(rep_H, dim=3).repeat_interleave(rep_W, dim=4)
            mask_t = mask_t.repeat_interleave(rep_D, dim=2).repeat_interleave(rep_H, dim=3).repeat_interleave(rep_W, dim=4)

            masked_aug_patch_A = aug_patch_A * np.array(mask_s)
            masked_aug_patch_B = aug_patch_B * np.array(mask_t)
            
            valid_A = masked_aug_patch_A[:,0,voxels_A[:,0],voxels_A[:,1],voxels_A[:,2]]!=0
            valid_B = masked_aug_patch_B[:,0,voxels_B[:,0],voxels_B[:,1],voxels_B[:,2]]!=0

            valid = valid_A & valid_B
            if(~valid.any() or len(np.where(valid[0])[0])<self.max_num_voxels):
                continue
            valid_indices = np.random.choice(np.where(valid[0])[0], self.max_num_voxels, replace=False)
            
        
        new_data_dict = {
            "aug_patch_A": masked_aug_patch_A,
            "patch_A": patch_A,
            "mask_A":mask_s,
            "aug_patch_B": masked_aug_patch_B,
            "patch_B": patch_B,
            "mask_B":mask_t,
            "voxel_A":voxels_A[valid_indices],
            "voxel_B":voxels_B[valid_indices],
        }
        return new_data_dict

    @staticmethod
    def mask_creation(
        batch_size: int,
        patch_size: tuple[int, int, int],
        mask_percentage: float,
        rng_seed: int | None = None,
        block_size: int = 16,
    ) -> torch.Tensor:
        """
        Creates a masking tensor with 1s (indicating no masking) and 0s (indicating masking).
        The mask has to be of same size like the input data (batch_size, 1, x, y, z).

        :param batch_size: batch size during training
        :param patch_size: The 3D shape information for the input patch.
        :param mask_percentage: percentage of the patch that should be masked
        :param block_size: size of the blocks that should be masked
        :return:
        """

        sparsity_factor = mask_percentage
        mask = [create_blocky_mask(patch_size, block_size, sparsity_factor) for _ in range(batch_size)]
        mask = torch.stack(mask)[:, None, ...]  # Add channel dimension
        return mask
    
    def get_global_and_local_crops(self, imgs: np.ndarray):
        batch_size, N, X, Y, Z = imgs.shape
        global_crops_A, global_crops_B= [], []
        global_voxels_A, global_voxels_B = [], []
        
        for i in range(batch_size):
            image = imgs[i]
            voxels_A = voxels_B = np.argwhere(image[0]!=0)
            g_patch_size_A = g_patch_size_B = self.patch_size
            big_bbox = self.get_rand_big_bbox((X, Y, Z), g_patch_size_A, g_patch_size_B, self.min_IoU)   # [x_start, y_start, z_start, x_end, y_end, z_end]
            g_bbox_A, g_bbox_B = self.get_global_bboxes(g_patch_size_A, g_patch_size_B, big_bbox)
            g_crop_A, g_crop_B = self.get_crop(image, g_bbox_A), self.get_crop(image, g_bbox_B)

            # The original codebase uses skimage.transform.resize, which is very slow, that's also (partly) why they
            # moved their augmentations to the data preprocessing step, resulting in limited augmentations.
            g_crop_A, g_crop_B = F.interpolate(torch.from_numpy(g_crop_A)[None, ...], self.patch_size).numpy(), \
                                 F.interpolate(torch.from_numpy(g_crop_B)[None, ...], self.patch_size).numpy()

            # get the minimum sized bounding box that holds both global bboxes
            min_bbox = self.get_min_bbox(g_bbox_A, g_bbox_B)
            shift_A = g_bbox_A[:3]
            shift_B = g_bbox_B[:3]
            voxels_A = voxels_A-shift_A
            voxels_B = voxels_B-shift_B
            valid_A = np.all((voxels_A >= 0) & (voxels_A < self.patch_size), axis=1)
            valid_B = np.all((voxels_B >= 0) & (voxels_B < self.patch_size), axis=1)
            
            valid = valid_A & valid_B
            indices = np.where(valid)[0]

            global_voxels_A.append(voxels_A[indices])
            global_voxels_B.append(voxels_B[indices])

            global_crops_A.append(g_crop_A)
            global_crops_B.append(g_crop_B)

        global_crops_A = np.concat(global_crops_A)                # [B, C, X, Y, Z]
        global_voxels_A = np.concat(global_voxels_A)              
        global_crops_B = np.concat(global_crops_B)                # [B, C, X, Y, Z]
        global_voxels_B = np.concat(global_voxels_B)              
        

        return global_crops_A, global_crops_B,global_voxels_A,global_voxels_B

    def get_global_bboxes(self, g_patch_size_A: Shape3D, g_patch_size_B: Shape3D, big_bbox: BoundingBox3D) -> tuple[
        BoundingBox3D, BoundingBox3D]:
        g_bbox_A = self.get_rand_inner_bbox(big_bbox, g_patch_size_A)
        # tries = 0
        while True:
            g_bbox_B = self.get_rand_inner_bbox(big_bbox, g_patch_size_B)
            # tries += 1
            if self.calculate_IoU(g_bbox_A, g_bbox_B) >= self.min_IoU:
                # print(tries)
                break
        return g_bbox_A, g_bbox_B


    def get_min_bbox(self, bbox_A: BoundingBox3D, bbox_B: BoundingBox3D) -> BoundingBox3D:
        min_bbox_starts = [min(bbox_A[i], bbox_B[i]) for i in range(3)]
        min_bbox_ends = [max(bbox_A[i] + bbox_A[3+i], bbox_B[i] + bbox_B[3+i]) for i in range(3)]
        min_bbox_shape = [min_bbox_ends[i] - min_bbox_starts[i] for i in range(3)]
        return BoundingBox3D(min_bbox_starts + min_bbox_shape)


    @staticmethod
    def calculate_IoU(bbox_1: BoundingBox3D, bbox_2: BoundingBox3D) -> float:
        overlaps_per_axis = [
            max(0, min(bbox_1[i] + bbox_1[3 + i], bbox_2[i] + bbox_2[3 + i]) - max(bbox_1[0 + i], bbox_2[0 + i]))
            for i in range(3)
        ]
        overlapping_volume = prod(overlaps_per_axis)
        v1, v2 = prod(bbox_1[3:]), prod(bbox_2[3:])
        return overlapping_volume / (v1 + v2 - overlapping_volume)


    @staticmethod
    def get_rand_big_bbox(img_shape: Shape3D, g_patch_size_A: Shape3D, g_patch_size_B: Shape3D,
                          min_IoU: float) -> BoundingBox3D:
        """
        The original implementation gets two global views with an IoU restriction by randomly sampling two global
        global views repeatedly from the image until the IoU threshold is reached. While this is not optimal, the smallest
        possible bbox from which you (1) can sample the two global views randomly, (2) still be above the IoU
        threshold and (3) still have all possible combinations is not a rectangular cuboid, making it difficult to
        calculate it.
        This function tries to provide a bbox where the possibility of not reaching the IoU threshold is minimized,
        so that if you randomly sample two global views from this bbox, the number of iterations it takes to meet the
        IoU restriction is minimized as well.
        """
        big_bbox_shape = []
        max_overlaps = [min(A, B) for A, B in zip(g_patch_size_A, g_patch_size_B)]
        max_areas = [side_1 * side_2 for side_1, side_2 in reversed(list(combinations(max_overlaps, r=2)))]
        volume_A, volume_B = prod(g_patch_size_A), prod(g_patch_size_B)

        # for each axis, calculate the minimum intersection of A and B, so that if the overlapping area of the
        # other two axis is maximal, the resulting IoU is above 'min_IoU'
        for i in range(3):
            side_A, side_B = g_patch_size_A[i], g_patch_size_B[i]

            # Q: How do we get the minimum intersection per axis/side, so that we are still over the IoU threshold
            #    given a maximal overlap area of the other two axis?
            # -> Start from this equation:
            #   ((min_side_intersect * max_area) / (V1 + V2 - min_side_intersect * max_area) > min_IoU
            # -> solve for min_side_intersect, then we get the following:
            min_side_intersection = ceil( (volume_A+volume_B)*min_IoU / ((1+min_IoU)*max_areas[i]) )
            big_bbox_shape.append(min(img_shape[i], side_A + side_B - min_side_intersection))

        big_bbox_starts = [randint(0, img_shape[i] - big_bbox_shape[i] + 1) for i in range(3)]
        return BoundingBox3D(big_bbox_starts + big_bbox_shape)

    @staticmethod
    def get_crop(image: np.ndarray, bbox: BoundingBox3D):
        x_start, y_start, z_start, xs, ys, zs = bbox
        return image[:, x_start:x_start+xs, y_start:y_start+ys, z_start:z_start+zs]


    @staticmethod
    def get_rand_inner_bbox(bbox: BoundingBox3D, inner_bbox_shape: Shape3D) -> BoundingBox3D:
        inner_bbox_start = tuple([bbox[i] + randint(0, bbox[3+i] - inner_bbox_shape[i]) for i in range(3)])
        return BoundingBox3D(inner_bbox_start + inner_bbox_shape)



if __name__ == "__main__":
    # bbox_1 = (2, 4, 3, 4, 4, 6)
    # bbox_2 = (1, 3, 2, 2, 4, 2)
    # print(PCRLv2Transform.calculate_IoU(bbox_1, bbox_2))

    trafo = Vox2VecTransform(
        global_patch_sizes = ((96, 96, 96), (128, 128, 96), (128, 128, 128), (160, 160, 128)),
        global_input_size = (128, 128, 128),
        local_patch_sizes = ((32, 32, 32), (64, 64, 32), (64, 64, 64)),
        local_input_size = (64, 64, 64),
        num_locals = 6,
        min_IoU = 0.3
    )

    # _ = trafo.get_rand_big_bbox((160, 160, 160), (96, 96, 96), (96, 96, 96), 0.3)

    bbox = (20, 31, 127, 112, 112, 64)

    import time

    with torch.no_grad():
        start = time.time()
        for i in range(4):
            imgs = np.zeros((8, 1, 180, 180, 180))
            _ = trafo(data=imgs)
        elapsed = time.time() - start
        print(f"Time per iteration: {elapsed/4:.3f}s")













