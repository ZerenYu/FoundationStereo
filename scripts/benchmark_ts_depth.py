import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
import pandas as pd
from dataclasses import dataclass
import json
from typing import List, Dict
import os
import argparse
import torch
import logging
import time
from omegaconf import OmegaConf
import random

from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder
from scripts.deserialize_depth_dataset import Boto3ResourceManager, deserialize_and_download_image, deserialize_and_download_tensor
from Utils import set_logging_format, set_seed


def load_model(args):
    """Loads the stereo model and checkpoint.

    Args:
        args: Command-line arguments.

    Returns:
        model: The loaded stereo model.
    """
    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        if k not in ['left_file', 'right_file']: # Avoid overwriting constructed paths
             cfg[k] = args.__dict__[k]
    current_args = OmegaConf.create(cfg) # Use a different name to avoid confusion
    logging.info(f"args for model loading:\n{current_args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")

    model = FoundationStereo(current_args)

    ckpt = torch.load(ckpt_dir)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])

    model.cuda()
    model.eval()
    return model


@dataclass
class DepthData:
    dataset_creator: str
    camera_names: List[str]
    item_id: int
    split: str
    image_paths: Dict[str, str]
    depth_map_paths: Dict[str, str]
    normal_map_paths: Dict[str, str]
    visible_mask_paths: Dict[str, str]
    world_from_camera_transforms_path: str
    camera_intrinsics_path: str

    @classmethod
    def from_row(cls, row):
        return cls(
            dataset_creator=row[0],
            camera_names=list(row[1]),
            item_id=row[2],
            split=row[3],
            image_paths=json.loads(row[4]),
            depth_map_paths=json.loads(row[5]),
            normal_map_paths=json.loads(row[6]),
            visible_mask_paths=json.loads(row[7]),
            world_from_camera_transforms_path=row[8],
            camera_intrinsics_path=row[9],
        )


def deserialize_data(data: DepthData, resource_manager: Boto3ResourceManager):
    """Deserialize all the data we need for a single benchmark item."""
    camera_ids = list(data.image_paths.keys())
    if len(camera_ids) < 2:
        raise ValueError(
            f"Need at least two images for inference, but got {len(camera_ids)}.")

    cam1_id, cam2_id = random.sample(camera_ids, 2)
    logging.info(f"Randomly selected cameras: {cam1_id}, {cam2_id}")

    # It's conventional to use bit_depth=8 for RGB images.
    img1 = deserialize_and_download_image(
        data.image_paths[cam1_id], bit_depth=8, resource_manager=resource_manager, dtype=torch.float32)
    img2 = deserialize_and_download_image(
        data.image_paths[cam2_id], bit_depth=8, resource_manager=resource_manager, dtype=torch.float32)

    intrinsics = deserialize_and_download_tensor(
        data.camera_intrinsics_path, resource_manager=resource_manager)

    # The model expects a batch dimension
    return img1[None], img2[None], intrinsics


def run_inference(model, img0_torch, img1_torch, args):
    """Runs inference on a pair of image tensors.

    Args:
        model: The stereo model.
        img0_torch: Left image tensor.
        img1_torch: Right image tensor.
        args: Command-line arguments.

    Returns:
        disp: The disparity map.
        inference_time: The time taken for inference.
    """
    H, W = img0_torch.shape[2:]
    padder = InputPadder(img0_torch.shape, divis_by=32, force_square=False)
    img0_padded, img1_padded = padder.pad(img0_torch.cuda(), img1_torch.cuda())

    start_time = time.time()
    with torch.cuda.amp.autocast(True):
        if not args.hiera:
            disp = model.forward(img0_padded, img1_padded, iters=args.valid_iters, test_mode=True, low_memory=True)
        else:
            disp = model.run_hierachical(img0_padded, img1_padded, iters=args.valid_iters, test_mode=True, small_ratio=0.5, low_memory=True)
    inference_time = time.time() - start_time

    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W)
    return disp, inference_time


def main(args):
    model = load_model(args)
    resource_manager = Boto3ResourceManager()

    def data_fn():
        df = pd.read_parquet(args.meta_data_path)
        for i in range(len(df)):
            yield DepthData.from_row(df.iloc[i])

    for data in data_fn():
        logging.info(f"Processing item {data.item_id}")
        img1, img2, intrinsics = deserialize_data(data, resource_manager)
        print(intrinsics)
        disp, inference_time = run_inference(model, img1, img2, args)
        
    #     logging.info(f"Inference time: {inference_time:.4f}s, Disparity map shape: {disp.shape}")
        
    #     # We break here for now to only test one item.
        break


if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_data_path', default="metadata/depth_live_1724981057", type=str, help='path to metadata parquet file')
    parser.add_argument('--basename_dir', default=f'{code_dir}/../data/', type=str, help='directory of input images, e.g. xxx_left/right.png')
    parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
    parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, help='the directory to save results')
    parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
    parser.add_argument('--hiera', default=1, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
    parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
    parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
    parser.add_argument('--denoise_cloud', type=int, default=0, help='whether to denoise the point cloud')
    parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
    parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
    args = parser.parse_args()

    print("Starting test...")
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    main(args)