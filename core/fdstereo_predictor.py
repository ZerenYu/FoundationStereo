import os
import torch
import numpy as np
from omegaconf import OmegaConf
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from .foundation_stereo import FoundationStereo
from .utils.utils import InputPadder

class FoundationStereoPredictor:
    def __init__(self, model_path, device='cuda', valid_iters=32, hiera=False, remove_invisible=True):
        """
        Initialize the FoundationStereo predictor.
        Args:
            model_path (str): Path to the pretrained model checkpoint.
            device (str): Device to run the model on.
            valid_iters (int): Number of refinement iterations.
            hiera (bool): Whether to use hierarchical inference for high-res images.
            remove_invisible (bool): Whether to remove non-overlapping regions.
        """
        self.device = device
        torch.autograd.set_grad_enabled(False)

        cfg_path = os.path.join(os.path.dirname(model_path), 'cfg.yaml')
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config file 'cfg.yaml' not found in the same directory as the model: {os.path.dirname(model_path)}")

        cfg = OmegaConf.load(cfg_path)
        
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
        
        cfg.valid_iters = valid_iters
        cfg.hiera = hiera
        cfg.remove_invisible = remove_invisible
        
        self.cfg = cfg
        self.model = FoundationStereo(self.cfg)
        
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.to(self.device)
        self.model.eval()

    def prepare_image(self, img_data):
        """
        Prepares a numpy image for the model.
        Args:
            img_data (np.ndarray): The input image in (H, W, C) format.
        Returns:
            torch.Tensor: The prepared image tensor.
        """
        if img_data.ndim == 2:
            img_data = np.stack([img_data]*3, axis=-1)
        
        img_tensor = torch.from_numpy(img_data).to(self.device).float()[None].permute(0, 3, 1, 2)
        return img_tensor

    def inference(self, image1, image2, intrinsics=None, baseline=None):
        """
        Performs inference on a pair of stereo images.
        Args:
            image1 (np.ndarray): The left image (H, W, 3).
            image2 (np.ndarray): The right image (H, W, 3).
            intrinsics (np.ndarray, optional): The 3x3 camera intrinsic matrix. Defaults to None.
            baseline (float, optional): The camera baseline. Defaults to None.
        Returns:
            tuple[np.ndarray, np.ndarray | None]: A tuple containing the disparity map and the depth map. 
                                                  The depth map is None if intrinsics or baseline are not provided.
        """
        H, W = image1.shape[:2]

        img1_tensor = self.prepare_image(image1)
        img2_tensor = self.prepare_image(image2)

        padder = InputPadder(img1_tensor.shape, divis_by=32, force_square=False)
        img1_padded, img2_padded = padder.pad(img1_tensor, img2_tensor)

        with torch.cuda.amp.autocast(True):
            if not self.cfg.hiera:
                disp = self.model.forward(img1_padded, img2_padded, iters=self.cfg.valid_iters, test_mode=True)
            else:
                disp = self.model.run_hierachical(img1_padded, img2_padded, iters=self.cfg.valid_iters, test_mode=True, small_ratio=0.5)

        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W)
        
        if self.cfg.remove_invisible:
            yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
            us_right = xx - disp
            invalid = us_right < 0
            disp[invalid] = np.inf

        depth = None
        if intrinsics is not None and baseline is not None:
            fx = intrinsics[0, 0]
            with np.errstate(divide='ignore', invalid='ignore'):
                depth = fx * baseline / disp
            if self.cfg.remove_invisible:
                depth[np.isinf(disp)] = 0
            depth[np.isneginf(disp)] = 0

        return disp, depth 