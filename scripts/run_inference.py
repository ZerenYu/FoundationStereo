import os
import cv2
import numpy as np
import argparse

# Ensure the script can find the model and utility files
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from models.FoundationStereo.core.fdstereo_predictor import FoundationStereoPredictor


def main(args):
    """
    Main function to run stereo inference.
    """
    # Model and data paths
    model_path = args.model_path
    data_dir = args.data_dir
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)

    # Hardcoded filenames from the provided test data
    left_image_path = os.path.join(data_dir, "20250709_062923_33467134_left_image.png")
    right_image_path = os.path.join(data_dir, "20250709_062923_33467134_right_image.png")
    left_intrinsic_path = os.path.join(data_dir, "20250709_062923_33467134_left_intrinsic.npy")
    left_pose_path = os.path.join(data_dir, "20250709_062923_33467134_left_pose.npy")
    right_pose_path = os.path.join(data_dir, "20250709_062923_33467134_right_pose.npy")

    # Load images
    print("Loading images...")
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)
    
    # Convert images from BGR (OpenCV default) to RGB
    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

    # Load intrinsics and poses
    print("Loading intrinsics and poses...")
    left_intrinsics = np.load(left_intrinsic_path)
    left_pose = np.load(left_pose_path)
    right_pose = np.load(right_pose_path)

    # Calculate baseline from camera poses.
    # Assumes poses are 4x4 camera-to-world transformation matrices.
    t_left = left_pose[:3, 3]
    t_right = right_pose[:3, 3]
    baseline = np.linalg.norm(t_left - t_right)
    
    # Initialize the predictor
    print("Initializing FoundationStereo predictor...")
    predictor = FoundationStereoPredictor(model_path=model_path)
    
    # Run inference
    print("Running inference...")
    disp, depth = predictor.inference(left_image, right_image, intrinsics=left_intrinsics, baseline=baseline)

    # Define output paths
    disp_path = os.path.join(output_dir, "disparity.npy")
    depth_path = os.path.join(output_dir, "depth.npy")
    depth_image_path = os.path.join(output_dir, "depth_image.png")

    # Save raw disparity and depth maps
    np.save(disp_path, disp)
    np.save(depth_path, depth)
    print(f"Saved disparity map to {disp_path}")
    print(f"Saved depth map to {depth_path}")

    # Visualize the depth map and save it as a PNG image
    print("Visualizing depth map...")
    valid_mask = (depth > 0) & np.isfinite(depth)
    if valid_mask.any():
        # Normalize depth for visualization, ignoring invalid pixels
        min_depth = np.min(depth[valid_mask])
        max_depth = np.percentile(depth[valid_mask], 95) # Clip to 95th percentile for better visualization
        
        normalized_depth = (depth - min_depth) / (max_depth - min_depth)
        normalized_depth = np.clip(normalized_depth, 0, 1)
        normalized_depth[~valid_mask] = 0
        
        depth_image = (normalized_depth * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        depth_colormap[~valid_mask] = 0  # Set invalid depth regions to black

        cv2.imwrite(depth_image_path, depth_colormap)
        print(f"Saved visualized depth map to {depth_image_path}")
    else:
        print("No valid depth values found to visualize.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run FoundationStereo inference on a pair of stereo images.")
    parser.add_argument('--model_path', type=str, 
                        default='data/checkpoints/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth',
                        help='Path to the pretrained model checkpoint.')
    parser.add_argument('--data_dir', type=str, default='data/test_data',
                        help='Path to the directory containing test data.')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save the output files.')
    
    args = parser.parse_args()
    main(args) 