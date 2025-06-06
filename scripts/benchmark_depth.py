import pandas as pd
from dataclasses import dataclass
import json
from typing import List, Dict


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




def main(meta_data_path):
    
    def data_fn():
        df = pd.read_parquet(meta_data_path)
        for i in range(len(df)):
            yield DepthData.from_row(df.iloc[i])
            
    for data in data_fn():
        print(f"data: {data}")
        break
    
    
    
    
if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--basename_dir', default=f'{code_dir}/../data/', type=str, help='directory of input images, e.g. xxx_left/right.png')
    parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
    parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, help='the directory to save results')
    parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
    parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
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

    model = load_model(args)
    main("metadata/depth_live_1724981057")