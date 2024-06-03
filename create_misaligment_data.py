## This script is adopted from https://github.com/ika-rwth-aachen/MultiCorrupt
## Thanks to the authors of multicorrupt
import os
import pickle
import argparse
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from nuscenes import NuScenes
import lidar
import warnings

warnings.filterwarnings('ignore')

def update_pathes_dev1x(infos):
    """
    For .pkl files from mmdetection3d framework on branch dev-1.x
    Update the pathes for the lidar files.
    """
    for info in infos:
        info['lidar_path'] = os.path.join('samples', 'LIDAR_TOP', info['lidar_points']['lidar_path'])
        sweep_pathes = []
        if 'lidar_sweeps' in info:
            for sweep_info in info['lidar_sweeps']:
                sweep_pathes.append({
                    "data_path": os.path.join(*sweep_info['lidar_points']['lidar_path'].split(os.path.sep)[-3:])
                })
        info['sweeps'] = sweep_pathes
    return infos

def update_pathes(infos):
    """
    For .pkl files from mmdetection3d framework older than 1.x.
    Remove the first 16 characters from the data_path.
    """
    for info in infos:
        info['lidar_path'] = os.path.join(*info['lidar_path'].split(os.path.sep)[-3:])
        for sweep in info['sweeps']:
            sweep['data_path'] =  os.path.join(*sweep['data_path'].split(os.path.sep)[-3:])
    return infos


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate corrupted nuScenes dataset for LiDAR')
    parser.add_argument('-c', '--n_cpus', help='number of CPUs that should be used', type=int,
                        default=4)
    parser.add_argument('-s', '--sweep', help='if apply for sweep LiDAR', type=bool,
                        default=False)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset',
                        default='/workspace/nuscenes/nuscenes')
    parser.add_argument('-d', '--dst_folder', help='savefolder of dataset', type=str,
                        default='/workspace/multicorrupt/beamsreducing/1')
    parser.add_argument('-f', '--severity', help='severity level {1,2,3}', type=int,
                        default=1)
    parser.add_argument('--seed', help='random seed', type=int,
                        default=1000)
    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':
    args = parse_arguments()

    print(f'using {args.n_cpus} CPUs')
    print(f'using {args.seed} as numpy random seed')
    np.random.seed(args.seed)

    nusc_info = NuScenes(version='v1.0-trainval', dataroot=args.root_folder, verbose=True)

    imageset = os.path.join(args.root_folder, 'nuscenes_infos_val.pkl')
    with open(imageset, 'rb') as f:
        infos = pickle.load(f)

    if 'infos' in infos:
        # mmdetection3d 1.4
        all_infos = update_pathes(infos['infos'])
    elif 'data_list' in infos:
        all_infos = update_pathes_dev1x(infos['data_list'])
    else:
        exit("This mmdetection3d version is not supported.")
    
    Path(args.dst_folder).mkdir(parents=True, exist_ok=True)
    lidar_save_root = os.path.join(args.dst_folder , 'samples/LIDAR_TOP')
    if not os.path.exists(lidar_save_root):
        os.makedirs(lidar_save_root)
    
    if args.sweep:
        sweep_root = os.path.join(args.dst_folder , 'sweeps/LIDAR_TOP')
        if not os.path.exists(sweep_root):
            os.makedirs(sweep_root)


    def sweep_map(i: int) -> None:
        info = all_infos[i]
        lidar_path = info['lidar_path']
        point = np.fromfile(os.path.join(args.root_folder, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        new_point = lidar.transform_points(point, args.severity)
        for n in info['sweeps']:
            sweep_path = n['data_path']
            sweep_point = np.fromfile(os.path.join(args.root_folder, sweep_path), dtype=np.float32, count=-1).reshape([-1, 5])
            new_sweep_point = lidar.transform_points(sweep_point, args.severity)
            sweep_path = os.path.join(args.dst_folder, sweep_path)
            new_sweep_point.astype(np.float32).tofile(sweep_path)

        lidar_save_path = os.path.join(args.dst_folder, lidar_path)
        new_point.astype(np.float32).tofile(lidar_save_path)


    def sample_map(i: int) -> None:
        info = all_infos[i]
        lidar_path = info['lidar_path']
        point = np.fromfile(os.path.join(args.root_folder, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
        
        
        new_point = lidar.transform_points(point, args.severity)


        lidar_save_path = os.path.join(args.dst_folder, lidar_path)
        new_point.astype(np.float32).tofile(lidar_save_path)


    length = len(all_infos)
    if args.sweep:
        with mp.Pool(args.n_cpus) as pool:
            l = list(tqdm(pool.imap(sweep_map, range(length)), total=length))

    else:
        with mp.Pool(args.n_cpus) as pool:
            l = list(tqdm(pool.imap(sample_map, range(length)), total=length))
