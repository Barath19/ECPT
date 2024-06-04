#!/bin/bash

severity_levels=("1" "2" "3")

# Directory paths
corrupt_dataset_root="/home/bk/Research/thesis/datasets/corrupted_data"
nuscenes_data_dir="/home/bk/Research/Thesis/datasets/nuScenes_mini"
logfile="evaluation_log.txt"

# Run basline on the original nuscenes dataset
echo "Original datatset"
# Create soft link in /workspace/data/nuscenes
ln -s "$corrupt_dataset_root/spatialmisalignment/orig/sweeps/LIDAR_TOP" "/home/bk/Research/thesis/datasets/nuScenes_mini/sweeps/LIDAR_TOP"
ln -s "$corrupt_dataset_root/spatialmisalignment/orig/samples/LIDAR_TOP" "/home/bk/Research/thesis/datasets/nuScenes_mini/samples/LIDAR_TOP"
# Perform model evaluation
output=$(python tools/test.py projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-mini-3d.py /home/bk/Research/thesis/benchmarks/mmdetection3d/checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth)
echo "$output" > "orig_output.txt"
# Extract NDS and mAP scores from the output
nds=$(echo "$output" | grep "NDS:" | awk '{print $2}')
map=$(echo "$output" | grep "mAP:" | awk '{print $2}')
# Save results to the logfile
echo "Severity (Orig): 0, NDS: $nds, mAP: $map"
echo "Severity (Orig): 0, NDS: $nds, mAP: $map" >> "$logfile"
# Remove soft link
rm "$nuscenes_data_dir/sweeps/LIDAR_TOP"
rm "$nuscenes_data_dir/samples/LIDAR_TOP"



# Loop over severity levels
for severity in "${severity_levels[@]}"; do
  # Log the current configuration in the terminal
  echo "Current Configuration: Severity=$severity"
  # Create soft link 
  ln -s "$corrupt_dataset_root/spatialmisalignment/$severity/sweeps/LIDAR_TOP" "$nuscenes_data_dir/sweeps/LIDAR_TOP"
  ln -s "$corrupt_dataset_root/spatialmisalignment/$severity/samples/LIDAR_TOP" "$nuscenes_data_dir/samples/LIDAR_TOP"
  # Perform model evaluation
  output=$(python tools/test.py projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-mini-3d.py /home/bk/Research/thesis/benchmarks/mmdetection3d/checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth)
 # Save the entire output to a separate text file
  echo "$output" > "${severity}_output.txt"
  # Extract NDS and mAP scores from the output
  nds=$(echo "$output" | grep "NDS:" | awk '{print $2}')
  map=$(echo "$output" | grep "mAP:" | awk '{print $2}')
  # Save results to the logfile
  echo "Severity: $severity, NDS: $nds, mAP: $map"
  echo "Severity: $severity, NDS: $nds, mAP: $map" >> "$logfile"
  # Remove soft link
  rm "$nuscenes_data_dir/sweeps/LIDAR_TOP"
  rm "$nuscenes_data_dir/samples/LIDAR_TOP"
done

