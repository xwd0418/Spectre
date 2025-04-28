import os
from pathlib import Path
curr_file_path = Path(__file__).resolve()
curr_file_dir = curr_file_path.parent
# root_dir = f'/root/gurusmart/MorganFP_prediction/reproduce_previous_works/Spectre/datasets/testing_compounds'
root_dir = str(curr_file_dir)
print(f"Root directory: {root_dir}")

for subdir, dirs, files in os.walk(root_dir):
    if 'HSQC.txt' in files:
        hsqc_path = os.path.join(subdir, 'HSQC.txt')
        normal_hsqc_path = os.path.join(subdir, 'normal_HSQC.txt')
        
        # Read HSQC.txt
        with open(hsqc_path, 'r') as f:
            lines = f.readlines()
        
        # Process and create normal_HSQC.txt
        with open(normal_hsqc_path, 'w') as f:
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    h, c, _ = parts
                    f.write(f"{h},{c}\n")

        print(f"Processed {hsqc_path} -> {normal_hsqc_path}")
