import os

root_dir = '/root/gurusmart/MorganFP_prediction/reproduce_previous_works/Spectre/datasets/testing_compounds'

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
                    f.write(f"{h},{c},0\n")

        print(f"Processed {hsqc_path} -> {normal_hsqc_path}")
