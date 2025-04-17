import subprocess
from pathlib import Path
# Change this to your root directory


root_dir_1 = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stop_on_cosine/all_data_possible")
root_dir_2 = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stop_on_cosine/larger_flexible_models_3072dim")
# root_dir = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/puuting_h_in_the_middle")
def rerun_test_for_dir(root_dir):
    root_dir = Path(root_dir)
    for ckpt_file in root_dir.rglob("*.ckpt"):
        ckpt_file = str(ckpt_file)
        if ckpt_file.endswith("last.ckpt"):
            continue
        
        # if not "entropy_radius" in ckpt_file:
        #     continue
        # if "R0_to_R1_"  in ckpt_file or "R0_to_R5_" in ckpt_file :
        #     pass
        # else:
        #     continue
    
    
        print(f"Testing {ckpt_file}")
        subprocess.run([
        "python", "/root/gurusmart/MorganFP_prediction/reproduce_previous_works/Spectre/train_ranker_transformer.py", 
        "transformer_2d1d", 
        "--test", "1", 
        "--checkpoint_path", ckpt_file,
        # "--test_on_deepsat_retrieval_set", "1",
        
    ], 
                    #    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                       )

# rerun_test_for_dir(root_dir_1)
# rerun_test_for_dir(root_dir_2)
rerun_test_for_dir("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/rank_on_entire_set/grid_search/select_dim_6144_all_info_trial_3_radius_4")
        
