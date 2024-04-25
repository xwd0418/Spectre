from train_ranker_transformer import main 

def objective(trial):
    
    optuna_params = {
        "num_layer" : trial.suggest_int('num_layer', 4, 16),
        "num_attention_head" : trial.suggest_int('num_attention_head', 4, 16),
        "dropout" : trial.suggest_float('dropout', 0.1, 0.5),
        "noam_factor" : trial.suggest_float('noam_factor', 0.01, 100, log=True),
        "weight_decay" : trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        "pos_weight" : trial.suggest_float('pos_weight', 1, 8, log=True),
        "wavelength_bounds": [
            [trial.suggest_float('encoder_1_min_wavelength', 0.001, 0.1, log=True), trial.suggest_float('encoder_1_max_wavelength', 100, 4000, log=True)],
            [trial.suggest_float('encoder_2_min_wavelength', 0.001, 0.1, log=True), trial.suggest_float('encoder_2_max_wavelength', 100, 4000, log=True)],
        ]
        
        # "encoder_min_wavelength" : trial.suggest_float('encoder_min_wavelength', 0.001, 1),
        # "encoder_max_wavelength" : trial.suggest_float('encoder_max_wavelength', 1000, 10000)
    }
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100)