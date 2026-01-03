import numpy as np

class CurriculumManager:
    """
    Manages the difficulty level of the training data.
    Progresses from small, sparse graphs to large, dense ones.
    """
    def __init__(self, min_vars=20, max_vars=50, max_levels=30, stability_patience=5):
        self.min_vars = min_vars
        self.max_vars = max_vars
        self.max_levels = max_levels
        self.stability_patience = stability_patience
        
        self.current_level = 0
        self.stability_counter = 0
        self.best_metric = float('inf')
        
    def get_current_params(self):
        """
        Returns parameters for the current level.
        """
        progress = self.current_level / (self.max_levels - 1)
        
        # Linear interpolation for variables
        n_vars = int(self.min_vars + (self.max_vars - self.min_vars) * progress)
        
        # Density scaling (Harder = denser)
        density_min = 0.15 + (0.10 * progress)
        density_max = 0.25 + (0.10 * progress)
        
        # Intervention Range (Harder = wider values)
        int_range = 2.0 + (8.0 * progress)
        
        return {
            "max_vars": n_vars,
            "density_min": density_min,
            "density_max": density_max,
            "intervention_range": int_range
        }
        
    def update(self, val_mae, val_f1):
        """
        Check if we should level up.
        Returns: (leveled_up: bool, reset_lr: bool)
        """
        # Simple gating logic based on MAE (since our main goal is Delta Prediction)
        # Relax threshold as difficulty increases
        params = self.get_current_params()
        n_vars = params['max_vars']
        
        if n_vars <= 25: thresh = 15.0
        elif n_vars <= 35: thresh = 25.0
        else: thresh = 40.0
        
        if val_mae < thresh:
            self.stability_counter += 1
        else:
            self.stability_counter = 0
            
        if self.stability_counter >= self.stability_patience and self.current_level < self.max_levels - 1:
            self.current_level += 1
            self.stability_counter = 0
            return True, True # Level Up, Reset LR
            
        return False, False

    def state_dict(self):
        return {
            "current_level": self.current_level,
            "stability_counter": self.stability_counter,
            "best_metric": self.best_metric
        }

    def load_state_dict(self, state_dict):
        self.current_level = state_dict.get("current_level", 0)
        self.stability_counter = state_dict.get("stability_counter", 0)
        self.best_metric = state_dict.get("best_metric", float('inf'))
