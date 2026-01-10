import joblib
import os
import numpy as np

class ModelWrapper:
    def __init__(self, model_path):
        """
        Loads the model from the given path.
        Handles both pipeline dictionaries (Scaler+LDA+Classifier) and standalone models.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        print(f"Loading model from {model_path}...")
        self.artifact = joblib.load(model_path)
        self.type = None
        
        # Determine structure
        if isinstance(self.artifact, dict):
            keys = self.artifact.keys()
            print(f"  Found pipeline components: {list(keys)}")
            
            # wrapper for dict-based pipelines (e.g. svm_lda_pipeline.joblib)
            if 'scaler' in keys and 'lda' in keys:
                self.scaler = self.artifact['scaler']
                self.lda = self.artifact['lda']
                
                # Identify classifier
                if 'svm' in keys:
                    self.classifier = self.artifact['svm']
                    self.type = 'svm_pipeline'
                elif 'nb' in keys:
                    self.classifier = self.artifact['nb']
                    self.type = 'nb_pipeline'
                else:
                    # Maybe just LDA + something else or custom?
                    # Fallback: look for any other key that is an estimator
                    pass
            else:
                raise ValueError(f"Unknown dictionary structure in model file: {keys}")
                
        elif hasattr(self.artifact, 'predict'):
            # Standalone model (e.g. lda_model.joblib)
            self.classifier = self.artifact
            self.type = 'standalone'
            self.scaler = None
            self.lda = None
            print("  Loaded standalone model.")
            
        else:
            raise ValueError("Could not recognize model structure.")

    def predict(self, features):
        """
        Predicts the class for the given features.
        Args:
            features: (n_samples, n_features) array or (n_features,)
        Returns:
            predictions: array of predicted labels
        """
        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        # Apply pipeline steps if they exist
        data = features
        
        if self.type in ['svm_pipeline', 'nb_pipeline']:
            if self.scaler:
                data = self.scaler.transform(data)
            if self.lda:
                data = self.lda.transform(data)
            return self.classifier.predict(data)
            
        elif self.type == 'standalone':
            # Assumes features are already what the model expects
            # (If lda_model.joblib was trained on raw features, fine. 
            #  If it matches the notebook, it might expect extracted features without scaling? 
            #  We'll verify.)
            return self.classifier.predict(data)
            
        return None

if __name__ == "__main__":
    # Simple test
    import sys
    
    model_dir = "../models"
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "svm_lda_pipeline.joblib"
        
    path = os.path.join(model_dir, model_name)
    try:
        model = ModelWrapper(path)
        print("Model loaded successfully.")
        
        # Test with dummy data
        # Feature count: 32 channels * 5 bands * 7 metrics = 1120 features
        dummy_features = np.zeros((1, 1120))
        pred = model.predict(dummy_features)
        print(f"Dummy prediction: {pred}")
        
    except Exception as e:
        print(f"Error: {e}")