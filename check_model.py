import joblib
import os

model_dir = 'models'
models = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]

for m in models:
    try:
        path = os.path.join(model_dir, m)
        model = joblib.load(path)
        print(f"Model: {m}")
        print(f"  Type: {type(model)}")
        if isinstance(model, dict):
            print(f"  Keys: {list(model.keys())}")
        if hasattr(model, 'steps'):
            print(f"  Steps: {[s[0] for s in model.steps]}")
        if hasattr(model, 'n_features_in_'):
            print(f"  n_features_in_: {model.n_features_in_}")
        elif hasattr(model, 'steps'): # Pipeline
            # Check the last step usually, or the first step if it's the estimator? 
            # Actually, the classifier at the end knows the input features *to it*, 
            # but the pipeline input might be different if steps change dimensions.
            # But usually we want to know what the pipeline expects.
            # If the first step is a scalar/selector, it expects the raw features.
            estimator = model.steps[-1][1]
            if hasattr(estimator, 'n_features_in_'):
                 print(f"  Pipeline estimator n_features_in_: {estimator.n_features_in_}")
            else:
                print("  Pipeline estimator does not have n_features_in_")
        else:
             print("  Could not determine n_features_in_")
    except Exception as e:
        print(f"  Error loading {m}: {e}")
