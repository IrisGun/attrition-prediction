import os
import subprocess
import json

def run_pipeline():
    """
    Orchestrate the entire ML pipeline.
    """
    print("Starting ML Pipeline...")
    
    # 1. Generate Data
    print("\n--- Step 1: Data Generation ---")
    subprocess.run(['python3', 'ml/data_gen.py'], check=True)
    
    # 2. Feature Engineering
    print("\n--- Step 2: Feature Engineering ---")
    subprocess.run(['python3', 'ml/feature_engineering.py'], check=True)
    
    # 3. Drift Detection
    print("\n--- Step 3: Drift Detection ---")
    drift_detected = subprocess.run(['python3', 'ml/drift_detection.py'], check=True, capture_output=True, text=True)
    print(drift_detected.stdout)
    
    # 4. Training and Evaluation
    print("\n--- Step 4: Training and Evaluation ---")
    # Always train for the first time, or if drift is detected
    if "Data drift detected" in drift_detected.stdout or not os.path.exists('ml/models/best_model.pkl'):
        subprocess.run(['python3', 'ml/train_eval.py'], check=True)
    else:
        print("Skipping re-training (no drift detected).")
        
    # 5. Inference
    print("\n--- Step 5: Inference ---")
    subprocess.run(['python3', 'ml/inference.py'], check=True)
    
    print("\nML Pipeline complete! Results saved to ml/artifacts/predictions.json")

if __name__ == "__main__":
    run_pipeline()
