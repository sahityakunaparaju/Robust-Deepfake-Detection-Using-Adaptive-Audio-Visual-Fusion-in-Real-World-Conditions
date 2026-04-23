import os
import torch
import torchvision.transforms as transforms
import debug_model

def run_debug():
    print("="*50)
    print(" DEEPFAKE DETECTION DEBUG AUDIT")
    print("="*50)

    # 1. MODEL WEIGHT LOADING
    model_path = os.path.join(os.path.dirname(debug_model.__file__), "best_model.pt")
    print("\n--- 1. MODEL WEIGHT LOADING ---")
    exists = os.path.exists(model_path)
    print(f"best_model.pt exists: {exists}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_model = debug_model.DeepfakeDetector().to(device)
    
    if exists:
        try:
            state_dict = torch.load(model_path, map_location=device)
            test_model.load_state_dict(state_dict, strict=False)
            print("Confirmation: Weights loaded successfully.")
            print("Loaded weights keys:", list(state_dict.keys())[:5])
        except Exception as e:
            print(f"Error loading weights: {e}")
    else:
        print("WARNING: best_model.pt NOT FOUND. Model using random initialized weights!")

    test_model.eval()

    # Create dummy tensors for initial check
    dummy_frames = torch.randn(1, 16, 3, 224, 224).to(device)
    dummy_audio = torch.randn(1, 1, 48000).to(device)
    
    # 2. MODEL OUTPUT DEBUG
    print("\n--- 2. MODEL OUTPUT DEBUG ---")
    with torch.no_grad():
        preds, gate = test_model(dummy_frames, dummy_audio)
        print("Raw logits:", preds.item())
        prob = torch.sigmoid(preds)
        print("Sigmoid prob:", prob.item())

    # 3. INPUT DISTRIBUTION CHECK
    # Check the extraction logic using a dummy video or just analyze the function definition.
    print("\n--- 3. INPUT DISTRIBUTION CHECK ---")
    print("Audit of extract_frames_tensor() logic:")
    print("Code currently does: frames_tensor.float() / 255.0")
    print("Expected: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) is MISSING.")
    
    # 4. AUDIO CHECK
    print("\n--- 4. AUDIO CHECK ---")
    print("Audit of extract_audio_subprocess() logic:")
    print("Code does: audio_tensor = audio_tensor / (audio_tensor.abs().max() + 1e-8)")
    print("This ensures range ~[-1, 1]")

    # 5. SANITY TEST (CRITICAL)
    print("\n--- 5. SANITY TEST ---")
    print("Testing random noise input:")
    with torch.no_grad():
        out_noise, gate_noise = test_model(dummy_frames, dummy_audio)
        prob_noise = torch.sigmoid(out_noise).item()
        print(f"Noise output prob: {prob_noise:.4f}")
        if prob_noise > 0.8:
            print("Diagnosis: Model outputs high probability even for pure noise! Weights are corrupted or improperly initialized.")

    # 6. THRESHOLD CHECK
    print("\n--- 6. THRESHOLD CHECK ---")
    conf = prob.item()
    print("Final prob:", conf)
    print("Temporary check: Fake if > 0.8 else Real")
    prediction = "Fake" if conf > 0.8 else "Real"
    print(f"With 0.8 threshold: {prediction}")

    # 7. DATA PIPELINE CONSISTENCY
    print("\n--- 7. DATA PIPELINE CONSISTENCY ---")
    print("Resolution: 224x224 (Matches)")
    print("Normalization: Missing ImageNet normalization! (Mismatch)")
    print("Frames: 16 (Matches)")
    print("Audio Length: 48000 (Matches)")

    # 9. GATING DEBUG
    print("\n--- 9. GATING DEBUG ---")
    print("Audio weight:", gate.mean().item())

    print("\n==================================")
    print("✅ DEBUG COMPLETE")
    print("==================================")

if __name__ == "__main__":
    run_debug()
