import torch
import os

ckpt_path = "final_chekpoint/checkpoint_epoch_253.pt"
if not os.path.exists(ckpt_path):
    print(f"Error: {ckpt_path} not found.")
else:
    try:
        # Load CPU to capture metadata
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print("Checkpoint Keys:", checkpoint.keys())
        if 'args' in checkpoint:
            print("Training Args:", checkpoint['args'])
        if 'epoch' in checkpoint:
            print("Epoch:", checkpoint['epoch'])
        if 'curriculum_state_dict' in checkpoint:
             print("Curriculum Level:", checkpoint['curriculum_state_dict'].get('current_level', 'Unknown'))
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
