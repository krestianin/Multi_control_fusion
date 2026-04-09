#Setup
1. python -m venv .venv
2. source .venv/bin/activate ( .venv\\Scripts\\activate on Windows)
3. 
    pip install --upgrade pip
    pip install torch torchvision torchaudio
    pip install diffusers transformers accelerate safetensors huggingface_hub pillow opencv-python

# Model Storage
Models are stored locally in the `models/` folder in this project directory (on E: drive).
This avoids filling up your C: drive cache. The first run will download models (~10GB) to this folder.

# Clear cache
Remove-Item -Path "$env:USERPROFILE\.cache\huggingface\hub" -Recurse -Force

# Clear local models
Remove-Item -Path "models" -Recurse -Force


# Datasets
python precompute_controls.py   # canny + depth (if not done yet)
python precompute_latents.py    # latents + text embeddings
python train_fusion_mlp.py