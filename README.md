#Setup
1. python -m venv .venv
2. source .venv/bin/activate ( .venv\\Scripts\\activate on Windows)
3. 
    pip install --upgrade pip
    pip install torch torchvision torchaudio
    pip install diffusers transformers accelerate safetensors huggingface_hub pillow opencv-python


# Clear cache
Remove-Item -Path "$env:USERPROFILE\.cache\huggingface\hub" -Recurse -Force