# Setup
Setup the environment
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
1. Place this dataset in your data/ folder (we took 10k from this): 
    - https://huggingface.co/datasets/poloclub/diffusiondb 
    - Use the instruction and the code provided on their website
2. Download the following dataset (we took 10k from here as well):
    - https://huggingface.co/datasets/nlphuji/flickr30k
    - This should create two folders - flickr30k-images/ and flickr30k-descriptions/
    - Use the instruction and the code provided on their website
3. All folder (data, and both flicker ones should be in the root directory)
4. To produce csv files, run: 
    - prepare_dataset.py

2. Then run the following to produce final data folders used for training:

    - python precompute_controls.py   # canny + depth (if not done yet)
    - python precompute_latents.py    # latents + text embeddings

# Training
1. Adjust hyperparameters as needed (use comments in TrainConfig class) in the train_fusion_mlp.py
2. Run python train_fusion_mlp.py

# Inferencing
1. Adjust user settings in main of the inference.py
2. Run python inference.py