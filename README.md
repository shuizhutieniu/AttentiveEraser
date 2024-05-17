# AttentiveEraser
*Invisible Edits: Using Attention Manipulation to Remove Objects in Diffusion-Based Image Synthesis*

## Requirements

The codebase is tested under **NVIDIA Tesla T4** with the python library **pytorch-2.2.1+cu121** and **diffusers-0.16.1.** We strongly recommend using a specific version of Diffusers as it is continuously evolving. For PyTorch, you could probably use other version under 2.x.x.

To install the required packages, you can run the following command in your terminal:
```bash
pip install -r requirements.txt
```

## Usage

```raw
python ./run.py \
       -p     "A squirrel and a cherry"  # The text prompt describing the image
       -s     2436247                    # The seed for reproducibility of results
       -i     5                          # The position index of the target word to remove from the image
       -s     1.0              # noise level
       -ns    5                # number of Attention Editing steps
       -ds    50               # number of diffusion steps (default: 50)
       -m                      # image annotation (optional, useful for debugging)
       -p   "A tiger sitting a on car"         # prompt
       -n   "Your note"                        # your note
       -dp  "/path/to/diffusion/model/folder"  # default: assets/models/clip-vit-large-patch14
       -cp  "/path/to/clip/model/folder"       # default: assets/models/stable-diffusion-v1-4
       -f   "/your/output/folder"              # experiment output folder
```
