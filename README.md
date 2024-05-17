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
       -p     "A squirrel and a cherry"  # The text prompt describing the image, e.g., "A squirrel and a cherry"
       -s     2436247                    # The seed number for reproducibility of results, e.g., 42.
       -i     5                          # The position index of the target word to remove from the image, e.g., 5 for "cherry" in the prompt.
       -t     1 20                       # The range of diffusion model layers to apply the modifications, e.g., 1 20 for layers 1 to 20.
       -w     1,2,3,4,5                  # number of Attention Editing steps
       -ds    50                         # List of word indices in the attention map to be modified, default is the target word index.
```

Or you can run the following command to understand the available command-line options for the script:

```bash
python run.py -h
```