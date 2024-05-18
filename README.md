# AttentiveEraser
*Invisible Edits: Using Attention Manipulation to Remove Objects in Diffusion-Based Image Synthesis*

## Overview

<table style="border-collapse: collapse;width: 100%;">

  <tr>
      <th colspan="2" style="text-align: center; vertical-align: middle; padding: 2px; border: 1px solid white;">和宝宝<del>吃如论</del>烤冷面</th>
  </tr>

  <tr>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.ap-guangzhou.myqcloud.com/2308052246_IMG_2024.jpg" alt="name" style="width: 100%; height: auto;"></td>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.ap-guangzhou.myqcloud.com/2308052246_IMG_2025.jpg" alt="name" style="width: 100%; height: auto;"></td>
  </tr>
</table>

## Requirements

The codebase is tested under **NVIDIA Tesla T4** with the python library **pytorch-2.2.1+cu121** and **diffusers-0.16.1** using pre-trained models through [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5). We strongly recommend using a specific version of Diffusers as it is continuously evolving. For PyTorch, you could probably use other version under 2.x.x.

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
       -w     1,2,3,4,5                  # List of word indices in the attention map to be modified, default is the target word index.
       -r     False                      # "Flag to replace the self-attention maps, default is False."
       -sa    [1, 50]                    # The layers where self-attention maps are replaced, e.g., 1 20. Effective only if -r is specified.
       -em    []                         # Indices of word embeddings to replace, default is empty, e.g., [5] to replace "cherry".
```

Or you can run the following command to understand the available command-line options for the script:

```bash
python run.py -h
```

and sample with
```bash
python run.py -p "A squirrel and a cherry" -s 2436247 -i 5 -t 1 50
```

