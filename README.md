# AttentiveEraser
*Invisible Edits: Using Attention Manipulation to Remove Objects in Diffusion-Based Image Synthesis*

## Overview


<table style="border-collapse: collapse;width: 100%;">

  <tr>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.ap-guangzhou.myqcloud.com/2436249A%20squirrel%20and%20a%20ball.jpg" alt="name" style="width: 100%; height: auto;"></td>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.ap-guangzhou.myqcloud.com/2436249A%20squirrel%20and%20a%20ball_edited.jpg" alt="name" style="width: 100%; height: auto;"></td>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.ap-guangzhou.myqcloud.com/852_A%20elephant%20and%20a%20lemon.jpg" alt="name" style="width: 100%; height: auto;"></td>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.ap-guangzhou.myqcloud.com/852_A%20elephant%20and%20a%20lemon_Edited.jpg" alt="name" style="width: 100%; height: auto;"></td>
  </tr>

  <tr>
  <td style="border: 1px solid white; padding: 0px; text-align: center;">A squirrel plays with a ball</td>
  <td style="border: 1px solid white; padding: 0px; text-align: center;">A squirrel plays with <del>a ball</del></td>
  <td style="border: 1px solid white; padding: 0px; text-align: center;">A photo of an elephant and Lemon Slice</td>
  <td style="border: 1px solid white; padding: 0px; text-align: center;">A photo of an elephant and <del>Lemon Slice</del></td>
  </tr>

  <tr>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.ap-guangzhou.myqcloud.com/2436247A%20squirrel%20and%20a%20cherry.jpg" alt="name" style="width: 100%; height: auto;"></td>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.ap-guangzhou.myqcloud.com/2436247A%20squirrel%20and%20a%20cherry_edited.jpg" alt="name" style="width: 100%; height: auto;"></td>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.ap-guangzhou.myqcloud.com/2436247A%20banana%20and%20a%20cherry.jpg" alt="name" style="width: 100%; height: auto;"></td>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.ap-guangzhou.myqcloud.com/2436247A%20banana%20and%20a%20cherry_edited.jpg" alt="name" style="width: 100%; height: auto;"></td>
  </tr>

  <tr>
  <td style="border: 1px solid white; padding: 0px; text-align: center;">A squirrel and a cherry</td>
  <td style="border: 1px solid white; padding: 0px; text-align: center;">A squirrel and <del>a cherry</del></td>
  <td style="border: 1px solid white; padding: 0px; text-align: center;">Bananas and Cherries on Green</td>
  <td style="border: 1px solid white; padding: 0px; text-align: center;">Bananas and <del>Cherries</del> on Green</td>
  </tr>

</table>


## Requirements

The codebase is tested under **NVIDIA Tesla T4** with the python library **pytorch-2.2.1+cu121** and **diffusers-0.16.1** using pre-trained models through [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5). We strongly recommend using a specific version of Diffusers as it is continuously evolving. For PyTorch, you could probably use other version under 2.x.x.

To install the required packages, you can run the following command in your terminal:
```bash
pip install -r requirements.txt
```

## Quickstart

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

You can also dig into the models and adjust the parameters you want
```python
import torch
from diffusers import DDIMScheduler
from torchvision.utils import save_image
from AttentiveEraser.tools import *
from AttentiveEraser import Pipeline, AttnCtrl, RegisterAttnCtrl

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_path = "runwayml/stable-diffusion-v1-5"
scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
)
pipe = Pipeline.AttentiveEraserPipeline.from_pretrained(
    model_path, scheduler=scheduler
).to(device)
NUM_DIFFUSION_STEPS = 50


def QuickStart(
    prompts: list,
    initial_latent: torch.Tensor,
    isSelfAttnReplace: bool = False,
    SelfAttnReplaceSteps: list = [1, 50],
    CrossAttnEditIndex: list = [],
    CrossAttnEditWord: list = [],
    CrossAttnEditSteps: list = [0, 0],
    EmbedCtrlIndex: list = [],
    isResetMask: bool = False,
    cross_replace_steps: list = None,
    isDisplay: bool = True,
):
    controller = AttnCtrl.AttentiveEraserAttentionControlEdit(
        prompts,
        NUM_DIFFUSION_STEPS,
        self_replace_steps=SelfAttnReplaceSteps,
        cross_replace_steps=cross_replace_steps,
    )
    RegisterAttnCtrl.AttentiveEraser_register_attention_control(
        pipe,
        controller,
        CrossAttnEditIndex=CrossAttnEditIndex,
        CrossAttnEditWord=CrossAttnEditWord,
        isSelfAttnReplace=isSelfAttnReplace,
        CrossAttnEditSteps=CrossAttnEditSteps,
        isResetMask=isResetMask,
    )
    results = pipe(
        prompts,
        latents=initial_latent,
        guidance_scale=7.5,
        EmbedCtrlIndex=EmbedCtrlIndex,
    )
    if isDisplay:
        display_tensors_as_images(results[0], results[1])

    return results, controller


GlobalSeed(2436247)
prompt = "A squirrel and a cherry"
initial_latent = torch.randn([1, 4, 64, 64], device=device)
prompts = [prompt, prompt]
initial_latent = initial_latent.expand(len(prompts), -1, -1, -1)

results, ctrler = QuickStart(
    prompts=prompts,
    initial_latent=initial_latent,
    isSelfAttnReplace=False,
    SelfAttnReplaceSteps=[1, 50],
    CrossAttnEditIndex=[4, 5],
    CrossAttnEditWord=[1, 2, 3, 4, 5],
    CrossAttnEditSteps=[1, 20],
    EmbedCtrlIndex=[],
    isResetMask=False,
)
```


## Attention Edits
Inspired by [p2p](https://github.com/google/prompt-to-prompt), we perform our main logic by implementing `AttentiveEraserAttentionControlEdit` inherits from the abstract class `AttentionControl`

The `forward` method is called in each attention layer of the diffusion model during the image generation, and we use it to extract the attention map of the target word, then create a mask. This mask, combined with a Gaussian blur technique, is applied to the attention maps corresponding to different layers and words."




<table style="border-collapse: collapse;width: 100%;">

  <tr>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.ap-guangzhou.myqcloud.com/2436249A%20squirrel%20and%20a%20ball.jpg" alt="name" style="width: 100%; height: auto;"></td>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.ap-guangzhou.myqcloud.com/2436249A%20squirrel%20and%20a%20ball_edited.jpg" alt="name" style="width: 100%; height: auto;"></td>
  </tr>

</table>


![alt text](https://xiaolan-1317307543.cos.ap-guangzhou.myqcloud.com/output.png)
