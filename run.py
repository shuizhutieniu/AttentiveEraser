import os
import torch
from diffusers import DDIMScheduler
from torchvision.utils import save_image
import warnings
import argparse

warnings.filterwarnings("ignore", category=FutureWarning)

from AttentiveEraser.tools import (
    GlobalSeed,
    display_tensors_as_images,
)

import AttentiveEraser.Pipeline as Pipeline
import AttentiveEraser.AttnCtrl as AttnCtrl
import AttentiveEraser.RegisterAttnCtrl as RegisterAttnCtrl


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
    SaveSteps: int | list = [],
    isResetMask: bool = False,
    CurrentTime: str = "",
    SavedTime: list = None,
    cross_replace_steps=None,
    isDisplay=True,
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
        SaveSteps=SaveSteps,
        isResetMask=isResetMask,
        CurrentTime=CurrentTime,
        SavedTime=SavedTime,
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


def main():
    parser = argparse.ArgumentParser(
        description="Remove specified objects from an image without changing the prompt, by manipulating the attention maps in a Diffusion model."
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help='The text prompt describing the image, e.g., "A squirrel and a cherry".',
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        required=True,
        help="The seed number for reproducibility of results, e.g., 42.",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        required=True,
        help='The position index of the target word to remove from the image, e.g., 5 for "cherry" in the prompt.',
    )
    parser.add_argument(
        "-t",
        "--step",
        type=int,
        nargs="+",
        required=True,
        help="The range of diffusion model layers to apply the modifications, e.g., 1 20 for layers 1 to 20.",
    )
    parser.add_argument(
        "-w",
        "--word",
        type=int,
        nargs="*",
        help="List of word indices in the attention map to be modified, default is the target word index.",
    )

    parser.add_argument(
        "-r",
        "--replace_self_attention",
        action="store_true",
        help="Flag to replace the self-attention maps, default is False.",
    )
    parser.add_argument(
        "-sa",
        "--self_attention_steps",
        type=int,
        nargs="+",
        default=[1, 50],
        help="The layers where self-attention maps are replaced, e.g., 1 20. Effective only if -r is specified.",
    )
    parser.add_argument(
        "-em",
        "--replace_embedding_indices",
        type=int,
        nargs="*",
        default=[],
        help='Indices of word embeddings to replace, default is empty, e.g., [5] to replace "cherry".',
    )

    args = parser.parse_args()
    run(
        args.prompt,
        args.seed,
        args.index,
        args.step,
        args.word,
        args.replace_self_attention,
        args.self_attention_steps,
        args.replace_embedding_indices,
    )


def run(
    prompt,
    seed,
    index,
    step,
    word,
    replace_self_attention,
    self_attention_steps,
    replace_embedding_indices,
):
    GlobalSeed(seed)
    initial_latent = torch.randn([1, 4, 64, 64], device=device)
    prompts = [prompt, prompt]
    initial_latent = initial_latent.expand(len(prompts), -1, -1, -1)

    results, _ = QuickStart(
        prompts=prompts,
        initial_latent=initial_latent,
        isSelfAttnReplace=replace_self_attention,
        SelfAttnReplaceSteps=self_attention_steps,
        CrossAttnEditIndex=index,
        CrossAttnEditWord=word,
        CrossAttnEditSteps=step,
        EmbedCtrlIndex=replace_embedding_indices,
        SaveSteps=[],
        isResetMask=False,
    )

    save_image(results[0], os.path.join(str(seed) + "_" + str(prompt) + ".jpg"))
    save_image(
        results[1], os.path.join("Edited_" + str(seed) + "_" + str(prompt) + ".jpg")
    )


if __name__ == "__main__":
    main()
