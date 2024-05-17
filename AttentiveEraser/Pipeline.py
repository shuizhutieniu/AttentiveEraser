import torch
import numpy as np
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
import torch.nn.functional as F


class AttentiveEraserPipeline(StableDiffusionPipeline):
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
    ):
        prev_timestep = (
            timestep
            - self.scheduler.config.num_train_timesteps
            // self.scheduler.num_inference_steps
        )
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep > 0
            else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def latent2image(self, latents, return_type="np"):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)["sample"]
        if return_type == "np":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)["sample"]

        return image

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
        EmbedCtrlIndex: list = [],
    ):
        DEVICE = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=77, return_tensors="pt"
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]

        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(latents_shape, device=DEVICE)
        else:
            assert latents.shape == latents_shape, "ERROR"

        if guidance_scale > 1.0:
            uc_text = ""
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt",
            )
            unconditional_embeddings = self.text_encoder(
                unconditional_input.input_ids.to(DEVICE)
            )[0]
            text_embeddings[1][EmbedCtrlIndex, :] = unconditional_embeddings[1][
                EmbedCtrlIndex, :
            ]

            text_embeddings = torch.cat(
                [unconditional_embeddings, text_embeddings], dim=0
            )

        self.scheduler.set_timesteps(num_inference_steps)

        for _, t in enumerate(tqdm(self.scheduler.timesteps, desc="...")):
            if guidance_scale > 1.0:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            noise_pred = self.unet(
                model_inputs, t, encoder_hidden_states=text_embeddings
            ).sample
            if guidance_scale > 1.0:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (
                    noise_pred_con - noise_pred_uncon
                )
            latents, pred_x0 = self.step(noise_pred, t, latents)

        image = self.latent2image(latents, return_type="pt")
        return image
