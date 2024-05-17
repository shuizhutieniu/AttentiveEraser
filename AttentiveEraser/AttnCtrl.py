import abc
from einops import rearrange
from typing import Optional, Union, Tuple, List, Callable, Dict
from torchvision.utils import save_image
import torch.nn.functional as F
from math import sqrt
import torch
import pickle
import os
from .tools import load_from_pkl, get_time_words_attention_alpha, get_replacement_mapper
import datetime


def GetTime():
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%H_%M_%S")
    return formatted_time


def save_as_pkl(var, file_name: str, folder_path: str = "pkls"):
    os.makedirs(folder_path, exist_ok=True)
    original_file_name = file_name
    counter = 1
    full_file_path = os.path.join(folder_path, f"{file_name}.pkl")

    while os.path.exists(full_file_path):
        counter += 1
        file_name = f"{original_file_name}({counter})"
        full_file_path = os.path.join(folder_path, f"{file_name}.pkl")

    with open(full_file_path, "wb") as file:
        pickle.dump(var, file)


def InfoSave(
    input=None, prefix: str = "", mark: bool = False, filename: str = "output"
):
    with open(f"{filename}.txt", "a") as file:
        if mark:
            file.write("#########  " + prefix + str(input) + "  ##########" + "\n")
        else:
            file.write(prefix + str(input) + "\n")


class EmptyControl:
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(
        self,
        attn,
        is_cross: bool,
        place_in_unet: str,
        CrossAttnEditIndex=[],
        CrossAttnEditWord = [],
        isSelfAttnReplace: bool = True,
        CrossAttnEditSteps: list = [],
        SaveSteps: int | list = [],
        isResetMask: bool = False,
        CurrentTime: str = "",
        SavedTime: list = None,
    ):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2 :] = self.forward(
                attn[h // 2 :],
                is_cross,
                place_in_unet,
                CrossAttnEditIndex,
                CrossAttnEditWord,
                isSelfAttnReplace,
                CrossAttnEditSteps,
                SaveSteps,
                isResetMask,
                CurrentTime,
                SavedTime,
            )
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.LOW_RESOURCE = False


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if attn.shape[1] <= 16**2:
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentiveEraserAttentionControlEdit(AttentionStore, abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32**2:
            attn_base = attn_base.unsqueeze(0).expand(
                att_replace.shape[0], *attn_base.shape
            )
            return attn_base
        else:
            return att_replace

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum("hpw,bwn->bhpn", attn_base, self.mapper)

    def forward(
        self,
        attn,
        is_cross: bool,
        place_in_unet: str,
        CrossAttnEditIndex: list = [],
        CrossAttnEditWord:list = [],
        isSelfAttnReplace: bool = True,
        CrossAttnEditSteps: list = [],
        SaveSteps: int | list = None,
        isResetMask: bool = False,
        CurrentTime: str = "",
        SavedTime: list = None,
    ):
        super(AttentiveEraserAttentionControlEdit, self).forward(
            attn, is_cross, place_in_unet
        )
        CrossAttnEditSteps = list(
            range(CrossAttnEditSteps[0], CrossAttnEditSteps[1] + 1)
        )
        if type(SaveSteps) is int:
            SaveSteps = [SaveSteps]

        if is_cross or (
            self.num_self_replace[0] <= self.cur_step <= self.num_self_replace[1]
        ):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])

            attn_base, attn_repalce = attn[0], attn[1:]
            if attn.shape[-1] == 77:
                isCrossAttnReplace = False
                if isCrossAttnReplace:
                    alpha_words = self.cross_replace_alpha[self.cur_step]
                    attn_repalce_new = (
                        self.replace_cross_attention(attn_base, attn_repalce)
                        * alpha_words
                        + (1 - alpha_words) * attn_repalce
                    )
                    attn[1:] = attn_repalce_new

                h_w = int(sqrt(attn[1].shape[1]))
                square0 = attn[0].reshape(8, h_w, h_w, 77)
                square = attn[1].reshape(8, h_w, h_w, 77)
                square = square.sum(0) / square.shape[0]
                square0 = square0.sum(0) / square0.shape[0]

                if self.cur_step in SaveSteps:
                    save_as_pkl(
                        square0.to("cpu"),
                        file_name=str(list(square.to("cpu").shape))
                        + f"_{self.cur_step} ",
                        folder_path="pkls_" + CurrentTime,
                    )

                if (self.cur_step + 1) in CrossAttnEditSteps:
                    if isResetMask:
                        mask = None
                        if square.shape[0] == 16:
                            mask_16 = load_from_pkl(
                                file_name="16x16_" + SavedTime[0]
                            ).to("cuda")
                            mask = mask_16
                        if square.shape[0] == 32:
                            mask_32 = load_from_pkl(
                                file_name="32x32_" + SavedTime[1]
                            ).to("cuda")
                            mask = mask_32

                        if mask is not None:
                            for Word in CrossAttnEditWord:
                                to_zero = attn[1][:, :, Word]
                                to_zero = to_zero.reshape(8, h_w, h_w)
                                for i in range(8):
                                    to_zero[i][mask] = 0.0
                                to_zero = to_zero.reshape(8, h_w * h_w)
                                attn[1][:, :, Word] = to_zero
                    else:
                        attn[1][:, :, CrossAttnEditIndex] = 0.0

            if is_cross:
                pass
            else:
                if isSelfAttnReplace:
                    attn[1:] = self.replace_self_attention(
                        attn_base, attn_repalce, place_in_unet
                    )

            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])

        return attn

    def __init__(
        self,
        prompts,
        num_steps: int,
        self_replace_steps,
        cross_replace_steps: Union[float, Tuple[float, float]] = None,
    ):
        super(AttentiveEraserAttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        if type(self_replace_steps) is int:
            self_replace_steps = 1, self_replace_steps
        self.num_self_replace = self_replace_steps[0] - 1, self_replace_steps[1] - 1
        if cross_replace_steps is not None:
            self.cross_replace_alpha = get_time_words_attention_alpha(
                prompts, num_steps, cross_replace_steps, tokenizer=None
            ).to("cuda")
            self.mapper = get_replacement_mapper(prompts=prompts, tokenizer=None).to(
                "cuda"
            )
