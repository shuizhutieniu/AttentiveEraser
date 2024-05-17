import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import os
import random
import torch
import pickle
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo


def get_current_time():
    china_zone = ZoneInfo("Asia/Shanghai")
    current_time = datetime.now(china_zone)
    return current_time.strftime("%H%M%S")


def save_as_pkl_simple(var, file_name: str):
    if not file_name.endswith(".pkl"):
        file_name += ".pkl"

    with open(file_name, "wb") as file:
        pickle.dump(var, file)


def create_mask(tensor, percent):
    flat_tensor = tensor.flatten()
    sorted_tensor, _ = torch.sort(flat_tensor, descending=True)

    num_elements_to_zero = int(len(flat_tensor) * percent / 100)

    if num_elements_to_zero > 0:
        threshold = sorted_tensor[num_elements_to_zero - 1]
    else:
        threshold = sorted_tensor[0] + 1

    mask = tensor >= threshold
    modified_tensor = tensor.clone()
    modified_tensor[mask] = 0

    return modified_tensor, mask


def SaveAsZip(filetype: str = "pkl", zipname: str = "output", path="./pkls"):
    extension = f".{filetype}"

    with zipfile.ZipFile(f"{zipname}.zip", "w") as myzip:
        for foldername, subfolders, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(extension):
                    filePath = os.path.join(foldername, filename)
                    myzip.write(filePath, arcname=filename)


def display_tensors_as_images(*tensors):
    for tensor in tensors:
        if tensor.shape != (3, 512, 512):
            raise ValueError("Error")
    n = len(tensors)
    _, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for i, tensor in enumerate(tensors):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        tensor_transformed = tensor.permute(1, 2, 0).numpy()
        axes[i].imshow(tensor_transformed)
        axes[i].axis("off")
    plt.show()


def GlobalSeed(seed: Optional[int] = None, workers: bool = False):
    if seed is None:
        seed = os.environ.get("PL_GLOBAL_SEED")
    seed = int(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"


import matplotlib.pyplot as plt


import matplotlib.pyplot as plt


def plot_tensors_heatmaps_advanced(
    *tensors,
    N=7,
    titles: list = None,
    isColorbar: bool = True,
    colormap: str = "hot",
    WordIndex: int = 5,
    saveToFile: bool = False,
    filePath: str = "heatmap",
    colormapValue: list = None,
    title_fontsize: int = 12,  # 新添加的参数用于设置标题字体大小
):
    if titles is None:
        titles = [f"{t.size(0)}x{t.size(1)}" for t in tensors]

    max_size = max(max(t.size(0), t.size(1)) for t in tensors)
    num_rows = (len(tensors) + N - 1) // N
    num_cols = min(len(tensors), N)
    figsize = (3 * num_cols, 3 * num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

    for ax, tensor, title in zip(axes.flatten(), tensors, titles):
        tensor = tensor[:, :, WordIndex] if tensor.dim() == 3 else tensor
        if not colormapValue:
            im = ax.imshow(
                tensor.numpy(), cmap=colormap, extent=(0, max_size, max_size, 0)
            )
        else:
            im = ax.imshow(
                tensor.numpy(),
                cmap=colormap,
                extent=(0, max_size, max_size, 0),
                vmin=colormapValue[0],
                vmax=colormapValue[1],
            )
        ax.set_xlim([0, max_size])
        ax.set_ylim([max_size, 0])
        ax.axis("off")
        if isColorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(str(title), fontsize=title_fontsize)  # 设置标题的字体大小

    extra_axes = len(axes.flatten()) - len(tensors)
    if extra_axes > 0:
        for ax in axes.flatten()[-extra_axes:]:
            ax.axis("off")

    plt.subplots_adjust(hspace=0.2)
    plt.tight_layout()

    if saveToFile:
        plt.savefig(filePath + ".png")
        plt.close(fig)
    else:
        plt.show()


def load_from_pkl(file_name):
    with open(f"{file_name}.pkl", "rb") as file:
        return pickle.load(file)


def update_alpha_time_word(
    alpha,
    bounds,
    prompt_ind: int,
    word_inds: Optional[torch.Tensor] = None,
):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[:start, prompt_ind, word_inds] = 0
    alpha[start:end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(
    prompts,
    num_steps,
    cross_replace_steps,
    tokenizer,
    max_num_words=77,
):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0.0, 1.0)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(
            alpha_time_words, cross_replace_steps["default_"], i
        )
    for key, item in cross_replace_steps.items():
        if key != "default_":
            pass
    alpha_time_words = alpha_time_words.reshape(
        num_steps + 1, len(prompts) - 1, 1, 1, max_num_words
    )
    return alpha_time_words


def get_replacement_mapper_(x: str, y: str, tokenizer, max_len=77):
    words_x = x.split(" ")
    words_y = y.split(" ")
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
    inds_source = [[5]]
    inds_target = [[5]]
    mapper = np.zeros((max_len, max_len))
    i = j = 0
    cur_inds = 0
    while i < max_len and j < max_len:
        if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:
            inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]
            if len(inds_source_) == len(inds_target_):
                mapper[inds_source_, inds_target_] = 1
            else:
                ratio = 1 / len(inds_target_)
                for i_t in inds_target_:
                    mapper[inds_source_, i_t] = ratio
            cur_inds += 1
            i += len(inds_source_)
            j += len(inds_target_)
        elif cur_inds < len(inds_source):
            mapper[i, j] = 1
            i += 1
            j += 1
        else:
            mapper[j, j] = 1
            i += 1
            j += 1

    return torch.from_numpy(mapper).float()


def get_replacement_mapper(prompts, tokenizer, max_len=77):
    x_seq = prompts[0]
    mappers = []
    for i in range(1, len(prompts)):
        mapper = get_replacement_mapper_(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
    return torch.stack(mappers)


def aggregate_attention(
    ctrler, res: int, from_where, is_cross: bool, select: int, prompts
):
    out = []
    attention_maps = ctrler.get_average_attention()
    num_pixels = res**2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[
                    select
                ]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()
