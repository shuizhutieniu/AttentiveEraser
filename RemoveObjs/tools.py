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


def average_tensors(*tensors):
    """
    使用更高效的方式计算形状为[x, x, 77]的多个张量的平均值。
    参数:
        *tensors: 可变数量的torch.Tensor对象，每个张量形状应为[x, x, 77]。
    返回:
        一个torch.Tensor对象，包含了输入张量的平均值，形状为[x, x, 77]。
    """
    # 检查输入
    if not tensors:
        raise ValueError("至少需要一个张量")
    
    # 将所有张量堆叠在一个新的维度上
    stacked_tensors = torch.stack(tensors, dim=0)
    
    # 沿着新的维度计算均值，得到最终的平均张量
    mean_tensor = torch.mean(stacked_tensors, dim=0)
    print(len(tensors))
    return mean_tensor


def get_current_time():
    """
    如其名
    """
    china_zone = ZoneInfo("Asia/Shanghai")
    current_time = datetime.now(china_zone)
    return current_time.strftime("%H%M%S")


def save_as_pkl_simple(var, file_name: str):
    """
    将变量保存为pkl文件
    """
    if not file_name.endswith(".pkl"):
        file_name += ".pkl"

    with open(file_name, "wb") as file:
        pickle.dump(var, file)


def zero_top_percent_with_mask(tensor, percent):
    """
    将输入的二维 torch.Tensor 的前百分比的元素设为 1（设置为1是为了方便查看，实际应用中是设为0），并返回修改后的张量及对应的mask。

    arguments:
        tensor (torch.Tensor): 输入的二维张量。
        percent (float): 需要置为 1 的元素所占的百分比。此值应介于 0 和 100 之间。

    returns:
        tuple: 包含两个元素的元组。
               第一个元素是修改后的张量，
               第二个元素是一个布尔掩码，标示哪些元素被设置为 1。
    """
    if tensor.dim() != 2:
        raise ValueError("输入必须是一个二维的torch.Tensor。")

    flat_tensor = tensor.flatten()
    sorted_tensor, indices = torch.sort(flat_tensor, descending=True)

    num_elements_to_zero = int(len(flat_tensor) * percent / 100)

    if num_elements_to_zero > 0:
        threshold = sorted_tensor[num_elements_to_zero - 1]
    else:
        threshold = sorted_tensor[0] + 1

    mask = tensor >= threshold
    modified_tensor = tensor.clone()
    modified_tensor[mask] = 1

    return modified_tensor, mask


def SaveAsZip(filetype: str = "pkl", zipname: str = "output", path="./pkls"):
    """
    遍历指定目录，查找特定扩展名的文件，并将这些文件打包为一个 ZIP 文件。

    arguments:
        filetype (str, 可选): 搜索的文件扩展名，默认为 'pkl'。
        zipname (str, 可选): 生成的 ZIP 文件的名称，默认为 'output'。
        path (str, 可选): 要搜索文件的起始目录，默认为 './pkls'。

    no returns
    """
    extension = f".{filetype}"

    with zipfile.ZipFile(f"{zipname}.zip", "w") as myzip:
        for foldername, subfolders, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(extension):
                    filePath = os.path.join(foldername, filename)
                    myzip.write(filePath, arcname=filename)


def display_tensors_as_images(*tensors):
    """
    如其名
    """
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
    """
    全局随机种子
    """
    if seed is None:
        seed = os.environ.get("PL_GLOBAL_SEED")
    seed = int(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"


def plot_tensors_heatmaps_advanced(
    *tensors,
    N=7,
    titles: list = None,
    isColorbar: bool = True,
    colormap: str = "hot",
    WordIndex: int = 5,
):
    """
    绘制一系列张量的热图，每个张量可以是二维正方形或具有形状 [x, x, 77] 的三维张量。
    对于三维张量，函数会根据 WordIndex 参数选择一个特定的切片进行绘制。

    arguments:
        *tensors: 一个或多个二维或三维张量，必须确保三维张量的最后一个维度大小为77。
        N (int, 可选): 每行显示的热图数量，默认为7。
        titles (list, 可选): 每个热图的标题列表，如果未提供，默认为每个张量的维度。
        isColorbar (bool, 可选): 是否在每个热图旁边显示颜色条，默认为True。
        colormap (str, 可选): 热图使用的颜色映射，默认为'hot'。
        WordIndex (int, 可选): 如果张量是三维的，选择哪一个切片进行绘制，默认为第五个切片。

    no returns
    """
    if titles is None:
        titles = [f"{t.size(0)}x{t.size(1)}" for t in tensors]

    max_size = max(max(t.size(0), t.size(1)) for t in tensors)
    num_rows = (len(tensors) + N - 1) // N
    num_cols = min(len(tensors), N)
    figsize = (20, 3 * num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

    for ax, tensor, title in zip(axes.flatten(), tensors, titles):
        tensor = tensor[:, :, WordIndex] if len(list(tensor.shape)) == 3 else tensor
        im = ax.imshow(tensor.numpy(), cmap=colormap, extent=(0, max_size, max_size, 0))
        ax.set_xlim([0, max_size])
        ax.set_ylim([max_size, 0])
        ax.axis("off")
        if isColorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)

    extra_axes = len(axes.flatten()) - len(tensors)
    if extra_axes > 0:
        for ax in axes.flatten()[-extra_axes:]:
            ax.axis("off")

    plt.subplots_adjust(hspace=0.2)
    plt.tight_layout()
    plt.show()


def plot_tensors_heatmaps(
    *tensors,
    figsize=(20, 3),
    titles=None,
    isColorbar: bool = True,
    colormap: str = "hot",
    WordIndex: int = 5,
):
    """
    老版本 弃用
    """
    if titles is None:
        titles = [f"{t.size(0)}x{t.size(1)}" for t in tensors]

    max_size = max(max(t.size(0), t.size(1)) for t in tensors)

    fig, axes = plt.subplots(1, len(tensors), figsize=figsize)
    if len(tensors) == 1:
        axes = [axes]

    for ax, tensor, title in zip(axes, tensors, titles):
        im = ax.imshow(
            tensor[:, :, WordIndex].numpy(),
            cmap=colormap,
            extent=(0, max_size, max_size, 0),
        )

        ax.set_xlim([0, max_size])
        ax.set_ylim([max_size, 0])
        ax.axis("off")
        if isColorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(title)

    plt.tight_layout()
    plt.show()


def load_from_pkl(file_name):
    """
    如其名
    """
    with open(f"{file_name}.pkl", "rb") as file:
        return pickle.load(file)



# 新增



def update_alpha_time_word(
    alpha,
    bounds,
    prompt_ind: int,
    word_inds: Optional[torch.Tensor] = None,
):
    """
    alpha, [51, 3, 77]
    bounds, (0,0.8)
    prompt_ind, 0-2
    word_inds, None
    """
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
    # [51, 3, 77]
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
    """
    x = "A painting of a squirrel eating a burger",
    y = "A painting of a lion eating a burger",
    """
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
    # inds_replace = [4]
    # inds_source = [get_word_inds(x, i, tokenizer) for i in inds_replace]
    # inds_target = [get_word_inds(y, i, tokenizer) for i in inds_replace]
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