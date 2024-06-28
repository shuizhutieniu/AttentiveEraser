import streamlit as st
from PIL import Image
import os
import json
import pandas as pd
import numpy as np

st.set_page_config(page_title="Homepage", layout="wide", page_icon="💕")


def show_markdown(filename: str):
    with open(f"{filename}.md", "r", encoding="utf-8") as file:
        markdown_text = file.read()
    st.markdown(markdown_text, unsafe_allow_html=True)


# 设置页面标题
st.title("AttentiveEraser-WebUI")


col1, col2 = st.columns(2)


def display_values():
    # 获取滑动条的值
    range1 = st.session_state.range1
    range2 = st.session_state.range2

    # 显示获取的值
    st.write(f"滑动条1的范围: {range1}")
    st.write(f"滑动条2的范围: {range2}")
    st.write("Loss:0.0156 画一个折线图")


# propmt
# seed
# wordIndex

# CrossAttnEditStep
# useMaskRange

# isReplaceEmbed + range
# isReplaceSelfAttn + range
# isLowResource
# isSGD
# 8 16 32 64


with col1:
    st.subheader("control panel")
    with st.container(border=True):
        with st.container(border=True):
            left_column, right_column, rr_column = st.columns([0.7, 0.2, 0.1])
            with left_column:
                st.text_input("prompt", key="sdcax")
            with right_column:
                st.text_input("seed", key="sdca")
            with rr_column:
                st.text_input("Index", key="sd")

        with st.container(border=True):
            slider_left_column, slider_right_column = st.columns(2)
            with slider_left_column:
                st.slider(
                    "CrossAttnEditStep(1-50)",
                    min_value=1,
                    max_value=50,
                    value=(4, 14),
                    key="range1",
                )
            with slider_right_column:
                st.slider(
                    "useMaskRange(1-77, 1-7word and trailling)",
                    min_value=1,
                    max_value=77,
                    value=(1, 7),
                    key="range2",
                )

            otherslider_left_column, otherslider_right_column = st.columns(2)
            with otherslider_left_column:
                st.slider(
                    "ReplaceEmbedding(1-50)",
                    min_value=1,
                    max_value=50,
                    value=(1, 40),
                    key="range15",
                )
            with otherslider_right_column:
                st.slider(
                    "ReplaceSelfAttn(1-50)",
                    min_value=1,
                    max_value=50,
                    value=(1, 10),
                    key="range16",
                )

        with st.container(border=True):
            co1, co2, co3, co4 = st.columns(4)
            with co1:
                st.checkbox(label="SGD", value=True)
            with co2:
                st.checkbox(label="LowResource", value=False)
            with co3:
                st.checkbox(label="ReplaceEmbed", value=False)
            with co4:
                st.checkbox(label="ReplaceSelfAttn", value=True)

            cco1, cco2, cco3, cco4 = st.columns(4)
            with cco1:
                st.checkbox(label="8x8", value=True)
            with cco2:
                st.checkbox(label="16x16", value=True)
            with cco3:
                st.checkbox(label="32x32", value=True)
            with cco4:
                st.checkbox(label="64x64", value=False)

        st.button("Generate Image", type="primary")

with col2:
    with st.container(border=True):
        show_markdown("test")
        # 创建一个简单的数据集
        data = pd.DataFrame(
            {
                "日期": pd.date_range(start="1/1/2020", periods=100),
                "值": np.random.randn(100).cumsum(),
            }
        )
        # 绘制折线图
        st.line_chart(data.set_index("日期"))
