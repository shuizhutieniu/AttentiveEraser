import streamlit as st
import pandas as pd

st.set_page_config(page_title="Homepage", layout="wide", page_icon="💕")

losses = [
    0.8060,
    0.6066,
    0.1067,
    0.0182,
    0.0101,
    0.0115,
    0.0209,
    0.0275,
    0.0293,
    0.0292,
    0.0299,
    0.0335,
    0.0498,
    0.0466,
    0.0509,
    0.0348,
    0.0320,
    0.0051,
    0.0079,
    0.0025,
    0.0104,
    0.0025,
    0.0062,
    0.0034,
    0.0088,
]


def show_markdown(filename: str):
    with open(f"{filename}.md", "r", encoding="utf-8") as file:
        markdown_text = file.read()
    st.markdown(markdown_text, unsafe_allow_html=True)


# 设置页面标题
st.title("AttentiveEraser-WebUI")


col1, col2 = st.columns(2)


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
                    "Cross-Attention Map Edit Steps(1-50)",
                    min_value=1,
                    max_value=50,
                    value=(4, 14),
                    key="range1",
                )
            with slider_right_column:
                st.slider(
                    "Words that use masks(1-77)",
                    min_value=1,
                    max_value=77,
                    value=(1, 7),
                    key="range2",
                )

            otherslider_left_column, otherslider_right_column = st.columns(2)
            with otherslider_left_column:
                st.slider(
                    "Replace Embedding Steps(1-50)",
                    min_value=1,
                    max_value=50,
                    value=(1, 40),
                    key="range15",
                )
            with otherslider_right_column:
                st.slider(
                    "Replace Self-Attention Map Steps(1-50)",
                    min_value=1,
                    max_value=50,
                    value=(1, 10),
                    key="range16",
                )

        with st.container(border=True):
            co1, co2, co3, co4 = st.columns(4)
            with co1:
                st.checkbox(label="Use Gradient Descent", value=True)
            with co2:
                st.checkbox(label="Low Resource Mode", value=False)
            with co3:
                st.checkbox(label="Replace WordEmbedding", value=False)
            with co4:
                st.checkbox(label="Replace Self-Attention Map", value=True)

            cco1, cco2, cco3, cco4 = st.columns(4)
            with cco1:
                st.checkbox(
                    label="extract from :red-background[[8x8]] AttnMap", value=False
                )
            with cco2:
                st.checkbox(
                    label="extract from :red-background[[16x16]] AttnMap", value=True
                )
            with cco3:
                st.checkbox(
                    label="extract from :red-background[[32x32]] AttnMap", value=True
                )
            with cco4:
                st.checkbox(
                    label="extract from :red-background[[64x64]] AttnMap", value=False
                )

        st.button("Generate Image", type="primary")

with col2:
    with st.container(border=True):
        show_markdown("test")
        data = pd.DataFrame({"Iteration": range(len(losses)), "Loss": losses})
        # 绘制折线图
        st.line_chart(
            data.set_index("Iteration"),
            x_label="Iteration",
            y_label="Loss",
        )
