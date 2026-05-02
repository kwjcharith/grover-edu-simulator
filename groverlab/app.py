"""Streamlit entry point for standalone local and online GroverLab use."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groverlab.grover_education import CORE_WARNING, MISCONCEPTION_WARNINGS


def main() -> None:
    """Render the Streamlit app shell."""

    import streamlit as st

    st.set_page_config(page_title="GroverLab", page_icon="G", layout="wide")
    st.title("GroverLab")
    st.caption("Research-grade educational simulator for Grover's Algorithm")
    st.warning(CORE_WARNING)

    st.subheader("Dataset")
    st.text_area("Comma-separated data", placeholder="apple, banana, cherry, date")
    st.file_uploader("Or upload CSV", type=["csv"])
    st.text_input("Target item")

    st.subheader("Misconception checks")
    for warning in MISCONCEPTION_WARNINGS:
        st.info(warning)

    st.info("The full simulation workflow will be connected in the next implementation step.")


if __name__ == "__main__":
    main()
