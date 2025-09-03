import pandas as pd
import streamlit as st

from .layout import setup_page


def main() -> None:
    setup_page()
    st.title("Sticky DataFrame Debug")

    df = pd.DataFrame({f"Col{i}": range(100) for i in range(20)})
    st.dataframe(df, height=200)
    st.dataframe(df)


if __name__ == "__main__":
    main()
