import pandas as pd
from pandas.io.formats.style import Styler


def _style_negatives(df: pd.DataFrame) -> Styler:
    """Return a Styler adding class "neg" to negative numeric cells."""
    classes = pd.DataFrame("", index=df.index, columns=df.columns)
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        classes.loc[df[col] < 0, col] = "neg"
    return df.style.set_td_classes(classes)

