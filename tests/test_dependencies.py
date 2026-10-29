"""Check third-party APIs whose metadata alone does not ensure compatibility."""

import janitor  # noqa: F401 - registers pandas extension methods
import matplotlib.pyplot as plt
import missingno
import pandas as pd


def test_pyjanitor_clean_names():
    frame = pd.DataFrame({"Column Name": [1]})
    assert frame.clean_names().columns.tolist() == ["column_name"]


def test_missingno_matrix():
    frame = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, 2.0, 3.0]})
    assert isinstance(missingno.matrix(frame, sparkline=False), plt.Axes)
