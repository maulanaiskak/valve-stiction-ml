import pandas as pd

from valve_stiction_ml.dataset import cap_windows_per_loop


def test_cap_windows_per_loop_caps_large_groups():
    df = pd.DataFrame({"loop_id": ["a"] * 10 + ["b"] * 3, "x": range(13)})

    capped = cap_windows_per_loop(df, max_per_loop=3, random_state=0)

    counts = capped["loop_id"].value_counts()
    assert counts["a"] == 3
    assert counts["b"] == 3  # already <= cap, kept whole


def test_cap_windows_per_loop_deterministic():
    df = pd.DataFrame({"loop_id": ["a"] * 10, "x": range(10)})

    first = cap_windows_per_loop(df, max_per_loop=4, random_state=1)
    second = cap_windows_per_loop(df, max_per_loop=4, random_state=1)

    pd.testing.assert_frame_equal(
        first.sort_values("x").reset_index(drop=True),
        second.sort_values("x").reset_index(drop=True),
    )


def test_cap_windows_per_loop_preserves_all_columns():
    df = pd.DataFrame({"loop_id": ["a"] * 5, "x": range(5), "label": ["yes"] * 5})

    capped = cap_windows_per_loop(df, max_per_loop=2)

    assert list(capped.columns) == ["loop_id", "x", "label"]
    assert len(capped) == 2
