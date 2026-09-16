import numpy as np
import pandas as pd

from valve_stiction_ml.dataset import iter_windows, load_signal, window_signal


def test_window_signal_non_overlapping_shapes():
    pv = np.arange(250, dtype=float)
    op = np.arange(250, dtype=float) * 2

    windows = window_signal(pv, op, window_size=100)

    assert len(windows) == 2  # 250 // 100, remainder dropped
    for pv_w, op_w in windows:
        assert pv_w.shape == (100,)
        assert op_w.shape == (100,)
    # non-overlapping: second window starts where the first ends
    assert windows[0][0][-1] == 99
    assert windows[1][0][0] == 100


def test_window_signal_too_short_returns_empty():
    pv = np.arange(50, dtype=float)
    op = np.arange(50, dtype=float)

    assert window_signal(pv, op, window_size=100) == []


def test_load_signal_selects_only_pv_op(tmp_path):
    df = pd.DataFrame(
        {
            "Time": [1, 2, 3],
            "SP": [10, 10, 10],
            "PV": [1.0, 2.0, 3.0],
            "OP": [4.0, 5.0, 6.0],
            "Error": [0, 0, 0],
        }
    )
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_signal(csv_path)

    assert list(loaded.columns) == ["PV", "OP"]
    assert len(loaded) == 3


def test_load_signal_raises_on_missing_columns(tmp_path):
    df = pd.DataFrame({"Time": [1, 2], "SP": [10, 10]})
    csv_path = tmp_path / "bad.csv"
    df.to_csv(csv_path, index=False)

    try:
        load_signal(csv_path)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_iter_windows_uses_manifest_metadata(tmp_path):
    raw_dir = tmp_path / "raw"
    (raw_dir / "ISDB" / "yes").mkdir(parents=True)
    df = pd.DataFrame({"PV": np.arange(200, dtype=float), "OP": np.arange(200, dtype=float)})
    df.to_csv(raw_dir / "ISDB" / "yes" / "loop1.csv", index=False)

    manifest = pd.DataFrame(
        [
            {
                "source_file": "loop1.csv",
                "relative_path": "ISDB/yes/loop1.csv",
                "origin_dataset": "ISDB",
                "folder_label": "yes",
                "loop_id": "ISDB__yes__loop1",
            }
        ]
    )

    windows = iter_windows(manifest, raw_dir, window_size=100)

    assert len(windows) == 2
    assert all(w.loop_id == "ISDB__yes__loop1" for w in windows)
    assert all(w.origin_dataset == "ISDB" for w in windows)
    assert [w.window_index for w in windows] == [0, 1]
