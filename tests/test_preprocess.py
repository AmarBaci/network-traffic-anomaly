import numpy as np
import pandas as pd
import pytest

from nta.preprocess import (
    build_features_and_labels,
    clean_dataframe,
    find_leakage_columns,
    split_train_val_test_for_anomaly,
)


def make_flows(n_normal=200, n_attack=100, seed=0):
    rng = np.random.default_rng(seed)
    n = n_normal + n_attack
    return pd.DataFrame(
        {
            " Flow Duration": rng.integers(1, 1000, n),
            "Total Fwd Packets": rng.integers(1, 50, n),
            "Constant": np.ones(n),
            "Source IP": ["10.0.0.1"] * n,
            "Label": ["BENIGN"] * n_normal + ["DDoS"] * n_attack,
        }
    )


def test_clean_dataframe_strips_names_and_drops_empty_columns():
    df = pd.DataFrame({" a ": [1.0, np.inf], "empty": [np.nan, np.nan]})

    cleaned = clean_dataframe(df)

    assert list(cleaned.columns) == ["a"]
    assert cleaned["a"].isna().sum() == 1  # inf replaced with NaN


def test_build_features_and_labels_drops_label_metadata_and_constants():
    df = clean_dataframe(make_flows())

    X, y = build_features_and_labels(df)

    assert set(X.columns) == {"Flow Duration", "Total Fwd Packets"}
    assert y.sum() == 100
    assert set(y.unique()) == {0, 1}


def test_build_features_and_labels_requires_label_column():
    with pytest.raises(ValueError):
        build_features_and_labels(pd.DataFrame({"x": [1, 2]}))


def test_find_leakage_columns_flags_label_like_columns():
    columns = ["Flow Duration", "Label", "attack_type", "is_malicious"]

    assert find_leakage_columns(columns, label_column="Label") == ["attack_type", "is_malicious"]


def test_split_trains_on_normal_only_without_overlap():
    X, y = build_features_and_labels(clean_dataframe(make_flows()))

    X_train, X_val, y_val, X_test, y_test = split_train_val_test_for_anomaly(X, y)

    assert (y.loc[X_train.index] == 0).all()
    assert set(y_val.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}

    indices = [set(X_train.index), set(X_val.index), set(X_test.index)]
    assert not indices[0] & indices[1]
    assert not indices[0] & indices[2]
    assert not indices[1] & indices[2]
    assert sum(len(i) for i in indices) == len(X)


def test_split_is_reproducible_with_same_seed():
    X, y = build_features_and_labels(clean_dataframe(make_flows()))

    first = split_train_val_test_for_anomaly(X, y, random_state=42)
    second = split_train_val_test_for_anomaly(X, y, random_state=42)

    assert first[0].index.equals(second[0].index)
    assert first[3].index.equals(second[3].index)


def test_split_rejects_invalid_sizes():
    X, y = build_features_and_labels(clean_dataframe(make_flows()))

    with pytest.raises(ValueError):
        split_train_val_test_for_anomaly(X, y, val_size=0.5, test_size=0.5)
