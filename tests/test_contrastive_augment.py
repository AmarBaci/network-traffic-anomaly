import pytest
import torch

from nta.contrastive_augment import TabularAugmentation


def test_returns_two_views_with_input_shape():
    torch.manual_seed(0)
    x = torch.randn(8, 5)

    v1, v2 = TabularAugmentation()(x)

    assert v1.shape == x.shape
    assert v2.shape == x.shape
    assert not torch.equal(v1, v2)


def test_does_not_modify_input():
    x = torch.randn(8, 5)
    original = x.clone()

    TabularAugmentation()(x)

    assert torch.equal(x, original)


def test_mask_only_zeroes_or_keeps_values():
    torch.manual_seed(0)
    x = torch.rand(64, 10) + 1.0  # strictly positive, so zeros come from masking

    view, _ = TabularAugmentation(strategy="mask", mask_prob=0.5)(x)

    kept = view != 0
    assert torch.equal(view[kept], x[kept])
    assert (~kept).any()


def test_jitter_scales_within_range():
    torch.manual_seed(0)
    x = torch.ones(32, 6)

    view, _ = TabularAugmentation(strategy="jitter", jitter_range=(0.8, 1.2))(x)

    assert view.min() >= 0.8
    assert view.max() <= 1.2


def test_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        TabularAugmentation(strategy="rotate")
