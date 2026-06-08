import torch
import pytest

from models.sthpf import FixedSTHPF


def test_fixed_sthpf_output_shape_matches_input():
    module = FixedSTHPF(fs=10, ft=2)
    video = torch.randn(2, 6, 3, 288, 144)

    output = module(video)

    assert output.shape == video.shape


def test_fixed_sthpf_output_is_finite():
    module = FixedSTHPF(fs=10, ft=2)
    video = torch.randn(2, 6, 3, 64, 32)

    output = module(video)

    assert torch.isfinite(output).all()


def test_fixed_sthpf_output_is_real_tensor():
    module = FixedSTHPF(fs=10, ft=2)
    video = torch.randn(2, 6, 3, 64, 32)

    output = module(video)

    assert not torch.is_complex(output)


def test_fixed_sthpf_rejects_wrong_shape():
    module = FixedSTHPF(fs=10, ft=2)
    wrong_input = torch.randn(2, 3, 288, 144)

    with pytest.raises(ValueError):
        module(wrong_input)


def test_fixed_sthpf_supports_small_cutoffs():
    module = FixedSTHPF(fs=1, ft=1)
    video = torch.randn(2, 6, 3, 64, 32)

    output = module(video)

    assert output.shape == video.shape
    assert torch.isfinite(output).all()
