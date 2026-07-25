"""Integration tests: StationaryField and TimeVaryingField with FiLM conditioning."""

import pytest
import torch

from src.resnet_lddmm.fields.time_varying import StationaryField, TimeVaryingField
from src.resnet_lddmm.conditioning.film import FiLMConditioning, ConcatConditioning
from src.resnet_lddmm.conditioning.factory import create_conditioning


class TestStationaryFieldWithFiLM:
    """Test StationaryField with FiLMConditioning."""

    def test_stationary_with_film(self):
        """Verify StationaryField works with FiLM conditioning."""
        film_cond = FiLMConditioning(n_z=256, output_dim=3)
        field = StationaryField(conditioning=film_cond)

        x = torch.randn(2, 10, 3)
        code = torch.randn(2, 256)

        v = field(x, code=code)

        assert v.shape == (2, 10, 3)
        assert torch.isfinite(v).all()

    def test_stationary_with_film_from_config(self):
        """Verify StationaryField with FiLM created from config."""
        config = {"method": "film", "n_z": 256, "output_dim": 3}
        conditioning = create_conditioning(config)
        field = StationaryField(conditioning=conditioning)

        x = torch.randn(2, 10, 3)
        code = torch.randn(2, 256)

        v = field(x, code=code)

        assert v.shape == (2, 10, 3)

    def test_stationary_film_vs_concat(self):
        """Verify both FiLM and concat produce outputs with correct shapes."""
        x = torch.randn(2, 10, 3)
        code = torch.randn(2, 256)

        # FiLM version
        film_cond = FiLMConditioning(n_z=256, output_dim=3)
        field_film = StationaryField(conditioning=film_cond)
        v_film = field_film(x, code=code)

        # Concat version
        concat_cond = ConcatConditioning(n_z=256)
        field_concat = StationaryField(conditioning=concat_cond)
        v_concat = field_concat(x, code=code)

        assert v_film.shape == v_concat.shape == (2, 10, 3)
        assert torch.isfinite(v_film).all()
        assert torch.isfinite(v_concat).all()

    def test_stationary_film_gradient_flow(self):
        """Verify gradients flow through StationaryField with FiLM."""
        film_cond = FiLMConditioning(n_z=256, output_dim=3)
        field = StationaryField(conditioning=film_cond)

        x = torch.randn(2, 10, 3, requires_grad=True)
        code = torch.randn(2, 256, requires_grad=True)

        v = field(x, code=code)
        loss = v.sum()
        loss.backward()

        assert x.grad is not None
        assert code.grad is not None
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(code.grad).all()


class TestTimeVaryingFieldWithFiLM:
    """Test TimeVaryingField with FiLMConditioning."""

    def test_time_varying_with_film(self):
        """Verify TimeVaryingField works with FiLM conditioning."""
        film_cond = FiLMConditioning(n_z=256, output_dim=3)
        field = TimeVaryingField(num_blocks=5, width=64, conditioning=film_cond)

        x = torch.randn(2, 10, 3)
        code = torch.randn(2, 256)

        v = field(x, step=0, code=code)

        assert v.shape == (2, 10, 3)
        assert torch.isfinite(v).all()

    def test_time_varying_with_film_from_config(self):
        """Verify TimeVaryingField with FiLM created from config."""
        config = {"method": "film", "n_z": 256, "output_dim": 3}
        conditioning = create_conditioning(config)
        field = TimeVaryingField(num_blocks=5, width=64, conditioning=conditioning)

        x = torch.randn(2, 10, 3)
        code = torch.randn(2, 256)

        v = field(x, step=0, code=code)

        assert v.shape == (2, 10, 3)

    def test_time_varying_film_steps(self):
        """Verify TimeVaryingField with FiLM across different steps."""
        film_cond = FiLMConditioning(n_z=256, output_dim=3)
        field = TimeVaryingField(num_blocks=5, width=64, conditioning=film_cond)

        x = torch.randn(2, 10, 3)
        code = torch.randn(2, 256)

        for step in range(5):
            v = field(x, step=step, code=code)
            assert v.shape == (2, 10, 3)
            assert torch.isfinite(v).all()
