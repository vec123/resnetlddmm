"""Null object code implementation."""

from src.resnet_lddmm.codes.base import ShapeCode


class NoCode(ShapeCode):
    """Null object: no codes, no regularisation.

    Used when training per-pair (Milestone A) with no learned shape embedding.
    Implements the Null Object pattern to eliminate if-branches in the stepper.
    """

    def forward(self, batch) -> None:
        """Returns None unconditionally (no codes available).

        Args:
            batch: ignored

        Returns:
            None
        """
        return None

    def penalty(self) -> None:
        """Returns None unconditionally (no regularisation term).

        Returns:
            None
        """
        return None
