import os
import sys
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim

from src.learning.logger.train_logs import TrainingLogger


class _Stepper:
    """Minimal stand-in for TrainingStepper: save_checkpoint pulls encoder/decoder/
    optimizer state dicts off it."""
    def __init__(self):
        self.encoder = nn.Linear(8, 4)
        self.decoder = nn.Linear(4, 8)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-3)


def test_save_checkpoint():
    print('=== test_save_checkpoint ===')
    with tempfile.TemporaryDirectory() as tmpdir:
        stepper = _Stepper()
        logger = TrainingLogger(log_dir=tmpdir)

        # current API: save_checkpoint(stepper, step) -> {encoder, decoder, optimizer}
        logger.save_checkpoint(stepper, step=1)
        checkpoint_path = os.path.join(tmpdir, 'checkpoints', 'step_1.pt')
        print('checkpoint path:', checkpoint_path)

        assert os.path.exists(checkpoint_path), 'Checkpoint file was not created.'
        checkpoint = torch.load(checkpoint_path)
        assert 'encoder' in checkpoint
        assert 'decoder' in checkpoint
        assert 'optimizer' in checkpoint


if __name__ == '__main__':
    test_save_checkpoint()
    print('test_logger.py completed successfully.')
