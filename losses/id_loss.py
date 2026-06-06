import torch
import torch.nn as nn


class IDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, pids):
        if logits.size(0) != pids.size(0):
            raise ValueError(
                f"Batch size mismatch: logits {logits.size(0)} vs pids {pids.size(0)}"
            )
        return self.ce(logits, pids)
