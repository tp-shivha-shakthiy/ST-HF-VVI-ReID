import torch


class Trainer:
    def __init__(self, model, criterion, optimizer, cfg):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.cfg = cfg

    def training_step(self, batch):
        frames = batch["frames"]
        pids = batch["pids"]

        self.optimizer.zero_grad()
        outputs = self.model(frames, modalities=batch.get("modalities"))
        losses = self.criterion(outputs, pids)
        total = losses["total"]

        if not torch.isfinite(total):
            raise RuntimeError(f"Loss is not finite: {total}")

        total.backward()
        self.optimizer.step()

        return losses

    def train_epoch(self, loader):
        return {}

    def fit(self, train_loader, val_loader, epochs):
        pass
