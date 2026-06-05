class Trainer:
    def __init__(self, model, criterion, optimizer, cfg):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.cfg = cfg

    def train_epoch(self, loader):
        return {}

    def fit(self, train_loader, val_loader, epochs):
        pass
