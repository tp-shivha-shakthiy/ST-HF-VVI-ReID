class IDLoss:
    def __init__(self, num_classes=100):
        self.num_classes = num_classes

    def __call__(self, logits, targets):
        return logits.sum()
