class Baseline:
    def __init__(self, backbone, feature_dim=2048, num_classes=100):
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def forward(self, x):
        return x
