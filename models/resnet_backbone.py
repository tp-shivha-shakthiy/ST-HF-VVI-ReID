class ResNetBackbone:
    def __init__(self, name="resnet50", pretrained=True):
        self.name = name
        self.pretrained = pretrained
        self.out_dim = 2048

    def forward(self, x):
        return x
