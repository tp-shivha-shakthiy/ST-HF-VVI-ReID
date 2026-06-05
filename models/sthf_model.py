class STHFModel:
    def __init__(self, backbone, sthpf, classifier):
        self.backbone = backbone
        self.sthpf = sthpf
        self.classifier = classifier

    def forward(self, x):
        return x
