class TripletLoss:
    def __init__(self, margin=0.3):
        self.margin = margin

    def __call__(self, features, targets):
        return features.sum()
