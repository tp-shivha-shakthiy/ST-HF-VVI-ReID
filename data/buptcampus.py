from .base_video_dataset import BaseVideoDataset


class BUPTcampus(BaseVideoDataset):
    def __init__(self, root, seq_len=8, transform=None, split="train"):
        super().__init__(root, seq_len, transform)
        self.split = split
