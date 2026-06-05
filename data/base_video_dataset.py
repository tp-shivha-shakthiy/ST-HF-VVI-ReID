class BaseVideoDataset:
    def __init__(self, root, seq_len=8, transform=None):
        self.root = root
        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
