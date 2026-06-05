class VideoSampler:
    def __init__(self, dataset, batch_size, num_workers=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
