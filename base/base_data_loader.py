from torch.utils.data import DataLoader, Dataset

class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.dataset = dataset

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }
        super().__init__(**self.init_kwargs)
