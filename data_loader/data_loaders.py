from torchvision import datasets, transforms
from base.base_data_loader import BaseDataLoader

class MnistDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, train=True):
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(root=self.data_dir, train=train, download=False, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
