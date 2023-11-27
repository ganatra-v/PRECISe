from constants import Constants
from medmnist import DermaMNIST, BreastMNIST, RetinaMNIST, PneumoniaMNIST, OCTMNIST, OrganSMNIST
from torch.utils.data import Dataset, DataLoader

class MedDataset(Dataset):
    def __init__(self, dataset_name, split = 'train', transform = None):        
        assert dataset_name in Constants.SUPPORTED_DATASETS, f"Unsupported dataset {dataset_name}"
        assert split in ['train', 'val', 'test'], f"Unsupported split {split}"
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.load_dataset()
        super(MedDataset, self).__init__()
    
    def load_dataset(self):
        if self.dataset_name == Constants.BREAST_MNIST:
            self.data = BreastMNIST(split = self.split, download = True)
        elif self.dataset_name == Constants.RETINA_MNIST:
            self.data = RetinaMNIST(split = self.split, download = True)
        elif self.dataset_name == Constants.PNEUMONIA_MNIST:
            self.data = PneumoniaMNIST(split = self.split, download = True)
        elif self.dataset_name == Constants.OCT_MNIST:
            self.data = OCTMNIST(split = self.split, download = True)
        
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


    
