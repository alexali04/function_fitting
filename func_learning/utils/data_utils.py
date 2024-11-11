from torch.utils.data import DataLoader, Dataset
import torch


class CircleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def generate_circle_data(
    num_points: int = 1000,
    low: float = -5.0,
    high: float = 5.0
    ):
    """
    Function is (sin(t) + 3, cos(t) + 3)
    """

    t = torch.linspace(low, high, num_points, dtype=torch.float32)    # 3rd argument cannot be int
    x = torch.sin(t) + 3.0
    y = torch.cos(t) + 3.0
    

    domain_dataset = CircleDataset(x, y)
    domain_dataloader = DataLoader(domain_dataset, batch_size=64, shuffle=True)

    return domain_dataloader, x, y