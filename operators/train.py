from dataset import DatasetConductivity
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader


dataset = DatasetConductivity(root="/data/Bei/atrium_conductivity_600")
train_data, test_data = train_test_split(dataset, test_size=0.1, random_state=42)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)