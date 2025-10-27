from ucimlrepo import fetch_ucirepo
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(57, 200),
            nn.Sigmoid(),
            nn.Linear(200, 150),
            nn.Sigmoid(),
            nn.Linear(150, 1),
        )

    def forward(self, X):
        return self.linear_sigmoid_stack(X)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")




if __name__ == '__main__':
    spambase = fetch_ucirepo(id=94)
    learning_rate = 1e-3
    batch_size = 16
    epochs = 1000

    X = spambase.data.features.to_numpy()
    y = spambase.data.targets.to_numpy()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Device:", device)
    model = NeuralNetwork().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for i in range(epochs):
        print(f"Epoch {i+1}\n-------------------------------")
        train_loop(loader, model, loss_fn, optimizer)

    