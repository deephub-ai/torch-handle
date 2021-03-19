import torch
from torchhandle.workflow import FP16Context
def make_model(in_size, out_size, num_layers):
    layers = []
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(in_size, in_size))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(in_size, out_size))
    return torch.nn.Sequential(*tuple(layers))


if __name__ == "__main__":
    batch_size = 128  # Try, for example, 128, 256, 513.
    in_size = 4096
    out_size = 4096
    num_layers = 3
    num_samples = 128
    X, Y = torch.rand(num_samples, in_size), torch.rand(num_samples, out_size)
    dataset = torch.utils.data.TensorDataset(X, Y)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion_fn = {"fn": torch.nn.MSELoss, }
    optimizer_fn = {"fn": torch.optim.Adam,
                    "args": dict(lr=0.0001)}
    model_fn = {"fn": make_model,
                "args": dict(in_size=in_size, out_size=out_size, num_layers=num_layers)}

    trn_loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)
    loaders = {"train": trn_loader, }
    c = FP16Context(model=model_fn,
                    criterion=criterion_fn,
                    optimizer=optimizer_fn,
                    context_tag="fp16")
    train = c.make_train_session(device, dataloader=loaders)
    train.train(epochs=10)