import torch

from torchhandle.workflow import BaseContext

if __name__ == "__main__":
    num_samples, num_features = int(1e4), int(1e1)
    print(num_samples, num_features)
    X, Y = torch.rand(num_samples, num_features), torch.rand(num_samples)
    dataset = torch.utils.data.TensorDataset(X, Y)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = {"fn": torch.nn.Linear,
             "args": {"in_features": 10, "out_features": 1}
             }
    criterion = {"fn": torch.nn.MSELoss
                 }
    optimizer = {"fn": torch.optim.Adam
                 }
    scheduler = {"fn": torch.optim.lr_scheduler.StepLR,
                 "args": {"step_size": 2, "gamma": 0.9}
                 }
    c = BaseContext(model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir="./outputs",
                    logging_file="log.txt",
                    context_tag="ex03")
    for i in range(5):
        # use all data just for  for demo , not actual Kford
        trn_loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=0)
        loaders = {"train": trn_loader, "valid": val_loader}

        session = c.make_train_session(device, dataloader=loaders, fold_tag=i)
        session.train(10)
