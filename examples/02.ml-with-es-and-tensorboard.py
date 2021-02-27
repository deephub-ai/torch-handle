import torch

from torchhandle.workflow import BaseContext, Metric


class C1(BaseContext):
    def init_state_fn(self):
        state = super().init_state_fn()
        state.es_current_step = 0
        state.es_metric = 1000
        return state

    def early_stopping_fn(self, session):
        """
        return true to stop
        """
        es_steps = 5
        valid_loss = session.epoch_metric["valid_loss"]
        session.state.es_current_step = session.state.es_current_step + 1
        if valid_loss < session.state.es_metric:
            session.state.es_metric = valid_loss
            session.state.es_current_step = 0
        elif session.state.es_current_step >= es_steps:
            return True

        return False


class RMSE(Metric):
    def __init__(self):
        self.diff = None

    def map(self, state):

        target = state.target_batch.cpu().detach().unsqueeze(dim=1)
        output = state.output_batch.cpu().detach()
        if self.diff is None:
            self.diff = torch.pow(target - output, 2)
        else:
            self.diff = torch.cat([self.diff, torch.pow(target - output, 2)], dim=0)

    def reduce(self):
        mse = torch.sum(self.diff) / self.diff.shape[0]
        rmse = torch.sqrt(mse)
        return [rmse]

    @property
    def name(self) -> list:
        return ["RMSE"]

    @property
    def best(self) -> list:
        return ["min"]


if __name__ == "__main__":
    num_samples, num_features = int(1e4), int(1e1)
    print(num_samples, num_features)
    X, Y = torch.rand(num_samples, num_features), torch.rand(num_samples)
    dataset = torch.utils.data.TensorDataset(X, Y)
    trn_loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=0)
    loaders = {"train": trn_loader, "valid": val_loader}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = {"fn": torch.nn.Linear,
             "args": {"in_features": 10, "out_features": 1}
             }
    criterion = {"fn": torch.nn.MSELoss
                 }
    optimizer = {"fn": torch.optim.Adam
                 }
    metric_fn = [{"fn": RMSE}]
    c = C1(model=model,
           criterion=criterion,
           optimizer=optimizer,
           metric_fn=metric_fn,
           output_dir="./outputs",
           logging_file="output.log",
           context_tag="ex02")
    train = c.make_train_session(device, dataloader=loaders)
    train.train(epochs=100)
    print("this line was not write to log file")
