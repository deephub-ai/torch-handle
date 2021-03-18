import numpy as np
import torch

class Metric:
    def __init__(self):
        self.metric_list = []

    def map(self, state): raise NotImplementedError

    def reduce(self) -> list: raise NotImplementedError

    @property
    def name(self) -> list: raise NotImplementedError

    @property
    def best(self) -> list: raise NotImplementedError


class metric_loss(Metric):

    def map(self, state):
        loss = state.loss.cpu().detach()
        # target=state.target_batch.cpu().detach()
        # output=state.output_batch.cpu().detach()
        if len(loss.size()) == 0:
            loss = loss.item()
        else:
            loss = torch.mean(loss).item()
        self.metric_list.append(loss)
        state.metric = {"loss": loss}

    def reduce(self) -> list:
        return [np.mean(self.metric_list)]

    @property
    def name(self) -> list:
        return ["loss"]

    @property
    def best(self) -> list:
        return ["min"]
