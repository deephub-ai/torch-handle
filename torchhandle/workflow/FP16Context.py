import torch

from torchhandle.workflow import Session
from torchhandle.workflow.BaseContext import BaseContext
class FP16Context(BaseContext):

    def train_start_fn(self, session: Session):
        session.scaler = torch.cuda.amp.GradScaler()

    def forward_fn(self, session: Session):
        with torch.cuda.amp.autocast():
            super().forward_fn(session)
        #assert self.session.state.output_batch.dtype is torch.float16

    def loss_fn(self, session: Session):
        with torch.cuda.amp.autocast():
            super().loss_fn(session)

    def backward_fn(self, session: Session):
        loss = session.scaler.scale(session.state.loss)
        gradients = torch.ones_like(loss)
        loss.backward(gradients)
        # Gradient Accumulation
        if self.ga_steps(session):
            session.scaler.step(session.optimizer)
            session.scaler.update()
            session.optimizer.zero_grad()


