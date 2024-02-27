from typing import Callable

import lightning as L
import torch


class Outputs:
    def __init__(self):
        self.y_pred = []
        self.y_true = []

    def clear(self):
        self.y_pred.clear()
        self.y_true.clear()


class LightningModel(L.LightningModule):

    def __init__(self,
                 model: torch.nn.Module,
                 loss: Callable = None,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0
                 ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay

        self.step_outputs = {"train": Outputs(), "val": Outputs(), "test": Outputs()}

    def forward(self, inputs):
        """Expects a batched input."""
        return self.model(inputs)

    def _step(self, kind, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        preds = output.detach()
        loss = self.loss(output, target)
        self.step_outputs[kind].y_pred.extend(preds)
        self.step_outputs[kind].y_true.extend(target)
        self.log(f"Loss/{kind}", loss, prog_bar=True)
        return loss

    def training_step(self, *args):
        return self._step("train", *args)

    def validation_step(self, *args):
        return self._step("val", *args)

    def test_step(self, *args):
        return self._step("test", *args)

    def _on_epoch_end(self, kind):
        if len(self.step_outputs[kind].y_pred) == 0:
            return
        preds = torch.stack(self.step_outputs[kind].y_pred).cpu()
        targets = torch.stack(self.step_outputs[kind].y_true).cpu()
        accuracy = ((preds.max(dim=1).indices == targets).sum() / len(preds)).item()
        self.log(f"Accuracy/{kind}", accuracy, prog_bar=True)
        self.step_outputs[kind].clear()

    def on_train_epoch_end(self):
        self._on_epoch_end("train")

    def on_validation_epoch_end(self):
        self._on_epoch_end("val")

    def on_test_epoch_end(self):
        self._on_epoch_end("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
