import json

import torch.nn as nn
import torch.optim.optimizer
from pytorch_lightning import LightningModule
from torch.nn import ModuleDict
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BaseModel(LightningModule):
    def __init__(self, targets, use_out_sequential=True,
                 optimizer=torch.optim.Adam, optimizer_parameters=None):
        super(BaseModel, self).__init__()

        self.targets = targets
        self.use_out_sequential = use_out_sequential

        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters or {}

        self.valid_losses = []
        self.train_losses = []

    def configure_out_layer(self):
        self.out_sequentials = ModuleDict()
        if self.use_out_sequential:
            self.output_dim = 0
            for target in self.targets:
                self.output_dim += target["dim"]
                if target["mode"] == "regression":
                    self.out_sequentials[target["name"]] = nn.Sequential(
                        nn.Linear(self.last_common_dim, target["dim"], device=self.device))
                elif target["mode"] == "binary_classification":
                    if target["dim"] != 1:
                        raise ValueError(
                            f"Target '{target['name']}': binary_classification requires dim=1, got {target['dim']} instead")
                    self.out_sequentials[target["name"]] = nn.Sequential(
                        nn.Linear(self.last_common_dim, target["dim"], device=self.device),
                        target["activation"] if "activation" in target else nn.Sigmoid())
                elif target["mode"] == "multiclass_classification":
                    self.out_sequentials[target["name"]] = nn.Sequential(
                        nn.Linear(self.last_common_dim, target["dim"], device=self.device),
                        target["activation"] if "activation" in target else nn.Softmax())
                else:
                    raise ValueError(
                        "Invalid mode value, only 'regression', 'binary_classification' or 'multiclass_classification' are allowed")
        else:
            self.output_dim = self.last_common_dim

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_parameters)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.2, patience=20, verbose=True),
                "monitor": "val_loss",
                "frequency": 1  # should be set to "trainer.check_val_every_n_epoch"
            },
        }

    def training_step(self, train_batch, *args, **kwargs):
        pred = self.forward(train_batch)
        true = train_batch.y
        loss = sum([target["loss"](pred[target["name"]], true[target["name"]]) for target in self.targets])
        self.log('train_loss', loss, batch_size=train_batch.batch.max() + 1, prog_bar=True)
        return loss

    def validation_step(self, val_batch, *args, **kwargs):
        pred = self.forward(val_batch)
        true = val_batch.y
        loss = sum([target["loss"](pred[target["name"]], true[target["name"]]) for target in self.targets])
        self.log('val_loss', loss, batch_size=val_batch.batch.max() + 1)
        return loss

    def get_model_structure(self):
        def make_jsonable(x):
            try:
                json.dumps(x)
                return x
            except (TypeError, OverflowError):
                if isinstance(x, dict):
                    return {key: make_jsonable(value) for key, value in x.items()}
                return str(x)

        return make_jsonable(self.config)
