import pytorch_lightning as pl
import torch, torch.nn as nn, numpy as np

class IdentityModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lr = 1
        self.v = nn.Linear(5, 1)

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        print("\n" + "-"*30)
        print(f"Train step with batch_idx {batch_idx}")
        print(f"Size of batch parameter: {len(batch)}")
        vals = []
        for v in batch:
            if type(v) is torch.Tensor:
                vals.append(f"(tens: {v.size()})")
            elif type(v) is np.ndarray:
                vals.append(f"(ndarray: {v.shape})")
            else:
                vals.append(f"(type: {str(type(v))})")
        print(f"\t{vals}")
        print(f"\tself.device = {self.device}")
        fake_data = torch.ones((32, 5), requires_grad=True, device=self.device)
        return torch.sum(self.v(fake_data))

    def validation_step(self, batch, batch_idx):
        print("\n" + "-"*30)
        print(f"Val step with batch_idx {batch_idx}")
        return 1
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)