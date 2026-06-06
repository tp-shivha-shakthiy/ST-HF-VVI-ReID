import torch
import torch.nn as nn

from .id_loss import IDLoss
from .triplet_loss import TripletLoss


class ReIDLossBuilder(nn.Module):
    def __init__(
        self,
        lambda_id: float = 1.0,
        lambda_tri: float = 1.0,
        lambda_int_id: float = 0.0,
        lambda_int_tri: float = 0.0,
    ):
        super().__init__()

        self.lambda_id = lambda_id
        self.lambda_tri = lambda_tri
        self.lambda_int_id = lambda_int_id
        self.lambda_int_tri = lambda_int_tri

        self.id_loss_fn = IDLoss()
        self.triplet_loss_fn = TripletLoss()

    def forward(self, outputs, pids):
        id_loss = self.id_loss_fn(outputs["logits"], pids)
        triplet_loss = self.triplet_loss_fn(outputs["features"], pids)

        int_id_loss = id_loss * 0.0
        int_triplet_loss = triplet_loss * 0.0

        if outputs.get("int_logits") is not None:
            int_id_loss = self.id_loss_fn(outputs["int_logits"], pids)
        if outputs.get("int_features") is not None:
            int_triplet_loss = self.triplet_loss_fn(outputs["int_features"], pids)

        total = (
            self.lambda_id * id_loss
            + self.lambda_tri * triplet_loss
            + self.lambda_int_id * int_id_loss
            + self.lambda_int_tri * int_triplet_loss
        )

        return {
            "total": total,
            "id_loss": id_loss.detach(),
            "triplet_loss": triplet_loss.detach(),
            "int_id_loss": int_id_loss.detach(),
            "int_triplet_loss": int_triplet_loss.detach(),
        }


def build_loss(config):
    loss_cfg = config.get("loss", {})
    return ReIDLossBuilder(
        lambda_id=loss_cfg.get("lambda_id", 1.0),
        lambda_tri=loss_cfg.get("lambda_tri", 1.0),
        lambda_int_id=loss_cfg.get("lambda_int_id", 0.0),
        lambda_int_tri=loss_cfg.get("lambda_int_tri", 0.0),
    )
