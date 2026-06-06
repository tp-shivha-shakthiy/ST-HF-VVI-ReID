import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, features, pids):
        features = F.normalize(features, p=2, dim=1)
        device = features.device
        batch_size = features.size(0)

        dist = torch.cdist(features, features)
        same_identity = pids.unsqueeze(0) == pids.unsqueeze(1)
        same_identity = same_identity.fill_diagonal_(False)

        losses = torch.zeros(batch_size, device=device)

        for i in range(batch_size):
            pos_mask = same_identity[i]
            neg_mask = ~same_identity[i]
            neg_mask[i] = False

            if not pos_mask.any() or not neg_mask.any():
                continue

            ap = dist[i][pos_mask].max()
            an = dist[i][neg_mask].min()
            loss = torch.clamp(ap - an + self.margin, min=0.0)
            losses[i] = loss

        return losses.mean() if losses.sum() > 0 else torch.zeros((), device=device)
