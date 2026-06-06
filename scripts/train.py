import argparse
import yaml
import torch
import torch.optim as optim

from models.baseline import BaselineModel
from losses.build_loss import build_loss
from engine.trainer import Trainer


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.debug:
        sequence_length = config.get("dataset", {}).get("sequence_length", 6)
        image_size = config.get("dataset", {}).get("image_size", [288, 144])
        batch_size = 4
        num_classes = config.get("model", {}).get("num_classes", 10)
        lr = config.get("train", {}).get("lr", 3.5e-4)
        height, width = image_size

        model = BaselineModel(num_classes=num_classes, pretrained=False)
        criterion = build_loss(config)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        trainer = Trainer(model, criterion, optimizer, config)

        dummy_batch = {
            "frames": torch.randn(batch_size, sequence_length, 3, height, width),
            "pids": torch.tensor([0, 0, 1, 1]),
            "camids": torch.zeros(batch_size, dtype=torch.long),
            "modalities": ["rgb"] * batch_size,
            "track_ids": [f"track_{i}" for i in range(batch_size)],
        }

        losses = trainer.training_step(dummy_batch)

        print("Debug training step completed successfully.")
        print(f"  id_loss:           {losses['id_loss'].item():.6f}")
        print(f"  triplet_loss:      {losses['triplet_loss'].item():.6f}")
        print(f"  int_id_loss:       {losses['int_id_loss'].item():.6f}")
        print(f"  int_triplet_loss:  {losses['int_triplet_loss'].item():.6f}")
        print(f"  total:             {losses['total'].item():.6f}")


if __name__ == "__main__":
    main()
