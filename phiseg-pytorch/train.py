from unittest.mock import call
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import subprocess
import argparse

from src.data import load_data
from src.models import PHISeg, ProbUNet, UNet, UNetMCDropout

Chaksu = "data/Chaksu_ROI.h5"
RIGA = "data/RIGA_ROI.h5"


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main trainer file for all models.")
    parser.add_argument(
        "--random-seed",
        dest="random_seed",
        action="store",
        default=0,
        type=int,
        help="Random seed for pl.seed_everything function.",
    )
    parser.add_argument(
        "--method",
        dest="method",
        action="store",
        default=None,
        type=str,
        help="The method should be used [probunet, phiseg, unet, unet-mcdropout]",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        action="store",
        default=4,
        type=int,
        help="Batch size for training.",
    )
    args = parser.parse_args()

    git_hash = get_git_revision_short_hash()
    human_readable_extra = ""
    experiment_name = "-".join(
        [
            git_hash,
            f"seed={args.random_seed}",
            args.method,
            human_readable_extra,
            f"bs={args.batch_size}",
        ]
    )

    pl.seed_everything(seed=args.random_seed)

    (train_loader, validation_loader,) = load_data.create_train_val_loaders(
        file_path1=Chaksu, file_path2=RIGA, batch_size=args.batch_size
    )

    logger = TensorBoardLogger(
        save_dir="./runs-roi", name=experiment_name, default_hp_metric=False
    )
    checkpoint_callbacks = [
        ModelCheckpoint(
            monitor="val/total_loss",
            filename="best-loss-{epoch}-{step}",
        ),
        ModelCheckpoint(
            monitor="val/dice",
            filename="best-dice-{epoch}-{step}",
            mode="max",
        ),
    ]

    if args.method != "unet":
        checkpoint_callbacks.append(
            ModelCheckpoint(monitor="val/ged^2", filename="best-ged2-{epoch}-{step}")
        )

    if args.method == "phiseg":
        model = PHISeg(
            total_levels=7, latent_levels=5, zdim=2, num_classes=3, beta=1.0
        )  # changed number of classes
    elif args.method == "probunet":
        model = ProbUNet(
            total_levels=7, zdim=6, num_classes=3, beta=1.0
        )  # changed number of classes
    elif args.method == "unet-mcdropout":
        model = UNetMCDropout(total_levels=7, num_classes=3)
    elif args.method == "unet":
        model = UNet(total_levels=7, num_classes=3)  # changed number of classes
    else:
        raise ValueError(f"Unknown method: {args.method}.")

    trainer = pl.Trainer(
        logger=logger,
        val_check_interval=0.25,
        log_every_n_steps=50,
        accelerator="gpu",
        devices=1,
        callbacks=checkpoint_callbacks,
    )
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader
    )
