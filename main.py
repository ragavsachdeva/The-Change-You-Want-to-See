import warnings
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from easydict import EasyDict
from loguru import logger as L
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import rank_zero_only

from data.datamodule import DataModule
from models.centernet_with_coam import CenterNetWithCoAttention
from utils.general import get_easy_dict_from_yaml_file

warnings.filterwarnings("ignore")


@rank_zero_only
def print_args(configs):
    L.log("INFO", configs)


def train(configs, model, logger, datamodule, callbacks=None):
    L.log("INFO", f"Training model.")
    trainer = pl.Trainer.from_argparse_args(
        configs,
        logger=logger,
        strategy=DDPPlugin(find_unused_parameters=False),
        log_every_n_steps=1,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        benchmark=False,
        profiler="simple",
    )
    trainer.fit(model, datamodule=datamodule)
    return trainer, trainer.checkpoint_callback.best_model_path


def test(configs, model, logger, datamodule, checkpoint_path, callbacks=None):
    L.log("INFO", f"Testing model.")
    tester = pl.Trainer.from_argparse_args(
        configs, logger=logger, callbacks=callbacks, benchmark=True
    )
    tester.test(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)

def get_logging_callback_manager(args):
    if args.method == "centernet":
        from models.centernet_with_coam import WandbCallbackManager

        return WandbCallbackManager(args)

    raise NotImplementedError(f"Given method ({args.method}) not implemented!")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--method", type=str)
    parser.add_argument("--no_logging", action="store_true", default=False)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--test_from_checkpoint", type=str, default="")
    parser.add_argument("--quick_prototype", action="store_true", default=False)
    parser.add_argument("--load_weights_from", type=str, default=None)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--experiment_name", type=str, default=None)
    args, _ = parser.parse_known_args()
    if args.method == "centernet":
        parser = CenterNetWithCoAttention.add_model_specific_args(parser)
    else:
        raise NotImplementedError(f"Unknown method type {args.method}")

    # parse configs from cmd line and config file into an EasyDict
    parser = DataModule.add_data_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args = EasyDict(vars(args))
    configs = get_easy_dict_from_yaml_file(args.config_file)

    # copy cmd line configs into config file configs, overwriting any duplicates
    for key in args.keys():
        if args[key] is not None:
            configs[key] = args[key]
        elif key not in configs.keys():
            configs[key] = args[key]

    if configs.quick_prototype:
        configs.limit_train_batches = 2
        configs.limit_val_batches = 2
        configs.limit_test_batches = 2
        configs.max_epochs = 1

    print_args(configs)

    pl.seed_everything(1, workers=True)

    datamodule = DataModule(configs)
    if configs.method == "centernet":
        model = CenterNetWithCoAttention(configs)

    logger = None
    callbacks = [get_logging_callback_manager(configs)]
    if not configs.no_logging:
        logger = WandbLogger(
            project="badlaav",
            id=configs.wandb_id,
            save_dir="/work/rs/logs",
            name=configs.experiment_name,
        )
        callbacks.append(ModelCheckpoint(monitor="val/overall_loss", mode="min", save_last=True))

    trainer = None
    if configs.test_from_checkpoint == "":
        # train the model and store the path to the best model (as per the validation set)
        # Note: multi-GPU training is supported.
        trainer, test_checkpoint_path = train(configs, model, logger, datamodule, callbacks)
        # test the best model exactly once on a single GPU
        torch.distributed.destroy_process_group()
    else:
        # test the given model checkpoint
        test_checkpoint_path = configs.test_from_checkpoint

    configs.gpus = 1
    if trainer is None or trainer.global_rank == 0:
        test(
            configs,
            model,
            logger,
            datamodule,
            test_checkpoint_path if test_checkpoint_path != "" else None,
            callbacks,
        )
