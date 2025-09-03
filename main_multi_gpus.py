import copy
import os
import resource

import pytorch_lightning as pl
##import wandb

from mg3d.config import ex
from mg3d.datamodules.multitask_datamodule import MTDataModule
from mg3d.models import PTUnifierTransformerSS

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (28672, rlimit[1]))

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    # Data modules
    dm = MTDataModule(_config, dist=True)

    # Module
    model = PTUnifierTransformerSS(_config)

    # Loggers
    os.makedirs(_config["log_dir"], exist_ok=True)
    exp_name = f'{_config["exp_name"]}'
    run_name = f'{exp_name}-seed{_config["seed"]}-from_{_config["load_path"].replace("/", "_")}'
##    wb_logger = pl.loggers.WandbLogger(project="PTUnifier", name=run_name, settings=wandb.Settings(start_method='fork'))
    tb_logger = pl.loggers.TensorBoardLogger(_config["log_dir"], name=run_name)
    loggers = [tb_logger]

    # Callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=2,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
        save_weights_only=True if "finetune" in exp_name else False
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    # Training Hyper-Parameters
    num_gpus = (_config["num_gpus"] if isinstance(_config["num_gpus"], int) else len(_config["num_gpus"]))
    grad_steps = max(_config["batch_size"] // (_config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]), 1)
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else -1
    max_epochs = _config["max_epoch"] if max_steps == -1 else 1000
    devices = _config["devices"]

    # Trainer
    trainer = pl.Trainer(
        gpus=num_gpus,
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=max_epochs,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=loggers,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        default_root_dir=_config["default_root_dir"]
    )
    
    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)   #
        if "finetune" in exp_name:
            trainer.test(ckpt_path="best" if "irtr" not in _config["exp_name"] else None, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
