import os
import glob
import subprocess
import tarfile
import wget
import copy
from omegaconf import OmegaConf, open_dict
import torch


import pytorch_lightning as pl
import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.utils import logging, exp_manager
import torch
from nemo.core.config import hydra_runner
import torch.nn as nn
import pytorch_lightning as ptl
from pytorch_lightning.loggers import WandbLogger
import json
from nemo.collections.asr.models import EncDecCTCModelBPE
from nemo.utils.exp_manager import exp_manager



@hydra_runner(
    config_path="/home/asds/ml_projects_mher/NeMo-SDP-ArmData/NeMo/examples/asr/conf/conformer/", config_name="conformer_ctc_bpe"
)
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    LANGUAGE = "hy-AM"

    # tokenizer_dir = os.path.join('tokenizers', LANGUAGE)
    # manifest_dir = os.path.join('asds/mozilla-foundation/common_voice_16_1', LANGUAGE)

    # def enable_bn_se(m):
    #     if type(m) == nn.BatchNorm1d:
    #         m.train()
    #         for param in m.parameters():
    #             param.requires_grad_(True)

    #     if 'SqueezeExcite' in type(m).__name__:
    #         m.train()
    #         for param in m.parameters():
    #             param.requires_grad_(True)
    
    manifest_dir = "asds/mozilla-foundation_copy/common_voice_17_0/hy-AM"
    train_manifest = f"{manifest_dir}/train/train_mozilla-foundation_common_voice_17_0_manifest.json.clean"
    dev_manifest = f"{manifest_dir}/validation/validation_mozilla-foundation_common_voice_17_0_manifest.json.clean"
    test_manifest_path = f"{manifest_dir}/test/test_mozilla-foundation_common_voice_17_0_manifest.json.clean"
    other_manifest = f"{manifest_dir}/other/other_mozilla-foundation_common_voice_17_0_manifest.json.clean"
    invalidated_manifest = f"{manifest_dir}/invalidated/invalidated_mozilla-foundation_common_voice_17_0_manifest.json.clean"
    # train_manifest_full = "asds/mozilla-foundation/common_voice_16_1/hy-AM/train_full_mozilla-foundation_common_voice_16_1_manifest.json"


    # if freeze_encoder:
    #     model.encoder.freeze()
    #     model.encoder.apply(enable_bn_se)
    #     logging.info("Model encoder has been frozen")
    # else:
    #     model.encoder.unfreeze()
    #     logging.info("Model encoder has been un-frozen")

    # TOKENIZER_TYPE = "bpe"
    # TOKENIZER_DIR = "/home/asds/mozilla-foundation/common_voice_16_1/hy-AM/armenian/tokenizers/tokenizer_spe_bpe_v130"


    # cfg.tokenizer.dir = TOKENIZER_DIR
    # cfg.tokenizer.type = "bpe"

    # # # Set tokenizer config
    # model.cfg.tokenizer = cfg.tokenizer

    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger


    trainer = pl.Trainer(**cfg.trainer)
    # exp_manager(trainer, cfg.get("exp_manager", None))
    model = EncDecCTCModelBPE(cfg=cfg.model, trainer=trainer)



    # Setup model with the trainer
    model.set_trainer(trainer)

    # # finally, update the model's internal config
    # model.cfg = model._cfg

    from nemo.utils import exp_manager

    # # Environment variable generally used for multi-node multi-gpu training.
    # # In notebook environments, this flag is unnecessary and can cause logs of multiple training runs to overwrite each other.
    # os.environ.pop('NEMO_EXPM_VERSION', None)

    config = exp_manager.ExpManagerConfig(
        exp_dir=f'experiments/lang-{LANGUAGE}/',
        name=f"ASR-Model-Language-{LANGUAGE}",
        create_wandb_logger=True,  # Enable WandB logging
        wandb_logger_kwargs={      # Provide arguments to WandB logger
            'name': f"ysu_asds_v5_mcv17.0_ctc_bpe_conformer",
            'project': 'nemo-fastconformer',  # Specify your WandB project name
            # You can add additional WandB arguments here, like `tags`, `notes`, etc.
        },
        checkpoint_callback_params=exp_manager.CallbackParams(
            monitor="val_wer",
            mode="min",
            always_save_nemo=True,
            save_best_model=True,
        ),
    )

    config = OmegaConf.structured(config)

    logdir = exp_manager.exp_manager(trainer, config)
    trainer.fit(model)
    
    with open(test_manifest_path, 'r') as f:
        test_manifest = [json.loads(line) for line in f]

    transcriptions = [entry['text'] for entry in test_manifest]
    audio_file_paths = [entry['audio_filepath'] for entry in test_manifest]  # List of audio file paths

    # Transcribe all audio files in a single batch
    hypotheses = model.transcribe(audio_file_paths)


    # print(hypotheses[0])
    print("hypotheses", len(hypotheses[0]))
    print("transcriptions", len(transcriptions))
    # Assert the lengths of hypotheses and transcriptions are the same
    assert len(hypotheses[0]) == len(transcriptions), "Length of hypotheses and transcriptions must match"

    # Calculate WER
    wer = word_error_rate(hypotheses[0], transcriptions)
    print(f"Word Error Rate: {wer:.2f}%")

if __name__=='__main__':
    main()
