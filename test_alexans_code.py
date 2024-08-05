import os
import glob
import subprocess
import tarfile
import wget
import copy
from omegaconf import OmegaConf, open_dict
import torch

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.utils import logging, exp_manager
import torch
import torch.nn as nn
import pytorch_lightning as ptl
from pytorch_lightning.loggers import WandbLogger
import json
import re


def preprocess_armenian_manifest(input_manifest_file, output_manifest_file):
    armenian_alphabet = "աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆուև "
    entries = []
    chars_to_ignore_regex = "[\.\,,\?\:\-!;՞()«»…\]\[/\*–‽+&_\\½√>€™$•¼}{~—=“\"”″‟„()՝,,։«»֊`՚’՜՛ehxlgnõ´av․'\u200b]"
    
    numbers_regex = "\d+"
    english_words_regex = "[a-zA-Z]{2,}"

    with open(input_manifest_file, 'r', encoding='utf-8') as f_in:
      for line in f_in:
          entry = json.loads(line)
          if re.search(numbers_regex, entry['text']) or entry['duration'] > 20 or entry['duration'] < 1:
                continue
          
          words_count = len(entry['text'].split())
          if words_count <= 3 or words_count > 30:
                continue
          
          processed_text = entry['text'].lower()
          processed_text = processed_text.replace("և", "եւ")
          processed_text = "".join(char for char in processed_text if char in armenian_alphabet)
          entry['text'] = processed_text
          entry['audio_filepath'] = entry['audio_filepath'].replace('mozilla-foundation_copy_2_copy_2/', 'mozilla-foundation_copy_2/')
        #   entry['audio_filepath'] = entry['audio_filepath'].replace('/workspace/nemo_capstone/asds', '/home/asds/ml_projects_mher/NeMo-SDP-ArmData/asds')

          entries.append(entry)
    with open(output_manifest_file, 'w', encoding='utf-8') as f_out:
      for entry in entries:
        # print(entry)
        json.dump(entry, f_out, ensure_ascii=False)
        f_out.write("\n")


def main():
    print(torch.cuda.is_available())


    VERSION = "mozilla-foundation/common_voice_16_1"
    LANGUAGE = "hy-AM"

    # tokenizer_dir = os.path.join('tokenizers', LANGUAGE)
    # manifest_dir = os.path.join('asds/mozilla-foundation/common_voice_16_1', LANGUAGE)

    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_en_fastconformer_transducer_large", map_location='cuda')
    # model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_en_fastconformer_hybrid_large_pc", map_location='cuda')

    freeze_encoder = False # set to False if dare lol

    def enable_bn_se(m):
        if type(m) == nn.BatchNorm1d:
            m.train()
            for param in m.parameters():
                param.requires_grad_(True)

        if 'SqueezeExcite' in type(m).__name__:
            m.train()
            for param in m.parameters():
                param.requires_grad_(True)

    if freeze_encoder:
        model.encoder.freeze()
        model.encoder.apply(enable_bn_se)
        logging.info("Model encoder has been frozen")
    else:
        model.encoder.unfreeze()
        logging.info("Model encoder has been un-frozen")
    
    # train_manifest = f"{manifest_dir}/train/train_mozilla-foundation_common_voice_16_1_manifest.json.clean"
    dev_manifest = "/home/asds/ml_projects_mher/NeMo-SDP-ArmData/asds/mozilla-foundation_copy_2/common_voice_16_1/hy-AM/validation/validation_mozilla-foundation_common_voice_16_1_manifest.json"
    test_manifest_path = "/home/asds/ml_projects_mher/NeMo-SDP-ArmData/asds/mozilla-foundation_copy_2/common_voice_16_1/hy-AM/test/test_mozilla-foundation_common_voice_16_1_manifest.json"
    # other_manifest = f"{manifest_dir}/other/other_mozilla-foundation_common_voice_16_1_manifest.json.clean"
    # invalidated_manifest = f"{manifest_dir}/invalidated/invalidated_mozilla-foundation_common_voice_16_1_manifest.json.clean"
    train_manifest_full = "/home/asds/ml_projects_mher/NeMo-SDP-ArmData/asds/mozilla-foundation_copy_2/common_voice_16_1/hy-AM/train/train_mozilla-foundation_common_voice_16_1_manifest.json"


    preprocess_armenian_manifest(train_manifest_full, train_manifest_full)
    preprocess_armenian_manifest(dev_manifest, dev_manifest)
    preprocess_armenian_manifest(test_manifest_path, test_manifest_path)


    TOKENIZER_TYPE = "bpe"
    TOKENIZER_DIR = "/home/asds/ml_projects_mher/NeMo-SDP-ArmData/asds/mozilla-foundation_copy_2/common_voice_16_1/hy-AM/armenian/tokenizers/tokenizer_spe_bpe_v130"

    model.change_vocabulary(new_tokenizer_dir=TOKENIZER_DIR, new_tokenizer_type=TOKENIZER_TYPE)

    cfg = copy.deepcopy(model.cfg)

    # Setup new tokenizer
    cfg.tokenizer.dir = TOKENIZER_DIR
    cfg.tokenizer.type = "bpe"

    # Set tokenizer config
    model.cfg.tokenizer = cfg.tokenizer

    print(OmegaConf.to_yaml(cfg.train_ds))

    # Setup train, validation, test configs
    with open_dict(cfg):
  # Train dataset
        cfg.train_ds.manifest_filepath = f"{train_manifest_full}"
        cfg.train_ds.batch_size = 16
        cfg.train_ds.num_workers = 8
        cfg.train_ds.pin_memory = True
        cfg.train_ds.use_start_end_token = False
        cfg.train_ds.trim_silence = False

        # Validation dataset
        cfg.validation_ds.manifest_filepath = dev_manifest
        cfg.validation_ds.batch_size = 8
        cfg.validation_ds.num_workers = 8
        cfg.validation_ds.pin_memory = True
        cfg.validation_ds.use_start_end_token = False
        cfg.validation_ds.trim_silence = False

        # Test dataset
        cfg.test_ds.manifest_filepath = test_manifest_path
        cfg.test_ds.batch_size = 8
        cfg.test_ds.num_workers = 8
        cfg.test_ds.pin_memory = True
        cfg.test_ds.use_start_end_token = False
        cfg.test_ds.trim_silence = False

    with open_dict(model.cfg.optim):
        model.cfg.optim.lr = 0.025
        model.cfg.optim.weight_decay = 0.002

        model.cfg.optim.sched.warmup_steps = None  # Remove default number of steps of warmup
        model.cfg.optim.sched.name = "CosineAnnealing"  # Remove default number of steps of warmup
        model.cfg.optim.sched.warmup_ratio = 0.10  # 10 % warmup
        model.cfg.optim.sched.min_lr = 1e-9

    with open_dict(model.cfg.spec_augment):
        model.cfg.spec_augment.freq_masks = 2
        model.cfg.spec_augment.freq_width = 25
        model.cfg.spec_augment.time_masks = 10
        model.cfg.spec_augment.time_width = 0.
        
    # setup model with new configs
    model.setup_training_data(cfg.train_ds)
    model.setup_multiple_validation_data(cfg.validation_ds)
    model.setup_multiple_test_data(cfg.test_ds)

    model.spec_augmentation = model.from_config_dict(model.cfg.spec_augment)

    use_cer = False
    log_prediction = True

    model.wer.use_cer = use_cer
    model.wer.log_prediction = log_prediction


    EPOCHS = 20

    # wandb_logger = WandbLogger(name='Adam-32-0.001',project='nemo-fastconformer')

    trainer = ptl.Trainer(devices=1,
                    accelerator='gpu',
                    max_epochs=EPOCHS,
                    accumulate_grad_batches=32,
                    enable_checkpointing=False,
                    logger= False,
                    log_every_n_steps=5,
                    check_val_every_n_epoch=2)

    # Setup model with the trainer
    model.set_trainer(trainer)

    from nemo.utils import exp_manager

    # Environment variable generally used for multi-node multi-gpu training.
    # In notebook environments, this flag is unnecessary and can cause logs of multiple training runs to overwrite each other.
    os.environ.pop('NEMO_EXPM_VERSION', None)

    config = exp_manager.ExpManagerConfig(
        exp_dir=f'experiments/lang-{LANGUAGE}/',
        name=f"ASR-Model-Language-{LANGUAGE}",
        create_wandb_logger=True,  # Enable WandB logging
        wandb_logger_kwargs={      # Provide arguments to WandB logger
            'name': f"ASR-Model-Language-{LANGUAGE}-freeze_encoder=False",
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

    # Prepare lists for transcriptions and hypotheses
    transcriptions = [entry['text'] for entry in test_manifest]
    hypotheses = []

    # Transcribe audio files
    for entry in test_manifest:
        transcript = model.transcribe(paths2audio_files=[entry['audio_filepath']])
        hypotheses.extend(transcript)

    # Calculate WER
    wer = word_error_rate(hypotheses, transcriptions)
    print(f"Word Error Rate: {wer:.2f}%")

if __name__=='__main__':
    main()