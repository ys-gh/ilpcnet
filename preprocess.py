import argparse
import json
import os
import random
from glob import glob

import librosa
import numpy as np
import pyworld
import torch
from attrdict import AttrDict
from tqdm import tqdm

from util import read_wav


def prepare_jsut(config):

    wav_dir = os.path.join(config.data_root, "wav")
    assert os.path.isdir(wav_dir)

    feature_dir = os.path.join(config.data_root, "feature")
    os.makedirs(feature_dir, exist_ok=True)

    random.seed(config.seed)
    all_files = glob(os.path.join(wav_dir, "**", "*wav"), recursive=True)
    random.shuffle(all_files)
    train_files = all_files[:config.n_train]
    valid_files = all_files[config.n_train:]

    with open(os.path.join(config.data_root, "train.txt"), 'w') as f:
        f.write('\n'.join(train_files))
    with open(os.path.join(config.data_root, "valid.txt"), 'w') as f:
        f.write('\n'.join(valid_files))

    os.makedirs(os.path.join(feature_dir, "wav"), exist_ok=True)
    os.makedirs(os.path.join(feature_dir, "f0"), exist_ok=True)
    os.makedirs(os.path.join(feature_dir, "melsp"), exist_ok=True)

    for idx, wav_path in enumerate(tqdm(all_files)):
        file_id = os.path.splitext(os.path.basename(wav_path))[0]

        wav, sr = read_wav(wav_path, config.sampling_rate)
        assert sr == config.sampling_rate

        f0s = []
        melsps = []
        wavs = []

        if len(wav) < config.segment_size:
            print(f"{wav_path} was skipped")

        _f0, _t = pyworld.dio(wav,
                              fs=config.sampling_rate,
                              frame_period=config.frame_shift / config.sampling_rate * 1000)
        f0 = pyworld.stonemask(wav, _f0, _t, config.sampling_rate)
        sp = librosa.stft(wav, n_fft=config.fft_length, hop_length=config.frame_shift)
        melfilter = librosa.filters.mel(sr=config.sampling_rate,
                                        n_fft=config.fft_length,
                                        n_mels=config.sp_dim)
        melsp = np.dot(melfilter, sp)

        # f0s.append(f0)
        # melsps.append(melsp)
        # wavs.append(wav)
        # melsp_path = wav_path.replace(".wav", ".melsp")
        # f0_path = wav_path.replace(".wav", ".f0")
        # torch.save(f0, f0_path)
        # torch.save(melsp,)

        np.save(os.path.join(feature_dir, "f0", f"{file_id}"), f0)
        np.save(os.path.join(feature_dir, "melsp", f"{file_id}"), melsp)
        np.save(os.path.join(feature_dir, "wav", f"{file_id}"), wav)

        # n_seg = len(wav) // (config.segment_size)
        # wav_segs = []
        # for i in range(n_seg):
        #     wav_seg = wav[i * (config.segment_size):(i + 1) * (config.segment_size)]
        #     _f0, _t = pyworld.dio(wav_seg.astype(np.doubl),
        #                           fs=config.sampling_rate,
        #                           frame_period=config.frame_shift / config.sampling_rate * 1000)
        #     f0 = pyworld.stonemask(wav_seg.astype(np.double), _f0, _t, config.sampling_rate)
        #     sp = librosa.stft(wav_seg.astype(np.double),
        #                       n_fft=config.fft_length,
        #                       hop_length=config.frame_shift)
        #     melfilter = librosa.filters.mel(sr=config.sampling_rate,
        #                                     n_fft=config.fft_length,
        #                                     n_mels=config.sp_dim)
        #     melsp = np.dot(melfilter, sp)
        #     f0s.append(f0[:config.segment_size // config.frame_shift])


def prepare_jvs(config):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./config.json')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = AttrDict(json.load(f))

    if config.corpus == "jsut":
        prepare_jsut(config)

    elif config.corpus == "jvs":
        prepare_jvs(config)

    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()
