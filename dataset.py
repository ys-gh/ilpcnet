import glob
import json
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


class MelDataset(torch.utils.data.Dataset):

    def __init__(self, config, file_paths, device):
        self.feature_dir = os.path.join(config.data_root, "feature")
        assert os.path.isdir(self.feature_dir)

        self.device = device
        self.file_paths = file_paths
        self.frame_shift = config.frame_shift

        self.segment_size = config.segment_size
        self.mel_segment_size = config.segment_size // config.frame_shift
        self.f0s, self.melsps, self.wavs = self.read_data(self.feature_dir, file_paths,
                                                          config.corpus)

    def read_data(self, feature_dir, file_paths, corpus):
        f0s = []
        melsps = []
        wavs = []

        for path in file_paths:
            if corpus == 'jsut':
                file_id = os.path.splitext(os.path.basename(path))[0]
            elif corpus == 'jvs':
                pass
            # todo
            else:
                NotImplementedError

            f0 = np.load(os.path.join(feature_dir, "f0", f"{file_id}.f0.npy"))
            melsp = np.load(os.path.join(feature_dir, "melsp", f"{file_id}.melsp.npy"))
            wav = np.load(os.path.join(feature_dir, "wav", f"{file_id}.wav.npy"))

            f0s.append(f0)
            melsps.append(melsp)
            wavs.append(wav)

        return f0s, melsps, wavs

    def __getitem__(self, idx):
        max_mel_start = self.melsps[idx].shape[1] - self.mel_segment_size
        mel_start = random.randint(0, max_mel_start)
        melsp = self.melsps[idx]
        melsp = melsp[:, mel_start:mel_start + self.mel_segment_size]

        wav_start = mel_start * self.frame_shift
        wav = self.wavs[idx]
        wav = wav[wav_start:wav_start + self.segment_size]

        f0_start = mel_start
        f0_segment_size = self.mel_segment_size
        f0 = self.f0s[idx]
        f0 = f0[f0_start:f0_start + f0_segment_size]

        return melsp, wav, f0

    def __len__(self):
        return len(self.file_paths)

    def collate_fn(self, batch):
        pass
        # batch_size = len(batch)
        # max_mel_start = self.melsps[idx].shape[1] - self.mel_segment_size
        # mel_start = random.randint(0, max_mel_start)

        # f0s = []
        # melsps = []
        # wavs = []

        # return [f0s, melsps, wavs]


# if __name__ == '__main__':
#     device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
#     train_dataset = MelDataset('train.txt', 'preprocessed', device)
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=16,
#         shuffle=True,
#         collate_fn=train_dataset.collate_fn,
#     )

#     n_batch = 0
#     for batchs in train_loader:
#         n_batch += 1
#     print(
#         "Training set  with size {} is composed of {} batches.".format(
#             len(train_dataset), n_batch
#         )
#     )
