import librosa
import matplotlib.pyplot as plt
import numpy as np


def read_wav(wav_path, sr):
    _wav, _sr = librosa.core.load(wav_path, sr=sr)
    _wav = librosa.util.normalize(_wav) * 0.99
    wav, _ = librosa.effects.trim(_wav, top_db=60)
    return wav.astype(np.float64), _sr


def plot_melsp(wav, sampling_rate=16000, frame_length=400, fft_length=1024, frame_shift=80):
    melsp = librosa.feature.melspectrogram(wav,
                                           sr=sampling_rate,
                                           hop_length=frame_shift,
                                           win_length=frame_length,
                                           n_mels=128,
                                           fmax=sampling_rate // 2)

    melsp_db = librosa.power_to_db(melsp, ref=np.max)
    fig, ax = plt.subplots()
    librosa.display.specshow(melsp_db,
                             x_axis='time',
                             y_axis='linear',
                             sr=sampling_rate,
                             hop_length=frame_shift,
                             fmax=sampling_rate // 2,
                             ax=ax,
                             cmap=plt.cm.jet)
    ax.set_title("Melspectrogram", fontsize="medium")
    return fig
