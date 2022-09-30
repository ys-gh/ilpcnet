import argparse
import json
import logging
import os

import torch
import torch.nn as nn
import yaml
from attrdict import AttrDict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MelDataset
# from model import iLPCNet
from util import plot_melsp


def save_checkpoint(path, model, optimizer, epoch, loss, logger):
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(state_dict, path)
    logger.info(f"Saved checkpoint to: {path}")


def plot_checkpoint(writer, step, loss, fig_wav, audio, sampling_rate=16000, tag=""):
    writer.add_scalar("Loss/total_loss", loss, step)
    audio = audio.detach().cpu().numpy()

    writer.add_audio(tag, audio / max(abs(audio)), sampling_rate=sampling_rate)
    fig_wav = fig_wav.detach().cpu().numpy()
    fig = plot_melsp(fig_wav / max(abs(fig_wav)))
    writer.add_figure(tag, fig)


def train(model, optimizer, loader, writer, logging, epoch):
    model.train()

    for step, (mel, wav, f0) in enumerate(loader, 1):
        mel, wav, f0 = mel.cuda(), wav.cuda(), f0.cuda()


def valid(model, optimizer, loader, writer, logging, epoch):
    pass
    model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./config.json')
    parser.add_argument('-l', '--log', type=str, default='./logs')
    parser.add_argument('-r', '--resume_path', type=str, default=None)
    parser.add_argument('-s', '--save_dir', type=str, default='./__pretrained')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = AttrDict(json.load(f))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tf_logger
    train_writer_path = os.path.join(args.log, "train")
    valid_writer_path = os.path.join(args.log, "valid")

    train_writer = SummaryWriter(train_writer_path)
    valid_writer = SummaryWriter(valid_writer_path)

    # entire_logger
    logger = logging.getLogger()

    # model
    # model = iLPCNet(config).to(device)

    # optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.betas)

    init_epoch = 0
    # # step = 0

    # if args.resume_path is not None:
    #     # logger.info("Resuming from checkpoint: %s" % args.resume_path)
    #     logger.info(f"Resuming from checkpoint: {args.resume_path}")

    #     checkpoint = torch.load(args.resume_path)
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     loss = checkpoint['loss']
    #     init_epoch = checkpoint['epoch']

    # else:
    #     logger.info("Starting new training run.")

    feature_dir = os.path.join(config.data_root, "feature")
    assert os.path.isdir(feature_dir)

    with open(os.path.join(config.data_root, "train.txt"), 'r') as f:
        train_files = [path.strip('\n') for path in f]

    with open(os.path.join(config.data_root, "valid.txt"), 'r') as f:
        valid_files = [path.strip('\n') for path in f]

    print(valid_files)
    print(type(valid_files))

    # data_loader
    train_dataset = MelDataset(config, train_files, device)
    valid_dataset = MelDataset(config, valid_files, device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        #   collate_fn=train_dataset.collate_fn,
        num_workers=0)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        #   collate_fn=train_dataset.collate_fn,
        num_workers=0)

    # for epoch in tqdm(range(init_epoch, config.epochs + 1)):
    #     train(model, optimizer, train_loader, train_writer, logging, epoch, loss)
    #     valid(model, optimizer, valid_loader, valid_writer, logging, epoch, loss)

    #     if epoch % config.save_step == 0:
    #         save_path = os.path.join(args.save_dir, f"{epoch}.pth")
    #         save_checkpoint(save_path, model, optimizer, epoch, loss, logger)

    for epoch in range(init_epoch, config.epochs + 1):
        for step, (mel, wav, f0) in enumerate(train_loader, 1):
            pass


if __name__ == '__main__':
    main()
