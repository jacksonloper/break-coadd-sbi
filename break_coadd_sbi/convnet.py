"""Very simple convolutional neural network for predicting n quanties from an image.

This convnet clamps images to be nonnegative and use a log1p transformation on the image data
before running it through the network.
"""

import torch
from torch import nn
import lightning
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class InferenceNetwork(lightning.LightningModule):
    def __init__(self,height,width,in_channels,out_channels,lr=1e-4):
        super().__init__()
        self.save_hyperparameters() # all input arguments saved as self.hparams

        # input is
        #   batch x band x height x width
        # output is
        #   batch x band

        # make sure downsampling by 8 won't result in fractional dims
        assert self.hparams.height % 4 == 0
        assert self.hparams.width % 4 == 0

        # conv encoder downsamples
        self.conv_encoder = torch.nn.Sequential(
            ConvBlock(self.hparams.in_channels, 16, stride=1),
            ConvBlock(16, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=1),
        )

        # second encoder flattens and uses linear layers
        self.linear_input_size = 128 * (self.hparams.height//4) * (self.hparams.width//4)
        self.linear_encoder = nn.Sequential(
            nn.Linear(self.linear_input_size, 512),
            nn.BatchNorm1d(512,eps=0.001, momentum=0.03),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256,eps=0.001, momentum=0.03),
            nn.ReLU(),
            nn.Linear(256, self.hparams.out_channels),
        )

    def preprocess(self,imgs):
        return torch.log1p(torch.clamp(imgs,0))

    def predict(self, imgs):
        # preprocess
        imgs = self.preprocess(imgs)

        # process
        x = self.conv_encoder(imgs)
        x = x.view(imgs.shape[0], self.linear_input_size)
        x = self.linear_encoder(x)

        return x

    def loss(self,batch):
        X,Y=batch
        Yhat = self.predict(X)
        loss = torch.mean((Y - Yhat)**2)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log("train_loss", loss,prog_bar=True,on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

