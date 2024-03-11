import torch
from torch import nn

class BasicBlock(nn.Module):
	def __init__(self, channel_size, downsample=True):
		super(BasicBlock, self).__init__()

		self.downsample = downsample
		self.input_downsampler = None
		if self.downsample:
			self.conv1 = nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=3, stride=2, padding=3)
			self.input_downsampler = nn.Sequential(
				nn.Conv2d(in_channels=channel_size*2, out_channels=channel_size, kernel_size=1, stride=2, bias=False),
				nn.BatchNorm2d(channel_size)
			)
		else:
			self.conv1 = nn.Conv2d(in_channels=channel_size*2, out_channels=channel_size, kernel_size=3, stride=1, padding=3)

		self.bn1 = nn.BatchNorm2d()
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=3, stride=1, padding=3)
		self.bn2 = nn.BatchNorm2d()


	def forward(self, x):
		identity = x # this is the residual thing

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample:
			identity = self.input_downsampler(identity)

		out += identity
		out = self.relu(out)

		return out

		


class Resnet34(torch.nn.Module):
	def __init__(self):
		super(Resnet34, self).__init__()
		self.common_stride = 2
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d()
		self.relu = nn.ReLU(inplace=True)
		self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.layer_1 = self._make_layer(channel_size=64, block_num=3, downsample=True)
		self.layer_2 = self._make_layer(channel_size=128, block_num=4)
		self.layer_3 = self._make_layer(channel_size=256, block_num=6)
		self.layer_4 = self._make_layer(channel_size=512, block_num=3)
		self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

	def _make_layer(self, channel_size, block_num, downsample=False):
		blocks = []
		blocks.append(BasicBlock(channel_size=channel_size, downsample=downsample))
		for i in range (1, block_num):
			blocks.append(BasicBlock(channel_size=channel_size))

		return nn.Sequential(*blocks)


	def forward(self, x):

		return x
