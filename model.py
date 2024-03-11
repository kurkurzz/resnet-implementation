import torch
from torch import nn

class BasicBlock(nn.Module):
	def __init__(self, prev_channel_size, channel_size, downsample=False):
		super(BasicBlock, self).__init__()
		# downsample means reduce the size image by half
		self.downsample = downsample
		self.input_downsampler = None
		if self.downsample:
			self.conv1 = nn.Conv2d(in_channels=prev_channel_size, out_channels=channel_size, kernel_size=3, stride=2, padding=1)
			self.input_downsampler = nn.Sequential(
				nn.Conv2d(in_channels=prev_channel_size, out_channels=channel_size, kernel_size=1, stride=2, bias=False),
				nn.BatchNorm2d(channel_size)
			)
		else:
			self.conv1 = nn.Conv2d(in_channels=prev_channel_size, out_channels=channel_size, kernel_size=3, stride=1, padding=1)

		self.bn1 = nn.BatchNorm2d(num_features=channel_size)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(num_features=channel_size)


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
	def __init__(self, num_classes):
		super(Resnet34, self).__init__()
		# accept 3 channels image. stride 2 generally means the output image size will be reduced by half. 
		# but need to tweak padding and kernel size to achieve the exact half
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(num_features=64)
		self.relu = nn.ReLU(inplace=True)
		self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.layer_1 = self._make_layer(prev_channel_size=64, channel_size=64, block_num=3)
		self.layer_2 = self._make_layer(prev_channel_size=64, channel_size=128, block_num=4, downsample=True)
		self.layer_3 = self._make_layer(prev_channel_size=128, channel_size=256, block_num=6, downsample=True)
		self.layer_4 = self._make_layer(prev_channel_size=256, channel_size=512, block_num=3, downsample=True)
		self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512, num_classes)

	def _make_layer(self, prev_channel_size, channel_size, block_num, downsample=False):
		blocks = []
		blocks.append(BasicBlock(prev_channel_size=prev_channel_size, channel_size=channel_size, downsample=downsample))
		for i in range (1, block_num):
			blocks.append(BasicBlock(prev_channel_size=channel_size, channel_size=channel_size))

		return nn.Sequential(*blocks)


	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.max_pool(out)
		out = self.layer_1(out)
		out = self.layer_2(out)
		out = self.layer_3(out)
		out = self.layer_4(out)
		out = self.avg_pool(out)
		out = torch.flatten(out, 1)
		out = self.fc(out)
		return out
