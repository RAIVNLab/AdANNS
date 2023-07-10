import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from typing import Type, Any, Callable, Union, List, Optional
from torchvision.models import *
from tqdm import tqdm
import numpy as np


def get_ckpt(path, arch):
	ckpt=path
	ckpt = torch.load(ckpt, map_location='cpu')
	if arch == 'convnext_tiny':
		ckpt = ckpt['model']
	plain_ckpt={}
	for k in ckpt.keys():
		plain_ckpt[k[7:]] = ckpt[k] # remove the 'module' portion of key if model is Pytorch DDP
	return plain_ckpt


class BlurPoolConv2d(torch.nn.Module):
	def __init__(self, conv):
		super().__init__()
		default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
		filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
		self.conv = conv
		self.register_buffer('blur_filter', filt)

	def forward(self, x):
		blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
						   groups=self.conv.in_channels, bias=None)
		return self.conv.forward(blurred)


def apply_blurpool(mod: torch.nn.Module):
	for (name, child) in mod.named_children():
		if isinstance(child, torch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16):
			setattr(mod, name, BlurPoolConv2d(child))
		else: apply_blurpool(child)


'''
Retrieval utility methods.
'''
activation = {}
fwd_pass_x_list = []
fwd_pass_y_list = []
path_list = []

def get_activation(name):
	"""
	Get the activation from an intermediate point in the network.
	:param name: layer whose activation is to be returned
	:return: activation of layer
	"""
	def hook(model, input, output):
		activation[name] = output.detach()
	return hook


def append_feature_vector_to_list(activation, label, rep_size, path):
	"""
	Append the feature vector to a list to later write to disk.
	:param activation: image feature vector from network
	:param label: ground truth label
	:param rep_size: representation size to be stored
	"""
	for i in range (activation.shape[0]):
		x = activation[i].cpu().detach().numpy()
		y = label[i].cpu().detach().numpy()
		fwd_pass_y_list.append(y)
		fwd_pass_x_list.append(x[:rep_size])

def dump_feature_vector_array_lists(config_name, rep_size,  random_sample_dim, output_path):
	"""
	Save the database and query vector array lists to disk.
	:param config_name: config to specify during file write
        :param rep_size: representation size for fixed feature model
	:param random_sample_dim: to write a subset of database if required, e.g. to train an SVM on 100K samples
	:param output_path: path to dump database and query arrays after inference
	"""

	# save X (n x 2048), y (n x 1) to disk, where n = num_samples
	X_fwd_pass = np.asarray(fwd_pass_x_list, dtype=np.float32)
	y_fwd_pass = np.asarray(fwd_pass_y_list, dtype=np.uint16).reshape(-1,1)

	if random_sample_dim < X_fwd_pass.shape[0]:
		random_indices = np.random.choice(X_fwd_pass.shape[0], size=random_sample_dim, replace=False)
		random_X = X_fwd_pass[random_indices, :]
		random_y = y_fwd_pass[random_indices, :]
		print("Writing random samples to disk with dim [%d x 2048] " % random_sample_dim)
	else:
		random_X = X_fwd_pass
		random_y = y_fwd_pass
		print("Writing %s to disk with dim [%d x %d]" % (str(config_name)+"_X", X_fwd_pass.shape[0], rep_size))
    
	print("Unique entries: ", len(np.unique(random_y)))
	np.save(output_path+str(config_name)+'-X.npy', random_X)
	np.save(output_path+str(config_name)+'-y.npy', random_y)


def generate_retrieval_data(model, data_loader, config, random_sample_dim, rep_size, output_path, model_arch):
	"""
	Iterate over data in dataloader, get feature vector from model inference, and save to array to dump to disk.
	:param model: ResNet50 model loaded from disk
	:param data_loader: loader for database or query set
	:param config: name of configuration for writing arrays to disk
	:param random_sample_dim: to write a subset of database if required, e.g. to train an SVM on 100K samples
	:param rep_size: representation size for fixed feature model
	:param output_path: path to dump database and query arrays after inference
	"""
	model.eval()
	if model_arch == 'vgg19':
		model.classifier[5].register_forward_hook(get_activation('avgpool'))
	elif model_arch == 'mobilenetv2':
		model.classifier[0].register_forward_hook(get_activation('avgpool'))
	else:
		model.avgpool.register_forward_hook(get_activation('avgpool'))
	print("Dataloader len: ", len(data_loader))

	with torch.no_grad():
		with autocast():
				for i_batch, (images, target) in enumerate(data_loader):
					output = model(images.cuda())
					path = None
					append_feature_vector_to_list(activation['avgpool'].squeeze(), target.cuda(), rep_size, path)
					if (i_batch) % int(len(data_loader)/5) == 0:
						print("Finished processing: %f %%" % (i_batch / len(data_loader) * 100))
				dump_feature_vector_array_lists(config, rep_size, random_sample_dim, output_path)

	# re-initialize empty lists
	global fwd_pass_x_list
	global fwd_pass_y_list
	global path_list
	fwd_pass_x_list = []
	fwd_pass_y_list = []
	path_list = []

'''
Load pretrained models saved with old notation.
'''
class SingleHeadNestedLinear(nn.Linear):
	"""
	Class for MRL-E model.
	"""

	def __init__(self, nesting_list: List, num_classes=1000, **kwargs):
		super(SingleHeadNestedLinear, self).__init__(nesting_list[-1], num_classes, **kwargs)
		self.nesting_list=nesting_list
		self.num_classes=num_classes # Number of classes for classification

	def forward(self, x):
		nesting_logits = ()
		for i, num_feat in enumerate(self.nesting_list):
			if not (self.bias is None):
				logit = torch.matmul(x[:, :num_feat], (self.weight[:, :num_feat]).t()) + self.bias
			else:
				logit = torch.matmul(x[:, :num_feat], (self.weight[:, :num_feat]).t())
			nesting_logits+= (logit,)
		return nesting_logits

class MultiHeadNestedLinear(nn.Module):
	"""
	Class for MRL model.
	"""
	def __init__(self, nesting_list: List, num_classes=1000, **kwargs):
		super(MultiHeadNestedLinear, self).__init__()
		self.nesting_list=nesting_list
		self.num_classes=num_classes # Number of classes for classification
		for i, num_feat in enumerate(self.nesting_list):
			setattr(self, f"nesting_classifier_{i}", nn.Linear(num_feat, self.num_classes, **kwargs))

	def forward(self, x):
		nesting_logits = ()
		for i, num_feat in enumerate(self.nesting_list):
			nesting_logits +=  (getattr(self, f"nesting_classifier_{i}")(x[:, :num_feat]),)
		return nesting_logits

def load_from_old_ckpt(model, efficient, nesting_list):
		if efficient:
			model.fc=SingleHeadNestedLinear(nesting_list)
		else:
			model.fc=MultiHeadNestedLinear(nesting_list)

		return model
