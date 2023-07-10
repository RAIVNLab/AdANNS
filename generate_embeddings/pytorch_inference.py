'''
Code to evaluate MRL models on different validation benchmarks.
'''
import sys
sys.path.append("../") # adding root folder to the path

import torch
import torchvision
from torchvision import transforms
from torchvision.models import *
from torchvision import datasets
from tqdm import tqdm

from MRL import *
from imagenetv2_pytorch import ImageNetV2Dataset
from argparse import ArgumentParser
from utils import *

# nesting list is by default from 8 to 2048 in powers of 2, can be modified from here.
BATCH_SIZE = 256
IMG_SIZE = 256
CENTER_CROP_SIZE = 224
#NESTING_LIST=[2**i for i in range(3, 12)]
ROOT="../../IMAGENET/" # path to IN1K
DATASET_ROOT="../../datasets/" #

parser=ArgumentParser()

# model args
parser.add_argument('--efficient', action='store_true', help='Efficient Flag')
parser.add_argument('--mrl', action='store_true', help='To use MRL')
parser.add_argument('--rep_size', type=int, default=2048, help='Rep. size for fixed feature model')
parser.add_argument('--path', type=str, required=True, help='Path to .pt model checkpoint')
parser.add_argument('--old_ckpt', action='store_true', help='To use our trained checkpoints')
parser.add_argument('--workers', type=int, default=12, help='num workers for dataloader')
parser.add_argument('--model_arch', type=str, default='resnet50', help='Loaded model arch')
# dataset/eval args
parser.add_argument('--tta', action='store_true', help='Test Time Augmentation Flag')
parser.add_argument('--dataset', type=str, default='V1', help='Benchmarks')
parser.add_argument('--save_logits', action='store_true', help='To save logits for model analysis')
parser.add_argument('--save_softmax', action='store_true', help='To save softmax_probs for model analysis')
parser.add_argument('--save_gt', action='store_true', help='To save ground truth for model analysis')
parser.add_argument('--save_predictions', action='store_true', help='To save predicted labels for model analysis')
# retrieval args
parser.add_argument('--retrieval', action='store_true', help='flag for image retrieval array dumps')
parser.add_argument('--random_sample_dim', type=int, default=4202000, help='number of random samples to slice from retrieval database')
parser.add_argument('--retrieval_array_path', default='', help='path to save database and query arrays for retrieval', type=str)


args = parser.parse_args()

if args.model_arch == 'convnext_tiny':
	convnext_tiny = create_model(
        	'convnext_tiny',
		pretrained=False,
		num_classes=1000,
		drop_path_rate=0.1,
		layer_scale_init_value=1e-6,
		head_init_scale=1.0
		)

model_arch_dict = {
    'vgg19': {'model': vgg19(False), 'nest_list': [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]},
    'resnet18': {'model': resnet18(False), 'nest_list': [8, 16, 32, 64, 128, 256, 512]},
    'resnet34': {'model': resnet34(False), 'nest_list': [8, 16, 32, 64, 128, 256, 512]},
    'resnet50': {'model': resnet50(False), 'nest_list': [8, 16, 32, 64, 128, 256, 512, 1024, 2048]},
    'resnet101': {'model': resnet101(False), 'nest_list': [8, 16, 32, 64, 128, 256, 512, 1024, 2048]},
    'mobilenetv2': {'model': mobilenet_v2(False), 'nest_list': [10, 20, 40, 80, 160, 320, 640, 1280]},
    'convnext_tiny': {'model': convnext_tiny, 'nest_list': [12, 24, 48, 96, 192, 384, 768]},
}

model = model_arch_dict[args.model_arch]['model']
print(model)

if not args.old_ckpt:
	if args.mrl:
		if args.model_arch == 'mobilenetv2':
			model.classifier[1] = MRL_Linear_Layer(model_arch_dict[args.model_arch]['nest_list'], num_classes=1000, efficient=args.efficient)
		elif args.model_arch == 'convnext_tiny':
			model.head = MRL_Linear_Layer(model_arch_dict[args.model_arch]['nest_list'], num_classes=1000, efficient=args.efficient)
		else:
			model.fc = MRL_Linear_Layer(model_arch_dict[args.model_arch]['nest_list'], efficient=args.efficient)
	else:
		model.fc=FixedFeatureLayer(args.rep_size, 1000) # RR model
else:
	if args.mrl:
		model = load_from_old_ckpt(model, args.efficient, model_arch_dict[args.model_arch]['nest_list'])
	else:
		model.fc=FixedFeatureLayer(args.rep_size, 1000)

print(model.fc)
apply_blurpool(model)
model.load_state_dict(get_ckpt(args.path, args.model_arch)) # Since our models have a torch DDP wrapper, we modify keys to exclude first 7 chars (".module").
model = model.cuda()
model.eval()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_transform = transforms.Compose([
				transforms.Resize(IMG_SIZE),
				transforms.CenterCrop(CENTER_CROP_SIZE),
				transforms.ToTensor(),
				normalize])

# Model Eval
if not args.retrieval:
	if args.dataset == 'V2':
		print("Loading Robustness Dataset")
		dataset = ImageNetV2Dataset("matched-frequency", transform=test_transform)
	elif args.dataset == '4K':
		train_path = DATASET_ROOT+"imagenet-4k/train/"
		test_path = DATASET_ROOT+"imagenet-4k/test/"
		train_dataset = datasets.ImageFolder(train_path, transform=test_transform)
		test_dataset = datasets.ImageFolder(test_path, transform=test_transform)
	else:
		print("Loading Imagenet 1K val set")
		dataset = torchvision.datasets.ImageFolder(ROOT+'val/', transform=test_transform)

	dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=False)

	if args.mrl:
		_, top1_acc, top5_acc, total_time, num_images, m_score_dict, softmax_probs, gt, logits = evaluate_model(
				model, dataloader, show_progress_bar=True, nesting_list=model_arch_dict[args.model_arch]['nest_list'], tta=args.tta, imagenetA=args.dataset == 'A', imagenetR=args.dataset == 'R')
	else:
		_, top1_acc, top5_acc, total_time, num_images, m_score_dict, softmax_probs, gt, logits = evaluate_model(
				model, dataloader, show_progress_bar=True, nesting_list=None, tta=args.tta, imagenetA=args.dataset == 'A', imagenetR=args.dataset == 'R')

	tqdm.write('Evaluated {} images'.format(num_images))
	confidence, predictions = torch.max(softmax_probs, dim=-1)
	if args.mrl:
		for i, nesting in enumerate(model_arch_dict[args.model_arch]['nest_list']):
			print("Rep. Size", "\t", nesting, "\n")
			tqdm.write('    Top-1 accuracy for {} : {:.2f}'.format(nesting, 100.0 * top1_acc[nesting]))
			tqdm.write('    Top-5 accuracy for {} : {:.2f}'.format(nesting, 100.0 * top5_acc[nesting]))
			tqdm.write('    Total time: {:.1f}  (average time per image: {:.2f} ms)'.format(total_time, 1000.0 * total_time / num_images))
	else:
		print("Rep. Size", "\t", args.rep_size, "\n")
		tqdm.write('    Evaluated {} images'.format(num_images))
		tqdm.write('    Top-1 accuracy: {:.2f}%'.format(100.0 * top1_acc))
		tqdm.write('    Top-5 accuracy: {:.2f}%'.format(100.0 * top5_acc))
		tqdm.write('    Total time: {:.1f}  (average time per image: {:.2f} ms)'.format(total_time, 1000.0 * total_time / num_images))


	# saving torch tensor for model analysis...
	if args.save_logits or args.save_softmax or args.save_predictions:
		save_string = f"mrl={args.mrl}_efficient={args.efficient}_dataset={args.dataset}_tta={args.tta}"
		if args.save_logits:
			torch.save(logits, save_string+"_logits.pth")
		if args.save_predictions:
			torch.save(predictions, save_string+"_predictions.pth")
		if args.save_softmax:
			torch.save(softmax_probs, save_string+"_softmax.pth")

	if args.save_gt:
		torch.save(gt, f"gt_dataset={args.dataset}.pth")


# Image Retrieval Inference
else:
	if args.dataset == '1K':
		train_dataset = datasets.ImageFolder(ROOT+"train/", transform=test_transform)
		test_dataset = datasets.ImageFolder(ROOT+"val/", transform=test_transform)
	elif args.dataset == 'V2':
		train_dataset = None  # V2 has only a test set
		test_dataset = ImageNetV2Dataset("matched-frequency", transform=test_transform)
	elif args.dataset == '4K':
		train_path = DATASET_ROOT+"imagenet-4k/train/"
		test_path = DATASET_ROOT+"imagenet-4k/test/"
		train_dataset = datasets.ImageFolder(train_path, transform=test_transform)
		test_dataset = datasets.ImageFolder(test_path, transform=test_transform)
	else:
		print("Error: unsupported dataset!")

	database_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=False)
	queryset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=False)

	config = args.model_arch+ "/" + args.dataset + "_val_mrl" + str(int(args.mrl)) + "_e" + str(int(args.efficient)) + "_rr" + str(int(args.rep_size))
	print("Retrieval Config: " + config)
	generate_retrieval_data(model, queryset_loader, config, args.random_sample_dim, args.rep_size, args.retrieval_array_path, args.model_arch)

	if train_dataset is not None:
		config = args.model_arch+ "/" + args.dataset + "_train_mrl" + str(int(args.mrl)) + "_e" + str(int(args.efficient)) + "_rr" + str(int(args.rep_size))
		print("Retrieval Config: " + config)
		generate_retrieval_data(model, database_loader, config, args.random_sample_dim, args.rep_size, args.retrieval_array_path, args.model_arch)
