#!/usr/bin/env python3

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys

import argparse

from enum import Enum

import datetime as dt


from fastai import *
from fastai.vision import *

defaults.device = torch.device('cpu')


def load_model(args):
	#data = ImageDataBunch(None, None)
	if args.export_pkl_model != None:
		data = ImageDataBunch.from_folder('/tmp')
		learn = cnn_learner(data, models.resnet50, metrics=accuracy)
		learn.load(args.model_name)
		learn.export(args.export_pkl_model)
	else:
		model_name = Path(args.model_name).name
		model_path = Path(args.model_name).parent
		print(f'About to load model: {model_path}/{model_name}')
		learn = load_learner(model_path, model_name, device='cuda:0')
		#learn = load_learner('/mnt/btrfs-data/backup-cnr-issia/rementa/the-bramati-axiom-generated-www/dynamic-hand-gestures-resnet-models', 'resnet-50-img_size-None-6b-2020-02-10_18.21.22_0.pkl')
		print(f'Successfully loaded model: {args.model_name}')
	return learn


def do_inference(learn, img, debug=True):
	# This is superflous, labels are embedded in the pkl file of the model...
	#classes = [ 'C', 'Catching', 'Catching-hands-up', 'Line', 'Pointing', 'Pointing-With-Hand-Raised', 'Resting', 'Rotating', 'Scroll-Finger', 'Shaking', 'Shaking-Low', 'Shaking-Raised-Fist', 'Slicing', 'Zoom' ]
	start = time.time()
	if debug:
		print(type(img))
	print(f'Doing inference on {img.shape} image...')
	raw_pred = learn.predict(img)
	end = time.time()
	prediction = raw_pred[0]
	if debug:
		print(80*'-')
		print(f'Raw prediction            : {raw_pred}')
		print(f'Prediction label          : {prediction}')
		print(f'Elapsed time for inference: {end - start}')
		print(80*'-')
	return raw_pred


def argument_parser():
	global args

	parser = argparse.ArgumentParser(description='Fast.ai inference module for the Leap Motion Dynamic Hand Gesture (LMDHG) dataset visualizer.')
	parser.add_argument('--model-name', default='models/resnet-50.pth')
	parser.add_argument('--test-img-filename', default='')
	parser.add_argument('--export-pkl-model', default=None)
	args = parser.parse_args()
	print(args)
	print(f'Loading model: {args.model_name}')
	print(f'Performing inference on image: {args.test_img_filename}')
	if args.export_pkl_model != None:
		print(f'Exporting model to: {args.export_pkl_model}')


args		= None


if __name__ == '__main__':
	argument_parser()
	learn = load_model(args)
	img   = open_image(args.test_img_filename)
	img.resize((3, 480, 640))
	img.refresh()
	print(f'Opened image {args.test_img_filename} with size: {img.shape}')

	do_inference(learn, img)


