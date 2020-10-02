#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
import signal
import time
import sys

import argparse

from vispy import app, scene, visuals
from vispy.scene import visuals as vs_visuals
import vispy.gloo
import vispy.io

from vispy.util.event import Event

import matplotlib.colors as mcolors

from enum import Enum

import datetime as dt

from threading import Thread

from inference import do_inference
from inference import load_model

from utils import current_milli_time
from utils import fps_data
from utils import calc_and_show_fps

def do_show_fps(text):
	global fpsviz
	fpsviz.text  = text
	fpsviz.color = 'white'


signal.signal(signal.SIGINT, signal.SIG_DFL)

show_only_scatter_points	= False
show_color			= True
show_axis			= False
show_edge_traces		= False
show_nodes			= True
show_tips_traces		= True
#show_node_labels		= True

pause				= False

segment_counter			= 0		# Useful for debugging the connection map

# build visuals
Plot3D = vs_visuals.create_visual_node(visuals.LinePlotVisual)

# build canvas
canvas = scene.SceneCanvas(keys='interactive', title='plot3d', app='PyQt5', show=True, fullscreen=True, vsync=False, size=(1920, 1080))

# Add a ViewBox to let the user zoom/rotate
view = canvas.central_widget.add_view()
view.camera = 'turntable'

def set_view(view, view_name, debug=False):
	if debug:
		print(f'Setting view to: {view_name}')
	if view_name == 'top':
		view.camera.fov = 0
		#view.camera.rotation = 45
		view.camera.elevation = 190
		view.camera.roll = 0
		view.camera.azimuth = 0
		view.camera.center = (0, 200)
		#view.camera.distance = 1000
		view.camera.distance = None
		view.camera.scale_factor = 400
	elif view_name == 'right':
		view.camera.fov = 0
		#view.camera.rotation = 45
		view.camera.elevation = 10
		view.camera.roll = 0
		view.camera.azimuth = 89
		view.camera.center = (0, 200)
		#view.camera.distance = 1000
		view.camera.distance = None
		view.camera.scale_factor = 400
	else:
		view.camera.fov = 45 
		view.camera.distance = 750

if show_only_scatter_points:
	scatter = vs_visuals.Markers()
	view.add(scatter)
else:
	lineviz = vs_visuals.Line()
	view.add(lineviz)

	if show_nodes:
		scatter = vs_visuals.Markers()
		view.add(scatter)
	if show_tips_traces:
		tips_scatter = vs_visuals.Markers()
		view.add(tips_scatter)

if show_axis:
	# just makes the axes
	axis = vs_visuals.XYZAxis(parent=view.scene)


finger_articulations = Enum('finger_articulations', 'l_palm l_wrist l_elbow l_thumb_metacarpal l_thumb_proximal l_thumb_intermediate l_thumb_distal l_index_metacarpal l_index_proximal l_index_intermediate l_index_distal l_middle_metacarpal l_middle_proximal l_middle_intermediate l_middle_distal l_ring_metacarpal l_ring_proximal l_ring_intermediate l_ring_distal l_pinky_metacarpal l_pinky_proximal l_pinky_intermediate l_pinky_distal r_palm r_wrist r_elbow r_thumb_metacarpal r_thumb_proximal r_thumb_intermediate r_thumb_distal r_index_metacarpal r_index_proximal r_index_intermediate r_index_distal r_middle_metacarpal r_middle_proximal r_middle_intermediate r_middle_distal r_ring_metacarpal r_ring_proximal r_ring_intermediate r_ring_distal r_pinky_metacarpal r_pinky_proximal r_pinky_intermediate r_pinky_distal ')

tips = 'l_thumb_distal l_index_distal l_middle_distal l_ring_distal l_pinky_distal r_thumb_distal r_index_distal r_middle_distal r_ring_distal r_pinky_distal'

fingers_colors = { 'l_thumb': 'red', 'l_index': 'lime', 'l_middle': 'blue', 'l_ring': 'yellow', 'l_pinky': 'deeppink',
			'r_thumb': 'lightskyblue', 'r_index': 'darksalmon', 'r_middle': 'lightgreen', 'r_ring': 'gold', 'r_pinky': 'darkviolet' }

CONNECTION_MAP = [[1,   2],
                  [2,   3],
                  [2,   4],
                  [2,  20],
                  [3,   4],
                  [3,  20],
                  [4,   5],
                  [5,   6],
                  [6,   7],
                  [4,   8],
                  [8,   9],
                  [9,  10],
                  [10, 11],
                  [8,  12],
                  [12, 13],
                  [13, 14],
                  [14, 15],
                  [12, 16],
                  [16, 17],
                  [17, 18],
                  [18, 19],
                  [16, 20],
                  [20, 21],
                  [21, 22],
                  [22, 23],
                  [24, 25],
                  [25, 26],
                  [25, 27],
                  [25, 43],
                  [26, 27],
                  [26, 43],
                  [27, 28],
                  [28, 29],
                  [29, 30],
                  [27, 31],
                  [31, 32],
                  [32, 33],
                  [33, 34],
                  [31, 35],
                  [35, 36],
                  [36, 37],
                  [37, 38],
                  [35, 39],
                  [39, 40],
                  [40, 41],
                  [41, 42],
                  [39, 43],
                  [43, 44],
                  [44, 45],
                  [45, 46]]

translation_map = {
			'#ATTRAPER': 'Catching',
			'#ATTRAPER_MAIN_LEVEE': 'Catching hands up',
			'#C': 'C',
			'#DEFILER_DOIGT': 'Scroll Finger',
			'#LIGNE': 'Line',
			'#PIVOTER': 'Rotating',
			'#POINTER': 'Pointing',
			'#POINTER_MAIN_LEVEE': 'Pointing With Hand Raised',
			'#REPOS': 'Resting',
			'#SECOUER': 'Shaking',
			'#SECOUER_BAS': 'Shaking Low',
			'#SECOUER_POING_LEVE': 'Shaking Raised Fist',
			'#TRANCHER': 'Slicing',
			'#ZOOM': 'Zoom',
		}

shrec_2020_gesture_dict = {			# http://www.andreagiachetti.it/shrec20gestures/
			"one":		1,
			"two":		2,
			"three":	3,
			"four":		4,
			"OK":		5,
			"pinch":	6,
			"grab":		7,
			"expand":	8,
			"tap":		9,
			"swipe-left":	10,
			"swipe-right":	11,
			"swipe-V":	12,
			"swipe-O":	13,
}



N = len(CONNECTION_MAP) * 2
# color array
color = np.ones((N, 4), dtype=np.float32)
color[:, 0] = np.linspace(0, 1, N)
color[:, 1] = color[::-1, 0]
color[:, 2] = np.linspace(0, 1, N)

debug = False

if debug:
	print(color)

def make_skeleton(fn='DataFile1.txt', dataset_path='dataset', debug=False):
	pfn     = Path(fn)
	stem    = pfn.name.replace(''.join(pfn.suffixes), '')#.lower()
	save_fn = dataset_path / (stem + '.csv.xz')
	if not Path(save_fn).exists():
		print(f'{save_fn} does not exists, creating it...')

		if not Path(dataset_path).exists():
			dataset_path.mkdir(exist_ok=True)

		df = pd.DataFrame([line.strip().split(' ') for line in open(fn, 'r')])
		if debug:
			print(df.head())
			print(df.index, df.columns)

		df['label'] = 'unlabeled'
	
		if debug:
			print(df.head())
			print(df.index, df.columns)
			print(df.iloc[:, -1], df.iloc[:, -2], df.iloc[:, -3])
			print(80*'-')
	
		rows_to_delete = []
	
		curr_label = '<NO LABEL>'
		for idx in df.index:
			if debug:
				print(df['label'])
				print(df.iloc[idx, -2])
				print(df.iloc[idx, -3])
				print(df.iloc[idx, -4])
				print(20*'-')
			if not df.iloc[idx, 2] and not df.iloc[idx, 3] and not df.iloc[idx, 4] and not df.iloc[idx, -2] and not df.iloc[idx, -3] and not df.iloc[idx, -4]:	# we don't have 138 entries,
				if debug:																	# this one must be a label...
					print(df.iloc[idx, 0])
				curr_label = translation_map[df.iloc[idx, 0]]
				rows_to_delete.append(idx)
			df['label'][idx] = curr_label
			if debug and idx > 5:
				break

		df.drop(index=rows_to_delete, inplace=True)						# drop the "label lines" with only #ATTRAPER or other funny french words :)
		df.loc[:, df.columns != 'label'] = df.loc[:, df.columns != 'label'].astype('float')	# convert all the table to float, except for the label column
		df.to_csv(save_fn)									# save in XZ format to save space and loading times
	
		print(df.head())

		skeleton = df
	else:
		print(f'Clean and compressed csv data file ({save_fn}) already exists, reloading it...')
		skeleton = pd.read_csv(save_fn, index_col=0)
		if debug:
			print(f'Skeleton: {skeleton} - columns: {skeleton.columns} - len: {len(skeleton.index)}')

		for col in skeleton.columns:
			if col != 'label':
				skeleton[col] = skeleton[col] * args.data_scale_factor
				if int(col) % 3 == 0:
					skeleton[col] = skeleton[col] + args.data_x_offset
				if int(col) % 3 == 1:
					skeleton[col] = skeleton[col] + args.data_y_offset
				if int(col) % 3 == 2:
					skeleton[col] = skeleton[col] + args.data_z_offset
		if debug:
			print(f'Skeleton: {skeleton} - columns: {skeleton.columns} - len: {len(skeleton.index)}')
	return skeleton


def do_edge_traces(pos, color, debug=False):
	global pos_array
	global col_array

	if debug:
		print(20*'-')
		print(pos_array.shape, pos_array.shape[1])
		print(col_array.shape, col_array.shape[1])

	pos_array = np.append(pos_array, pos  , axis=0)
	col_array = np.append(col_array, color, axis=0)
	#col_array[:, 3] = np.linspace(0, 1, col_array.shape[0])
	col_array[:, 2] = np.logspace(0, 1, col_array.shape[0]) / 10
	col_array[0:50, 0] = 1
	col_array[0:50, 1] = 0
	col_array[0:50, 2] = 0

	if debug:
		print(pos_array.shape, pos_array.shape[1])
		print(col_array.shape, col_array.shape[1])
		print(20*'-')
	if debug:
		print(col_array)


def do_tips_traces(sk_row, pos, color, idx, reset_history, debug=False):
	global pos_array
	global col_array
	global tips_array
	global tipscol_array
	global label

	make_colored_fingers()

	pos_array  = pos
	col_array  = np.array(color)
	if idx == 0 or reset_history:
		tips_array    = pos[0].reshape(1,3)
		tipscol_array = np.array([0.9, 0.2, 0.1, 1.]).reshape(1,4)
	else:
		for art in tips.split():
			if debug:
				print(art, finger_articulations[art].value)
			tip = np.array(sk_row[finger_articulations[art].value-1])
			if debug:
				print(20*'*')
				print(pos_array.shape)
				print(col_array.shape)
				print(tip, tip.shape, tip.reshape(tip.shape[0], 1))
				print(20*'*')
			tips_array    = np.append(tips_array,    tip.reshape(1,3), axis=0)
			#tipscol_array = np.append(tipscol_array, np.array([0.9, 0.2, 0.1, 1.]).reshape(1,4), axis=0)

			finger = art.replace('_distal', '')
			c, n = get_rgb_normalized_color_from_finger_name(finger)
			c_arr = [cl for cl in c]
			c_arr.append(1.0)
			if debug:
				print(finger, c, n, c_arr)
			tipscol_array = np.append(tipscol_array, np.array(c_arr).reshape(1,4), axis=0)

			max_len   = args.clear_history_older_than_n_frames

			# if we have specified a value for --clear-history-older-than-n-frames, truncate the fingertips history array
			if max_len:
				if tips_array.shape[0] >= max_len and tipscol_array.shape[0] >= max_len:
					tips_array    = tips_array[-max_len:]		# most recent "tip traces" are appended at the end
					tipscol_array = tipscol_array[-max_len:]

			if debug:
				label = str(tips_array.shape) + ' - ' + str(tipscol_array.shape)

			if debug:
				print(20*'°')
				print(tips_array.shape)
				print(tipscol_array.shape)
				print(pos_array.shape)
				print(col_array.shape)
				print(20*'°')
			if debug:
				time.sleep(1)

def one_dot_of_noise(r_palm, tip, noise_array, noisecol_array, c_arr):
	center = (r_palm - tip)
	if debug:
		print('c', center, center.shape)
		print('t', tip*2, (tip*2).shape)
	noise = np.array([0.0, 0.0, 0.0])
	for i in range(3):
		noise[i] = random.uniform(center[i], tip[i]*2)
		if debug:
			print('n', noise, noise.shape)

	noise_array = np.append(noise_array, noise.reshape(1,3), axis=0)
	noisecol_array = np.append(noisecol_array, np.array(c_arr).reshape(1,4), axis=0)

	if debug:
		print(f'noise_array: {noise_array}')
	return noise_array, noisecol_array

def do_noise(sk_row):
		global noise_array
		global noisecol_array
		# if we have specified a value for --add-noise, generate points and add them to the array
		if args.add_noise or True:
			for noise in range(int(args.add_noise / 10)):
				for art in tips.split():
					tip = np.array(sk_row[finger_articulations[art].value-1])
					finger = art.replace('_distal', '')
					c, n = get_rgb_normalized_color_from_finger_name(finger)
					c_arr = [cl for cl in c]
					c_arr.append(1.0)
					if tip[0] != args.data_x_offset and tip[1] != args.data_y_offset and tip[2] != args.data_z_offset: # the original value wasn't 0.0
						if debug:
							print(tip, sk_row[finger_articulations['l_palm'].value-1], sk_row[finger_articulations['r_palm'].value-1])
						l_palm = sk_row[finger_articulations['l_palm'].value-1]
						r_palm = sk_row[finger_articulations['r_palm'].value-1]
						if l_palm[0] != args.data_x_offset and l_palm[1] != args.data_y_offset and l_palm[2] != args.data_z_offset:
							noise_array, noisecol_array = one_dot_of_noise(l_palm, tip, noise_array, noisecol_array, c_arr)
						if r_palm[0] != args.data_x_offset and r_palm[1] != args.data_y_offset and r_palm[2] != args.data_z_offset:
							noise_array, noisecol_array = one_dot_of_noise(r_palm, tip, noise_array, noisecol_array, c_arr)

		if noise_array.shape[0] >= args.add_noise and noisecol_array.shape[0] >= args.add_noise:
			noise_array    = noise_array[-args.add_noise:]
			noisecol_array = noisecol_array[-args.add_noise:]


def create_fingers_dictionary(tips, finger_articulations, debug=False):
	fingers = {}
	for this_finger in tips.split():
		if debug:
			print('0.', this_finger)
		finger = this_finger.replace('_distal','')
		f = []
		for art in finger_articulations:
			if debug:
				print('1.', art)
			if finger in art.name:
				if debug:
					print('2.', art.name, art.value)
				f.append(art.value)
		fingers[finger] = f
	if debug:
		print(fingers)

	#fingers -> {'l_thumb': [4, 5, 6, 7], 'l_index': [8, 9, 10, 11], 'l_middle': [12, 13, 14, 15], 'l_ring': [16, 17, 18, 19], 'l_pinky': [20, 21, 22, 23], 'r_thumb': [27, 28, 29, 30], 'r_index': [31, 32, 33, 34], 'r_middle': [35, 36, 37, 38], 'r_ring': [39, 40, 41, 42], 'r_pinky': [43, 44, 45, 46]}

	return fingers

def get_rgb_normalized_color_from_finger_name(finger, debug=False):
	color_name = fingers_colors[finger]
	hex_color = mcolors.CSS4_COLORS[color_name]
	rgb_color = mcolors.to_rgb(hex_color)
	if debug:
		print(color_name, hex_color, rgb_color)
	return rgb_color, color_name

def make_colored_fingers(debug=False):
	global color
	global fingers

	if debug:
		for name, color in mcolors.CSS4_COLORS.items():
			print(name, color)
	for f_idx, finger in enumerate(fingers):
		finger_joints = fingers[finger]
		if debug:
			print(f_idx, finger, finger_joints)
		for joint_idx in range(len(finger_joints)):
			c, n = get_rgb_normalized_color_from_finger_name(finger)
			this_joint = finger_joints[joint_idx]
			if debug:
				print(this_joint, c, n)

			cm_left  = [ item[0] for item in CONNECTION_MAP ]	# collect the leftmost column of the joints  (e.g. 2  in the fourth pair)
			cm_right = [ item[1] for item in CONNECTION_MAP ]	# collect the rightmost column of the joints (e.g. 20 in the fourth pair)

			if debug:
				print(20*'-')
				print(cm_left)
				print(20*'-')
				print(cm_right)
				print(20*'-')

			offset = 0

			for left_idx, left_node in enumerate(cm_left):
				if left_node == this_joint:
					if debug:
						print(this_joint, left_node, c, n, ' --> ', left_idx, left_idx*2, left_idx*2 + offset)
					color[left_idx*2 + offset, :-1] = c
					color[left_idx*2 + offset + 1, :-1] = c
			










def draw_one_frame(sk_row, idx, pos_array, col_array, tips_array, tipscol_array, reset_history, debug=False):
	global segment_counter

	if debug:
		print(CONNECTION_MAP)			# these are the joints of the hand
	cm_1 = [ item[0]-1 for item in CONNECTION_MAP ]	# collect the leftmost column of the joints  (e.g. 1), subtract 1 to make it zero-based
	cm_2 = [ item[1]-1 for item in CONNECTION_MAP ]	# collect the rightmost column of the joints       2   (e.g. 2), subtract 1 to make it zero-based
	#print(len(cm_1), '-', cm_1)                                                                     # 2         3
	#print(len(cm_2), '-', cm_2)                                                                     # 2         4
                                                                                                                         # 3         20
	x = [sk_row[cm_1][:,0], sk_row[cm_2][:,0]]	# grab the columns, address the skeleton data    # 3         4
	#print(x)					# and take just the first element of the triplet # 4         20
	#break                                                                                           # 5         5
                                                                                                                         # 6         6 ---> hint, you can select columns with Ctrl-V in vim :)
	y = [sk_row[cm_1][:,1], sk_row[cm_2][:,1]]	# grab the columns, address the skeleton data
							# and take just the second element of the triplet
	z = [sk_row[cm_1][:,2], sk_row[cm_2][:,2]]	# grab the columns, address the skeleton data
							# and take just the third element of the triplet
	pos = []
	for jdx, _ in enumerate(x[0]):
		if debug:
			print(f'x: {x} - y: {y} - z: {z}')
			print(len(x), len(y), len(z))		# 2 2 2
			print(x[0], x[1])			# all the x column 1 (start), all the x column 2 (end)
			print(x[0][0], x[1][0])			# -104.48100000000001 -114.766
		start_x = x[0]
		end_x   = x[1]
		start_y = y[0]	# - 500
		end_y   = y[1]	# - 500
		start_z = z[0]
		end_z   = z[1]

		pos.append([start_x[jdx], start_y[jdx], start_z[jdx]])
		pos.append([end_x[jdx], end_y[jdx], end_z[jdx]])

	pos = np.array(pos)
	if debug:
		print(pos.shape)
	if debug:
		print(pos)

	if not show_edge_traces and not show_tips_traces:	# because this one resets the graph with the "current hand"
		pos_array  = pos
		col_array  = np.array(color)

	if show_edge_traces:					# we want that edges (= hands lines) leave a trace: append pos to pos_array
		if idx == 0 or reset_history:			# except when we change gesture (or idx == 0 because there's no history)
			pos_array  = pos
			col_array  = np.array(color)
		else:
			do_edge_traces(pos, color)
			if debug:
				print(20*'#')
				print(pos_array.shape, pos_array.shape[1])
				print(col_array.shape, col_array.shape[1])
				print(20*'#')

	if show_tips_traces:
		do_tips_traces(sk_row, pos, color, idx, reset_history)
		if args.add_noise:
			do_noise(sk_row)

	if debug:
		print(pos_array.shape)
		print(col_array.shape)
		print(pos.shape, pos.shape[1])

	if show_only_scatter_points:
		scatter.set_data(pos_array, edge_color=None, face_color=(0.4, 0.7, 1, 1), size=10)
	else:
		if args.debug_segments:
			col_array[segment_counter] = (1,1,1,1)
			segment_counter += 1
			if segment_counter >= 100:
				segment_counter = 0
			do_show_label(str(segment_counter))

		vispy.gloo.wrappers.set_line_width(width=args.line_width)
		if show_color:
			lineviz.set_data(pos_array, color=col_array, width=args.line_width, connect='segments')				# always remember that both pos_array and color_array
		else:													# must have the same length! (e.g. 100x3 vs. 100x4
			lineviz.set_data(pos_array, color=(1, 0.5, 0.3, 1), width=args.line_width, connect='segments')				# or 600x3 vs. 600x4 - where 4 is the AA value)
		if show_nodes:
			scatter.set_data(pos_array, edge_color=None, face_color=(0.4, 0.7, 1, 1), size=10)
		if show_tips_traces:
			#tipscol_array[:, 3] = np.logspace(0, 1, tipscol_array.shape[0]) / 10
			tipscol_array[:, 3] = np.linspace(0, 1, tipscol_array.shape[0])
			tips_scatter.set_data(tips_array, edge_color=None, face_color=tipscol_array, size=10)
		if args.add_noise:
			noisecol_array[:, 3] = np.linspace(0, 1, noisecol_array.shape[0])
			noise_scatter.set_data(noise_array, edge_color=None, face_color=noisecol_array, size=10)

	calc_and_show_fps(draw_one_frame_fps_data)


def write_csv_predictions(inf_counter, label, idx):
	global write_csv_predictions_df

	df = write_csv_predictions_df
	if len(df.columns) == 2:
		write_csv_predictions_df = df.append({-1: 0, 'filename': Path(args.filename).stem.lower(), 'num detected': 0}, ignore_index=True)
		df = write_csv_predictions_df
		df.index = df[-1]
		df.index.names = [None]
		df.drop(-1, axis=1, inplace=True)
	print(df)
	this_pred_str   = 'pred-'+str(inf_counter)
	last_pred_str   = 'pred-'+str(inf_counter-1)
	this_pred_cl    = str(shrec_2020_gesture_dict[label])
	this_pred_start = idx - args.inference_every_n_frames if idx >= args.inference_every_n_frames else 0
	print(f'pred-{inf_counter} = {this_pred_cl} from {this_pred_start} to {idx} - previous: {last_pred_str}')
	print(df)
	if inf_counter > 1 and df[last_pred_str+'-end'][0] == this_pred_start:
		df[last_pred_str+'-end']   = idx				# we merged two gestures, we must decrement inf_counter
		return inf_counter - 1
	else:
		df[this_pred_str]          = this_pred_cl
		df[this_pred_str+'-start'] = this_pred_start
		df[this_pred_str+'-end']   = idx
		df['num detected']         = inf_counter
		return inf_counter




def perform_inference(debug=True):
	global write_csv_predictions_df
	global images_outdir
	global inf_counter
	global label
	global idx

	do_show_label('')				# empty the label to avoid spoilers to the classifiers
	img		= canvas.render()
	png		= vispy.io._make_png(img)
	if debug:					# when getting: ValueError: ndarray is not C-contiguous
		print(img.flags)			# https://stackoverflow.com/questions/26778079/valueerror-ndarray-is-not-c-contiguous-in-cython
	fastai_img	= open_image(BytesIO(png))
	raw_pred	= do_inference(learn, fastai_img)
	pred_label	= raw_pred[0]
	label		= str(pred_label)

	raw_probs       = raw_pred[2]
	highest_prob_cl = raw_pred[1]
	confidence      = raw_probs[int(highest_prob_cl)]

	print(80*'=')
	print(raw_pred)
	print(80*'=')

	filename = str(dt.datetime.now()).replace('-', '').replace(' ', '-').replace(':', '')  + '-' +	\
			str(Path(args.filename).stem.lower()) + '-' +					\
			"{:.3f}".format(confidence) + '-' + "{:03d}".format(idx) + '-' +		\
			str(int(highest_prob_cl)) + '-' + label + '.png'					# 20200218-125513.291016

	outfn  = images_outdir / filename

	if confidence >= args.save_image_only_when_prob_greater_than:
		inf_counter += 1
		with open(outfn, 'wb') as f:
			f.write(png)
		if args.write_csv_predictions:
			inf_counter = write_csv_predictions(inf_counter, label, idx)

	probs        = []

	for prob in raw_probs:
		if prob > 0.1:
			probs.append("{:.3f}".format(prob))
	probs.sort()

	print(str(inf_counter) + ' - ' + str(pred_label) + ' - ' + "{:.3f}".format(confidence) + ' - ' + str(highest_prob_cl) + ' - ' + '/'.join(probs))
	print(80*'=')

	if debug:
		label        = str(inf_counter) + ' - ' + str(pred_label) + ' - ' + "{:.3f}".format(confidence) + '\n' + str(highest_prob_cl) + ' - ' + '/'.join(probs)
	else:
		label        = str(inf_counter) + ' - ' + str(pred_label)

	return label, confidence


def quit_app_if_ros_skeleton_not_updated():
	curr_time = current_milli_time()
	if debug:
		print(f'{curr_time} - {ros_skeleton_update_time_ms[0]}')
	if curr_time - ros_skeleton_update_time_ms[0] >= 10000:			# TODO: make this a parameter
		print(f'Not received skeleton updates via ROS topic for {(1.0*(curr_time - ros_skeleton_update_time_ms[0])/1000):.2f} seconds. Quitting the GUI...')
		from ros_receiver import save_skeleton_to_file
		save_skeleton_to_file()
		app.quit()							# this is just a wrapper for the Qt's app.exec() loop
		return True
	else:
		return False



def update_ros(ev, debug=False):
	global idx
	global tt_idx
	global label
	global rate_it
	global pos_array
	global col_array
	global tips_array
	global tipscol_array
	global inf_counter

	global learn

	timetable_speed = 100
	timetable = [
			(timetable_speed * 1, 'Hi! Welcome in Demo Mode.'),
			(timetable_speed * 2, 'Are you ready to perform some dynamic gesture?'),
			(timetable_speed * 3, 'Ok then, let\'s get started!'),
			(timetable_speed * 4, 'Please perform {curr_demo_gesture_label}...'),
			(timetable_speed * 5, '3'),
			(timetable_speed * 6, '2'),
			(timetable_speed * 7, '1'),
			(timetable_speed * 8, 'Go!'),
		]

	try:
		if args.enable_ros:
			# Also implement a mockup of "main ROS loop"
			if not rospy.is_shutdown():
				rate_it.sleep()
				idx += 1

			if args.demo_mode:
				if idx < timetable[tt_idx][0]:
					do_show_demo_text(timetable[tt_idx][1])
				if idx == timetable[tt_idx][0]:
					tt_idx += 1

			if args.do_inference and idx % args.inference_every_n_frames == 0:	# make a prediction every 5 seconds, also save the canvas for writing a PNG image...
				label, confidence = perform_inference()
				reset_history = True
			else:
				confidence    = -1.001						# if we're not performing inference, we don't have confidence
				reset_history = False

			if debug:
				print(dt.datetime.now())
				print(20*'-', ros_skeleton)

			if args.show_label:
				do_show_label(label, confidence)
			draw_one_frame(ros_skeleton, idx, pos_array, col_array, tips_array, tipscol_array, reset_history)

			quit_app_if_ros_skeleton_not_updated()
	finally:
		pass


def save_canvas_on_disk(args, idx, lastlabel, debug=False):
	global images_outdir
	# Use render to generate an image object
	img    = canvas.render()
	fname  = lastlabel.replace(' ', '-') + '-' + str(idx) + '-' + str(Path(args.filename).stem.lower())
	outfn  = images_outdir / fname

	# Use write_png to export your wonderful plot as png!
	print(f'Saving image: {outfn}.png')
	vispy.io.write_png(f'{outfn}.png', img)
	if args.double_view:
		set_view(view, 'right')
		img2 = canvas.render()
		if debug:
			print(type(img2), img2.shape, img2)
		vispy.io.write_png(f'{fname}-right.png', img2)
		double_img = np.concatenate((img, img2), axis=0)
		vispy.io.write_png(f'{fname}-double.png', double_img)
		set_view(view, 'top')


def update(ev, debug=False):
	global idx
	global pos_array
	global col_array
	global tips_array
	global tipscol_array
	global label
	global lastlabel
	global confidence
	global pause

	if debug:
		from pprint import pprint
		print(dt.datetime.now())
		print(type(ev), ev)
		pprint(vars(ev))

	try:
		while pause:
			time.sleep(0.1)
			return

		if skeleton is None:
			return

		if idx < len(skeleton.index):
			# skeleton -> 13756 x 1
			# size(skeleton{1}) -> 46 x 3	Matlab, what a shit! Why skeleton is 13756 x 1 and not 13756 x 46 x 3 if you know the sizes of the objects inside each row?!?!
			# CONNECTION_MAP -> 50 x 2
			# hline -> 50  x 1
			# x/y/z -> 50  x 2
			# pos   -> 100 x 3 (x, y, z)
			# color -> 100 x 4 (rgba)


			if args.do_inference:
				reset_history = False
				if idx % args.inference_every_n_frames == 0:		# make a prediction every 5 seconds, also save the canvas for writing a PNG image...
					lastlabel         = label
					label, confidence = perform_inference()
					if confidence >= args.reset_history_only_when_prob_greater_than:
						reset_history = True			# our guess is solid! reset the screen and let's start with the next gesture!
			else:
				confidence = -1.002					# usually we're not performing inference, we're just reading labels from a file
				reset_history = False
	
				if debug:
					print(len(skeleton.iloc[idx]), '-----------', skeleton.iloc[idx])
				lastlabel  = label
				label      = skeleton.iloc[idx]['label']
	
				if label != lastlabel:
					reset_history = True
					if lastlabel:
						save_canvas_on_disk(args, idx, lastlabel)
				if args.screenshot_every_n_frames != -1 and idx % args.screenshot_every_n_frames == 0 and idx != 0:
					save_canvas_on_disk(args, idx, lastlabel)

			sk_row    = skeleton.iloc[idx][:-1]		# normally 139 elements (138 + our label), drop the label and go on
			sk_row    = sk_row.values.reshape(46, 3)	# 138 now reshaped as 46 x 3 (type numpy.ndarray)

			if args.show_label:
				do_show_label(label, confidence)

			draw_one_frame(sk_row, idx, pos_array, col_array, tips_array, tipscol_array, reset_history)

			idx = idx + 1
			if idx >= len(skeleton.index) and args.exit_on_eof and lastlabel:
				if not args.do_inference:
					save_canvas_on_disk(args, idx, lastlabel)

				print(f'Quitting the GUI...')
				app.quit()				# this is just a wrapper for the Qt's app.exec() loop



	finally:
		pass



def add_line():
	vispy.gloo.wrappers.set_line_width(width=args.line_width)

	pos  = np.array([(0+idx,0+idx), (100+idx,100+idx)])
	line = scene.Line(pos=pos,
			color='green',
			method='gl',
			width=8,
			connect='strip',
			parent=view.scene)
	line.antialias=1

def do_show_label(label, confidence=-1.0):
	global textviz

	textviz.text  = label # + '--->' + str(confidence)
	if int(confidence) == -1:
		textviz.color = 'white'
	elif confidence <= 0.5:
		textviz.color = 'red'
	elif confidence <= args.save_image_only_when_prob_greater_than:
		textviz.color = 'yellow'
	elif confidence <= 1.00:
		textviz.color = 'green'
	else:
		textviz.color = 'cyan'

def add_label_to_scene():
	global textviz
	# Put up a text visual to display labels
	textviz = scene.Text('<Label>', parent=canvas.scene, color='black', pos=[225, 50, 0], font_size=24.)

def add_fps_to_scene():
	global fpsviz
	# Put up a text visual to display labels
	fpsviz = scene.Text('<FPS>', parent=canvas.scene, color='black', pos=[350, 1000, 0], font_size=24.)

def do_show_demo_text(text):
	global demotextviz

	demotextviz.text  = f'' + str(text)
	demotextviz.color = 'white'
	#demotextviz.anchors = ('top', 'left')
	#demotextviz.anchors = ('bottom', 'right')
	#demotextviz.anchors = ('center', 'center')
	#demotextviz.transforms.dpi = dpi
	#demotextviz.draw()

def add_demo_text_to_scene():
	global demotextviz
	# Put up a text visual to display labels
	demotextviz = scene.Text('<Demo Text>', parent=canvas.scene, color='white', pos=[960, 540, 0], font_size=48., anchor_x = 'center', anchor_y = 'center')
	#demotextviz = visuals.TextVisual('<Demo Text>', parent=canvas.scene, color='white', bold=True, pos=(0., 0.), font_size=48.)

@canvas.events.key_press.connect
def on_key_press(event):
	global pause

	if event.key == 'Escape':
		print('Exiting annotation GUI normally!')
		return
	if event.key == 'P':
		print('Pausing...')
		if pause:
			pause = False
		else:
			pause = True
		return



def threaded_update():
	global skeleton

	'''
	#ev._blocked = False
	ev.iteration = i
	'''
	'''
	{'_blocked': False,
	 '_handled': False,
	 '_native': None,
	 '_sources': [<vispy.app.timer.Timer object at 0x7fb54c1925f8>],
	 '_type': 'timer_timeout',
	 'count': 121,
	 'dt': 0.03757810592651367,
	 'elapsed': 2.6360816955566406,
	 'iteration': 121}
	'''

	if skeleton is not None:
		for i in skeleton.index:
			ev = vispy.util.event.Event('dummy_event')
			update(ev)
	else:	# we don't have skeleton, just ros_skeleton (== skeleton_frame, a single row of 46x3 data representing a single frame with two hands)
		while(True):
			ev = vispy.util.event.Event('dummy_event')
			update(ev)
			if quit_app_if_ros_skeleton_not_updated():
				break

def run_app(app):
	if sys.flags.interactive != 1:
		app.run()

def argument_parser():
	global args
	global images_outdir

	global noise_scatter

	parser = argparse.ArgumentParser(description='Visualizer and classifier for the Leap Motion Dynamic Hand Gesture (LMDHG) dataset.')

	parser.add_argument('filename',  nargs='?', default='DataFile1.txt')
	parser.add_argument('--exit-on-eof', default=True)
	parser.add_argument('--dataset-path', default='./dataset', help='output dataset directory: compressed .csv.xz files will be saved here')
	parser.add_argument('--captured-images-output-dir', default='./<date-time.ms>', help='captured images output directory')

	# ------------
	# -- LABELS --
	# ------------
	parser.add_argument('--show-label', dest='show_label', action='store_true', help='show the label while drawing the gesture: hide it to save images for deep learning training/inference')
	parser.add_argument('--no-show-label', dest='show_label', action='store_false')
	# ------------
	# --- INFO ---
	# ------------
	parser.add_argument('--show-fps', dest='show_fps', action='store_true', help='show FPS (both instant and average) while drawing the gesture')
	parser.add_argument('--no-show-fps', dest='show_fps', action='store_false')

	# ---------------
	# -- DEBUGGING --
	# ---------------
	parser.add_argument('--debug-segments', dest='debug_segments', action='store_true', help='add visual debugging information (colors) to help debugging edges in the connection map')
	parser.add_argument('--no-debug-segments', dest='debug_segments', action='store_false')

	# ---------------------
	# -- ROS INTEGRATION --
	# ---------------------
	parser.add_argument('--enable-ros', dest='enable_ros', action='store_true', help='use ROS to acquire Leap Motion data in real time or from ROSbags')
	parser.add_argument('--no-enable-ros', dest='enable_ros', action='store_false')
	parser.add_argument('--ros-topic', default='/leap_motion/leap_filtered', help='specify the ROS topic to subscribe to')
	parser.add_argument('--save-to-file', default='', help='specify the filename where to save the skeleton produced by ros_receiver.py')
	parser.add_argument('--label-to-file', default='', help='use this to give a label to the file containing the skeleton saved by ros_receiver.py')
	parser.add_argument('--write-to-file-every-n-rows',  default=100000, type=int, help='while saving skeleton rows generated by ros_receiver.py, write to file every n rows')
	parser.add_argument('--write-to-file-every-n-msecs', default=120000, type=int, help='while saving skeleton rows generated by ros_receiver.py, write to file every n milliseconds')
	parser.add_argument('--show-ros-callback-fps',    dest='show_ros_callback_fps', action='store_true', help='show FPS (both instant and average) while receiving ROS messages of the hand pose')
	parser.add_argument('--no-show-ros-callback-fps', dest='show_ros_callback_fps', action='store_false')


	# ---------------
	# -- INFERENCE --
	# ---------------
	parser.add_argument('--do-inference', dest='do_inference', action='store_true', help='Perform inference with the model loaded specifying --model-name')
	parser.add_argument('--no-do-inference', dest='do_inference', action='store_false')

	parser.add_argument('--model-name', default='models/resnet-50.pth', help='the model for performing inference in online mode')
	parser.add_argument('--export-pkl-model', default=None, help='load pth (pytorch model) for inference and save it as pickle file (.pkl)')
	parser.add_argument('--cuda-device', default='cuda:0', help='change the CUDA device for performing inference (e.g. cuda:0 or cpu)')
	parser.add_argument('--inference-every-n-frames', default=300, type=int, help='change the time interval on which inference is performed (useful for "quasi real-time" inference)')
	parser.add_argument('--save-image-only-when-prob-greater-than', default=0.85, type=float, help='save a PNG image only when last inference performed was "very confident" about the classification performed')
	parser.add_argument('--reset-history-only-when-prob-greater-than', default=0.75, type=float, help='reset fingertips history only when last inference performed was "very confident" about the classification performed. This + a low value of --inference-every-n-frames should do the trick to have real-time inference')
	parser.add_argument('--write-csv-predictions', dest='write_csv_predictions', action='store_true', help='Write a CSV with the predicted classes and the start/end frame number of the recognized gesture')
	parser.add_argument('--no-write-csv-predictions', dest='write_csv_predictions', action='store_false')

	# ----------------------
	# -- BATCH PROCESSING --
	# ----------------------
	parser.add_argument('--batch-mode', dest='batch_mode', action='store_true', help='run in batch mode (without a timer to draw the frames) to process data as fast as possible')
	parser.add_argument('--no-batch-mode', dest='batch_mode', action='store_false')

	# ---------------
	# -- VIEW MODE --
	# ---------------
	parser.add_argument('--view-name', default='top', help='change the view: only top and right implemented right now')
	parser.add_argument('--double-view', dest='double_view', action='store_true', help='before saving the png at the end of the gesture, switch to the other view, save the canvas and join the two views (e.g. top+right) into a double view png file')
	parser.add_argument('--no-double-view', dest='double_view', action='store_false')

	# ------------------
	# -- VIEW OPTIONS --
	# ------------------
	parser.add_argument('--fps',			default=60,  type=int,   help='render the 3D scene at this rate')
	parser.add_argument('--line-width',		default=12.0,type=float, help='draw hands with this line width')
	parser.add_argument('--data-scale-factor',	default=1.0, type=float, help='scale each column of data (except the label obviously) by a scale factor (e.g. 100.0)')
	parser.add_argument('--data-x-offset',		default=0.0, type=float, help='apply an offset to x/y/z coordinates after having them multiplied by the data scale factor')
	parser.add_argument('--data-y-offset',		default=0.0, type=float, help='apply an offset to x/y/z coordinates after having them multiplied by the data scale factor')
	parser.add_argument('--data-z-offset',		default=0.0, type=float, help='apply an offset to x/y/z coordinates after having them multiplied by the data scale factor')
	parser.add_argument('--screenshot-every-n-frames', default=-1, type=int, help='take a screenshot every n frames, not just at the end of the gesture. Useful for capturing partial gestures')
	parser.add_argument('--clear-history-older-than-n-frames', default=0, type=int, help='clear fingertips history older than n frames')
	parser.add_argument('--add-noise',		default=0,  type=int, help='add "history" noise to frames, like another old noisy gesture was performed before the actual one (up to <value> points)')

	# -------------------------------
	# -- DEMO MODE (unimplemented) --
	# -------------------------------
	parser.add_argument('--demo-mode', dest='demo_mode', action='store_true', help='WIP: propose a gesture to the user and ask him/her to replay it (useful to collect gestures for the dataset)')
	parser.add_argument('--no-demo-mode', dest='demo_mode', action='store_false')

	parser.set_defaults(show_label=True)
	parser.set_defaults(show_fps=False)
	parser.set_defaults(debug_segments=False)
	parser.set_defaults(enable_ros=False)
	parser.set_defaults(batch_mode=False)
	parser.set_defaults(double_view=False)
	parser.set_defaults(demo_mode=False)
	parser.set_defaults(do_inference=False)

	args = parser.parse_args()
	if not args.enable_ros and not args.demo_mode:
		print('')
		print('')
		print(f'Opening file: {args.filename}')
		if args.batch_mode:
			print(f'Running in batch mode...')
	else:
		print('')
		print('')
		print(f'Enabling ROS integration via topic: {args.ros_topic} - rendering and receiving topics @ {args.fps} fps...')
		if args.demo_mode:
			print(f'Enabling Demo Mode...')

	if args.captured_images_output_dir != './<date-time.ms>':
		images_outdir = Path(args.captured_images_output_dir)

	# TODO: ok, it's time to have a proper init function... too messy
	if args.add_noise:
		noise_scatter = vs_visuals.Markers()
		view.add(noise_scatter)

	if args.save_image_only_when_prob_greater_than >= 1.0:
		print('')
		print(f'Error: parameter passed to --save-image-only-when-prob-greater-than is greater than 1.0, you probably wrote {args.save_image_only_when_prob_greater_than} instead of {args.save_image_only_when_prob_greater_than / 100.0}')
		print('')
		sys.exit(0)




args		= None
idx		= 0
tt_idx		= 0
inf_counter	= 0
skeleton	= None
textviz		= None
fpsviz		= None
demotextviz	= None
pos_array	= np.empty((N, 3))
col_array	= np.empty((N, 4))
tips_array	= np.empty((N, 3))
tipscol_array	= np.empty((N, 4))
noise_array	= np.empty((1, 3))
noisecol_array	= np.empty((1, 4))
label		= ''
lastlabel	= ''
confidence      = -1.003			# funny eh? :) but it's useful to track who sets confidence the last in case of problems :)
dataset_path	= Path('dataset')
fingers		= create_fingers_dictionary(tips, finger_articulations)
rate_it		= None
learn		= None
images_outdir   = Path('./' + str(dt.datetime.now()).replace('-', '').replace(' ', '-').replace(':', ''))

write_csv_predictions_df = pd.DataFrame(columns = [-1, 'filename'])

# Init data structure to show FPS count
draw_one_frame_fps_data = fps_data()


if __name__ == '__main__':
	argument_parser()

	if not args.enable_ros:
		dataset_path = Path(args.dataset_path)
		skeleton = make_skeleton(args.filename, dataset_path)
		timer = None
		if not args.batch_mode:
			timer = app.Timer(interval=1.0/args.fps, connect=update, start=True)
	else:
		import rospy

		from ros_receiver import init_ros
		from ros_receiver import skeleton_frame as ros_skeleton
		from ros_receiver import skeleton_frame_last_updated_ms as ros_skeleton_update_time_ms

		rate_it = init_ros(args)
		timer   = app.Timer(interval=0.001, connect=update_ros, start=True)
		if args.demo_mode:
			add_demo_text_to_scene()

	if args.do_inference:
		learn   = load_model(args)

	set_view(view, args.view_name, debug=True)	# just to have some feedback at the starting...
	canvas.show()
	add_label_to_scene()
	add_fps_to_scene()

	images_outdir.mkdir(exist_ok=True)

	draw_one_frame_fps_data.show_fps_text    = 'Draw One Frame Func - '
	draw_one_frame_fps_data.show_fps_textual = args.show_fps
	draw_one_frame_fps_data.show_fps_graphic = args.show_fps
	draw_one_frame_fps_data.show_fps_graphic_func = do_show_fps


	if args.batch_mode:
		thread = Thread(target=run_app, args = (app, ))
		thread.start()
		threaded_update()
	else:
		run_app(app)

	if args.batch_mode:
		thread.join()

	if args.write_csv_predictions:
		filename = str(dt.datetime.now()).replace('-', '').replace(' ', '-').replace(':', '')  + '-' +	\
			str(Path(args.filename).stem.lower())
		outfn  = images_outdir / filename
		print(f'Writing predictions to CSV file: {outfn}')
		write_csv_predictions_df.to_csv(outfn)

	
