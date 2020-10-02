#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys

import argparse

import rospy
import roslib
import rosbag

from std_msgs.msg import Bool
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import Int32MultiArray

from leap_motion.msg import Human

from enum import Enum

import datetime as dt

from threading import Thread, Lock

from utils import current_milli_time
from utils import fps_data
from utils import calc_and_show_fps

human_callback_fps_data = fps_data()



scale_factor                   = 1000								# between leap_motion ROS driver and Unity
skeleton_frame                 = np.zeros(shape=(46, 3))
skeleton_frame_list            = []
skeleton_frame_last_updated_ms = [current_milli_time()]
last_saved_time                = 0
first_topic_time               = 0

global_args                    = None

human_callback_counter         = 0
human_message_queue            = []
human_message_queue_idx        = 0
human_message_queue_prev_perc  = 0

def read_rosbag():
	fname = '/tmp/2020-02-21-16-39-15-Shaking.bag'
	bag = rosbag.Bag(fname)
	for topic, msg, t in bag.read_messages(topics=['/leap_motion/leap_filtered']):
		print(msg)
	bag.close()

def print_every_n(text):
	#if True or human_callback_fps_data.last_upd_time % 1000 == 0:
	print(text)

def human_callback_dummy(human, debug=False):
	global human_callback_counter
	skeleton_frame_last_updated_ms[0] = current_milli_time()						# let the main app know when the skeleton was last updated in the callback
	calc_and_show_fps(human_callback_fps_data)
	human_callback_counter+=1
	if human_callback_counter % 1000 == 0:
		print(f'Received: {human_callback_counter} callbacks...')

def human_callback(human, debug=False):				# Receives ROS topics and translates the messages into the 46x3 numpy array needed by the visualizer
	global first_topic_time					# The body of the callback should be simple to be executed very quickly to avoid loosing messages
	global human_callback_counter				# at least if the user wants so...
	global human_message_queue

	start_of_callback = current_milli_time()

	human_callback_counter+=1
	if human_callback_counter % 1000 == 0:
		print(f'Received: {human_callback_counter} callbacks...')

	if first_topic_time == 0:
		first_topic_time = current_milli_time()
		human_callback_impl_thread = Thread(target=human_callback_impl_thread_func, args=(human_message_queue, debug))
		human_callback_impl_thread.start()

	human_message_queue.append(human)			# TODO: place these two calls in a if block with parameter so that the user can decide
	#human_callback_impl(human_message_queue)		# whether the ROS coupling should be "tight and lossless" or "loose and lossy"

	end_of_callback = current_milli_time()
	if debug:
		print(end_of_callback-start_of_callback)

def human_callback_impl_thread_func(human_message_queue, debug=False):
	global human_message_queue_idx
	global human_message_queue_prev_perc
	print('Human Callback Impl Worker Thread started...')
	while True:
		if len(human_message_queue):
			perc = f'{human_message_queue_idx/len(human_message_queue) * 100.0:.0f}'
		else:
			perc = '0'
		if perc != human_message_queue_prev_perc:
			print(f'Perc: {perc} - IDX: {human_message_queue_idx} - QLEN: {len(human_message_queue)}')
			human_message_queue_prev_perc = perc
		if human_message_queue_idx < len(human_message_queue):
			human = human_message_queue[human_message_queue_idx]
			human_message_queue_idx += 1
			human_callback_impl(human, debug=debug)

def human_callback_impl(human, debug=False):
	global skeleton_frame
	global skeleton_frame_last_updated_ms
	global last_saved_time


	# http://docs.ros.org/melodic/api/leap_motion/html/msg/Human.html
	'''
	std_msgs/Header header
	
	# A unique ID for this Frame.
	int32 lmc_frame_id
	
	# How many hands were detected in the frame
	int32 nr_of_hands
	
	# How many fingers were detected in the frame
	int32 nr_of_fingers
	
	# How many gestures were detected in the frame
	int32 nr_of_gestures
	
	# The rate at which the Leap Motion software is providing frames of data (in frames per second). 
	float32 current_frames_per_second
	
	# A string containing a brief, human readable description of the Frame object. 
	string to_string
	
	# If there were any hands detected in the frame then the 
	# hand.msg will be added here
	Hand right_hand
	Hand left_hand
	'''

	if debug:
		print(f'Hands: {human.nr_of_hands} - Fingers: {human.nr_of_fingers} - FPS: {human.current_frames_per_second}')

	# Now process each hand
	# http://docs.ros.org/melodic/api/leap_motion/html/msg/Hand.html

	lhand = human.left_hand
	rhand = human.right_hand

	hands = [lhand, rhand]


	#l_palm l_wrist l_elbow
	#l_thumb_metacarpal l_thumb_proximal l_thumb_intermediate l_thumb_distal
	#l_index_metacarpal l_index_proximal l_index_intermediate l_index_distal
	#l_middle_metacarpal l_middle_proximal l_middle_intermediate l_middle_distal
	#l_ring_metacarpal l_ring_proximal l_ring_intermediate l_ring_distal
	#l_pinky_metacarpal l_pinky_proximal l_pinky_intermediate l_pinky_distal

	if debug:
		cp1 = current_milli_time()

	for idx, hand in enumerate(hands):
		offset = idx * 23					# because we have 23 "elements" for each hand = 46 elements total = 46x3 points
		if hand.is_present:
			if debug:
				hand_letter = 'R' if idx else 'L'

			flist = hand.finger_list
			fingerlist = [f.type for f in flist]
			#fingernames = [f.to_string for f in flist]	# useless list of 'Finger Id:1202', 'Finger Id:1203', 'Finger Id:1204'
			if debug:
				print(f'{hand_letter}Hand fingers: {len(flist)} - {fingerlist}')

			#elbow_list = np.array([0.2, 0.2, 0.2])		# always, we have no data about this...

			ep = hand.elbow_position
			ep_list = np.array([ep[0], ep[1], ep[2]])
			wp = hand.wrist_position
			wp_list = np.array([wp[0], wp[1], wp[2]])
			pc = hand.palm_center
			pc_list = np.array([pc.x, pc.y, pc.z])
			if debug:
				print(f'{hand_letter}Hand elbow - {ep_list}')
				print(f'{hand_letter}Hand wrist - {wp_list}')
				print(f'{hand_letter}Hand palm  - {pc_list}')

			skeleton_frame[offset + 0] = ep_list * scale_factor
			skeleton_frame[offset + 1] = wp_list * scale_factor
			skeleton_frame[offset + 2] = pc_list * scale_factor
			if debug:
				print(skeleton_frame)

			idx_counter = 3
			for finger in flist:
				for bone in finger.bone_list:
					bs = bone.bone_start.position
					be = bone.bone_end.position
					if debug:
						bs_list = np.abs(np.array([bs.x, bs.y, bs.z]))		# use abs to avoid noisy minus signs in front of numbers (just for visualization purposes)
						be_list = np.abs(np.array([be.x, be.y, be.z]))
					else:
						bs_list = np.array([bs.x, bs.y, bs.z])
						be_list = np.array([be.x, be.y, be.z])
					if debug:
						print(f'{finger.type} - {bone.type} - {bone.bone_start} - {bone.bone_end}')
						print(f'{finger.type} - {bone.type} - {bs_list} - {be_list}')

					skeleton_frame[offset + idx_counter] = be_list * scale_factor

					idx_counter += 1
		else:
			skeleton_frame[offset:offset+23] = np.zeros(shape=(23, 3))

	if debug:
		cp2 = current_milli_time()

	skeleton_frame_list.append(np.copy(skeleton_frame))		# make a copy and append that to the list, otherwise the next time sk_frame
									# will be overwritten and we'll end up with a list ALL made of identical elements!
	if debug:
		print(80*'=')
		print(skeleton_frame)
		print(80*'=')
		print(skeleton_frame_list[-1])
		print(80*'=')
		print(skeleton_frame_list[int(len(skeleton_frame_list)/4)])
		print(80*'=')
		print(skeleton_frame_list[int(len(skeleton_frame_list)/3)])
		print(80*'=')
		print(skeleton_frame_list[int(len(skeleton_frame_list)/2)])
		print(80*'=')

	skeleton_frame_last_updated_ms[0] = current_milli_time()						# let the main app know when the skeleton was last updated in the callback

	calc_and_show_fps(human_callback_fps_data)

	if global_args and global_args.save_to_file != '' and len(skeleton_frame_list) % global_args.write_to_file_every_n_rows == 0:
		save_thread = Thread(target=save_skeleton_to_file, args=(debug,))
		save_thread.start()

	if debug:
		cp3 = current_milli_time()
		print(cp3-cp2, cp2-cp1)


def save_skeleton_to_file(debug=False):
	global last_saved_time 
	global first_topic_time

	if debug:
		print(80*'+')
		print(len(skeleton_frame_list))
		print(80*'+')
		print(skeleton_frame_list[0].shape)
		print(80*'+')
		print(global_args.save_to_file)
		print(80*'+')
		print(skeleton_frame_list[int(len(skeleton_frame_list)/4)])
		print(skeleton_frame_list[int(len(skeleton_frame_list)/3)])
		print(skeleton_frame_list[int(len(skeleton_frame_list)/2)])
		print(80*'+')

	df_save = pd.DataFrame(list(map(np.ravel, skeleton_frame_list)))

	if len(df_save.index) == 3 and df_save.iloc[0, 0] == 0.0 and df_save.iloc[0, -1] == 0.0 and df_save.iloc[0, -1] == 0.0:
		df_save = df_save.iloc[1:]							# delete just the first blank row we used to give shape to the array

	if debug:
		print(80*'-')
		print(len(df_save.index))
		print(len(df_save.columns))
		print(80*'-')
		print(df_save.iloc[int(len(df_save.index)/4)])
		print(df_save.iloc[int(len(df_save.index)/3)])
		print(df_save.iloc[int(len(df_save.index)/2)])
		print(80*'-')

	curr_time = current_milli_time()
	if global_args:
		if (len(df_save.index) % global_args.write_to_file_every_n_rows == 0
			or curr_time - last_saved_time >= global_args.write_to_file_every_n_msecs):

			rows = len(df_save.index)
			secs = (curr_time-first_topic_time)/1000.0
			if secs == 0:
				secs = 0.001
			print(80*'-')
			print(f'Saving to file: {global_args.save_to_file} because num rows: {rows} or delta t: {(current_milli_time() - last_saved_time)/1000.0}')
			print(f'Saving {rows} rows to file, captured in {secs} seconds (an avg of {rows/secs} rows/s)')
			print(80*'-')
			save_thread = Thread(target=perform_actual_save, args=(df_save, global_args.label_to_file, global_args.save_to_file,))
			save_thread.start()
			last_saved_time = current_milli_time()

def perform_actual_save(df_save, label_to_file, save_to_file):
	start_save = current_milli_time()
	df_save.rename(columns={138: 'label'}, inplace=True)
	df_save['label'] = label_to_file
	df_save.to_csv(save_to_file, float_format='%.2f')
	end_save = current_milli_time()
	secs = (end_save-start_save)/1000.0
	rows = len(df_save.index)
	print(f'Saved {rows} rows to {save_to_file} in {secs} seconds.')


def subscribe_to_hands_topic(topic):
	hands_subscriber = rospy.Subscriber(topic, Human, human_callback, queue_size=100000, buff_size=2**27)
	return hands_subscriber

def init_ros(args):
	global global_args
	np.set_printoptions(precision=3)		# to allow humans to watch reasonable numbers...

	# ROS initialization
	rosnode_name = "dynamic_hand_gestures_ros_receiver"
	rospy.init_node(rosnode_name)

	print(f'Subscribing to topic: {args.ros_topic} @ {args.fps} FPS')
	if args.save_to_file != '':
		print(f'Saving skeleton to file: {args.save_to_file}')
		if args.label_to_file == '':
			print(f'WARNING: no label provided, use --label-to-file to give a label to the saved skeleton')

	if args.ros_topic != '':
		hands_subscriber = subscribe_to_hands_topic(args.ros_topic)
	elif args.rosbag_filename != '':
		read_rosbag()
	rate_it = rospy.Rate(args.fps)

	rospy.loginfo('%s node successfully initialized.' % rosnode_name)

	human_callback_fps_data.show_fps_textual = args.show_ros_callback_fps
	human_callback_fps_data.show_fps_text    = 'Hands Callback - '
	human_callback_fps_data.show_fps_textual_func = print_every_n

	global_args = args

	return rate_it


def argument_parser():
	global args

	parser = argparse.ArgumentParser(description='ROS receiver for the Leap Motion Dynamic Hand Gesture (LMDHG) dataset visualizer.')
	parser.add_argument('--ros-topic', default='/leap_motion/leap_filtered')
	parser.add_argument('--rosbag-filename', default='')
	parser.add_argument('--fps', default=60, type=int)
	parser.add_argument('--save-to-file',  default='', help='specify the filename where to save the skeleton produced by ros_receiver.py')
	parser.add_argument('--label-to-file', default='', help='use this to give a label to the file containing the skeleton saved by ros_receiver.py')
	parser.add_argument('--write-to-file-every-n-rows',  default=100000, type=int, help='while saving skeleton rows generated by ros_receiver.py, write to file every n rows')
	parser.add_argument('--write-to-file-every-n-msecs', default=120000, type=int, help='while saving skeleton rows generated by ros_receiver.py, write to file every n milliseconds')
	parser.add_argument('--show-ros-callback-fps',    dest='show_ros_callback_fps', action='store_true', help='show FPS (both instant and average) while receiving ROS messages of the hand pose')
	parser.add_argument('--no-show-ros-callback-fps', dest='show_ros_callback_fps', action='store_false')

	args = parser.parse_args()


args = None


if __name__ == '__main__':
	argument_parser()
	rate_it = init_ros(args)

	rospy.spin()
	# Main ROS loop
	#while (not rospy.is_shutdown()):
	#	rate_it.sleep()
	#	sleep(0.001)

