#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys

import argparse

import roslib
import rospy
from std_msgs.msg import Bool
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import Int32MultiArray

from leap_motion.msg import Human

from enum import Enum

import datetime as dt


scale_factor   = 1000								# between leap_motion ROS driver and Unity
skeleton_frame = np.zeros(shape=(46, 3))

def human_callback(human, debug=False):
	global skeleton_frame

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


	for idx, hand in enumerate(hands):
		offset = idx * 23					# because we have 23 "elements" for each hand = 46 elements total = 46x3 points
		if hand.is_present:
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

					#skeleton_frame[offset + idx_counter    ] = bs_list
					skeleton_frame[offset + idx_counter] = be_list * scale_factor

					idx_counter += 1
		else:
			skeleton_frame[offset:offset+23] = np.zeros(shape=(23, 3))
	if debug:
		print(skeleton_frame)



def subscribe_to_hands_topic(topic):
	hands_subscriber = rospy.Subscriber(topic, Human, human_callback)
	return hands_subscriber

def init_ros(args):
	np.set_printoptions(precision=3)		# to allow humans to watch reasonable numbers...

	# ROS initialization
	rosnode_name = "dynamic_hand_gestures_ros_receiver"
	rospy.init_node(rosnode_name)

#	# detected_phrase is what the classifier publishes (= Arturo sends to us)
#	sub_gesture_sequence_topic = "diver/command/detected_phrase"
#	# command is what is sent to the mission controller (= what we send to Marco)
#	pub_command_topic = "diver/command/command"
#	print('Subscribing to topic:', sub_gesture_sequence_topic)
#	subscriber = rospy.Subscriber(sub_gesture_sequence_topic, Int32MultiArray, gesture_sequence_callback)
#	print('Publishing topic:', pub_command_topic)
#	publisher_command = rospy.Publisher(pub_command_topic, syntax_checker_cmd, queue_size=10)

	print(f'Subscribing to topic: {args.ros_topic}')
	hands_subscriber = subscribe_to_hands_topic(args.ros_topic)
	rate_it = rospy.Rate(args.fps)

	rospy.loginfo('%s node successfully initialized.' % rosnode_name)

	return rate_it


def argument_parser():
	global args

	parser = argparse.ArgumentParser(description='ROS receiver for the Leap Motion Dynamic Hand Gesture (LMDHG) dataset visualizer.')
	parser.add_argument('--ros-topic', default='/leap_motion/leap_filtered')
	parser.add_argument('--fps', default=60, type=int)
#	parser.add_argument('--view_name', default='top')
#	parser.add_argument('--exit_on_eof', default=True)
#	parser.add_argument('--line_width', default=12.0, type=float)
#	parser.add_argument('--dataset_path', default='./dataset')
#	parser.add_argument('--show_label', dest='show_label', action='store_true')
#	parser.add_argument('--no-show_label', dest='show_label', action='store_false')
#	parser.add_argument('--debug_segments', dest='debug_segments', action='store_true')
#	parser.add_argument('--no-debug_segments', dest='debug_segments', action='store_false')
#	parser.set_defaults(show_label=True)
#	parser.set_defaults(debug_segments=False)
	args = parser.parse_args()
	print(f'Subscribing to topic: {args.ros_topic}')



args		= None


if __name__ == '__main__':
	argument_parser()
	rate_it = init_ros(args)

	# Main ROS loop
	while (not rospy.is_shutdown()):
		rate_it.sleep()

