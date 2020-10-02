#!/usr/bin/env python3

import sys
import csv
import math

import argparse

from pathlib import Path

import pandas as pd

# region global variable
l_palm = [0.0] * 3
l_wrist = [0.0] * 3
l_elbow = [0.0] * 3
l_thumb_metacarpal = [0.0] * 3
l_thumb_proximal = [0.0] * 3
l_thumb_intermediate = [0.0] * 3
l_thumb_distal = [0.0] * 3
l_index_metacarpal = [0.0] * 3
l_index_proximal = [0.0] * 3
l_index_intermediate = [0.0] * 3
l_index_distal = [0.0] * 3
l_middle_metacarpal = [0.0] * 3
l_middle_proximal = [0.0] * 3
l_middle_intermediate = [0.0] * 3
l_middle_distal = [0.0] * 3
l_ring_metacarpal = [0.0] * 3
l_ring_proximal = [0.0] * 3
l_ring_intermediate = [0.0] * 3
l_ring_distal = [0.0] * 3
l_pinky_metacarpal = [0.0] * 3
l_pinky_proximal = [0.0] * 3
l_pinky_intermediate = [0.0] * 3
l_pinky_distal = [0.0] * 3

r_palm = []
r_wrist = []
r_elbow = []
r_thumb_metacarpal = []
r_thumb_proximal = []
r_thumb_intermediate = []
r_thumb_distal = []
r_index_metacarpal = []
r_index_proximal = []
r_index_intermediate = []
r_index_distal = []
r_middle_metacarpal = []
r_middle_proximal = []
r_middle_intermediate = []
r_middle_distal = []
r_ring_metacarpal = []
r_ring_proximal = []
r_ring_intermediate = []
r_ring_distal = []
r_pinky_metacarpal = []
r_pinky_proximal = []
r_pinky_intermediate = []
r_pinky_distal = []

label = ""

gesture_dict = {
  1: "one (static)",
  2: "two (static)",
  3: "three (static)",
  4: "four (static)",
  5: "OK (static)",
  6: "pinch",
  7: "grab",
  8: "expand",
  9: "tap",
  10: "swipe left",
  11: "swipe right",
  12: "swipe V",
  13: "swipe O"
}

# endregion

def getGestureFeatures(personID, gestureID, itemID):
    csvFileGestures = 'C:\\Users\\Matteo\\Desktop\\StaticHandGestureRecognitionLeap_copy\\kinect_leap_dataset\\acquisitions\\P{}\\G{}\\{}_leap_motion.csv'.format(personID, gestureID, itemID)

    csvGestures = []
    with open(csvFileGestures, mode='r') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(l_palm)

        #for row in csv_reader:
        #    csvGestures.append(list(row))
        #return csvGestures[2][1:]




def getFingerDataFromId(gestureID, itemID):
	#txtFileGestures = 'D:\\Users\\Katia\\GestureRecognition\\Dictionary\\Dictionary\\{}\\{}.txt'.format(gestureID, itemID)
	txtFileGestures = Path(str(args.batch_mode_dataset_in_path + '/{}/{}.txt'.format(gestureID, itemID)))
	return getFingerData(txtFileGestures)

def getFingerData(txtFileGestures, debug=False):
    txt_file = open(txtFileGestures, mode='r+')
    txt_reader = txt_file.read()
    txt_lines = txt_reader.split('\n')
    txt_lines.pop()
    gesturesList = []
    lineIndex = 1
    for row in txt_lines:
        values = row.split(args.csv_separator)

        #if len(values) < 2:
        #    print "salta gesture " + str(gestureID) + " item " + str(itemID)
        #    break
        #    break

        r_palm = values[0:3]
        r_wrist = [0.0] * 3
        r_elbow = [0.0] * 3
        r_thumb_metacarpal = [0.0] * 3
        r_thumb_proximal = values[14:17]
        r_thumb_intermediate = values[7:10]
        r_thumb_distal = values[21:24]
        r_index_metacarpal = values[28:31]
        r_index_proximal = values[35:38]
        r_index_intermediate = values[42:45]
        r_index_distal = values[49:52]
        r_middle_metacarpal = values[56:59]
        r_middle_proximal = values[63:66]
        r_middle_intermediate = values[70:73]
        r_middle_distal = values[77:80]
        r_ring_metacarpal = values[84:87]
        r_ring_proximal = values[91:94]
        r_ring_intermediate = values[98:101]
        r_ring_distal = values[105:108]
        r_pinky_metacarpal = values[112:115]
        r_pinky_proximal = values[119:122]
        r_pinky_intermediate = values[126:129]
        r_pinky_distal = values[133:136]

        ####################### r_wrist ##########################################
        if debug:
            print(f'r_palm: {r_palm} - r_middle_metacarpal: {r_middle_metacarpal}')
        if len(r_palm) < 3 or len(r_middle_metacarpal) < 3:
            print(f'Empty values detected, probably end of line. Quitting...')
            break

        palm_middle_direction = [float(r_palm[i]) - float(r_middle_metacarpal[i]) for i in range(3)]
        r_wrist = [float(r_palm[i]) - float(palm_middle_direction[i]) for i in range(3)]

        #Check the computed wrist is pointed at the correct versor (i.e.wrist does not coincide with the middle joint
        wrist_middle_direction = [math.pow(float(r_wrist[i]) - float(r_middle_metacarpal[i]), 2) for i in range(3)]
        sum = math.fsum(wrist_middle_direction[i] for i in range(3))
        wrist_middle_distance = math.sqrt(sum)

        if(math.fabs(wrist_middle_distance) < 0.01):
            r_wrist = [float(r_palm[i]) + float(palm_middle_direction[i]) for i in range(3)]

        joint_copy = r_wrist
        r_wrist = r_palm
        r_palm = joint_copy

        ####################### r_thumb_intermediate ##########################################
        r_thumb_metacarpal = r_thumb_intermediate

        # wrist_thumb_direction = [float(r_wrist[i]) - float(r_thumb_intermediate[i]) for i in range(3)]
        # r_thumb_metacarpal = [float(r_wrist[i]) - 0.5 * float(wrist_thumb_direction[i]) for i in range(3)]
        #
        # # Check the computed thumb_metacarpal is pointed at the correct versor (i.e. the thumb_metacarpal - wrist distance is lesser than the thumb_intermediate - wrist distance
        # wrist_metacarpal_direction = [math.pow(float(r_wrist[i]) - float(r_thumb_metacarpal[i]), 2) for i in range(3)]
        # sum = math.fsum(wrist_metacarpal_direction[i] for i in range(3))
        # wrist_metacarpal_distance = math.sqrt(sum)
        #
        # wrist_intermediate_direction = [math.pow(float(r_wrist[i]) - float(r_thumb_intermediate[i]), 2) for i in range(3)]
        # sum = math.fsum(wrist_intermediate_direction[i] for i in range(3))
        # wrist_intermediate_distance = math.sqrt(sum)
        #
        # if (wrist_metacarpal_distance > wrist_intermediate_distance):
        #     r_thumb_metacarpal = [float(r_wrist[i]) + 0.5 * float(wrist_thumb_direction[i]) for i in range(3)]
        #
        # joint_copy = r_thumb_intermediate
        # r_thumb_intermediate = r_thumb_metacarpal
        # r_thumb_metacarpal = joint_copy


        ####################### r_elbow ##########################################
        r_elbow = [float(5.0)*float(r_palm[i]) - float(4.0)*float(r_middle_metacarpal[i]) for i in range(3)]



        gestures = l_palm + l_wrist + l_elbow + l_thumb_metacarpal + l_thumb_proximal + l_thumb_intermediate + l_thumb_distal + l_index_metacarpal + l_index_proximal + l_index_intermediate + l_index_distal + l_middle_metacarpal + l_middle_proximal + l_middle_intermediate + l_middle_distal + l_ring_metacarpal + l_ring_proximal + l_ring_intermediate + l_ring_distal + l_pinky_metacarpal + l_pinky_proximal + l_pinky_intermediate + l_pinky_distal + r_palm + r_wrist + r_elbow + r_thumb_metacarpal + r_thumb_proximal + r_thumb_intermediate + r_thumb_distal + r_index_metacarpal + r_index_proximal + r_index_intermediate + r_index_distal + r_middle_metacarpal + r_middle_proximal + r_middle_intermediate + r_middle_distal + r_ring_metacarpal + r_ring_proximal + r_ring_intermediate + r_ring_distal + r_pinky_metacarpal + r_pinky_proximal + r_pinky_intermediate + r_pinky_distal
        gestures = [lineIndex] + gestures
        gesturesList.append(gestures)
        lineIndex += 1

    return gesturesList

def getGestureRow():
    gestureList = l_palm + l_wrist + l_elbow + l_thumb_metacarpal + l_thumb_proximal + l_thumb_intermediate + l_thumb_distal + l_index_metacarpal + l_index_proximal + l_index_intermediate + l_index_distal + l_middle_metacarpal + l_middle_proximal + l_middle_intermediate + l_middle_distal + l_ring_metacarpal + l_ring_proximal + l_ring_intermediate + l_ring_distal + l_pinky_metacarpal + l_pinky_proximal + l_pinky_intermediate + l_pinky_distal + r_palm + r_wrist + r_elbow + r_thumb_metacarpal + r_thumb_proximal + r_thumb_intermediate + r_thumb_distal + r_index_metacarpal + r_index_proximal + r_index_intermediate + r_index_distal + r_middle_metacarpal + r_middle_proximal + r_middle_intermediate + r_middle_distal + r_ring_metacarpal + r_ring_proximal + r_ring_intermediate + r_ring_distal + r_pinky_metacarpal + r_pinky_proximal + r_pinky_intermediate + r_pinky_distal

    return gestureList

def writeCsvFileFromId(gestureID, itemID, gesturesList, debug=False):
	#csvFileGestures = 'D:\\Users\\Katia\\GestureRecognition\\Dictionary\\InputFileConversion\\datafiles\\datafile_{}_{}.csv'.format(gestureID, itemID)
	outdir  = Path(args.batch_mode_dataset_out_path)
	outdir.mkdir(exist_ok=True)
	if gestureID != -1:
		save_fn = outdir / str('datafile-{}-{}.csv.xz'.format(gesture_dict[gestureID].replace(' (static)',''), itemID))
	else:
		save_fn = outdir / str('datafile-{}-{}.csv.xz'.format('unknown', itemID))
	writeCsvFile(save_fn, gesturesList, gestureID, debug)

def writeCsvFile(save_fn, gesturesList, gestureID, debug=False):
	firstRow = [i for i in range(-1,138)] # + ['label']
	if debug:
		print(firstRow)
	df = pd.DataFrame(gesturesList, columns=firstRow)
	df.index = df[-1]
	df.index.names = [None]
	df.drop(-1, axis=1, inplace=True)
	if gestureID != -1:
		df['label'] = gesture_dict[gestureID]
	else:
		df['label'] = 'unknown'

	if debug:
		print(firstRow)
		print(df)
	df.to_csv(save_fn)					# save in XZ format to save space and loading times

##############################################################################################


def argument_parser():
	global args

	parser = argparse.ArgumentParser(description='Converter from the SHREC contest dataset format to the Leap Motion Dynamic Hand Gesture (LMDHG) dataset format.')

	parser.add_argument('--filename',  nargs='?', default='1.txt',				help='only used if not in batch mode')
	parser.add_argument('--gesture-id', default=-1, type=int)
	parser.add_argument('--csv-separator', default=';',					help='useful when someone publishes the training set with sep=; and the test set with sep=,')

	parser.add_argument('--batch-mode',    dest='batch_mode', action='store_true',		help='run in batch mode to process all the files in the dataset directory')
	parser.add_argument('--no-batch-mode', dest='batch_mode', action='store_false')
	parser.add_argument('--batch-mode-dataset-in-path',       default='./input-dataset',	help='input dataset directory: nn.txt files will be read from here')
	parser.add_argument('--batch-mode-dataset-out-path',      default='./output-dataset',	help='output dataset directory: datafile-nn.csv.xz files will be written here')
	parser.set_defaults(batch_mode=False)

	args = parser.parse_args()



if __name__ == '__main__':
	argument_parser()

	if args.batch_mode:
		gestureIDList = range(1,14)
		itemIDList = range(1,37)

		for gestureID in gestureIDList:
			for itemID in itemIDList:
				print(f'Processing gesture no. {gesture_dict[gestureID].replace(" (static)","")} {itemID}')
				gesturesList = getFingerDataFromId(gestureID, itemID)
				writeCsvFileFromId(gestureID, itemID, gesturesList)
	else:
		print(f'Processing gesture file: {args.filename}')
		gesturesList = getFingerData(args.filename)

		outdir  = Path('./')
		pfn     = Path(args.filename)
		stem    = pfn.name.replace(''.join(pfn.suffixes), '').lower()
		if args.gesture_id != -1:
			save_fn = outdir / (str(args.gesture_id) + '-' + stem + '.csv.xz')
		else:
			save_fn = outdir / ('unknown-' + stem + '.csv.xz')

		print(f'Writing file: {save_fn}')
		writeCsvFile(save_fn, gesturesList, gestureID=args.gesture_id)
