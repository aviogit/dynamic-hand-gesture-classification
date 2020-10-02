#!/usr/bin/env python3

import sys
import time
#from threading import Lock

current_milli_time = lambda: int(round(time.time() * 1000))

#calc_and_show_fps_mutex = Lock()

class fps_data:
	last_upd_time		= current_milli_time()	# ms since last update
	show_fps_text		= ''			# text to show before "FPS: 25.00 - AVG: 55.06"
	show_fps_textual	= False			# bool to decide whether to call show_fps_textual_func()
	show_fps_graphic	= False			# bool to decide whether to call show_fps_graphic_func()
	show_fps_textual_func	= print			# show_fps_textual_func()
	show_fps_graphic_func	= print			# show_fps_graphic_func()

	def __init__(self, debug=False):
		if debug:
			print(f'Initializing fps_data with default values...')
		self.last_fps_values = []		# list of last n values (must be here otherwise it's "static" = shared between objects!!! O__o)
							# https://docs.python.org/3/tutorial/classes.html

def calc_and_show_fps(fps_data_struct, debug=False):
	#global calc_and_show_fps_mutex
	#calc_and_show_fps_mutex.acquire()
	last_upd_time         = fps_data_struct.last_upd_time		# ms since last update
	last_fps_values       = fps_data_struct.last_fps_values		# list of last n values
	show_fps_text         = fps_data_struct.show_fps_text		# text to show before "FPS: 25.00 - AVG: 55.06"
	show_fps_textual      = fps_data_struct.show_fps_textual	# bool to decide whether to call show_fps_textual_func()
	show_fps_graphic      = fps_data_struct.show_fps_graphic	# bool to decide whether to call show_fps_graphic_func()
	show_fps_textual_func = fps_data_struct.show_fps_textual_func	# show_fps_textual_func()
	show_fps_graphic_func = fps_data_struct.show_fps_graphic_func	# show_fps_graphic_func()

	curr_upd_time  = current_milli_time()
	delta_t        = curr_upd_time - last_upd_time
	if delta_t == 0:
		delta_t = 1
	curr_fps       = 1000.0 / delta_t
	last_fps_values.append(curr_fps)
	if len(last_fps_values) >= 250:
		last_fps_values.pop(0)

	if debug or False:
		print(80*'-')
		print(show_fps_text)
		print(last_fps_values[:5], last_fps_values[-5:])
		print(show_fps_text)
		print(80*'-')

	acc = 0
	for v in last_fps_values:
		acc += v
	fps_text = f'{show_fps_text}FPS: {curr_fps:.2f} - AVG: {(1.0*acc/len(last_fps_values)):.2f}'

	if debug:
		print(f'FPS: {curr_fps:.2f} - delta_t: {delta_t}')
	fps_data_struct.last_upd_time = curr_upd_time
	#calc_and_show_fps_mutex.release()
	if show_fps_textual:
		show_fps_textual_func(fps_text)
	if show_fps_graphic:
		show_fps_graphic_func(fps_text)




if __name__ == '__main__':
	start = current_milli_time()
	var   = 0
	nloop = 10000000
	for i in range(nloop):
		var = var + 1
	end   = current_milli_time()
	print(f'Variable incremented for {nloop} iterations (value = {var}) - elapsed time: {end-start} milliseconds...')
