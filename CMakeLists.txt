cmake_minimum_required(VERSION 2.8.3)

project(dynamic-hand-gestures)

######################################################################################
######################################################################################
######################################################################################

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(Reset       "${Esc}[m")
  set(Bold        "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()

MESSAGE("")
MESSAGE("-= ${BoldYellow}CMAKE ${PROJECT_NAME} BEGIN${Reset} =-")
MESSAGE("")

######################################################################################
######################################################################################
######################################################################################

find_package(catkin REQUIRED COMPONENTS rospy roscpp std_msgs message_generation genmsg)

#add_message_files(FILES syntax_checker_cmd.msg dispatcher_complete_mission_cmd.msg)

#generate_messages(DEPENDENCIES std_msgs)

catkin_package(CATKIN_DEPENDS rospy roscpp std_msgs message_runtime)

######################################################################################
######################################################################################
######################################################################################

MESSAGE("-= ${BoldYellow}CMAKE ${PROJECT_NAME} END${Reset} =-")
MESSAGE("")
