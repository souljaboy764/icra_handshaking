#!/usr/bin/python
import numpy as np

import intprim
from intprim.util.kinematics import BaseKinematicsClass
from intprim.probabilistic_movement_primitives import ProMP

import time
import sys
import os
import argparse

from util import *
from networks import Model
from nuitrack_skeleton_predictor import NuitrackSkeletonPredictor

# import rospy
import rospkg

import datetime
import time

import qi


rospy.init_node('joint_reader')
IP = "192.168.100.186"
PORT = 9559
session = qi.Session()
try:
	session.connect("tcp://" + IP + ":" + str(PORT))
except Exception as e:
	print(e)
except RuntimeError:
	print ("Can't connect to Naoqi at ip \"" + IP + "\" on port " + str(PORT) +".\n"
			"Please check your script arguments. Run with -h option for help.")
	sys.exit(1)

motion_service = session.service("ALMotion")
rate = rospy.Rate(50)

joint_names = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']

sensor_readings = []
while not rospy.is_shutdown():
	sensor_readings.append(motion_service.getAngles(joint_names, True))
	rate.sleep()
sensor_readings = np.array(sensor_readings)
savefile = datetime.datetime.now().strftime("%m%d%H%M")+'_'+str(np.random.randint(1000,9999))+'_pepper_readings.npz'

np.savez_compressed(os.path.join(rospkg.RosPack().get_path('icra_handshaking'), 'sensor_readings', savefile), sensor_readings=sensor_readings)