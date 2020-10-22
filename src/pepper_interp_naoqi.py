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
import motion


rospy.init_node('skeleton_predictor')
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

limits_max = [2.08567, -0.00872665, 2.08567, 1.56207]
limits_min = [-2.08567, -1.56207, -2.08567, 0.00872665]
bounds = ((limits_min[0], limits_max[0]),(limits_min[1], limits_max[1]),(limits_min[2], limits_max[2]),(limits_min[3], limits_max[3]))

motion_service = session.service("ALMotion")
if not motion_service.robotIsWakeUp():
	motion_service.wakeUp()


joint_names = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']
hand_joint_name = joint_names[-1]

motion_service.setStiffnesses(joint_names, 0.9)

Sigma_cartesian = 1e-4**2*np.eye(3)
fwd_kin = PepperKinematics(lambda_theta=0., lambda_x=1.)

times = np.linspace(0.,1.,45)
rate = rospy.Rate(1./(times[1]*2))

predictor = NuitrackSkeletonPredictor()

while predictor.goal_pred is None:
	rate.sleep()

start = datetime.datetime.now()
for i in range(1,len(times)):
	# 1. Get current robot hand position
	curr_q = motion_service.getAngles(joint_names, True)

	# 2. Generate trajectory from current position to target position
	mu_q, Sigma_q = fwd_kin.inv_kin(mu_theta=np.zeros(4), sig_theta=np.eye(4),
									mu_x = predictor.goal_pred, sig_x = Sigma_cartesian, 
									method='L-BFGS-B', jac=None, bounds=bounds)
	rate.sleep()
	traj = np.linspace(curr_q[:-1], mu_q, len(times)-i+1)
	print(traj[1],mu_q)
	# 3. Publish sample 
	joint_values = np.hstack([traj[1], [np.pi/2]])
	motion_service.setAngles(joint_names, joint_values.tolist(), 0.2)
	if rospy.is_shutdown():
		break

print(datetime.datetime.now() - start)
rate.sleep()
pos_frame = motion.FRAME_TORSO
current_pos = motion_service.getPosition(joint_names[-1], pos_frame, True)[:3]
print(current_pos)
print(predictor.goal_pred)
print 'diff:',np.linalg.norm(current_pos - predictor.goal_pred)

pos_frame = motion.FRAME_ROBOT
current_pos = motion_service.getPosition(joint_names[-1], pos_frame, True)[:3]
print(current_pos)
print(predictor.goal_pred)
print 'diff:',np.linalg.norm(current_pos - predictor.goal_pred)

pos_frame = motion.FRAME_WORLD
current_pos = motion_service.getPosition(joint_names[-1], pos_frame, True)[:3]
print(current_pos)
print(predictor.goal_pred)
print 'diff:',np.linalg.norm(current_pos - predictor.goal_pred)

As = fwd_kin.forward_kinematics(traj[1])
current_pos = As[-1][0:3,3]
print(current_pos)
print(predictor.goal_pred)
print 'diff:',np.linalg.norm(current_pos - predictor.goal_pred)


time.sleep(5)
motion_service.setAngles(joint_names, [np.pi/2, -0.109, 0, 0.009, np.pi/2], 0.2)
time.sleep(1.5)
motion_service.setStiffnesses(joint_names, 0.)

rospy.signal_shutdown('Finished')