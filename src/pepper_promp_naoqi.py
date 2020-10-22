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

fwd_kin = PepperKinematics()
# basis_model = intprim.basis.PolynomialModel(3, ['joint'+str(i) for i in range(4)])
basis_model = intprim.basis.GaussianModel(3, 0.1, ['joint'+str(i) for i in range(4)])
promp = ProMP(basis_model)
promp.import_data(os.path.join(rospkg.RosPack().get_path('icra_handshaking'), 'skeleton_promp_gaussian_3der.pkl'))
joint_names = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']

motion_service = session.service("ALMotion")
if not motion_service.robotIsWakeUp():
	motion_service.wakeUp()
motion_service.setStiffnesses(joint_names, 0.9)

Sigma_cartesian = 1e-4**2*np.eye(3)
times = np.linspace(0.,1.,45)
rate = rospy.Rate(1./(times[1]*2))


# for trial in range(1):
last_q = np.array([np.pi/2, -0.109, 0, 0.009])
mu_w_init, Sigma_w_init = promp.get_basis_weight_parameters()

predictor = NuitrackSkeletonPredictor()

while predictor.goal_pred is None:
	rate.sleep()

start = datetime.datetime.now()
# mu_cartesian = np.array([0.25212134, -0.07043525, 0.07195387]) 
for i in range(1,len(times)):
	rate.sleep()

	#1. condition promp to reach LSTM predicted hand pose starting from pose at current time
	mu_w_init, Sigma_w_init = promp.get_conditioned_weights(times[i-1], last_q, mean_w=mu_w_init, var_w=Sigma_w_init)
	prior_mu_q, prior_Sigma_q = promp.get_marginal(1.0, mu_w_init, Sigma_w_init)
	mu_q, Sigma_q = fwd_kin.inv_kin(mu_theta=prior_mu_q, sig_theta=prior_Sigma_q,
									mu_x = predictor.goal_pred, sig_x = Sigma_cartesian, 
									method='L-BFGS-B', jac=None, bounds=bounds)
	if hasattr(Sigma_q, 'todense'):
		Sigma_q = Sigma_q.todense()
		if np.allclose(Sigma_q,np.eye(4)):
			Sigma_q = None
	try:
		mu_w_task, Sigma_w_task = promp.get_conditioned_weights(1.0, mean_q=mu_q, var_q=Sigma_q,
													mean_w=mu_w_init, var_w=Sigma_w_init)
	except Exception as e:
		print(e)
		print(Sigma_q)
	
	#2. sample from conditioned promp
	samples, w = promp.generate_probable_trajectory(np.array([times[i]]), mu_w_task, Sigma_w_task)
	last_q = np.clip(samples.T[0], limits_min, limits_max)
	joint_values = np.hstack([last_q, [np.pi/2]])
		
	#3. publish sample 
	motion_service.setAngles(joint_names, joint_values.tolist(), 0.2)
	sensor_readings = motion_service.getAngles(joint_names, True)
	print(joint_values,sensor_readings)
	if rospy.is_shutdown():
		print'SHUTDOWN!!'
		break

print(last_q)
print(datetime.datetime.now() - start)
As = fwd_kin.forward_kinematics(last_q)
promp_pos = As[-1][0:3,3]
print 'promp pos:',promp_pos
print 'predictor.goal_pred:', predictor.goal_pred
# print 'mu_cartesian:', mu_cartesian
print 'diff:',np.linalg.norm(promp_pos - predictor.goal_pred)

As = fwd_kin.forward_kinematics(np.clip(mu_q, limits_min, limits_max))
promp_pos = As[-1][0:3,3]
print 'mu_q pos:',promp_pos
print 'predictor.goal_pred:', predictor.goal_pred
# print 'mu_cartesian:', mu_cartesian
print 'diff:',np.linalg.norm(promp_pos - predictor.goal_pred)
print np.clip(mu_q, limits_min, limits_max)

time.sleep(2)
motion_service.setAngles(joint_names, [np.pi/2, -0.109, 0, 0.009, np.pi/2], 0.2)
time.sleep(2)
motion_service.setStiffnesses(joint_names, 0.)
rospy.signal_shutdown('Finished')