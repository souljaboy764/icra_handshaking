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

import tf2_ros
from geometry_msgs.msg import TransformStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState


rospy.init_node('skeleton_predictor')
robot_traj_publisher = rospy.Publisher("/pepper_dcm/RightArm_controller/command", JointTrajectory, queue_size=1)

limits_max = [2.08567, -0.00872665, 2.08567, 1.56207]
limits_min = [-2.08567, -1.56207, -2.08567, 0.00872665]
bounds = ((limits_min[0], limits_max[0]),(limits_min[1], limits_max[1]),(limits_min[2], limits_max[2]),(limits_min[3], limits_max[3]))

fwd_kin = PepperKinematics(lambda_theta=0.5, lambda_x=0.5)
# basis_model = intprim.basis.PolynomialModel(3, ['joint'+str(i) for i in range(4)])
basis_model = intprim.basis.GaussianModel(3, 0.1, ['joint'+str(i) for i in range(4)])
promp = ProMP(basis_model)
promp.import_data(os.path.join(rospkg.RosPack().get_path('icra_handshaking'), 'skeleton_promp_gaussian_3der.pkl'))
joint_names = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']

Sigma_cartesian = 1e-4**2*np.eye(3)
times = np.linspace(0.,1.,70)
rate = rospy.Rate(1./(times[1]*1.5))

joint_trajectory = JointTrajectory()
joint_trajectory.header.frame_id = "base_link"
joint_trajectory.joint_names = joint_names
joint_trajectory.points.append(JointTrajectoryPoint())
joint_trajectory.points[0].time_from_start = rospy.Duration.from_sec(times[1]*1.5)

hand_loc_TF = TransformStamped()
hand_loc_TF.header.frame_id = "base_link"
hand_loc_TF.transform.rotation.x = hand_loc_TF.transform.rotation.y = hand_loc_TF.transform.rotation.z = 0
hand_loc_TF.transform.rotation.w = 1

broadcaster = tf2_ros.StaticTransformBroadcaster()

last_q = np.array([np.pi/2, -0.109, 0, 0.009])
mu_w_init, Sigma_w_init = promp.get_basis_weight_parameters()

predictor = NuitrackSkeletonPredictor()

while predictor.goal_pred is None and not rospy.is_shutdown():
	rate.sleep()

if rospy.is_shutdown():
	exit(-1)

start = datetime.datetime.now()

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
	
	#2. sample from conditioned promp
	samples, w = promp.generate_probable_trajectory(np.array([times[i]]), mu_w_task, np.zeros(Sigma_w_task.shape))
	last_q = np.clip(samples.T[0], limits_min, limits_max)
	joint_values = np.hstack([last_q, [np.pi/2]])
	joint_trajectory.points[0].positions = joint_values
		
	#3. publish sample 
	joint_trajectory.header.stamp = rospy.Time.now()
	robot_traj_publisher.publish(joint_trajectory)
	
	if rospy.is_shutdown():
		break

rate.sleep()

# Validating locations in rviz
As = fwd_kin.forward_kinematics(last_q)
promp_pos = As[-1][0:3,3]

hand_loc_TF.transform.translation.x = promp_pos[0]
hand_loc_TF.transform.translation.y = promp_pos[1]
hand_loc_TF.transform.translation.z = promp_pos[2]
hand_loc_TF.child_frame_id = "promp_pose"
broadcaster.sendTransform(hand_loc_TF)

As = fwd_kin.forward_kinematics(mu_q)
promp_pos = As[-1][0:3,3]

hand_loc_TF.transform.translation.x = promp_pos[0]
hand_loc_TF.transform.translation.y = promp_pos[1]
hand_loc_TF.transform.translation.z = promp_pos[2]
hand_loc_TF.child_frame_id = "inv_kin_pose"
broadcaster.sendTransform(hand_loc_TF)

# Resetting Pepper
rospy.Rate(0.2).sleep()
joint_trajectory.points[0].positions = [np.pi/2, -0.109, 0, 0.009, np.pi/2]
joint_trajectory.header.stamp = rospy.Time.now()
robot_traj_publisher.publish(joint_trajectory)
rospy.Rate(0.5).sleep()
rospy.signal_shutdown('Finished')