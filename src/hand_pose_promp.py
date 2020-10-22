#!/usr/bin/python
from tf.transformations import *
import tf2_ros
from geometry_msgs.msg import TransformStamped

import intprim
from intprim.util.kinematics import BaseKinematicsClass
from intprim.probabilistic_movement_primitives import ProMP

# MoveIt
from moveit_commander.robot import RobotCommander
from moveit_msgs.srv import GetPositionIKRequest, GetPositionIKResponse, GetPositionIK, GetPositionFKRequest, GetPositionFKResponse, GetPositionFK
from moveit_msgs.msg import DisplayRobotState, RobotTrajectory, DisplayTrajectory, OrientationConstraint
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import TransformStamped, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import JointState

import json

import scipy.optimize as opt
from dtw import *

import os
import sys

import numpy as np
import rospy
import rospkg

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from util import *



data = np.load(os.path.join(rospkg.RosPack().get_path('icra_handshaking'),'handreach_data.npz'), allow_pickle=True)
time = data['time']
Q = data['Q']
num_samples = len(Q)/3
Q1 = Q[:num_samples]
Q2 = Q[num_samples:2*num_samples]
skeletons = Q[2*num_samples:]
time1 = time[:num_samples]
num_joints = Q1[0].shape[1]

hand_locations = []

for i in range(num_samples):
	q1_diff = np.diff(Q1[i],axis=0).T
	avg_vel = np.linalg.norm(q1_diff,axis=0)
	first = np.where(avg_vel>0.002)[0][0]
	Q1[i] = Q1[i][first:]
	time1[i] = time1[i][first:]
	Q2[i] = Q2[i][first:]
	skeletons[i] = skeletons[i][first:]
	skeletons[i] = np.array(skeletons[i])
	skeletons[i] = skeletons[i].T[:3].T
	skeletons[i] = skeletons[i][:,UPPERBODY,:]
	hand_locations.append([])
	rotMat = skeleton_rotation(skeletons[i][0])
	# for j in range(len(skeletons[i])):
	# 	hand_locations[-1].append(rotMat[:3,:3].dot(skeletons[i][j][HANDRIGHT][:3]) + rotMat[:3,3])
	# hand_locations[-1] = np.array(hand_locations[-1])
	for j in range(len(skeletons[i])):
		skeletons[i][j] = (rotMat[:3,:3].dot(skeletons[i][j].T) + np.expand_dims(rotMat[:3,3],-1)).T
	seq_len, joints, vals = skeletons[i].shape
	skeletons[i] = skeletons[i].reshape(seq_len, joints*vals)
	print(seq_len, joints, vals)
print 'Finished getting hand locs'

# data = np.load('/home/elenoide/playground/vinayavekhin_lstm/handreach_skeletons.npz', allow_pickle=True, encoding='bytes')
# skeletons = data['Q']

# hand_locations = []
# for i in range(len(skeletons)):
# 	skeletons[i] = np.array(skeletons[i])
# 	hand_locations.append([])
# 	rotMat = skeleton_rotation(skeletons[i][0])
# 	print skeletons[i].shape
# 	for j in range(len(skeletons[i])):
# 		# hand_locations[-1].append(rotMat[:3,:3].dot(skeletons[i][j][HANDRIGHT][:3]) + rotMat[:3,3])
# 		skeletons[i][j] = (rotMat[:3,:3].dot(skeletons[i][j].T) + np.expand_dims(rotMat[:3,3],-1)).T
# 	seq_len, joints, vals = skeletons[i].shape
# 	skeletons[i] = skeletons[i].reshape(seq_len, joints*vals)
# 	# hand_locations[-1] = np.array(hand_locations[-1])
print 'Finished getting hand locs'


max_len = 0
max_len_index = 0
for i in range(len(skeletons)):
	if len(skeletons[i])>max_len:
		max_len = len(skeletons[i])
		max_len_index = i
template = skeletons[max_len_index]
num_joints = skeletons[-1].shape[-1]
basis_model = intprim.basis.GaussianModel(30, 0.0001, ['joint'+str(i) for i in range(num_joints)])
# basis_model = intprim.basis.PolynomialModel(2, ['joint'+str(i) for i in range(num_joints)])
promp = ProMP(basis_model)

for i in range(len(skeletons[:50])):
	# alignment = dtw(hand_locations[i], template, lambda x,y:np.linalg.norm(x-y))
	# warped = hand_locations[i][alignment[3][0]]
	promp.add_demonstration(skeletons[i].T)

# promp.export_data('hand_pose_promp.pkl')
# promp.import_data('hand_pose_promp.pkl')
observation = np.random.choice(skeletons)
alignment = dtw(observation, template, lambda x,y:np.linalg.norm(x-y))
observation = observation[alignment[3][0]]
mu_w_cond, Sigma_w_cond = promp.get_conditioned_weights(0.0, observation[0])
times = np.linspace(0,1,len(observation))
preds = []
for i in range(1,len(observation)):
	mu_w_cond, Sigma_w_cond = promp.get_conditioned_weights(times[i], mean_q=observation[i], mean_w=mu_w_cond, var_w=Sigma_w_cond)
	pred, _ = promp.generate_probable_trajectory(np.array([1.]), mu_w_cond, np.zeros(Sigma_w_cond.shape))
	preds.append(pred[:,0][11*3:12*3])
	print preds[-1], observation[-1][11*3:12*3]
visualize_skeleton(observation.reshape(-1,15,3)[1:], preds)

# rospy.init_node("pepper_handreach", sys.argv)
# robot_traj_publisher = rospy.Publisher("/pepper_dcm/RightArm_controller/command", JointTrajectory, queue_size=1)
# # robot_traj_publisher = rospy.Publisher("tutorial_robot_traj", DisplayTrajectory, queue_size=1)
# robot_state_publisher = rospy.Publisher("final_robot_state", DisplayRobotState, queue_size=1)

# broadcaster = tf2_ros.StaticTransformBroadcaster()
# tfBuffer = tf2_ros.Buffer()
# listener = tf2_ros.TransformListener(tfBuffer)

# robot_model_loader = RobotCommander("robot_description")
# move_group = robot_model_loader.get_group("right_arm")

# times = np.linspace(0.,1.,100)
# fwd_kin = PepperKinematics()

# joint_names = move_group.get_active_joints() # ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']
# robotTrajectory = RobotTrajectory()
# robotTrajectory.joint_trajectory.header.frame_id = BASE_LINK
# robotTrajectory.joint_trajectory.joint_names = joint_names
# print robotTrajectory.joint_trajectory.joint_names

# for i in range(len(times)):
# 	robotTrajectory.joint_trajectory.points.append(JointTrajectoryPoint())
# 	robotTrajectory.joint_trajectory.points[-1].time_from_start = rospy.Duration.from_sec(times[i]*1.5)

# traj_msg = DisplayTrajectory()
# traj_msg.trajectory_start.joint_state.header.frame_id = BASE_LINK
# traj_msg.trajectory_start.joint_state.name = robotTrajectory.joint_trajectory.joint_names

# state_msg = DisplayRobotState()

# current_state = robot_model_loader.get_current_state()
# current_state.joint_state.position = list(current_state.joint_state.position)

# hand_traj = np.linspace(np.pi/2, np.pi/2 , len(times))

# rospy.loginfo("SLEEPING")
# rospy.Rate(0.2).sleep()
# rospy.loginfo("STARTING")

# # while not rospy.is_shutdown():
# for spin in range(10):
# 	# Condition on the initial position
# 	q_init = [np.pi/2, -0.109, 0, 0.009] 
# 	mu_w_task, Sigma_w_task = promp.get_conditioned_weights(0.0, q_init)#np.array([q_init[1], q_init[0], q_init[2], q_init[3]]))

# 	prior_mu_q, prior_Sigma_q = promp.get_marginal(1.0, mu_w_task, Sigma_w_task)

# 	try:
# 		trans = tfBuffer.lookup_transform("base_link", "perception_nui_1_1_right_hand", rospy.Time(0), rospy.Duration(5))
# 	except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
# 		rospy.logerr("COULD NOT GET HUMAN HAND LOCATION")
# 		exit
# 	mu_cartesian = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])

# 	# mu_cartesian = np.array([0.25212134, -0.07043525, 0.07195387]) 
# 	# mu_cartesian = np.array([0.27716177, -0.14197945, 0.10070093])

# 	Sigma_cartesian = 1e-4**2*np.eye(3)

# 	print 'prior_mu_q:', prior_mu_q
# 	print 'prior_Sigma_q:', prior_Sigma_q
# 	bounds = ((limits_min[1], limits_max[1]),(limits_min[0], limits_max[0]),(limits_min[2], limits_max[2]),(limits_min[3], limits_max[3]))
# 	mu_q, Sigma_q = fwd_kin.inv_kin(mu_theta=prior_mu_q, sig_theta=prior_Sigma_q,
# 			mu_x = mu_cartesian, sig_x = Sigma_cartesian, method='L-BFGS-B', jac=None, bounds=bounds)
# 	if hasattr(Sigma_q, 'todense'):
# 		Sigma_q = Sigma_q.todense()
# 	print 'Sigma_q:'
# 	print Sigma_q
# 	print 'mu_q:', mu_q
# 	print 'mu_cartesian:', mu_cartesian
# 	As = fwd_kin.forward_kinematics(mu_q)
# 	end_eff = As[-1]
# 	pos = end_eff[0:3,3]
# 	quat = quaternion_from_matrix(end_eff)
# 	print 'mu_q pos:',pos
# 	print 'diff:',np.linalg.norm(pos-mu_cartesian)

# 	mu_w_task, Sigma_w_task = promp.get_conditioned_weights(1.0, mean_q=mu_q, var_q=Sigma_q,
# 															mean_w=mu_w_task, var_w=Sigma_w_task)

# 	samples, w = promp.generate_probable_trajectory(times, mu_w_task, Sigma_w_task)
# 	samples = samples.T

# 	i = 0
# 	stamp = rospy.Time.now()
# 	robotTrajectory.joint_trajectory.header.stamp = stamp
# 	traj_msg.trajectory_start.joint_state.header.stamp = stamp
# 	samples[i][0] = np.clip(samples[i][0], -2.08567, 2.08567)
# 	samples[i][1] = np.clip(samples[i][1], -1.56207, -0.00872665)
# 	samples[i][2] = np.clip(samples[i][2], -2.08567, 2.08567)
# 	samples[i][3] = np.clip(samples[i][3], 0.00872665, 1.56207)
# 	traj_msg.trajectory_start.joint_state.position = np.hstack([samples[i], hand_traj[i]])

	
# 	for i in range(len(samples)):
# 		samples[i][0] = np.clip(samples[i][0], -2.08567, 2.08567)
# 		samples[i][1] = np.clip(samples[i][1], -1.56207, -0.00872665)
# 		samples[i][2] = np.clip(samples[i][2], -2.08567, 2.08567)
# 		samples[i][3] = np.clip(samples[i][3], 0.00872665, 1.56207)
# 		robotTrajectory.joint_trajectory.points[i].positions = np.hstack([samples[i], [hand_traj[i]]])

# 	joint_state = JointState()
# 	joint_state.header.frame_id = BASE_LINK
# 	joint_state.header.stamp = stamp
# 	joint_state.name = robotTrajectory.joint_trajectory.joint_names
# 	joint_state.position = np.hstack([samples[-1], [hand_traj[-1]]])
# 	state_msg.state.joint_state = joint_state
	
# 	# pose = ros_fk(joint_state, ['r_wrist'])
# 	# if pose is not None:
# 	# 	pose.header.frame_id = BASE_LINK
# 	# 	fk_poseTF = poseStampedToTransformStamped(pose, "ros_fk_result")
# 	# 	broadcaster.sendTransform(fk_poseTF)
	
# 	As = fwd_kin.forward_kinematics(samples[-1])
# 	end_eff = As[-1]
# 	promp_pos = end_eff[0:3,3]
# 	quat = quaternion_from_matrix(end_eff)
# 	print 'promp pos:',promp_pos
# 	print 'diff:',np.linalg.norm(promp_pos-mu_cartesian)
# 	pose = PoseStamped()
# 	pose.header.frame_id = BASE_LINK
# 	pose.pose.position.x = promp_pos[0]
# 	pose.pose.position.y = promp_pos[1]
# 	pose.pose.position.z = promp_pos[2]
# 	pose.pose.orientation.x = quat[0]
# 	pose.pose.orientation.y = quat[1]
# 	pose.pose.orientation.z = quat[2]
# 	pose.pose.orientation.w = quat[3]
	
# 	promp_result_poseTF = poseStampedToTransformStamped(pose, "promp_result")
# 	broadcaster.sendTransform(promp_result_poseTF)

# 	As = fwd_kin.forward_kinematics(mu_q)
# 	end_eff = As[-1]
# 	mu_q_pos = end_eff[0:3,3]
# 	quat = quaternion_from_matrix(end_eff)

# 	pose.pose.position.x = mu_q_pos[0]
# 	pose.pose.position.y = mu_q_pos[1]
# 	pose.pose.position.z = mu_q_pos[2]
# 	pose.pose.orientation.x = quat[0]
# 	pose.pose.orientation.y = quat[1]
# 	pose.pose.orientation.z = quat[2]
# 	pose.pose.orientation.w = quat[3]
	
# 	ik_result_poseTF = poseStampedToTransformStamped(pose, "ik_result")
# 	broadcaster.sendTransform(ik_result_poseTF)

# 	pose.pose.position.x = mu_cartesian[0]
# 	pose.pose.position.y = mu_cartesian[1]
# 	pose.pose.position.z = mu_cartesian[2]
# 	pose.pose.orientation.x = 0
# 	pose.pose.orientation.y = 0
# 	pose.pose.orientation.z = 0
# 	pose.pose.orientation.w = 1
	
# 	target_poseTF = poseStampedToTransformStamped(pose, "target")
# 	broadcaster.sendTransform(target_poseTF)
	
# 	# traj_msg.trajectory.append(robotTrajectory)
# 	# robot_traj_publisher.publish(traj_msg)
# 	robot_traj_publisher.publish(robotTrajectory.joint_trajectory)
# 	robot_state_publisher.publish(state_msg)
# 	rospy.Rate(0.2).sleep()

# 	move_group.set_joint_value_target([np.pi/2, -0.109, 0, 0.009, np.pi/2])
# 	move_group.go()
# 	rospy.Rate(0.2).sleep()
# 	if rospy.is_shutdown():
# 		break