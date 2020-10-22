#!/usr/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import intprim
from intprim.util.kinematics import BaseKinematicsClass
from intprim.probabilistic_movement_primitives import ProMP

# import qi

import time
import sys
import os
import argparse

from util import *
from networks import Model

# MoveIt
from moveit_commander.robot import RobotCommander
from moveit_msgs.srv import GetPositionIKRequest, GetPositionIKResponse, GetPositionIK, GetPositionFKRequest, GetPositionFKResponse, GetPositionFK
from moveit_msgs.msg import DisplayRobotState, RobotTrajectory, DisplayTrajectory, OrientationConstraint
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import TransformStamped, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

import rospy
import rospkg
import tf2_ros




class NuitrackSkeletonPredictor:
	def __init__(self, input_dim=15*3, hidden_dim=64, checkpoint=os.path.join(rospkg.RosPack().get_path('icra_handshaking'),'final.pth')):#, promp_data, ):
		self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print('Creating Model')
		self._model = Model(input_dim, hidden_dim).to(self._device)
		try:
			checkpoint = torch.load(checkpoint)
			self._model.load_state_dict(checkpoint['model'])
		except Exception as e:
			print('Exception occurred while loading checkpoint:')
			print(e)
			exit(-1)
		self.skeleton_trajectory = []
		self._model.eval()

		# Transform from base_link to camera
		self.baseTransform = quaternion_matrix((0.168729717779, -0.619205892655, 0.683988627175, -0.346805280719))
		self.baseTransform[:3,3] = np.array([-0.12, 0.49, 0.19])
		self.first_msg = None

		self.goal_pred = None
		self.goalTF = TransformStamped()
		self.goalTF.header.frame_id = "base_link"
		self.goalTF.child_frame_id = "lstm_pred"
		self.goalTF.transform.rotation.x = self.goalTF.transform.rotation.y = self.goalTF.transform.rotation.z = 0
		self.goalTF.transform.rotation.w = 1

		self._broadcaster = tf2_ros.StaticTransformBroadcaster()
		self._nui_sub = rospy.Subscriber("/perception/skeletondata_0", Float32MultiArray, self.skeletonCb)
		rospy.loginfo('NuitrackSkeletonPredictor Ready!')

	def skeletonCb(self, msg):
		msg.data = np.array(msg.data)
		points = msg.data.reshape(-1,3)
		hand_loc = points[11]

		# Compute matrix for pose normalization
		if len(self.skeleton_trajectory) == 0:
			self._rotMat = skeleton_rotation(points)
			print(self._rotMat)
			self._rotmatinv = np.linalg.inv(self._rotMat)
		points = self._rotMat[:3,:3].dot(points.T) + np.expand_dims(self._rotMat[:3,3],-1)

		self.skeleton_trajectory.append(points.T.flatten())

		# Forward pass based on the observed skeleton trajectory till now
		x = torch.Tensor(self.skeleton_trajectory).to(self._device)
		x = x.view(1, len(self.skeleton_trajectory), len(self.skeleton_trajectory[0]))
		pred = self._model(x)
		pred = pred[0,-1].cpu().detach().numpy()
		
		# Invert the predicted pose from the normalized frame to world frame
		pred = self._rotmatinv[:3,:3].dot(pred) + self._rotmatinv[:3,3]


		# phase = (len(self.skeleton_trajectory) - 25)
		factor = 1/(1 + np.exp(15 - len(self.skeleton_trajectory)))

		self.goal_pred = factor*hand_loc + (1-factor)*pred
		self.goal_pred = self.baseTransform[:3,:3].dot(self.goal_pred) + self.baseTransform[:3,3]
		
		self.goalTF.transform.translation.x = self.goal_pred[0]
		self.goalTF.transform.translation.y = self.goal_pred[1]
		self.goalTF.transform.translation.z = self.goal_pred[2]

		self.goalTF.header.stamp = rospy.Time.now()
		self._broadcaster.sendTransform(self.goalTF)

		if len(self.skeleton_trajectory)>70:
			rospy.signal_shutdown('done')

if __name__=='__main__':
	rospy.init_node('skeleton_predictor')
	obj = NuitrackSkeletonPredictor()
	rospy.spin()