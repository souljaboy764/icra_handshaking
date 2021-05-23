from tf.transformations import *

from intprim.util.kinematics import BaseKinematicsClass

# MoveIt
from moveit_msgs.srv import GetPositionFKRequest, GetPositionFKResponse, GetPositionFK
from geometry_msgs.msg import TransformStamped

import numpy as np
import rospy

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

SPINEBASE = 0
SPINEMID = 1
NECK = 2
HEAD = 3
SHOULDERLEFT = 4
ELBOWLEFT = 5
WRISTLEFT = 6
HANDLEFT = 7
SHOULDERRIGHT = 8
ELBOWRIGHT = 9
WRISTRIGHT = 10
HANDRIGHT = 11
HIPLEFT = 12
KNEELEFT = 13
ANKLELEFT = 14
FOOTLEFT = 15
HIPRIGHT = 16
KNEERIGHT = 17
ANKLERIGHT = 18
FOOTRIGHT = 19
SPINESHOULDER = 20
HANDTIPLEFT = 21
THUMBLEFT = 22
HANDTIPRIGHT = 23
THUMBRIGHT = 24

UPPERBODY = np.array([SPINEBASE, SPINEMID, NECK, HEAD, SHOULDERLEFT, ELBOWLEFT, WRISTLEFT, HANDLEFT, SHOULDERRIGHT, ELBOWRIGHT, WRISTRIGHT, HANDRIGHT, HIPLEFT, HIPRIGHT, SPINESHOULDER])

BASE_LINK = "base_link"

class PepperKinematics(BaseKinematicsClass):
	""" Forward kinematics object for Pepper's right arm
	"""

	def __init__(self, lambda_x = 0.5, lambda_theta = 0.5):
		self.lambda_x = lambda_x
		self.lambda_theta = lambda_theta
		pass

	# q = [RShoulderPitch, RShoulderRoll, RElbowYaw, RElbowRoll]
	def _link_matrices(self,q):

		m00 = np.eye(4) # euler_matrix(0.05235987755982987, -0.0, 0.0) #base_link to right shoulder joint
		

		m01 = euler_matrix(0, q[0],0)
		m01[:3,3] = np.array([-0.057, -0.14974, 0.08682])

		m12 = euler_matrix(0, 0, q[1])
		
		m23 = euler_matrix(q[2], -0.157079, 0)
		m23[:3,3] = np.array([0.1812, -0.015, 0.00013])

		m34 = euler_matrix(0, 0, q[3])

		m45 = np.eye(4)
		m45[0,3] = 0.15
		return [m00,m01,m12,m23,m34,m45]

	def __laplace_cost_and_grad(self, theta, mu_theta, inv_sigma_theta, mu_x, inv_sigma_x):
		f_th, jac_th, ori = self.position_and_jac(theta)
		jac_th = jac_th[0:3,:]
		diff1 = theta - mu_theta
		tmp1 = np.dot(inv_sigma_theta, diff1)
		diff2 = f_th - mu_x
		tmp2 = np.dot(inv_sigma_x, diff2)
		nll = self.lambda_theta*np.dot(diff1,tmp1) + self.lambda_x*np.dot(diff2,tmp2)
		grad_nll = tmp1 + np.dot(jac_th.T,tmp2)

		return nll, grad_nll

def ros_fk(robot_state, fk_link_names):
	rospy.wait_for_service("compute_fk")
	fk_service_client = rospy.ServiceProxy("compute_fk", GetPositionFK)

	fk_service_request = GetPositionFKRequest()
	fk_service_response = GetPositionFKResponse()

	fk_service_request.header.frame_id=BASE_LINK
	fk_service_request.robot_state.joint_state = robot_state
	fk_service_request.fk_link_names = fk_link_names
	# Call the service
	try:
		fk_service_request.header.stamp = rospy.Time.now()
		fk_service_response = fk_service_client(fk_service_request)
		rospy.loginfo("FK Result: " + str((fk_service_response.error_code.val == fk_service_response.error_code.SUCCESS)) +' '+ str(fk_service_response.error_code.val))
		return fk_service_response.pose_stamped[0]
	except Exception as e:
		rospy.logerr("FK Service call failed: %s"%e)
		return None

def skeleton_rotation(body_transforms):
	
	xAxisHelper = body_transforms[SPINEBASE] - body_transforms[SHOULDERRIGHT]
	yAxis = body_transforms[SHOULDERLEFT] - body_transforms[SHOULDERRIGHT] # right to left
	xAxis = np.cross(xAxisHelper, yAxis) # out of the human(like an arrow from the back to front)
	zAxis = np.cross(xAxis, yAxis) # straight up from spine base to head
	
	xAxis /= np.linalg.norm(xAxis)
	yAxis /= np.linalg.norm(yAxis)
	zAxis /= np.linalg.norm(zAxis)

	transformationMatrixTF2 = np.zeros((4,4))
	transformationMatrixTF2[:3,:3] = np.array([[xAxis[0], xAxis[1], xAxis[2]],
									 [yAxis[0], yAxis[1], yAxis[2]],
									 [zAxis[0], zAxis[1], zAxis[2]]])

	transformationMatrixTF2[:3,3] = np.dot(transformationMatrixTF2[:3,:3],-body_transforms[SPINEBASE])

	return transformationMatrixTF2

def angle(a,b):
	dot = np.dot(a,b)
	return np.arccos(dot/(np.linalg.norm(a)*np.linalg.norm(b)))

def projectToPlane(plane, vec):
	return (vec - plane)*np.dot(plane,vec)
	
def transform_matrix(transform):
	m = quaternion_matrix([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])
	m[:3,3] = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
	return m

def transform_from_matrix(m):
	transform = TransformStamped().transform
	transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w = quaternion_from_matrix(m)
	transform.translation.x, transform.translation.y, transform.translation.z = m[:3,3]
	return transform

def poseStampedToTransformStamped(pose_stamped, child_frame_id):
	tf_stamped = TransformStamped()
	tf_stamped.header = pose_stamped.header
	tf_stamped.child_frame_id = child_frame_id
	tf_stamped.transform.translation.x = pose_stamped.pose.position.x
	tf_stamped.transform.translation.y = pose_stamped.pose.position.y
	tf_stamped.transform.translation.z = pose_stamped.pose.position.z

	tf_stamped.transform.rotation = pose_stamped.pose.orientation

	return tf_stamped


def skeleton2joints(body_transforms):
	# Adapted from https://github.com/robertocalandra/firstperson-teleoperation and https://github.com/souljaboy764/skeleton_teleop
	transforms = []
	for i in range(len(body_transforms)):
		transforms.append(body_transforms[i][:3])
	

	xAxisHelper = transforms[SPINEBASE] - transforms[SHOULDERRIGHT]
	yAxis = transforms[SHOULDERLEFT] - transforms[SHOULDERRIGHT] # right to left
	xAxis = np.cross(xAxisHelper, yAxis) # out of the human(like an arrow in the back)
	zAxis = np.cross(xAxis, yAxis) # like spine, but straight
	
	xAxis /= np.linalg.norm(xAxis)
	yAxis /= np.linalg.norm(yAxis)
	zAxis /= np.linalg.norm(zAxis)

	transformationMatrixTF2 = np.array([[xAxis[0], xAxis[1], xAxis[2]],
									 [yAxis[0], yAxis[1], yAxis[2]],
									 [zAxis[0], zAxis[1], zAxis[2]]])

	for i in range(len(transforms)):
		transforms[i] = np.dot(transformationMatrixTF2,transforms[i])
	
	#######################################################################################
	#										Right Arm									  #
	#######################################################################################
	
	#Recreating arm with upper and under arm
	rightUpperArm = transforms[ELBOWRIGHT] - transforms[SHOULDERRIGHT]
	rightUnderArm = transforms[HANDRIGHT] - transforms[ELBOWRIGHT]

	rightYaw = 0
	rightPitch = 0
	rightRoll = 0
	rightElbowAngle = angle(rightUpperArm, rightUnderArm)

	rightYaw = np.arctan2(rightUpperArm[1],-rightUpperArm[2]) # Comes from robot structure
	rightYaw -=0.009
	rightPitch = np.arctan(rightUpperArm[0]/rightUpperArm[2]) #Comes from robot structure
	rightPitch += np.pi/2

	#Recreating under Arm Position with known Angles(without roll)
	rightRotationAroundY = euler_matrix(0, rightPitch, 0)[:3,:3]
	rightRotationAroundX = euler_matrix(0, 0, rightYaw)[:3,:3]
	rightElbowRotation = euler_matrix(0, 0, rightElbowAngle)[:3,:3]

	
	rightUnderArmInZeroPos = np.array([np.linalg.norm(rightUnderArm), 0, 0.])
	rightUnderArmWithoutRoll = np.dot(rightRotationAroundY,np.dot(rightRotationAroundX,np.dot(rightElbowRotation,rightUnderArmInZeroPos)))

	#calculating the angle betwenn actual under arm position and the one calculated without roll
	rightRoll = angle(rightUnderArmWithoutRoll, rightUnderArm)
	
	
	#This is a check which sign the angle has as the calculation only produces positive angles
	rightRotationAroundArm = euler_matrix(0, 0, -rightRoll)[:3, :3]
	rightShouldBeWristPos = np.dot(rightRotationAroundY,np.dot(rightRotationAroundX,np.dot(rightRotationAroundArm,np.dot(rightElbowRotation,rightUnderArmInZeroPos))))
	r1saver = np.linalg.norm(rightUnderArm - rightShouldBeWristPos)
	
	rightRotationAroundArm = euler_matrix(0, 0, rightRoll)[:3, :3]
	rightShouldBeWristPos = np.dot(rightRotationAroundY,np.dot(rightRotationAroundX,np.dot(rightRotationAroundArm,np.dot(rightElbowRotation,rightUnderArmInZeroPos))))
	r1 = np.linalg.norm(rightUnderArm - rightShouldBeWristPos)
	
	if (r1 > r1saver):
		rightRoll = -rightRoll
	
	return np.array([rightYaw, rightPitch, rightRoll, rightElbowAngle])

def visual_skeleton(skeletons, pred=None):
	fig = plt.figure()
	ax = Axes3D(fig)

	ax.view_init(20, -45)
	plt.ion()
	print(len(skeletons))
	# show every frame 3d skeleton
	for frame_idx in range(len(skeletons)):

		plt.cla()
		plt.title("Frame: {}".format(frame_idx))

		ax.set_xlim3d([-0.5, 0.5])
		ax.set_ylim3d([-0.5, 0.5])
		ax.set_zlim3d([-0.1, 0.8])

		x = skeletons[frame_idx].T[0]
		y = skeletons[frame_idx].T[1]
		z = skeletons[frame_idx].T[2]

		if pred is not None:
			ax.scatter(preds[frame_idx][0], preds[frame_idx][1], preds[frame_idx][2], marker='o', color='r')
			
		ax.scatter(x, y, z, marker='o', color='b')
		ax.text(x[HANDRIGHT], y[HANDRIGHT], z[HANDRIGHT], "HANDRIGHT", None)
		ax.text(x[HANDLEFT], y[HANDLEFT], z[HANDLEFT], "HANDLEFT", None)

		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')

		ax.set_facecolor('none')
		plt.pause(0.03)

	plt.ioff()
	# plt.close("all")
	plt.show()