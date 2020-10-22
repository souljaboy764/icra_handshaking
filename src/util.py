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


def skeleton_rotation(body_transforms):
	
	zAxisHelper = body_transforms[SPINEBASE][:3] - body_transforms[SHOULDERRIGHT][:3]
	xAxis = body_transforms[SHOULDERLEFT][:3] - body_transforms[SHOULDERRIGHT][:3] #right to left
	zAxis = np.cross(zAxisHelper, xAxis) #out of the human(like an arrow in the back)
	yAxis = np.cross(zAxis, xAxis) #like spine, but straight
	
	xAxis /= np.linalg.norm(xAxis)
	yAxis /= np.linalg.norm(yAxis)
	zAxis /= np.linalg.norm(zAxis)

	transformationMatrixTF2 = np.eye(4)
	transformationMatrixTF2[:3,:3] = np.array([[xAxis[0], xAxis[1], xAxis[2]],
									 [yAxis[0], yAxis[1], yAxis[2]],
									 [zAxis[0], zAxis[1], zAxis[2]]])

	transformationMatrixTF2[:3,3] = np.dot(transformationMatrixTF2[:3,:3],-body_transforms[SPINEBASE][:3])

	return transformationMatrixTF2



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


def angle(a,b):
	dot = np.dot(a,b)
	return np.arccos(dot/(np.linalg.norm(a)*np.linalg.norm(b)))
def non_homogeneous(v):
	return v[:3]/v[3]

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

def skeleton2joints(body_transforms):
	transforms = []
	for i in range(len(body_transforms)):
		transforms.append(body_transforms[i][:3])
	

	xAxisHelper = transforms[SPINEBASE] - transforms[SHOULDERRIGHT]
	yAxis = transforms[SHOULDERLEFT] - transforms[SHOULDERRIGHT] #right to left
	xAxis = np.cross(xAxisHelper, yAxis) #out of the human(like an arrow in the back)
	zAxis = np.cross(xAxis, yAxis) #like spine, but straight
	
	# Coordinate System in the room
	gravity = np.array([0, 1, 0])
	groundX = np.array([-1, 0, 0])
	groundZ = np.array([0, 0, -1])

	xAxis /= np.linalg.norm(xAxis)
	yAxis /= np.linalg.norm(yAxis)
	zAxis /= np.linalg.norm(zAxis)

	transformationMatrixTF2 = np.array([[xAxis[0], xAxis[1], xAxis[2]],
									 [yAxis[0], yAxis[1], yAxis[2]],
									 [zAxis[0], zAxis[1], zAxis[2]]])												 
	
	
	gravity = np.dot(transformationMatrixTF2,gravity)
	groundX = np.dot(transformationMatrixTF2,groundX)
	groundZ = np.dot(transformationMatrixTF2,groundZ)

	for i in range(len(transforms)):
		transforms[i] = np.dot(transformationMatrixTF2,transforms[i])
	
	xAxis = np.array([1, 0, 0])
	yAxis = np.array([0, 1, 0])
	zAxis = np.array([0, 0, 1])

	# create help planes
	frontView = np.cross(xAxis, yAxis) #Plane for front view: normal is zAxis
	frontView = frontView/np.linalg.norm(frontView)
	
	sideView = np.cross(yAxis, zAxis) #Plane for side view: normal is xAxis
	sideView = sideView/np.linalg.norm(sideView)
	
	topView = np.cross(zAxis, xAxis) #Plane for top view: normal is yAxis
	topView = topView/np.linalg.norm(topView)

	ground = np.cross(groundZ, groundX)
	ground = ground/np.linalg.norm(ground)
	
	#######################################################################################
	#										Left Arm									  #
	#######################################################################################
	
	#Recreating arm with upper and under arm
	leftUpperArm = transforms[ELBOWLEFT] - transforms[SHOULDERLEFT]
	leftUnderArm = transforms[HANDLEFT] - transforms[ELBOWLEFT]

	leftYaw = 0
	leftPitch = 0
	leftRoll = 0
	leftElbowAngle = angle(leftUpperArm, leftUnderArm)

	leftYaw = np.arcsin(leftUpperArm[1]/np.linalg.norm(leftUpperArm)) #Comes from robot structure
	leftYaw += 0.009
	leftPitch = np.arctan2(leftUpperArm[0], leftUpperArm[2]) #Comes from robot structure
	leftPitch += np.pi/2
	
	#Recreating under Arm Position with known Angles(without roll)
	leftRotationAroundY = euler_matrix(0, leftPitch, 0)[:3,:3]
	leftRotationAroundX = euler_matrix(0, 0, leftYaw)[:3,:3]
	leftElbowRotation = euler_matrix(0, 0, leftElbowAngle)[:3,:3]

	leftUnderArmInZeroPos = np.array([np.linalg.norm(leftUnderArm), 0, 0.])
	leftUnderArmWithoutRoll = np.dot(leftRotationAroundY,np.dot(leftRotationAroundX,np.dot(leftElbowRotation,leftUnderArmInZeroPos)))

	#calculating the angle betwenn actual under arm position and the one calculated without roll
	leftRoll = angle(leftUnderArmWithoutRoll, leftUnderArm)
	
	
	#This is a check which sign the angle has as the calculation only produces positive angles
	leftRotationAroundArm = euler_matrix(0, 0, -leftRoll)[:3, :3]
	leftShouldBeWristPos = np.dot(leftRotationAroundY,np.dot(leftRotationAroundX,np.dot(leftRotationAroundArm,np.dot(leftElbowRotation,leftUnderArmInZeroPos))))
	l1saver = np.linalg.norm(leftUnderArm - leftShouldBeWristPos)
	
	leftRotationAroundArm = euler_matrix(0, 0, leftRoll)[:3, :3]
	leftShouldBeWristPos = np.dot(leftRotationAroundY,np.dot(leftRotationAroundX,np.dot(leftRotationAroundArm,np.dot(leftElbowRotation,leftUnderArmInZeroPos))))
	l1 = np.linalg.norm(leftUnderArm - leftShouldBeWristPos)
	
	if l1 > l1saver:
		leftRoll = -leftRoll
	
	# print "LEFT ANGLES: %.3f %.3f %.3f %.3f"%(leftYaw,leftPitch, leftRoll, leftElbowAngle)
	
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

	rightYaw = np.arcsin(rightUpperArm[1]/np.linalg.norm(rightUpperArm)) #Comes from robot structure
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
	
	return np.array([rightYaw, rightPitch, rightRoll, rightElbowAngle, leftYaw,leftPitch, leftRoll, leftElbowAngle])

# def visualize_skeleton(trajectory, preds=None):
# 	fig = plt.figure()
# 	ax = Axes3D(fig)

# 	ax.view_init(20, -45)
# 	plt.ion()
# 	print(len(trajectory))
# 	# show every frame 3d skeleton
# 	for frame_idx in range(1,len(trajectory)):

# 		plt.cla()
# 		plt.title("Frame: {}".format(frame_idx))

# 		ax.set_xlim3d([-1, 1])
# 		ax.set_ylim3d([-1, 1])
# 		ax.set_zlim3d([-0.8, 0.8])
# 		x = trajectory[:frame_idx,0]
# 		y = trajectory[:frame_idx,1]
# 		z = trajectory[:frame_idx,2]

# 		ax.plot(x, y, z, color='b')
# 		if preds is not None:
# 			ax.scatter(preds[frame_idx-1][0], preds[frame_idx-1][1], preds[frame_idx-1][2], marker='o', color='r')

# 		ax.set_xlabel('X')
# 		ax.set_ylabel('Y')
# 		ax.set_zlabel('Z')

# 		ax.set_facecolor('none')
# 		plt.pause(0.03)

# 	plt.ioff()
# 	# plt.close("all")
# 	plt.show()

def visualize_skeleton(skeletons, preds=None):
	fig = plt.figure()
	ax = Axes3D(fig)

	ax.view_init(20, -45)
	plt.ion()
	print(len(skeletons))
	# show every frame 3d skeleton
	for frame_idx in range(len(skeletons)):

		plt.cla()
		plt.title("Frame: {}".format(frame_idx))

		ax.set_xlim3d([-1, 1])
		ax.set_ylim3d([-1, 1])
		ax.set_zlim3d([-0.8, 0.8])

		x = skeletons[frame_idx].T[0]
		y = skeletons[frame_idx].T[1]
		z = skeletons[frame_idx].T[2]

		# for part in range(len(x)):
		# 	x_plot = x[part]
		# 	y_plot = y[part]
		# 	z_plot = z[part]
		labels = [str(i) for i in range(len(x))]
		ax.scatter(x[:-4], y[:-4], z[:-4], marker='o', color='b')
		for i in range(len(x[:-4])):
			ax.text(x[i], y[i]+0.05, z[i], labels[i], None)
		
		if preds is not None:
			ax.scatter(preds[frame_idx][0], preds[frame_idx][1], preds[frame_idx][2], marker='o', color='r')

		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')

		ax.set_facecolor('none')
		plt.pause(0.03)

	plt.ioff()
	# plt.close("all")
	plt.show()