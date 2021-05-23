import re
import numpy as np
from glob import glob
import os
import warnings
import argparse

from util import *

warnings.filterwarnings("error")
	

parser = argparse.ArgumentParser(description='Skeleton (V)AE')
parser.add_argument('--src-dir', type=str, metavar='S', required=True,
                    help='Directory where the NTURGBD raw skeleton files are stored.')
parser.add_argument('--dst-dir', type=str, metavar='D', required=True,
                    help='Directory where the processed skeleton npz files should be stored.')
parser.add_argument('--regex', type=str, metavar='R', default="*A058.skeleton",
                    help='Regex to filter skeletons with (should be compatible with glob) (default: *A058.skeleton (for handhshaking data))')					
args = parser.parse_args()

skeleton_files = []
skeleton_files = glob(args.src_dir + args.regex)
index = 0
skipped = 0
final_skeletons = []
final_angles = []
for fname in skeleton_files[:10]:
	try:
		with open(fname,'r') as f:
			text = map(float, re.sub('\r\n',' ',f.read()).split())
		frame_count = int(text.pop(0))
		if frame_count==0:
			continue
		dist = 0
		skeletons = []
		angles = []
		rotMat = None
		untracked = False
		for frame in range(frame_count):
			body_count = int(text.pop(0))
			if body_count <2:
				untracked = True
				break
			
			body_transforms = np.zeros((body_count*25,3))
			# print "%d bodies present in frame %d"%(body_count, frame)
			for body_num in range(body_count):
				bodyID = int(text.pop(0))
				clipedEdges = int(text.pop(0))
				handLeftConfidence = int(text.pop(0))
				handLeftState = int(text.pop(0))
				handRightConfidence = int(text.pop(0))
				handRightState = int(text.pop(0))
				isResticted = int(text.pop(0))

				leanX = float(text.pop(0))
				leanY = float(text.pop(0))

				bodyTrackingState = int(text.pop(0))
				if not bodyTrackingState:
					untracked = True
					break
				joint_count = int(text.pop(0))
				# print "%d joints found in body %d in frame %d"%(joint_count, body_num, frame)
				for joint_num in range(joint_count):

					x = float(text.pop(0))
					y = float(text.pop(0))
					z = float(text.pop(0))
					depthX = float(text.pop(0))
					depthY = float(text.pop(0))
					colorX = float(text.pop(0))
					colorY = float(text.pop(0))
					orientationW = float(text.pop(0))
					orientationX = float(text.pop(0))
					orientationY = float(text.pop(0))
					orientationZ = float(text.pop(0))
					
							
					jointTrackingState = int(text.pop(0))
					
					if bodyTrackingState and jointTrackingState:
						body_transforms[body_num*25 + joint_num] = [x,y,z]
					else:
						untracked = True
						break
				
			if untracked:
				break

			if rotMat is None:
				rotMat = skeleton_rotation(body_transforms)
			
			for i in range(len(body_transforms)):
				body_transforms[i] = np.dot(rotMat[:3,:3],body_transforms[i]) + rotMat[:3,3]
			
			skeletons.append(body_transforms[UPPERBODY])
			angles.append(skeleton2joints(body_transforms))
			curr_dist = (body_transforms[HANDRIGHT][0] - body_transforms[25+HANDRIGHT][0])**2 + \
						(body_transforms[HANDRIGHT][1] - body_transforms[25+HANDRIGHT][1])**2 + \
						(body_transforms[HANDRIGHT][2] - body_transforms[25+HANDRIGHT][2])**2
			
			if(curr_dist <0.05 or (frame>frame_count/2 and (dist - curr_dist)**2<1e-4)):
				break
			dist = curr_dist

		r_dist = (body_transforms[HANDRIGHT][0] - body_transforms[25+HANDRIGHT][0])**2 + \
						(body_transforms[HANDRIGHT][1] - body_transforms[25+HANDRIGHT][1])**2 + \
						(body_transforms[HANDRIGHT][2] - body_transforms[25+HANDRIGHT][2])**2
		l_dist = (body_transforms[HANDLEFT][0] - body_transforms[25+HANDLEFT][0])**2 + \
						(body_transforms[HANDLEFT][1] - body_transforms[25+HANDLEFT][1])**2 + \
						(body_transforms[HANDLEFT][2] - body_transforms[25+HANDLEFT][2])**2

		if l_dist < r_dist or np.allclose(l_dist, r_dist):
		 	skipped += 1
		 	print 'Non Right HS skipping',fname
		 	continue

		if untracked:
		 	skipped += 1
		 	print 'skipping',fname
		 	continue
		
		final_skeletons.append(skeletons)
		final_angles.append(angles)
		index += 1
		print "Finished %d/%d\tSkipped %d/%d"%(index,len(skeleton_files),skipped,len(skeleton_files))			
		
	except RuntimeWarning:
		skipped += 1
		print 'skipping',fname
		continue

# np.savez_compressed(os.path.join(args.dst_dir,'handreach_data.npz'), skeletons=np.array(final_skeletons), joint_angles=np.array(final_angles))
print(np.shape(final_skeletons[0]))
print(np.shape(final_angles[0]))