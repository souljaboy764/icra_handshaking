import intprim
from intprim.probabilistic_movement_primitives import ProMP
import numpy as np

parser = argparse.ArgumentParser(description='Training ProMP from extracted joints')
parser.add_argument('--src', type=str, metavar='S', required=True,
                    help='Path to the processed npz file after running preprocessing.py.')
parser.add_argument('--num-samples', type=int, metavar='N', default=20,
                    help='Number of samples to train the ProMP with.')
parser.add_argument('--basis-degree', type=int, metavar='D', default=4,
                    help='Degree of the ProMP basis functions.')
parser.add_argument('--basis-scale', type=float, metavar='S', default=0.001,
                    help='Scale of the ProMP basis functions.')
args = parser.parse_args()

with open(args.src,'r') as f:
	data = np.load(f,allow_pickle=True)
	Q = data['joint_angles']
num_joints = np.shape(Q[0])[1]
for i in range(num_samples):
	q_diff = np.diff(Q[i],axis=0).T
	avg_vel = np.linalg.norm(q_diff,axis=0)
	first = np.where(avg_vel>0.002)[0][0]
	Q[i] = Q[i][first:]
	
basis_model = intprim.basis.GaussianModel(args.basis_degree, args.basis_scale, ['joint'+str(i) for i in range(num_joints)])
promp = ProMP(basis_model)

for q in np.random.choice(Q,args.num_samples):
	promp.add_demonstration(np.array(q).T)

promp.export_data(os.path.join(os.path.dirname(args.src), 'skeleton_promp.pkl'))