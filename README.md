# Learning Human-like Hand Reaching for Human-Robot Handshaking
This project contains the code for the ICRA 2021 paper "Learning Human-like Hand Reaching for Human-Robot Handshaking". Please cite our paper if you use this work either in part or whole.

## Prerequisites
- [pytorch](https://pytorch.org/get-started/locally/)
- [intprim](https://github.com/souljaboy764/intprim)
- [pepper_moveit_config](https://github.com/ros-naoqi/pepper_moveit_config) or [naoqi](http://doc.aldebaran.com/2-5/dev/python/install_guide.html)

This codebase was developed and tested with ROS melodic on Ubuntu 18.04. 

## Installation
1. Clone this package to your catkin workspace `git clone https://github.com/souljaboy764/icra_handshaking`
2. Go to the `src/` folder and run `chmod a+x *.py`
3. Run `catkin_make` or `catkin build` 

## Data Preprocessing
This codebase assumes that you have downloaded the skeleton data from the [NTU RGB+D Dataset](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp).  
The preprocessing step selects the reaching phases of the right handed handshakes. It saves the upper body skeletons to train an LSTM and the joint angles to train the ProMP model in an npz file `handreach_data.npz`.
```bash
python src/preprocessing.py --src-dir /path/to/dataset --dst-dir /path/to/destination
``` 

The data can be loaded with:
```python
with open('/path/to/handreach_data.npz','rb') as f:
	data = np.load(f,allow_pickle=True, encoding='bytes')
	skeletons = data['skeletons'] # Array of sequences of shape Tx15x3 containing 3D positions of 15 upper body skeleton joints for T timesteps
	joint_angles = data['joint_angles'] # Array of sequences of shape Tx4 containing right arm joint angles (excluding the wrist) for T timesteps
```

## Running
Start the nodes for nuitrack and pepper moveit and have the human interaction partner visible in nuitrack. Then run
```bash
roslaunch icra_handshaking experiment.launch
```

In case the pepper ROS stack is unavailable, run 
```bash
rosrun icra_handshaking pepper_promp_naoqi.py
```

## Citation
```bibtex
@inproceedings{prasad2021learning,
  title={Learning Human-like Hand Reaching for Human-Robot Handshaking},
  author={Prasad, Vignesh and Stock-Homburg, Ruth and Peters, Jan},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year = {2021}
}
```

## TODO
- Add training codes
- Add nuitrack node information
- Update README