# LeNo
The official implementation of AAAI 2023 "LeNo: Adversarial Robust Salient Object Detection Networks with Learnable Noise" 
## Environment
Our code is tested on python3.6, pytorch1.9.0 and cuda11.6
## Train
1.Set the path of training sets in config.py
2.Run train_ours.py to get the results of the first phase
3.Comment out the training code of phase 1 and replace it with phase 2. Specifically, add the code block starting at line 89 and modify the code block starting at line 108. One of the two train functions should be commented out as appropriate. In the model_ours.py, output of model should also be replaced.
4.Load the weight of phase1 and run train_ours.py again.
## Test
1.Set the path of testing sets in config.py
2.Run test.py.
## Citation
## Acknowledgement
We implement this project based on the code of 'Learn2perturb: an end-to-end feature perturbation learning to improve adversarial robustness', proposed by Jeddi, A.; Shafiee, M. J.; Karg, M.; Scharfenberger, C.; and Wong, A. in CVPR. 
