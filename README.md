# Deep Q Network Safety Decision-making Model of Autonomous Vehicle Based on Trajectory Prediction
Shucheng Zhang

To overcome the shortcomings of handcrafted decision methods in field of autonomous vehicle, we developed a Safety Decision-making Model based on Deep
Q Network (DQN) to complete safe and high-speed driving in the highway environment. The experiments showed that our model greatly increased average
speed while keeping vehicles safe. Moreover, we added the predicted trajectories of surrounding vehicles into the original input and proved their importance in improving the risk forecast ability.

More details are shown in the report.

## Install
Training and testing environmrnt is based on OpenAI/Gym and highway-env packages.

```pip install gym```

```pip install highway-env```

RL models are based on Baseline3.

```pip install baselines```

After that, replace the all related files in highway-env with my implements, including kinematics.py, obeservation.py and my_highway_env.py.

## Code Description
code/data -> all experiments results

code/_*input_lowhighspeedreward_v* -> model

code/DQN_training.py -> train & test DQN

code/my_highway_env.py -> toy exvironment

code/observation.py -> model input with prediction information

code/kenimatics.py -> predict trajectory and controll

code/env_test.py & model_test.py -> test model & environment

code/result_record.py -> record experiment results


## Demo

https://user-images.githubusercontent.com/103908146/235839735-7130ee63-de80-40c5-8d2d-f91e2ab4f147.mp4













demo -> demo video

The Demo shows the results of reference model, RL model, RL model with t stage prediction input, RL model with 2t stage prediction input and RL model with 3t stage prediction input.

## Reference
[1]	J. Garcıa and F. Fernandez, “A Comprehensive Survey on Safe Reinforcement Learning,” p. 44.

[2]	E. Yurtsever, J. Lambert, A. Carballo, and K. Takeda, “A Survey of Autonomous Driving: Common Practices and Emerging Technologies,” IEEE Access, vol. 8, pp. 58443–58469, 2020, doi: 10.1109/ACCESS.2020.2983149.

[3]	C.-J. Hoel, K. Wolff, and L. Laine, “Automated Speed and Lane Change Decision Making using Deep Reinforcement Learning,” in 2018 21st International Conference on Intelligent Transportation Systems (ITSC), Nov. 2018, pp. 2148–2155. doi: 10.1109/ITSC.2018.8569568.

[4]	S. Nageshrao, E. Tseng, and D. Filev, “Autonomous Highway Driving using Deep Reinforcement Learning.” arXiv, Mar. 29, 2019. Accessed: Oct. 21, 2022. [Online]. Available: http://arxiv.org/abs/1904.00035

[5]	X. Xiong, J. Wang, F. Zhang, and K. Li, “Combining Deep Reinforcement Learning and Safety Based Control for Autonomous Driving.” arXiv, Dec. 01, 2016. Accessed: Oct. 19, 2022. [Online]. Available: http://arxiv.org/abs/1612.00147

[6]	A. Baheri, S. Nageshrao, H. E. Tseng, I. Kolmanovsky, A. Girard, and D. Filev, “Deep Reinforcement Learning with Enhanced Safety for Autonomous Highway Driving,” in 2020 IEEE Intelligent Vehicles Symposium (IV), Oct. 2020, pp. 1550–1555. doi: 10.1109/IV47402.2020.9304744.

[7]	J. Chen, B. Yuan, and M. Tomizuka, “Model-free Deep Reinforcement Learning for Urban Autonomous Driving.” arXiv, Oct. 21, 2019. Accessed: Oct. 21, 2022. [Online]. Available: http://arxiv.org/abs/1904.09503

[8]	B.-Q. Huang, G.-Y. Cao, and M. Guo, “Reinforcement Learning Neural Network to the Problem of Autonomous Mobile Robot Obstacle Avoidance,” in 2005 International Conference on Machine Learning and Cybernetics, Aug. 2005, vol. 1, pp. 85–89. doi: 10.1109/ICMLC.2005.1526924.

[9]	A. Baheri, “Safe Reinforcement Learning with Mixture Density Network: A Case Study in Autonomous Highway Driving.” arXiv, Nov. 17, 2020. Accessed: Oct. 21, 2022. [Online]. Available: http://arxiv.org/abs/2007.01698

[10]  Shai Shalev-Shwartz, Shaked Shammah, and Amnon Shashua. On a formal model of safe and scalable self-driving cars. CoRR, abs/1708.06374, 2017.

[11]  https://github.com/Stable-Baselines-Team/stable-baselines

[12]  https://github.com/eleurent/highway-env

[13]  https://github.com/openai/gym
