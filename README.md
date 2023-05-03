# DDQN Safety Decision-making Model of Autonomous Vehicles based on Object Trajectory Tracking and Prediction



In this study, we presented a safety decision-making model for autonomous vehicles that utilized the DDQN to overcome the limitations of traditional handcrafted decision-making methods. Our model considers objects' behavior tracking and prediction as inputs to determine the appropriate behavior for our vehicle in high-speed driving and collision avoidance scenarios. To capture essential temporal and spatial information, we integrated LSTM and Self-Attention mechanisms into our network architecture, which could make our network interpretable and make decisions like humans. The ablation experiments we conducted showed the effectiveness of our architecture design. The experimental results demonstrated that our proposed model outperformed the reference model and could be robust in complex and challenging environments and scenarios.

<img width="481" alt="demo" src="https://user-images.githubusercontent.com/103908146/235842695-2312e7a1-d4f4-4010-b852-1204be0acfd5.png">


More details are shown in the report.

## Installation
Training and testing environmrnt is based on OpenAI/Gym and highway-env packages.

```pip install gym```

```pip install highway-env```

```git clone https://github.com/Shuchengzhang888/Safety-Decision-Making-of-Autonomous-Vehicle.git```


## Demo Videos

### IDM & MOBIL Reference Model in Straight Highway Environment
https://user-images.githubusercontent.com/103908146/235841830-a0ef13a1-8db3-4563-a59b-07498a2975a1.mp4


### FCN-based Model in Straight Highway Environment
https://user-images.githubusercontent.com/103908146/235840707-fe2f74dd-72ac-43e2-8d22-2f31050cb30d.mp4



### LSTM-Attention-based Model in Straight Highway Environment
https://user-images.githubusercontent.com/103908146/235840762-0ec4d676-b112-4211-bd18-590d9cf54782.mp4



### LSTM-Attention-based Model in Rectangular Highway Environment
https://user-images.githubusercontent.com/103908146/235841008-be8f00c8-8285-448a-87c5-c591a560d9b1.mp4



### LSTM-Attention-based Model in RaceTrack Highway Environment
https://user-images.githubusercontent.com/103908146/235841232-6aca0695-f93c-41ed-b9d7-1b037bf301c0.mp4


## Partial Reference
[1] Chen, Yilun, et al. "Attention-based hierarchical deep reinforcement learning for lane change behaviors in autonomous driving." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2019.
[2] Hoel, Carl-Johan, Krister Wolff, and Leo Laine. "Automated speed and lane change decision making using deep reinforcement learning." 2018 21st International Conference on Intelligent Transportation Systems (ITSC). IEEE, 2018.
[3] Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning with double q-learning.”  Proceedings of the AAAI conference on artificial intelligence. Vol. 30. No. 1. 2016.
[4] M. Treiber, A. Hennecke, and D. Helbing, “Congested Traffic States in Empirical Observations and Microscopic Simulations,” Phys. Rev. E, vol. 62, pp. 1805–1824, 2000.
[5] A. Kesting, M. Treiber, and D. Helbing, “General lane-changing model mobil for car-following models,” Transportation Research Record, vol. 1999, pp. 86–94, 2007.
[6] Admin (2022) Connected vehicle: Features & trends, Telematics Wire. Available at: https://www.telematicswire.net/connected-vehicle-features-trends/ (Accessed: March 21, 2023). 
[7] Leurent, E. (2018). An Environment for Autonomous Driving Decision-Making (Version 1.4) [Computer software]. https://github.com/eleurent/highway-env


