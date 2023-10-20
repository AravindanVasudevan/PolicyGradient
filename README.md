# Policy Gradient

This repo contains the implementation of policy gradient using PyTorch. The algorithm was tested in the Mujoco environment.

The graph below shows the total reward per episode for the REINFORCE algorithm:

![alt text](https://github.com/AravindanVasudevan/PolicyGradient/blob/main/pics/PolicyGradient_Reinforce.png)

The graph below shows the total reward per episode for the REINFORCE with baseline algorithm:

![alt text](https://github.com/AravindanVasudevan/PolicyGradient/blob/main/pics/PolicyGradient_ReinforceBaseline.png)

Comparing the 2 approcahes - REINFORCE and REINFORCE with Baseline, the latter offers several advantages that make it a more robust and efficient choice for certain scenarios due to:

1. Variance reduction
2. Stability in learning
3. Improved sample efficiency
4. Enhanced convergence speed