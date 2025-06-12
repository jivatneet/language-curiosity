# Grounded Question Answering for Curiosity-driven exploration
Code associated with the ICLR 2021 workshop paper [Ask & Explore: Grounded Question Answering for Curiosity-driven exploration](https://arxiv.org/abs/2104.11902).

## Abstract
In many real-world scenarios where extrinsic rewards to the agent are extremely sparse, curiosity has emerged as a useful concept providing intrinsic rewards that enable the agent to explore its environment and acquire information to achieve its goals. Despite their strong performance on many sparse-reward tasks, existing curiosity approaches rely on an overly holistic view of state transitions, and do not allow for a structured understanding of specific aspects of the environment. In this paper, we formulate curiosity based on grounded question answering by encouraging the agent to ask questions about the environment and be curious when the answers to these questions change. We show that natural language questions encourage the agent to uncover specific knowledge about their environment such as the physical properties of objects as well as their spatial relationships with other objects, which serve as valuable curiosity rewards to solve sparse-reward tasks more efficiently.

## Prerequisites (in order)
[Mujoco License](https://www.roboti.us/license.html) (For instructions to set up, refer to readme of DeepMind's [dm\_control](https://github.com/deepmind/dm_control))

[CLEVR-Robot Environment](https://github.com/google-research/clevr_robot_env)

## Installation and Usage
1. This code is based on PyTorch. To set up the repository in your local machine, use these commands
```
git clone https://github.com/jivatneet/language-curiosity.git
cd language-curiosity/
virtualenv curiosity
pip install -r requirements.txt
```
2. For training
 ```
cd ..
python language-curiosity/train_qa.py
 ```
