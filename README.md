# Language-Curiosity
Code associated with the ICLR 2021 workshop paper [Ask & Explore: Grounded Question Answering for Curiosity-driven exploration](https://arxiv.org/abs/2104.11902).

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
