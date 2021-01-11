# Language-Curiosity


## Prerequisites (in order)
[Mujoco License](https://www.roboti.us/license.html) (For instructions to set up, refer to readme of DeepMind's [dm\_control](https://github.com/deepmind/dm_control))
[Clevr-robot environment](https://github.com/google-research/clevr_robot_env)

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
python language-curiosity/train.py
 ```