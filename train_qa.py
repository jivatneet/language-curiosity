r"""Train policy using curiosity."""
r"""Train with using QA. (Ext + int reward)"""
from agents_ext import *
from config import *
from utils import *
from torch.multiprocessing import Pipe

from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

from video_utils import *

import numpy as np
import copy

from clevr_robot_env import ClevrEnv

import argparse
import iep.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('--program_generator', default=None)
parser.add_argument('--execution_engine', default=None)
parser.add_argument('--baseline_model', default=None)
parser.add_argument('--use_gpu', default=1, type=int)

# For running on a preprocessed dataset
parser.add_argument('--input_question_h5', default='data/val_questions.h5')
parser.add_argument('--input_features_h5', default='data-ssd/val_features.h5')
parser.add_argument('--use_gt_programs', default=0, type=int)

# This will override the vocab stored in the checkpoint;
# we need this to run CLEVR models on human data
parser.add_argument('--vocab_json', default=None)

# For running on a single example
parser.add_argument('--question', default=None)
parser.add_argument('--image', default=None)
parser.add_argument('--cnn_model', default='resnet101')
parser.add_argument('--cnn_model_stage', default=3, type=int)
parser.add_argument('--image_width', default=64, type=int)
parser.add_argument('--image_height', default=64, type=int)

parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_samples', default=None, type=int)
parser.add_argument('--family_split_file', default=None)

parser.add_argument('--sample_argmax', type=int, default=1)
parser.add_argument('--temperature', default=1.0, type=float)

# If this is passed, then save all predictions to this file
parser.add_argument('--output_h5', default=None)

from scripts.run_model import run_single_example


def main(args):
    print({section: dict(config[section]) for section in config.sections()})
    train_method = default_config['TrainMethod']
    env_id = default_config['EnvID']

    env = ClevrEnv()

    # program_generator, _ = utils.load_program_generator(args.program_generator)
    # execution_engine, _ = utils.load_execution_engine(args.execution_engine, verbose=False)
    # model = (program_generator, execution_engine)
    # model = utils.load_baseline(args.baseline_model)

    input_size = env.observation_space.shape  # (64,64,3)
    output_size = env.action_space.n  # 4

    is_load_model = False
    is_render = False
    model_path = './models/cnnlstm02-gt/{}.model'.format(env_id)
    icm_path = './models/cnnlstm02-gt/{}.icm'.format(env_id)

    writer = SummaryWriter()

    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    save_video = default_config.getboolean('SaveVideo')
    video_interval = 500
    save_dir = './videos-epoch3-gt'

    lam = float(default_config['Lambda'])
    num_worker = int(default_config['NumEnv'])

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    eta = float(default_config['ETA'])

    clip_grad_norm = float(default_config['ClipGradNorm'])

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 3, 64, 64)) #???

    pre_obs_norm_step = int(default_config['ObsNormStep'])
    discounted_reward = RewardForwardFilter(gamma)

    agent = ICMAgent

    agent = agent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        eta=eta,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net
    )
    if is_load_model:
        if use_cuda:
            agent.model.load_state_dict(torch.load(model_path))
        else:
            agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))

    #torch.save(agent.model.state_dict(), model_path)
    #torch.save(agent.icm.state_dict(), icm_path)

    # set a fixed goal
    goal_text = 'There is a blue rubber sphere; are there any green rubber spheres to the left of it?'
    goal_program = [{'type': 'scene', 'inputs': []}, {'type': 'filter_color', 'inputs': [0], 'side_inputs': ['blue']}, {'type': 'filter_material', 'inputs': [1], 'side_inputs': ['rubber']}, {'type': 'filter_shape', 'inputs': [2], 'side_inputs': ['sphere']}, {'type': 'exist', 'inputs': [3]}, {'type': 'relate', 'inputs': [3], 'side_inputs': ['left']}, {'type': 'filter_color', 'inputs': [5], 'side_inputs': ['green']}, {'type': 'filter_material', 'inputs': [6], 'side_inputs': ['rubber']}, {'type': 'filter_shape', 'inputs': [7], 'side_inputs': ['sphere']}, {'type': 'exist', 'inputs': [8]}]

    env.set_goal(goal_text, goal_program)

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    sample_i_rall = 0
    global_update = 0
    global_step = 0

    num_iterations = 0

    next_obs = []
    steps = 0

    # introducing language objective
    counter_questions = {}
    total_questions = []
    threshold = 0.9
    max_ques = 256

    while len(counter_questions) < max_ques:
        goal_text, goal_program = env.sample_goal()
        if goal_text not in counter_questions:
            total_questions.append(goal_text)
        counter_questions[goal_text] = [0, goal_program]

    questions_set = total_questions[:num_step]
    total_questions = total_questions[num_step:]
    end_training = 0

    while True:
        
        if global_update%100 == 0:
            print("ENV GOAL: ", env.current_goal_text)
            print("ENV GOAL PROGRAM: ", env.current_goal)

        total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_values, total_log_prob, total_policy, total_int_ext  = \
            [], [], [], [], [], [], [], [], [], [], []
        all_frames = []
        global_step += num_step
        global_update += 1

        states = env.reset()
        while env.answer_question(env.current_goal) == 1:
            states = env.reset()
        states = states.reshape(1, 3, 64, 64)

        eps_reward = 0

        # shuffling question set
        sample_range = np.arange(num_step)
        np.random.shuffle(sample_range)

        num_episodes_pre_update = 0

        while num_episodes_pre_update < 10:

            num_episodes_pre_update += 1
            print('Starting rollout no. {}'.format(num_episodes_pre_update))
            # Step 1. n-step rollout
            for idx in range(num_step):
                num_iterations += 1

                all_frames.append(pad_image(env.render(mode='rgb_array')))

                       
                sample_idx = sample_range[idx]
                question = questions_set[sample_idx]
                program = counter_questions[question][1]
                ans_pre_step = env.answer_question(program)
                
                
                #ans_pre_step = run_single_example(args, model, question, states.reshape(64, 64, 3))
                #print("QUESTION: ", question)
                #print("ANS PRE STEP: ", ans_pre_step)

                actions, value, policy = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var))

                # for parent_conn, action in zip(parent_conns, actions):
                #     parent_conn.send(action)
                #
                # next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
                # for parent_conn in parent_conns:
                #     s, r, d, rd, lr = parent_conn.recv()
                #     next_states.append(s)
                #     rewards.append(r)
                #     dones.append(d)
                #     real_dones.append(rd)
                #     log_rewards.append(lr)

                next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
                episode_over = 0

                for action in actions:
                   # act = 2*np.random.rand(4) - 1
                   # act = env.sample_random_action()
                    s, r, d, info = env.step(action, record_achieved_goal = True)
                    episode_over = r
                    eps_reward += r
                    #achieved_goal_text = env.get_achieved_goals()
                    #print("ACHIEVED: ", achieved_goal_text)
                    s = s.reshape(1, 3, 64, 64)
                    next_states = s
                    rewards.append(r)
                    dones = d

                ans_post_step = env.answer_question(program)
                #ans_post_step = run_single_example(args, model, question, next_states.reshape(64, 64, 3))
                #print("ANS POST STEP: ", ans_post_step)

                # next_states = np.stack(next_states)
                # rewards = np.hstack(rewards)
                # dones = np.hstack(dones)

                # total reward = int reward

                writer.add_scalar('data/reward_per_step', episode_over, num_iterations)
                intrinsic_reward = agent.compute_intrinsic_reward(
                    (states - obs_rms.mean) / np.sqrt(obs_rms.var),
                    (next_states - obs_rms.mean) / np.sqrt(obs_rms.var),
                    actions)
        
                
                if ans_pre_step != ans_post_step:
                    print("Pre and post answer change")
                    intrinsic_reward += 10
                    counter_questions[question][0] += 1
                    if counter_questions[question][0]/(sample_episode+1) > threshold:
                        if len(total_questions) > 0:
                            questions_set[sample_idx] = total_questions[0]
                            total_questions.pop(0)
                        else:
                            end_training = 1
                            break
                

                #print('intrinsic:{}'.format(intrinsic_reward))
                # print('val:{}'.format(value))
                sample_i_rall += intrinsic_reward[sample_env_idx]

                total_int_reward.append(intrinsic_reward)
                total_state.append(states)
                total_next_state.append(next_states)
                total_reward.append(rewards)
                total_done.append(dones)
                total_action.append(actions)
                total_values.append(value)
                #total_log_prob.append(log_prob)
                total_policy.append(policy)

                states = next_states[:, :, :, :]

                sample_rall += rewards[0]
                sample_step += 1
                if sample_step >= num_step or episode_over:
                    sample_episode += 1
                    writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                    writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                    writer.add_scalar('data/step', sample_step, sample_episode)
                    print("Episode: %d Sum of rewards: %.2f. Length: %d." % (sample_episode, sample_rall, sample_step))
                    obs = env.reset()
                    while env.answer_question(env.current_goal) == 1:
                        obs = env.reset()
                    obs = obs.reshape(1, 3, 64, 64)
                    states = obs
                    sample_rall = 0
                    sample_step = 0
                    sample_i_rall = 0
                    # break

                # sample_rall += log_rewards[sample_env_idx]
                #
                # sample_step += 1
                # if real_dones[sample_env_idx]:
                #     sample_episode += 1
                #     writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                #     writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                #     writer.add_scalar('data/step', sample_step, sample_episode)
                #     sample_rall = 0
                #     sample_step = 0
                #     sample_i_rall = 0

            # language model exploited by agent -> end training
        if end_training:
            print('Ending training')
            break

        # calculate last next value
        _, value, _ = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var))
        total_values.append(value)
        # --------------------------------------------------
        # Save video
        if save_video and global_update % video_interval == 0:
            video_dir = os.path.join(save_dir, 'episode_{}.mp4'.format(global_update))
            print('Saving video to {}'.format(video_dir))
            save_video_file(np.uint8(all_frames), video_dir, fps=5)
            print('Video saved...')

        if save_video and eps_reward !=0 and len(total_reward) > 1: #policy improving
            video_dir = os.path.join(save_dir, 'reward_episode_{}.mp4'.format(global_update))
            print('Saving reward episode to {}'.format(video_dir))
            save_video_file(np.uint8(all_frames), video_dir, fps=5)

        # --------------------------------------------------
        total_reward = np.stack(total_reward).transpose()
        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 3, 64, 64])
        total_next_state = np.stack(total_next_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 3, 64, 64])
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose()
        total_values = np.stack(total_values).transpose()
        # total_logging_policy = torch.stack(total_policy).view(-1, output_size).cpu().numpy()

        # Step 2. calculate intrinsic reward
        # running mean intrinsic reward
        total_int_reward = np.stack(total_int_reward).transpose()
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                         total_int_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        # normalize intrinsic reward
        total_int_reward /= np.sqrt(reward_rms.var)
        writer.add_scalar('data/int_reward_per_epi', np.sum(total_int_reward) / num_worker, sample_episode)
        writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)
        # -------------------------------------------------------------------------------------------

        # logging Max action probability
        # writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)

        # Step 3. make target and advantage
        
        target, adv = make_train_data(total_reward + total_int_reward,
                                      np.zeros_like(total_int_reward),
                                      total_values,
                                      gamma,
                                      num_step*num_episodes_pre_update, #num_step,
                                      num_worker)

        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)

        # -----------------------------------------------

        # Step 5. Training!
        agent.train_model((total_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                           (total_next_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                           target, total_action,
                           adv,
                           total_policy)

        '''
        agent.train_model_continuous((total_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                          (total_next_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                          target, total_action,
                          adv,
                          total_log_prob,
                          global_update)
    
        agent.clear_actions()
        '''

        if global_step % (num_worker * num_step * 100) == 0:
            print('Now Global Step :{}'.format(global_step))
            #torch.save(agent.model.state_dict(), model_path)
            #torch.save(agent.icm.state_dict(), icm_path)

    writer.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
