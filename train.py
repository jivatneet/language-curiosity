r"""Train policy using curiosity."""
from agents import *
from config import *
from utils import *
from torch.multiprocessing import Pipe

from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

import numpy as np
import copy

from clevr_robot_env import ClevrEnv

def main():
    print({section: dict(config[section]) for section in config.sections()})
    train_method = default_config['TrainMethod']
    env_id = default_config['EnvID']

    env = ClevrEnv()

    input_size = env.observation_space.shape  # (64,64,3)
    output_size = env.action_space.shape[0]  # 4

    is_load_model = False
    is_render = False
    model_path = '/home/paul/jivat/language-curiosity/models/{}.model'.format(env_id)
    icm_path = '/home/paul/jivat/language-curiosity/models/{}.icm'.format(env_id)

    writer = SummaryWriter()

    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

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

    torch.save(agent.model.state_dict(), model_path)
    torch.save(agent.icm.state_dict(), icm_path)

    states = np.zeros([1, 3, 64, 64])
    obs = env.reset()
    obs = obs.reshape(1, 3, 64, 64)
    states = obs

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    sample_i_rall = 0
    global_update = 0
    global_step = 0

    next_obs = []
    steps = 0

    print(env.current_goal_text)
    while True:
        total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_values, total_log_prob = \
            [], [], [], [], [], [], [], [], []
        global_step += num_step
        global_update += 1

        # Step 1. n-step rollout
        for _ in range(num_step):
            actions, value, log_prob = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var))

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
            for action in actions:
                s, r, d, info = env.step(action)
                s = s.reshape(1, 3, 64, 64)
                next_states = s
                rewards = r
                dones = d

            # next_states = np.stack(next_states)
            # rewards = np.hstack(rewards)
            # dones = np.hstack(dones)

            # total reward = int reward
            intrinsic_reward = agent.compute_intrinsic_reward(
                (states - obs_rms.mean) / np.sqrt(obs_rms.var),
                (next_states - obs_rms.mean) / np.sqrt(obs_rms.var),
                actions)

            # print('intrinsic:{}'.format(intrinsic_reward))
            # print('val:{}'.format(value))
            sample_i_rall += intrinsic_reward[sample_env_idx]

            total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_next_state.append(next_states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_values.append(value)
            total_log_prob.append(log_prob)
            # total_policy.append(policy)

            states = next_states[:, :, :, :]

            sample_rall += rewards
            sample_step += 1
            if sample_step >= num_step:
                sample_episode += 1
                writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                print("Episode: %d Sum of rewards: %.2f. Length: %d." % (sample_episode, sample_rall, sample_step))
                obs = obs.reshape(1, 3, 64, 64)
                sample_rall = 0
                sample_step = 0
                break


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

        # calculate last next value
        _, value, _ = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var))
        total_values.append(value)
        # --------------------------------------------------

        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 3, 64, 64])
        total_next_state = np.stack(total_next_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 3, 64, 64])
        total_action = np.stack(total_action).transpose().reshape([-1, output_size])
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
        target, adv = make_train_data(total_int_reward,
                                      np.zeros_like(total_int_reward),
                                      total_values,
                                      gamma,
                                      num_step,
                                      num_worker)

        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
        # -----------------------------------------------

        # Step 5. Training!
        # agent.train_model((total_state - obs_rms.mean) / np.sqrt(obs_rms.var),
        #                   (total_next_state - obs_rms.mean) / np.sqrt(obs_rms.var),
        #                   target, total_action,
        #                   adv,
        #                   total_policy)

        agent.train_model_continuous((total_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                          (total_next_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                          target, total_action,
                          adv,
                          total_log_prob)

        agent.clear_actions()

        if global_step % (num_worker * num_step * 100) == 0:
            print('Now Global Step :{}'.format(global_step))
            torch.save(agent.model.state_dict(), model_path)
            torch.save(agent.icm.state_dict(), icm_path)

    writer.close()


if __name__ == '__main__':
    main()
