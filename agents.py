import numpy as np
import math
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable

from torch.distributions.categorical import Categorical

from model import CnnActorCriticNetwork, ICMModel

from agent_utils import normal

class ICMAgent(object):
    def __init__(
            self,
            input_size,
            output_size,
            num_env,
            num_step,
            gamma,
            lam=0.95,
            learning_rate=1e-4,
            ent_coef=0.01,
            clip_grad_norm=0.5,
            epoch=3,
            batch_size=128,
            ppo_eps=0.1,
            eta=0.01,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False):
        self.model = CnnActorCriticNetwork(input_size, output_size, use_noisy_net)
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.epoch = epoch
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.ent_coef = ent_coef
        self.eta = eta
        self.ppo_eps = ppo_eps
        self.clip_grad_norm = clip_grad_norm
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.icm = ICMModel(input_size, output_size, use_cuda)
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.icm.parameters()),
                                    lr=learning_rate)
        self.icm = self.icm.to(self.device)

        self.model = self.model.to(self.device)

        # adding from https://github.com/dgriff777/a3c_continuous/blob/master/player_util.py
        self.log_probs = []
        self.entropies = []

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        # policy, value = self.model(state)
        value, mu, sigma = self.model(state)

        # continuous action space
        mu = torch.clamp(mu, -1.0, 1.0)
        sigma = F.softplus(sigma) + 1e-5
        eps = torch.randn(mu.size())
        pi = np.array([math.pi])
        pi = torch.from_numpy(pi).float()

        eps = torch.Tensor(eps).to(self.device)
        pi = torch.Tensor(pi).to(self.device)
        pi = pi.float()

        action = (mu + sigma.sqrt() * eps).detach()
        act = Variable(action) # find alternate command
        prob = normal(act, mu, sigma, self.device)
        action = torch.clamp(action, -1.0, 1.1)
        entropy = 0.5 * ((sigma * 2 * pi.expand_as(sigma)).log() + 1)
        self.entropies.append(entropy)
        log_prob = (prob + 1e-6).log()
        log_prob = log_prob.sum()
        # self.log_probs.append(log_prob)

        # discrete action space
        # action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()

        # action = self.random_choice_prob_index(action_prob)

        return action.cpu().numpy(), value.data.cpu().numpy().squeeze(), log_prob
        # return action, value.data.cpu().numpy().squeeze(), policy.detach()

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def compute_intrinsic_reward(self, state, next_state, action):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        # action = torch.LongTensor(action).to(self.device)
        action = torch.FloatTensor(action).to(self.device)

        # action_onehot = torch.FloatTensor(
        #     len(action), self.output_size).to(
        #     self.device)
        # action_onehot.zero_()
        # action_onehot.scatter_(1, action.view(len(action), -1), 1)

        # real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
        #     [state, next_state, action_onehot])
        real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
            [state, next_state, action])
        intrinsic_reward = self.eta * F.mse_loss(real_next_state_feature, pred_next_state_feature, reduction='none').mean(-1)
        return intrinsic_reward.data.cpu().numpy()

    def train_model(self, s_batch, next_s_batch, target_batch, y_batch, adv_batch, old_policy):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        next_s_batch = torch.FloatTensor(next_s_batch).to(self.device)
        target_batch = torch.FloatTensor(target_batch).to(self.device)
        y_batch = torch.LongTensor(y_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)

        sample_range = np.arange(len(s_batch))
        ce = nn.CrossEntropyLoss()
        forward_mse = nn.MSELoss()

        with torch.no_grad():
            policy_old_list = torch.stack(old_policy).permute(1, 0, 2).contiguous().view(-1, self.output_size).to(
                self.device)

            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)
            # ------------------------------------------------------------

        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                # --------------------------------------------------------------------------------
                # for Curiosity-driven
                action_onehot = torch.FloatTensor(self.batch_size, self.output_size).to(self.device)

                action_onehot.zero_()
                action_onehot.scatter_(1, y_batch[sample_idx].view(-1, 1), 1)
                real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
                    [s_batch[sample_idx], next_s_batch[sample_idx], action_onehot])

                inverse_loss = ce(
                    pred_action, y_batch[sample_idx])

                forward_loss = forward_mse(
                    pred_next_state_feature, real_next_state_feature.detach())
                # ---------------------------------------------------------------------------------

                policy, value = self.model(s_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])

                ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(
                    value.sum(1), target_batch[sample_idx])

                entropy = m.entropy().mean()

                self.optimizer.zero_grad()
                loss = (actor_loss + 0.5 * critic_loss - 0.001 * entropy) + forward_loss + inverse_loss
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

    def train_model_continuous(self, s_batch, next_s_batch, target_batch, y_batch, adv_batch, log_prob_old):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        next_s_batch = torch.FloatTensor(next_s_batch).to(self.device)
        target_batch = torch.FloatTensor(target_batch).to(self.device)
        y_batch = torch.FloatTensor(y_batch).to(self.device)    # action is float in our case (not one-hot vector)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)

        # print('STATE')
        # print(s_batch)
        # print('\n\nNEXT STATE')
        # print(next_s_batch)
        # print('\n\nTARGET')
        # print(target_batch)
        # print('\n\nACTION')
        # print(y_batch)
        # print('\n\nADV')
        # print(adv_batch)

        sample_range = np.arange(len(s_batch))
        ce = nn.CrossEntropyLoss()
        forward_mse = nn.MSELoss()

        with torch.no_grad():
            log_prob_old = torch.stack(log_prob_old, dim = 0).to(self.device)
            # ------------------------------------------------------------

        x = []
        y = []

        for i in range(self.epoch):
            x.append(i)
            loss_sum = 0
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]
                # print('sample range: ', sample_range)
                # print('s batch len: ', len(s_batch))
                # print('s batch shape: ', s_batch.shape)
                # print('batch size ', self.batch_size)
                # print('sample index: ', sample_idx)
                # --------------------------------------------------------------------------------
                # for Curiosity-driven
                action_input = torch.FloatTensor(self.batch_size, self.output_size).to(self.device)
                action_input = y_batch[sample_idx]

                # print(action_input)
                real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
                    [s_batch[sample_idx], next_s_batch[sample_idx], action_input])

                inverse_loss = forward_mse(
                    pred_action, y_batch[sample_idx])

                forward_loss = forward_mse(
                    pred_next_state_feature, real_next_state_feature.detach())
                # ---------------------------------------------------------------------------------

                value, mu, sigma = self.model(s_batch[sample_idx])

                mu = torch.clamp(mu, -1.0, 1.0)
                sigma = F.softplus(sigma) + 1e-5
                pi = np.array([math.pi])
                pi = torch.from_numpy(pi).float()
                pi = torch.Tensor(pi).to(self.device)
                pi = pi.float()

                # act = Variable(action)  # find alternate command
                # prob = normal(act, mu, sigma, self.device)
                # action = torch.clamp(action, -1.0, 1.1)
                # entropy = 0.5 * ((sigma * 2 * pi.expand_as(sigma)).log() + 1)
                # self.entropies.append(entropy)
                # log_prob = (prob + 1e-6).log()

                log_probs = []
                entropies = []
                action_sample = y_batch[sample_idx]
                for index in range(len(sample_idx)):
                    prob = normal(action_sample[index], mu[index], sigma[index], self.device)
                    log_prob = (prob + 1e-6).log()
                    log_probs.append(log_prob.sum())
                    entropy = 0.5 * ((sigma[index] * 2 * pi.expand_as(sigma[index])).log() + 1)
                    entropies.append(entropy.mean())

                log_probs = torch.stack(log_probs, dim = 0) #.transpose().reshape([-1, self.output_size])
                entropies = torch.stack(entropies, dim = 0)

                # print('LOG PROBBBBB: ', log_probs)
                # print('\n\nLOG PROBBBBB OLLLLD: ', log_prob_old)
                # print('\n\nLOG PROBBBBB sh: ', log_probs.shape)
                # print('\n\nLOG PROBBBBB OLLLL sh: ', log_prob_old.shape)

                ratio = torch.exp(log_probs - log_prob_old[sample_idx])

                # print(log_probs.shape)
                # print(adv_batch[sample_idx])
                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(
                    value.sum(1), target_batch[sample_idx])

                entropy = entropies.mean()

                # print('ENTROPYYY: ', entropy)
                # print('\nACTOR LOSS: ', actor_loss)
                # print('\nCRITIC LOSS: ', critic_loss)
                # print('\nFWD LOSS: ', forward_loss)
                # print('\nINV LOSS: ', inverse_loss)

                self.optimizer.zero_grad()
                loss = (actor_loss + 0.5 * critic_loss - 0.001 * entropy) + forward_loss + inverse_loss
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                loss_sum += loss

            # print("epoch: {} loss:{}".format(i, loss_sum))
            y.append(loss_sum)
        plt.plot(x,y)


    def clear_actions(self):
        self.log_probs = []
        self.entropies = []
        return self
