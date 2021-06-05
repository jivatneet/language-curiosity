import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim

from torch.distributions.categorical import Categorical

from model import CnnActorCriticNetwork, ICMModel


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

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        policy, value = self.model(state)
        action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()

        action = self.random_choice_prob_index(action_prob)

        return action, value.data.cpu().numpy().squeeze(), policy.detach()

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def compute_intrinsic_reward(self, state, next_state, action):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)

        action_onehot = torch.FloatTensor(
            len(action), self.output_size).to(
            self.device)
        action_onehot.zero_()
        action_onehot.scatter_(1, action.view(len(action), -1), 1)

        real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
            [state, next_state, action_onehot])
        intrinsic_reward = self.eta * F.mse_loss(real_next_state_feature, pred_next_state_feature, reduction='none').mean(-1)
        return intrinsic_reward.data.cpu().numpy()

    def train_model(self, s_batch, next_s_batch, rep_batch, next_rep_batch, target_batch, y_batch, adv_batch, old_policy, action_ques):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        next_s_batch = torch.FloatTensor(next_s_batch).to(self.device)
        rep_batch = torch.FloatTensor(rep_batch).to(self.device)
        next_rep_batch = torch.FloatTensor(next_rep_batch).to(self.device)
        target_batch = torch.FloatTensor(target_batch).to(self.device)
        y_batch = torch.LongTensor(y_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)
        action_ques = torch.LongTensor(action_ques).to(self.device)

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
                mul_factor = int(len(rep_batch)/len(s_batch))
                # print("MUL: ", mul_factor)
                qa_sample_idx = [index * mul_factor for index in sample_idx]
                index_list = []
                for index in qa_sample_idx:
                    for k in range(mul_factor):
                        index_list.append(index + k)
                index = 0
                total_pred_action, total_real_feature, total_pred_feature = [], [], []
                inv_loss, fwd_loss = [], []
                while index < len(index_list):

                    action_onehot = torch.FloatTensor(self.batch_size, self.output_size).to(self.device)
                    action_onehot.zero_()
                    sample_idx_ques = index_list[index: index + self.batch_size]
                    # print("SAMPLE: ", len(sample_idx_ques), len(sample_idx))
                    action_onehot.scatter_(1, action_ques[sample_idx_ques].view(-1, 1), 1)
                    # real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
                    #    [s_batch[sample_idx], next_s_batch[sample_idx], action_onehot])
                    real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
                    [rep_batch[sample_idx_ques], next_rep_batch[sample_idx_ques], action_onehot])
                    inverse_loss = ce(
                        pred_action, action_ques[sample_idx_ques])

                    forward_loss = forward_mse(
                        pred_next_state_feature, real_next_state_feature.detach())
                    inv_loss.append(inverse_loss)
                    fwd_loss.append(forward_loss)

                    index += self.batch_size

                    '''
                    for l in range(len(pred_action)):
                        total_pred_action.append(pred_action[l])
                        total_real_feature.append(real_next_state_feature[l].detach())
                        total_pred_feature.append(pred_next_state_feature[l])

                    index += self.batch_size
                
                total_pred_action = np.stack(total_pred_action)
                print("TOT PRED STAKC: ", total_pred_action.shape)
                total_pred_action = torch.FloatTensor(total_pred_action).to(self.device)
                total_pred_feature = torch.FloatTensor(total_pred_feature).to(self.device)
                total_real_feature = torch.FloatTensor(total_real_feature)

                inverse_loss = ce(
                    total_pred_action, action_ques[index_list])

                forward_loss = forward_mse(
                    total_pred_feature, total_real_feature) #pred_next_state_feature, real_next_state_feature.detach())
                '''

                inverse_loss = sum(inv_loss) / len(inv_loss)
                forward_loss = sum(fwd_loss)/ len(fwd_loss)
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
