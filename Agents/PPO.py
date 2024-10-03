from Networks.actor_critic import Actor, Critic
from utills.utills import make_batch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, args):
        super(PPO, self).__init__()  # 부모 클래스 초기화 추가
        self.data = []
        self.args = args
        self.actor = Actor(state_dim, self.args.hidden_dim, action_dim, self.args.activation_function)
        self.critic = Critic(state_dim, self.args.hidden_dim, action_dim)

        self.optimizer_a = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.optimizer_c = optim.Adam(self.critic.parameters(), lr=self.args.critic_lr)

    def get_action(self, x):
        x = self.actor.forward(x)
        mu, std = self.actor.pi(x)
        return mu, std

    def get_value(self, x):
        x = self.critic.forward(x)
        value = self.critic.v(x)
        return value

    def put_data(self, transition):
        self.data.append(transition)
    def make_batch(self):
        # self.data에서 배치를 무작위로 샘플링
        # self.data는 경험 튜플 (s, a, r, s_prime, done_mask, old_log_prob)로 구성되어야 함
        batch = random.sample(self.data, self.args.batch_size)

        # 배치를 unpacking하여 각각의 텐서로 변환
        s, a, r, s_prime, old_log_prob, done_mask = zip(*batch)

        # 각 텐서를 PyTorch 텐서로 변환
        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.float32)
        r = torch.tensor(r, dtype=torch.float32)
        s_prime = torch.tensor(s_prime, dtype=torch.float32)
        done_mask = torch.tensor(done_mask, dtype=torch.float32)
        old_log_prob = torch.tensor(old_log_prob, dtype=torch.float32)

        return s, a, r, s_prime, old_log_prob, done_mask
    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                # neural network가 생각하는 s_prime의 value에 대한 예측값에 현재 얻은 reward를 더해줌
                td_target = r + self.args.gamma * self.critic.v(s_prime) * done_mask
                # neural network가 생각하는 s에서 s_prime으로 갈 때의 advantage
                delta = td_target - self.critic.v(s)
            delta = delta.numpy()  # tensor를 numpy로 바꿔줌

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.args.gamma * self.args.lamda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))

        return data_with_adv

    def train_net(self):
        if len(self.data) == self.args.batch_size * self.args.buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)
            for i in range(self.args.train_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch

                    mu, std = self.pi(s)
                    a, log_prob = self.sample_action(mu, std)

                    ratio = torch.exp(torch.from_numpy(log_prob).float() - old_log_prob)  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantage
                    actor_loss = -torch.min(surr1, surr2)
                    critic_loss = F.smooth_l1_loss(self.v(s), td_target)

                    self.optimizer_a.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    self.optimizer_a.step()

                    self.optimizer_c.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(),1.0)
                    self.optimizer_c.step()

