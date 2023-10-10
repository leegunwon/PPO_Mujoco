from Networks.network import Network
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(Network):
    """
    actor-critic 구조 중 actor 네트워크 부분
    입력 받은 state에서 가장 좋다고 생각하는 action space의 mu(평균값)과 std(표준 편차값)을 출력하는 신경망
    """
    def __init__(self, state_dim, hidden_dim, action_dim, activation_function=F.relu):
        super(Actor, self).__init__(state_dim, hidden_dim, action_dim, activation_function=F.relu)
        self.fc = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.activation_function = activation_function

    def forward(self, x):
        return self.activation_function(self.fc(x))

    def pi(self, x):
        mu = 1.0 * torch.tanh(self.fc_mu(x))  # 평균 생성
        std = torch.softplus(self.fc_std(x))  # 표준 편차
        return mu, std

class Critic(Network):
    """
    actor-critic 구조 중 critic 네트워크 부분
    입력 받은 state의 Q-value를 출력하는 신경망
    """
    def __init__(self, state_dim, hidden_dim, activation_function=torch.relu ):
        super(Critic, self).__init__(state_dim, hidden_dim, activation_function=F.relu)
        self.fc = nn.Linear(state_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, 1)
        self.activation_function = activation_function

    def forward(self, x):
        return self.activation_function(self.fc(x))

    def v(self, x):
        value = self.fc_v(x)
        return value
