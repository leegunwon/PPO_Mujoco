
from configparser import ConfigParser
from argparse import ArgumentParser

from utills.utills import Dict


def parse():
    parser = ArgumentParser('parameters')
    parser.add_argument("--env_name", type=str, default = 'Hopper-v4', help = "'Ant-v4','HalfCheetah-v4','Hopper-v4','Humanoid-v4','HumanoidStandup-v4',\
              'InvertedDoublePendulum-v4', 'InvertedPendulum-v4' (default : Hopper-v4)")
    parser.add_argument("--algo", type=str, default = 'ppo', help = 'algorithm to adjust (default : ppo)')
    parser.add_argument('--train', type=bool, default=True, help="(default: True)")
    parser.add_argument('--render', type=bool, default=False, help="(default: False)")
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs, (default: 1000)')
    parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')  # 모델 학습 그래프 등 시각화 해주는 툴
    parser.add_argument("--load", type=str, default = 'no', help = 'load network name in ./model_weights')  # 이전에 학습 시켰던 모델 불러오는 기능
    parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval(default: 100)')  # 일정 간격 별로 모델 저장
    parser.add_argument("--print_interval", type=int, default = 1, help = 'print interval(default : 20)')  # score 중간 중간 찍는 기능
    parser.add_argument("--use_cuda", type=bool, default = True, help = 'cuda usage(default : True)')  # 딥러닝 기본 연산 가속화
    parser.add_argument("--reward_scaling", type=float, default = 0.1, help = 'reward scaling(default : 0.1)')
    args = parser.parse_args()
    parser = ConfigParser()
    parser.read('config.ini')
    agent_args = Dict(parser, args.algo)

    return args,  agent_args

