from parse_params.aparse import parse
from Agents.PPO import PPO
import torch

args, agent_args = parse()
#
# if args.tensorboard:
#     from torch.utils.tensorboard import SummaryWriter
#
#     writer = SummaryWriter()
# else:
#     writer = None

# mujoco 환경변수가 설정이 안되서 패키지 내부에서 직접 모듈을 불러옴
# mujoco의 env 설정 부분
# ---------------------------------------------------------------------
if args.env_name == "HalfCheetah-v4":
    from gym.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
    env = HalfCheetahEnv()

elif args.env_name == "Ant-v4":
    from gym.envs.mujoco.ant_v4 import AntEnv
    env = AntEnv()

elif args.env_name == "Hopper-v4":
    from gym.envs.mujoco.hopper_v4 import HopperEnv
    env = HopperEnv()

elif args.env_name == "Humanoid-v4":
    from gym.envs.mujoco.humanoid_v4 import HumanoidEnv
    env = HumanoidEnv()

elif args.env_name == "InvertedPendulumEnv-v4":
    from gym.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv
    env = InvertedPendulumEnv()

elif args.env_name == "InvertedDoublePendulumEnv-v4":
    from gym.envs.mujoco.inverted_double_pendulum_v4 import InvertedDoublePendulumEnv
    env = InvertedDoublePendulumEnv()
# ---------------------------------------------------------------------

# 설정한 env별 action space shape 저장
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]

if args.algo == 'ppo':
    agent = PPO(state_dim, action_dim, agent_args)
# elif args.algo == 'ddpg':
#     from utils.noise import OUNoise

#     noise = OUNoise(action_dim, 0)  # DDPG의 경우 exploration 방법으로 noise를 사용
#     agent = DDPG(writer, device, state_dim, action_dim, agent_args, noise)

if args.load != 'no':
    agent.load_state_dict(torch.load("./model_weights/" + args.load))

score_lst = []
state_lst = []

# on policy 알고리즘

if agent_args.on_policy == True:
    score = 0.0
    print_interval = 20
    rollout = []
    for n_epi in range(agent_args.n_epi):
        env.render()
        s = env.reset()
        done = False
        count = 0
        while count < 1000 and not done:
            for t in range(agent_args.rollout_len):
                mu, std = agent.get_action(torch.from_numpy(s).float())
                dist = torch.distributions.Normal(mu, std[0])
                a = dist.sample()  # action space에서 action 샘플링
                log_prob = dist.log_prob(a).sum(-1, keepdim=True)

                s_prime, r, done, truncated = env.step(a)
                rollout.append((s, a, r, s_prime, log_prob, done))
                if len(rollout) == agent_args.rollout_len:
                    agent.put_data(rollout)
                    rollout = []
                s = s_prime
                score += r
                count += 1
            agent.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0

    torch.save(agent.state_dict(), './model_weights/agent_'+str(n_epi))
    env.close()





