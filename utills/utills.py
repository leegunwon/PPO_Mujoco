import numpy as np
import torch

class Dict(dict):
    def __init__(self,config,section_name,location = False):
        super(Dict,self).__init__()
        self.initialize(config, section_name,location)
    def initialize(self, config, section_name,location):
        for key,value in config.items(section_name):
            if location :
                self[key] = value
            else:
                self[key] = eval(value)
    def __getattr__(self,val):
        return self[val]

def make_batch(data, args, action_dim, state_dim):
    buffer_size = args.buffer_size
    minibatch_size = args.minibatch_size
    rollout_len = args.rollout_len
    data_t = []
    for j in range(buffer_size):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = np.array([]), np.array([]), np.array(
            []), np.array([]), np.array([]), np.array([])
        for i in range(minibatch_size):
            rollout = data.pop()
            s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = np.array([]), np.array([]), np.array([]), np.array(
                []), np.array([]), np.array([])

            for transition in rollout:
                s, a, r, s_prime, prob_a, done = transition

                s_lst = np.append(s_lst, s)
                a_lst = np.append(a_lst, a)
                r_lst = np.append(r_lst, r)
                s_prime_lst = np.append(s_prime_lst, s_prime)
                prob_a_lst = np.append(prob_a_lst, prob_a)
                done_mask = 0 if done else 1
                done_lst = np.append(done_lst, [done_mask])

            s_batch = np.append(s_batch, s_lst)
            a_batch = np.append(a_batch, a_lst)
            r_batch = np.append(r_batch, r_lst)
            s_prime_batch = np.append(s_prime_batch, s_prime_lst)
            prob_a_batch = np.append(prob_a_batch, prob_a_lst)
            done_batch = np.append(done_batch, done_lst)

        s_batch = s_batch.reshape(minibatch_size, rollout_len, action_dim)
        s_prime_batch = s_prime_batch.reshape(minibatch_size, rollout_len, action_dim)
        a_batch = a_batch.reshape(minibatch_size, rollout_len, state_dim)
        r_batch = r_batch.reshape(minibatch_size, rollout_len, 1)
        prob_a_batch = prob_a_batch.reshape(minibatch_size, rollout_len, 1)
        done_batch = done_batch.reshape(minibatch_size, rollout_len, 1)

        mini_batch = torch.from_numpy(s_batch).float(), torch.from_numpy(a_batch).float(), \
            torch.from_numpy(r_batch).float(), torch.from_numpy(s_prime_batch).float(), \
            torch.from_numpy(done_batch).float(), torch.from_numpy(prob_a_batch).float()

        data_t.append(mini_batch)

    return data_t