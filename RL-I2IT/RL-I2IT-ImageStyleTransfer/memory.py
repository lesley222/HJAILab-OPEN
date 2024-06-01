import numpy as np
import torch

from collections import deque


class ReplayMemory:
    def __init__(self, max_size=300, look_forward_steps=5, gamma=0.9):
        self.buffer = deque(maxlen=max_size)
        self.look_forward_steps = look_forward_steps
        self.gamma = gamma

    def __len__(self):
        return len(self.buffer)

    def store(self, memory):
        self.buffer.append(memory)

    def sample(self, batch_size):
        sample_indices = np.random.choice(range(len(self.buffer)), batch_size, replace=False)

        cc, tt, ss, aa, rr, ss_, dd = [], [], [], [], [], [], []
        max_index = len(self.buffer)
        for i in sample_indices:
            content = self.buffer[i].content  # content
            style = self.buffer[i].style  # target
            state = self.buffer[i].state  # state
            action = self.buffer[i].action  # action
            reward = self.buffer[i].reward  # reward
            next_state = self.buffer[i].next_state  # next_state
            done = self.buffer[i].done  # done

            for step in range(self.look_forward_steps):
                if max_index > i + step:
                    reward += self.gamma * self.buffer[i + step].reward
                else:
                    break

            cc.append(content)
            tt.append(style)
            ss.append(state)
            aa.append(action)
            rr.append(reward)
            ss_.append(next_state)
            dd.append(done)

        cc = np.concatenate(cc, axis=0)
        tt = np.concatenate(tt, axis=0)
        ss = np.concatenate(ss, axis=0)
        aa = np.concatenate(aa, axis=0)
        rr = np.array(rr)
        ss_ = np.concatenate(ss_, axis=0)
        dd = np.array(dd)

        # sample_indices = np.random.choice(range(len(rr)), batch_size, replace=False)

        return cc, tt, ss, aa, rr, ss_, dd
