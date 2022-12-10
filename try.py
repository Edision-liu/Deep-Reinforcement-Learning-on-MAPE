# -- coding:utf-8 --
# From: 刘志浩
# Date: 2020/12/16  21:24

import torch
from gym import envs
print(envs.registry.all())
import numpy as np
import math
something = [1, 2]
b = [2, 5]
c = something
print(c)
c = b       # Python 中 c = something 应该理解为给 something 贴上了一个标签 c。当再赋值给 c 的时候，就好象把 c 这个标签从原来的
            # something 上拿下来，贴到其他对象上，建立新的 reference
print(c)
print(something)

target_radius = np.sqrt(3 * 6 / np.sin(2 * np.pi / 12))
print(target_radius)

reward_old = []
reward = []
data = np.array([[[4], [4]], [[0], [2]]])
print(data)
reward = [[-0.0585, -0.1012], [-0.1139, -0.0577], [-0.1139, -0.0577]]
reward_old = [[-0.08, -0.1012], [-0.1139, -0.0577], [-0.1139, -0.0577]]
print(reward)
weight_0 = np.array([list(map(lambda x, y:1 if x > y else 0, x, y)) for (x,y) in zip(reward, reward_old)])
print(weight_0)
# np.reshape(weight_0, (1, 4))
# print(weight_0)
cc = np.array([[[y] for y in x] for x in weight_0])
print(cc)
sum_agents = 25
dist_thres = 0.1 / np.pi * (math.atan(20 / 35 - 25/35) + np.pi / 2)
print(dist_thres)


Categorical = torch.distributions.Categorical
probs = torch.FloatTensor([[-1,  -1, -1, -1, -1]])
# logits = probs - probs.logsumexp(dim=-1, keepdim=True)
# print(logits)
dist = Categorical(probs)
print(dist)
# Categorical(probs: torch.Size([2, 3]))
index = dist.sample()
print(index.numpy())
# [2 2]

# logits = logits - logits.logsumexp(dim=-1, keepdim=True)

print(1/np.log(0.6))

print(3.95/0.8)
print(0.05/4.9375)
print(np.log(0.83))

p_force_a = None
if p_force_a==None:
    p_force_a = np.array([-1.000520505250, -2])
print(sum(abs(p_force_a)))