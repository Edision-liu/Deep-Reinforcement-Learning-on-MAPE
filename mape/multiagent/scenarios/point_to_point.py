# -- coding:utf-8 --
# From: 刘志浩
# Date: 2021/3/11  10:36
# 本环境适用于单个智能体和单个目标点进行点到点训练的场景，训练出的模型希望能够实现点到点的强化学习，若用此文件，在learner.py文件中需要定义此环境下entities数目

import numpy as np
from mape.multiagent.core import World, Agent, Landmark
from mape.multiagent.scenario import BaseScenario
from scipy.optimize import linear_sum_assignment


def get_thetas(poses):
    # compute angle (0,2pi) from horizontal
    thetas = [None] * len(poses)
    for i in range(len(poses)):
        # (y,x)
        thetas[i] = find_angle(poses[i])
    return thetas


def find_angle(pose):
    # compute angle from horizontal
    angle = np.arctan2(pose[1], pose[0])
    if angle < 0:
        angle += 2 * np.pi
    return angle


class Scenario(BaseScenario):
    def __init__(self, num_agents=1, dist_threshold=0.1, arena_size=1, identity_size=0):
        self.num_agents = num_agents
        # self.target_radius = 6/np.pi  # fixing the target radius for now  default 0.5
        # target_radius在模型训练的时候有用，模型训练完成之后不会随环境参数设定改变而改变
        # self.ideal_theta_separation = (2 * np.pi) / self.num_agents  # ideal theta difference between two agents
        self.arena_size = arena_size
        self.dist_thres = 0.05
        # self.theta_thres = 0.1
        self.identity_size = identity_size
        #            dsafdfadsfadsfdasfadsfADSFadsfa'dDASf
        # self.delta_dists_old = np.zeros(self.num_agents)

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = self.num_agents
        num_landmarks = num_agents
        world.collaborative = False

        # add agents
        world.agents = [Agent(iden=i) for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.03  # 0.05
            agent.adversary = False

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.01

        # make initial conditions
        self.reset_world(world)
        world.dists = []
        return world

    def reset_world(self, world):
        # random properties for agents
        # colors = [np.array([0,0,0.1]), np.array([0,1,0]), np.array([0,0,1]), np.array([1,1,0]), np.array([1,0,0])]
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
            # agent.color = colors[i]

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-self.arena_size, self.arena_size, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        world.ss = np.array([[-1, -0.2], [-1, -0.4], [-1, -0.6], [-1, -0.8],
                             [-1, 0.2], [-1, 0.4], [-1, 0.6], [-1, 0.8],
                             [1, -0.2], [1, -0.4], [1, -0.6], [1, -0.8],
                             [1, 0.2], [1, 0.4], [1, 0.6], [1, 0.8],
                             [-0.8, -0.8], [-0.6, -0.8], [-0.4, -0.8], [-0.2, -0.8],
                             [0.8, -0.8], [0.6, -0.8], [0.4, -0.8], [0.2, -0.8],
                             [-0.8, 0.8], [-0.6, 0.8], [-0.4, 0.8], [-0.2, 0.8],
                             [0.8, 0.8], [0.6, 0.8], [0.4, 0.8], [0.2, 0.8]])
        for i, landmark in enumerate(world.landmarks):
            # bound on the landmark position less than that of the environment for visualization purposes
            # landmark.state.p_pos = np.random.uniform(-.8 * self.arena_size, .8 * self.arena_size, world.dim_p)
            landmark.state.p_pos = world.ss[i]
            landmark.state.p_vel = np.zeros(world.dim_p)

        world.steps = 0
        world.dists = []

    def reward(self, agent, world):
        # if agent.iden == 0:
        landmark_pose = world.landmarks[agent.iden].state.p_pos
        relative_poses = [agent.state.p_pos - landmark_pose for agent in world.agents]
        # thetas = get_thetas(relative_poses)
        # anchor at the agent with min theta (closest to the horizontal line)
        # theta_min = min(thetas)
        expected_poses = [landmark_pose]

        dists = np.array([[np.linalg.norm(agent.state.p_pos - pos) for pos in expected_poses] ])
        # print(dists)
        # print(np.min(dists))
        # optimal 1:1 agent-landmark pairing (bipartite matching algorithm)
        self.delta_dists = dists
        # print(self.delta_dists)
        # 计算全部智能体到目标点的距离
        all_dists = np.array([[np.linalg.norm(a.state.p_pos - l.state.p_pos) for l in world.landmarks] for a in world.agents])
        world.dists = self._bipartite_min_dists(all_dists)
        # print(world.dists)

        total_penalty = np.sum(self.delta_dists)

        self.joint_reward = -total_penalty

        return self.joint_reward

    def _bipartite_min_dists(self, dists):
        ri, ci = linear_sum_assignment(dists)
        min_dists = dists[ri, ci]
        return min_dists

    def observation(self, agent, world):

        all_dists = np.array([[np.linalg.norm(a.state.p_pos - l.state.p_pos) for l in world.landmarks]
                                for a in world.agents])
        ccc = np.array([[(a.state.p_pos - l.state.p_pos) for l in world.landmarks]
                        for a in world.agents])
        ri, ci = linear_sum_assignment(all_dists)
        # print(ri)
        # positions of all entities in this agent's reference frame
        entity_pos = [entity.state.p_pos - agent.state.p_pos for entity in world.landmarks]
        # print(entity_pos)
        # entity_pos2 = entity_pos[0]
        # print(entity_pos[ci[agent.iden]])
        # print(ccc[agent.iden][ci[agent.iden]])
        default_obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [entity_pos[ci[agent.iden]]])
        # default_obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [ccc[agent.iden][ci[agent.iden]]])
        if self.identity_size != 0:
            identified_obs = np.append(np.eye(self.identity_size)[agent.iden], default_obs)
            return identified_obs
        return default_obs

    def done(self, agent, world):
        condition1 = world.steps >= world.max_steps_episode
        self.is_success = np.all(self.delta_dists < self.dist_thres)
        return condition1 or self.is_success

    def info(self, agent, world):
        return {'is_success': self.is_success, 'world_steps': world.steps,
                'reward': self.joint_reward, 'dists': self.delta_dists.mean()}
