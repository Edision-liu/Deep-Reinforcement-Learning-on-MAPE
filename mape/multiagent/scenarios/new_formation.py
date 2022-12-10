# -- coding:utf-8 --
# From: 刘志浩
# Date: 2021/4/7  10:50
import numpy as np
from mape.multiagent.core import World, Agent, Landmark
from mape.multiagent.scenario import BaseScenario
from scipy.optimize import linear_sum_assignment


def get_thetas(poses):
    # compute angle (0,2pi) from horizontal
    thetas = [None]*len(poses)
    for i in range(len(poses)):
        # (y,x)
        thetas[i] = find_angle(poses[i])
    return thetas


def find_angle(pose):
    # compute angle from horizontal
    angle = np.arctan2(pose[1], pose[0])
    if angle<0:
        angle += 2*np.pi
    return angle


class Scenario(BaseScenario):
    def __init__(self, num_agents=4, dist_threshold=0.1, arena_size=1, identity_size=0):
        self.num_agents = num_agents
        self.ideal_theta_separation = (2*np.pi)/self.num_agents # ideal theta difference between two agents
        # self.target_radius = np.sqrt(3 * agent.size / np.sin(self.ideal_theta_0separation)) # fixing the target radius for now  default 0.5
        self.target_radius = 5 * 0.05 * self.num_agents / (2 * np.pi)
        # target_radius在模型训练的时候有用，模型训练完成之后不会随环境参数设定改变而改变
        self.arena_size = arena_size
        self.dist_thres = 0.06    # 需要与环境大小变化相对应，，，环境1 圆圈0.5时用0.05
        self.theta_thres = 0.1
        self.identity_size = identity_size

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = self.num_agents
        num_landmarks = 1
        world.collaborative = False

        # add agents
        world.agents = [Agent(iden=i) for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05    # 0.05
            agent.adversary = False

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03

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
        for i, landmark in enumerate(world.landmarks):
            # bound on the landmark position less than that of the environment for visualization purposes
            landmark.state.p_pos = np.random.uniform(-.3*self.arena_size, .3*self.arena_size, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        world.steps = 0
        world.dists = []

    def reward(self, agent, world):
        if agent.iden == 0:
            landmark_pose = world.landmarks[0].state.p_pos
            relative_poses = [agent.state.p_pos - landmark_pose for agent in world.agents]
            thetas = get_thetas(relative_poses)
            # anchor at the agent with min theta (closest to the horizontal line)
            theta_min = min(thetas)
            # expected_poses = [landmark_pose + np.array(
            #                   [-0.6,
            #                    (-1)**i*(i-1)*0.1])
            #                   for i in range(self.num_agents/2)]

            y = 0
            expected_poses = []
            # print(self.num_agents//2)
            for i in range(self.num_agents//2):
                # print(i)
                delta = (-1)**i*(i)*0.15
                # print('jjjjj')
                # print(delta)
                y = y + delta
                expected_poses.append(landmark_pose+np.array([-0.6, y]))
                expected_poses.append(landmark_pose+np.array([0.6, y]))
                # np.append(expected_poses, landmark_pose+np.array([-0.6, y]))
                # np.append(expected_poses, landmark_pose+np.array([0.6, y]))
            # print(expected_poses)
            dists = np.array([[np.linalg.norm(a.state.p_pos - pos) for pos in expected_poses] for a in world.agents])
            # optimal 1:1 agent-landmark pairing (bipartite matching algorithm)
            self.delta_dists = self._bipartite_min_dists(dists)
            world.dists = self.delta_dists

            total_penalty = np.mean(np.clip(self.delta_dists, 0, 2))
            self.joint_reward = -total_penalty

        return self.joint_reward

    def _bipartite_min_dists(self, dists):
        ri, ci = linear_sum_assignment(dists)
        min_dists = dists[ri, ci]
        return min_dists

    def observation(self, agent, world):
        # positions of all entities in this agent's reference frame
        entity_pos = [entity.state.p_pos - agent.state.p_pos for entity in world.landmarks]
        default_obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos)
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