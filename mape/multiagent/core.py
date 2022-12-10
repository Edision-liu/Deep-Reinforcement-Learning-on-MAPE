import numpy as np

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.05
        # entity can move / be pushed
        self.movable = True
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()
        self.size = 0.05

# properties of obstacle entities
class Obstacle(Entity):
    def __init__(self):
        super(Obstacle, self).__init__()
        self.size = 0.05

# properties of agent entities
class Agent(Entity):
    def __init__(self, iden=None):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        if iden is not None:
            self.iden = iden

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        ########################
        self.obstacles = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 16     # 1e+2
        self.contact_margin = 0.06    # 1e-2  应该是两个小球间最近的距离
        # number of steps that have been taken
        self.steps = 0
        self.max_steps_episode = 70   # default=50

    # return all entities in the world
    @property
    def entities(self):
        ###########################
        return self.agents + self.landmarks + self.obstacles

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # print(p_force)
        # print(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)
        self.steps += 1

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise
                # print(p_force[i])
        p_force[i+1] = np.array([1.2, 0])
        p_force[i+2] = np.array([1.2, 0])
        p_force[i+3] = np.array([1.2, 0])
        # p_force[i+4] = np.array([1, 0])
        # p_force[i+5] = np.array([1, 0])
        # p_force[i+6] = np.array([1, 0])
        # p_force[i+4] = np.array([3, 0])
        # p_force[i+5] = np.array([1, 0])
        # p_force[i+6] = np.array([1, 0])
        # p_force[i+7] = np.array([1, 0])
        # p_force[i+8] = np.array([1, 0])
        # p_force[i+9] = np.array([1, 0])
        # p_force[i+10] = np.array([1, 0])
        # p_force[i+21] = np.array([1, 0])
        # p_force[i+13]= np.array([1, 0])
        # p_force[i+14] =np.array([1, 0])
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        #################
        pure = p_force[:]
        # print(pure)
        for a, entity_a in enumerate(self.agents):
            for b, entity_b in enumerate(self.entities):
                if(b <= a): continue
                # [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                # print(p_force[a])
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b, pure[a], pure[b])
                if f_a is not None:
                    if p_force[a] is None: p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if f_b is not None:
                    if p_force[b] is None: p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b, p_force_a, p_force_b):
        # print(p_force_a)
        # print('llll')
        # print(p_force_b)
        # if np.size(p_force_a)==0:
        #     p_force_a = np.array([0, 0])
        # if np.size(p_force_b)==0:
        #     p_force_b = np.array([0, 0])
        #     p_force_a = [0, 0]
        # if p_force_b==None:
        #     p_force_b = [0, 0]
        # print(max(sum(abs(p_force_a)), sum(abs(p_force_b))))
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        # if p_force_a==None:
        #     p_force_a = np.array([0, 0])
        # if p_force_b==None:
        #     p_force_b = np.array([0, 0])
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        ################
        # entity_a.state.p_vel = p_force_a/1*0.01 + entity_a.state.p_vel
        # if all(entity_b.state.p_vel):
        #     entity_b.state.p_vel = entity_b.state.p_vel + p_force_b/1*0.01
        #     delta_v = entity_a.state.p_vel - entity_b.state.p_vel
        # else :
        #     delta_v = entity_a.state.p_vel
        #     print(delta_v)
        ################
        if all(entity_b.state.p_vel):
            delta_v = entity_a.state.p_vel - entity_b.state.p_vel
        else :
            delta_v = entity_a.state.p_vel
            # print(delta_v)
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        if max(sum(abs(p_force_a)), sum(abs(p_force_b))) > 2.5:
            dist_min = entity_a.size + entity_b.size +0.05
        else:
            dist_min = entity_a.size + entity_b.size +0.03
        # softmax penetration
        f = max(sum(abs(p_force_a)), sum(abs(p_force_b)))
        # a = f/1
        # v = a* self.dt

        # c = 1*v * self.dt / 1    # c为安全距离，除以1.5，意思是可能不是想向而行的，保持距离过大影响cover点
        # k = c*10 + dist_min
        c = 1*np.sqrt(sum(abs(delta_v)**2))*5 * self.dt / 1   ##!!!!!!后续考虑将V计算成当前速度加决策力产生的速度之和
        k = c + dist_min

        if c!=0:
            cc = f / (-dist_min*dist_min + k*k)
        else:
            cc = 0
        x = dist
        if dist > k:
            penetration = 0
        else:
            penetration = (-x*x + k*k) * cc
            if dist < dist_min and penetration==0:   #or????
                penetration = 1.5
                            # if c != 0:
                            #     cc = - max(sum(abs(p_force_a)), sum(abs(p_force_b))) / np.log(1-c)
                            # else:
                            #     cc = 0
                            # x = dist+1-k
                            # if dist > k:
                            #     penetration = 0
                            # else:
                            #     penetration = -np.log(x) * cc
                            #     if dist < dist_min and penetration==0:
                            #         penetration = 1
        # if penetration > max(sum(abs(p_force_a)), sum(abs(p_force_b))):
        #     penetration = max(sum(abs(p_force_a)), sum(abs(p_force_b)))
                    # print(p_force_a[0])
                    # print(p_force_b[0])
                    # c = (abs(p_force_a[0]-p_force_b[0]) + abs(p_force_a[1]-p_force_b[1]))  * self.dt
                    # print(c)
                    # k = c + dist_min
                    # if c != 0:
                    #     cc = - (abs(p_force_a[0]-p_force_b[0]) + abs(p_force_a[0]-p_force_b[0])) / np.log(1-c)
                    # else:
                    #     cc = 0
                    # x = dist+1-k
                    # if dist > k:
                    #     penetration = 0
                    # else:
                    #     penetration = -np.log(x) * cc
                    # if penetration > (abs(p_force_a[0]-p_force_b[0]) + abs(p_force_a[0]-p_force_b[0])):
                    #     penetration = (abs(p_force_a[0]-p_force_b[0]) + abs(p_force_a[0]-p_force_b[0]))



        # print(penetration)
        # penetration = -np.log(dist + 1 - k) * 1.95761518897
        # if penetration < 0:
        #     penetration = 0
        # elif penetration > sum(abs(p_force_a)):
        #     penetration = sum(abs(p_force_a))
        # print(penetration)
        # penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        # print(sum(abs(p_force_a)))
        # contact_force = p_force_a*3.3*self.contact_force
        # print(penetration)
        # force = delta_pos / dist * penetration
        # if sum(abs(delta_v))!=0:
        #     force = -abs(delta_v)/sum(abs(delta_v)) * penetration
        # else:
        force = delta_pos / dist * penetration

        if dist < dist_min:
            entity_a.state.p_vel = np.array([0.0, 0.0])
            if all(entity_b.state.p_vel):
                entity_b.state.p_vel = np.array([0.0, 0.0])
            print(dist)
            print('%%')
            print(penetration)
            print(entity_a.state.p_vel)
            # if penetration == (-x*x + k*k) * cc:
            #     print(x)
            #     print(cc)
            #     print('dsfadsfadsf')
        # print('%%')
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    # get collision forces for any contact between two entities
    # def get_collision_force(self, entity_a, entity_b):
    #     if (not entity_a.collide) or (not entity_b.collide):
    #         return [None, None] # not a collider
    #     if (entity_a is entity_b):
    #         return [None, None] # don't collide against itself
    #     # compute actual distance between entities
    #     delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
    #     dist = np.sqrt(np.sum(np.square(delta_pos)))
    #     # minimum allowable distance
    #     dist_min = entity_a.size + entity_b.size
    #     # softmax penetration
    #     k = self.contact_margin
    #     penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
    #     force = self.contact_force * delta_pos / dist * penetration
    #     force_a = +force if entity_a.movable else None
    #     force_b = -force if entity_b.movable else None
    #     return [force_a, force_b]