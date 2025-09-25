from ursina import *
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import copy
import time

# ---- PARAMETERS ----
WORLD_SIZE = 10
MAX_SPEED = 0.2
TAG_DISTANCE = 1.5
POPULATION_SIZE = 25     # number of arenas (1 blue + 1 red each)
ACTIONS = [
    Vec3(1,0,0), Vec3(-1,0,0), Vec3(0,0,1), Vec3(0,0,-1),
    Vec3(1,0,1), Vec3(1,0,-1), Vec3(-1,0,1), Vec3(-1,0,-1),
    Vec3(0,1,0), Vec3(0,-1,0)
]
STATE_DIM = 6
LR = 0.001
GAMMA = 0.9
EPSILON = 0.2
MEMORY_SIZE = 10000
BATCH_SIZE = 64
SURVIVAL_TIME = 5
GENERATION_TIME = 15
MUTATION_RATE = 0.02
GRAVITY = 0.05
MAX_REPEAT_GENERATIONS = 3
REPEAT_PENALTY = 5
DOT_LIFETIME = 5.0  # seconds

# ---- DQN NETWORK ----
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self,x):
        return self.fc(x)

# ---- AI AGENT ----
class DQNPlayer(Entity):
    def __init__(self, color, start_pos, is_chaser=False):
        super().__init__(model='cube', color=color, scale=(1,1,1), position=start_pos, alpha=0.3)
        self.is_chaser = is_chaser
        self.net = DQN(STATE_DIM,len(ACTIONS))
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.velocity = Vec3(0,0,0)
        self.last_tag_time = time.time()
        self.last_survival_reward = time.time()
        self.fitness = 0
        self.visited_positions = set()
        self.behavior_history = deque(maxlen=MAX_REPEAT_GENERATIONS)
        self.trails = []
        self.last_distance = None
    
    def get_state(self, other_pos):
        return np.array([self.x,self.y,self.z, other_pos.x, other_pos.y, other_pos.z], dtype=np.float32)
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0,len(ACTIONS)-1)
        state_t = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.net(state_t)
        return int(torch.argmax(q_values).item())
    
    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
    
    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        q_values = self.net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            q_next = self.net(next_states).max(1)[0]
        q_target = rewards + GAMMA * q_next
        loss = nn.MSELoss()(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def move(self, other_pos):
        state = self.get_state(other_pos)
        action_idx = self.choose_action(state)
        desired_velocity = ACTIONS[action_idx].normalized() * MAX_SPEED
        
        # Apply velocity and gravity
        self.velocity = self.velocity * 0.8 + desired_velocity * 0.2
        self.velocity.y -= GRAVITY
        self.position += self.velocity
        
        # Clamp to world bounds
        self.x = clamp(self.x, -WORLD_SIZE, WORLD_SIZE)
        self.y = clamp(self.y, 0, 5)
        self.z = clamp(self.z, -WORLD_SIZE, WORLD_SIZE)
        
        # Exploration reward
        pos_tuple = (round(self.x,1), round(self.y,1), round(self.z,1))
        new_behavior_reward = 0
        if pos_tuple not in self.visited_positions:
            new_behavior_reward = 0.05
            self.visited_positions.add(pos_tuple)
        
        # Add trail
        trail_seg = Entity(model='cube', scale=0.1, color=self.color, position=self.position, alpha=0.5)
        self.trails.append((trail_seg, time.time()))
        
        return state, action_idx, new_behavior_reward

# ---- MUTATE NETWORK ----
def mutate_network(net):
    new_net = copy.deepcopy(net)
    with torch.no_grad():
        for param in new_net.parameters():
            param.add_(torch.randn_like(param) * MUTATION_RATE)
    return new_net

# ---- ARROW CREATION ----
def make_arrow(start, end):
    direction = end - start
    length = direction.length()
    if length == 0:
        return None
    
    shaft = Entity(model='cylinder', scale=(0.05, length/2, 0.05),
                   position=start + direction/2,
                   rotation=look_at(start, end).rotation,
                   color=color.yellow)
    head = Entity(model='cone', scale=(0.2, 0.4, 0.2),
                  position=end,
                  rotation=look_at(start, end).rotation,
                  color=color.orange)
    arrow = Entity()
    shaft.parent = arrow
    head.parent = arrow
    return arrow

# ---- GAME SETUP ----
app = Ursina()
arenas = []
prev_gen_dots = []
arrows = []

grid_size = int(POPULATION_SIZE ** 0.5) + 1
spacing = 25

for i in range(POPULATION_SIZE):
    gx = i % grid_size
    gz = i // grid_size
    offset = Vec3(gx * spacing, 0, gz * spacing)

    # Create one chaser + one runner per arena
    ch = DQNPlayer(color=color.red, start_pos=offset + Vec3(-5,0,0), is_chaser=True)
    ru = DQNPlayer(color=color.blue, start_pos=offset + Vec3(5,0,0), is_chaser=False)

    # Coral walls
    wall_height = 4
    wall_thickness = 0.2
    arena_size = spacing / 2
    Entity(model='cube', scale=(spacing, wall_height, wall_thickness),
           position=offset + Vec3(0, wall_height/2, -arena_size),
           color=color.rgba(100, 100, 255, 80))
    Entity(model='cube', scale=(spacing, wall_height, wall_thickness),
           position=offset + Vec3(0, wall_height/2, arena_size),
           color=color.rgba(100, 100, 255, 80))
    Entity(model='cube', scale=(wall_thickness, wall_height, spacing),
           position=offset + Vec3(-arena_size, wall_height/2, 0),
           color=color.rgba(100, 100, 255, 80))
    Entity(model='cube', scale=(wall_thickness, wall_height, spacing),
           position=offset + Vec3(arena_size, wall_height/2, 0),
           color=color.rgba(100, 100, 255, 80))

    arenas.append((ch, ru))

camera = EditorCamera()
generation_start = time.time()

def distance(a,b):
    return np.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

def compute_behavior_signature(agent):
    positions = list(agent.visited_positions)
    positions_rounded = [(round(p[0]*2)/2, round(p[1]*2)/2, round(p[2]*2)/2) for p in positions]
    return hash(frozenset(positions_rounded))

# ---- UPDATE LOOP ----
def update():
    global generation_start, prev_gen_dots, arrows
    current_time = time.time()

    # Clear old arrows
    for a in arrows:
        destroy(a)
    arrows = []

    for ch, ru in arenas:
        ch_state, ch_action, ch_new_reward = ch.move(ru.position)
        ru_state, ru_action, ru_new_reward = ru.move(ch.position)
        
        dist = distance(ch, ru)
        ch_reward = -0.01 + max(0,5-dist)*0.1 + ch_new_reward
        ru_reward = 0.01 + dist*0.1 + ru_new_reward

        # Movement incentive
        if ch.velocity.length() < 0.05: ch_reward -= 0.1
        if ru.velocity.length() < 0.05: ru_reward -= 0.1

        # Distance change reward
        if ch.last_distance is not None:
            dist_change = ch.last_distance - dist
            ch_reward += dist_change * 0.2
            ru_reward -= dist_change * 0.2
            if ch.last_distance < 3 and dist > 3 and dist_change < 0:
                ru_reward += 8
                ch_reward -= 2
        ch.last_distance = dist
        ru.last_distance = dist

        # Jump reward
        if ch.velocity.y > 0.1: ch_reward += 0.1
        if ru.velocity.y > 0.1: ru_reward += 0.1

        # Tag event
        if dist < TAG_DISTANCE:
            ch_reward += 10
            ru_reward -= 10
            ch.position += Vec3(-5,0,0)
            ru.position += Vec3(5,0,0)
            ch.velocity = ru.velocity = Vec3(0,0,0)

        # Survival
        if current_time - ru.last_survival_reward >= SURVIVAL_TIME:
            ru_reward += 5
            ru.last_survival_reward = current_time
        if current_time - ch.last_tag_time >= SURVIVAL_TIME:
            ch_reward -= 5
            ch.last_tag_time = current_time

        ch.fitness += ch_reward
        ru.fitness += ru_reward

        ch_next = ch.get_state(ru.position)
        ru_next = ru.get_state(ch.position)
        ch.store_transition(ch_state, ch_action, ch_reward, ch_next)
        ru.store_transition(ru_state, ru_action, ru_reward, ru_next)
        ch.learn()
        ru.learn()

        arrow = make_arrow(ch.position, ru.position)
        if arrow: arrows.append(arrow)

    # Fade trails
    for ch, ru in arenas:
        for agent in (ch,ru):
            for trail, created in agent.trails[:]:
                age = current_time - created
                if age > DOT_LIFETIME:
                    destroy(trail)
                    agent.trails.remove((trail, created))
                else:
                    trail.alpha = max(0, 0.5*(1-age/DOT_LIFETIME))

    # Generation rollover
    if current_time - generation_start >= GENERATION_TIME:
        for ch, ru in arenas:
            prev_gen_dots.append((Entity(model='sphere', scale=0.2, color=color.red, position=ch.position), current_time))
            prev_gen_dots.append((Entity(model='sphere', scale=0.2, color=color.blue, position=ru.position), current_time))
        generation_start = current_time

app.run()
