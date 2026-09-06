import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from ga import evaluate_schedule, is_valid_order, repair_individual, can_assign

# Sử dụng numba để tăng tốc
try:
    from numba import njit
except ImportError:
    def njit(func):
        return func  # Fallback nếu không có numba


# 1. Xây dựng môi trường RL (dựa trên OpenAI Gym)
class MSRCPSPEvn(gym.Env):
    metadata = {'render.modes': ['human']}  # Thêm metadata để tuân thủ Gym

    def __init__(self, schedule, tasks, resources):
        super(MSRCPSPEvn, self).__init__()
        self.schedule = schedule.copy()
        self.tasks = tasks
        self.resources = resources
        self.n_tasks = len(tasks)

        # State: vector 7 chiều (makespan, hiệu suất tài nguyên, độ trễ, vi phạm ràng buộc, cân bằng tài nguyên, 2 chiều dự phòng)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(7,), dtype=np.float32)

        # Action: 0 = hoán đổi 2 task, 1 = điều chỉnh thời gian, 2 = thay resource (giới hạn 2-3 resource/task)
        self.action_space = spaces.Discrete(3)  # 3 action cơ bản

        self.state = self._get_state()
        self.step_count = 0
        self.max_steps = 100  # Giới hạn 100 bước mỗi episode
        self.resource_usage = None
        self.cached_makespan = None  # Khởi tạo caching
        self.cached_resource_usage = None  # Khởi tạo caching

  # @numba.jit(forceobj=True) --> Đánh đổi hiệu suất, Tăng tốc với numba (tùy chọn)
    def _get_state(self):
        # Kiểm tra và khởi tạo caching nếu chưa tồn tại
        if not hasattr(self, 'cached_makespan') or self.cached_makespan is None:
            self.cached_makespan = None
        if not hasattr(self, 'cached_resource_usage') or self.cached_resource_usage is None:
            self.cached_resource_usage = None

        if self.cached_makespan is not None and self.cached_resource_usage is not None:
            resource_usage = self.cached_resource_usage.copy()
            makespan = self.cached_makespan
        else:
            makespan = evaluate_schedule(self.schedule, self.tasks, self.resources)
            if makespan == float("inf"):
                makespan = 10000.0  # Giá trị mặc định nếu không hợp lệ
            self.resource_usage = {res: 0 for res in self.resources}
            completed = set()  # Khai báo completed trước khi sử dụng
            for task_id in self.schedule:
                resource_id = min([r for r in self.resources if can_assign(task_id, r, self.tasks, self.resources)],
                                  key=lambda r: self.resource_usage[r])
                start_time = max([self.resource_usage[resource_id]] +
                                    [self.resource_usage[r] for r in self.resources if
                                     any(t in completed for t in self.tasks[task_id]["predecessors"])],
                                    default=0)
                end_time = start_time + self.tasks[task_id]["duration"]
                self.resource_usage[resource_id] = end_time
                completed.add(task_id)
            resource_usage = self.resource_usage
        self.cached_makespan = makespan
        self.cached_resource_usage = resource_usage.copy()

        total_resource_usage = sum(resource_usage.values())
        resource_util = total_resource_usage / (len(self.resources) * makespan) if makespan > 0 else 0.0

        # Độ trễ: tổng chênh lệch giữa thời gian thực và thời gian lý tưởng (duration)
        delay = 0.0
        completed = set()  # Khai báo lại completed cho độ trễ
        for task_id in self.schedule:
            ideal_time = self.tasks[task_id]["duration"]
            start_time = max([resource_usage[r] for r in self.resources if
                                 any(t in completed for t in self.tasks[task_id]["predecessors"])],
                                default=0)
            actual_time = start_time + self.tasks[task_id]["duration"] - ideal_time
            delay += max(0, actual_time)  # Chỉ tính độ trễ dương
            completed.add(task_id)  # Cập nhật completed sau khi tính

        # Vi phạm ràng buộc: số vi phạm precedence
        violations = 0
        for i, task_id in enumerate(self.schedule):
            for pred in self.tasks[task_id]["predecessors"]:
                if pred not in self.schedule[:i]:
                    violations += 1

        # Mức độ cân bằng tài nguyên: độ lệch chuẩn của mức sử dụng tài nguyên
        resource_loads = np.array(list(resource_usage.values()))
        balance = np.std(resource_loads) if len(resource_loads) > 0 else 0.0

        # State vector
        return np.array([makespan, resource_util, delay, violations, balance, 0.0, 0.0], dtype=np.float32)

    def _get_available_resources(self, task_id):
        return [r for r in self.resources if can_assign(task_id, r, self.tasks, self.resources)]

    #@njit Tăng tốc với numba (tùy chọn)
    def _check_resource_violation(self):
        if not hasattr(self, 'resource_usage') or self.resource_usage is None:
            self.resource_usage = {res: 0 for res in self.resources}
        resource_usage = self.resource_usage.copy()
        completed = set()  # Khai báo completed trước khi sử dụng
        for task_id in self.schedule:
            resource_id = min([r for r in self.resources if can_assign(task_id, r, self.tasks, self.resources)],
                              key=lambda r: resource_usage[r])
            start_time = max([resource_usage[resource_id]] +
                                [resource_usage[r] for r in self.resources if
                                 any(t in completed for t in self.tasks[task_id]["predecessors"])],
                                default=0)
            end_time = start_time + self.tasks[task_id]["duration"]
            if 0 < resource_usage[resource_id] < end_time:
                return True
            resource_usage[resource_id] = end_time
            completed.add(task_id)
        return False

    def step(self, action):
        self.step_count += 1
        reward = 0.0

        if self.step_count >= self.max_steps:
            done = True
            return self.state, reward, done, {}

        if action == 0:  # Swap two tasks
            i, j = random.sample(range(len(self.schedule)), 2)
            task1, task2 = self.schedule[i], self.schedule[j]
            if not any(t in self.tasks[task2]["predecessors"] for t in self.tasks[task1]["predecessors"]):
                self.schedule[i], self.schedule[j] = self.schedule[j], self.schedule[i]
                self.cached_makespan = None
                self.cached_resource_usage = None

        elif action == 1:  # Adjust start time (giảm hoặc tăng 1-5 đơn vị, nếu khả thi)
            task_idx = random.randint(0, len(self.schedule) - 1)
            task_id = self.schedule[task_idx]
            if not hasattr(self, 'resource_usage') or self.resource_usage is None:
                self.resource_usage = {res: 0 for res in self.resources}
            resource_id = min([r for r in self.resources if can_assign(task_id, r, self.tasks, self.resources)],
                              key=lambda r: self.resource_usage[r])
            completed = set(self.schedule[:task_idx])
            start_time = np.max([self.resource_usage[resource_id]] +
                                [self.resource_usage[r] for r in self.resources if
                                 any(t in completed for t in self.tasks[task_id]["predecessors"])],
                                initial=0)
            # Điều chỉnh thời gian (giảm 1-5 đơn vị nếu có thể)
            adjustment = random.randint(-5, -1)  # Giảm 1-5 đơn vị
            if start_time > 0:
                new_start = max(0, start_time + adjustment)
                self.resource_usage[resource_id] = new_start + self.tasks[task_id]["duration"]
                self.cached_makespan = None
                self.cached_resource_usage = None

        elif action == 2:  # Change resource allocation
            task_idx = random.randint(0, len(self.schedule) - 1)
            task_id = self.schedule[task_idx]
            available_resources = self._get_available_resources(task_id)
            if len(available_resources) > 1:
                if not hasattr(self, 'resource_usage') or self.resource_usage is None:
                    self.resource_usage = {res: 0 for res in self.resources}
                current_res = min([r for r in self.resources if can_assign(task_id, r, self.tasks, self.resources)],
                                  key=lambda r: self.resource_usage[r])
                new_res = random.choice([r for r in available_resources if r != current_res][:2])
                completed = set(self.schedule[:task_idx])
                start_time = np.max([self.resource_usage[new_res]] +
                                    [self.resource_usage[r] for r in self.resources if
                                     any(t in completed for t in self.tasks[task_id]["predecessors"])],
                                    initial=0)
                end_time = start_time + self.tasks[task_id]["duration"]
                self.resource_usage[new_res] = end_time
                self.cached_makespan = None
                self.cached_resource_usage = None

        # Tính reward
        new_makespan = evaluate_schedule(self.schedule, self.tasks, self.resources)
        if new_makespan < float("inf"):
            old_makespan = self._get_state()[0]
            reward += 5 * (old_makespan - new_makespan)  # Tăng reward cho giảm makespan
            if not is_valid_order(self.schedule, self.tasks):
                reward -= 100  # Phạt nặng hơn cho vi phạm precedence
            if self._check_resource_violation():
                reward -= 50  # Phạt nặng hơn cho vi phạm tài nguyên
            # Bonus cho cân bằng tài nguyên hoặc giảm độ trễ
            new_state = self._get_state()
            if new_state[4] < self.state[4]:  # Cân bằng tài nguyên cải thiện
                reward += 2
            if new_state[2] < self.state[2]:  # Độ trễ giảm
                reward += 1
            #print(f"Step {self.step_count}, Action: {action}, Reward: {reward}, New Makespan: {new_makespan}")
        else:
            reward -= 100  # Phạt nặng nếu giải pháp không hợp lệ

        self.state = self._get_state()
        done = new_makespan < float("inf") and is_valid_order(self.schedule, self.tasks)
        return self.state, reward, done, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.schedule = list(range(1, self.n_tasks + 1))
        random.shuffle(self.schedule)
        while not is_valid_order(self.schedule, self.tasks):
            random.shuffle(self.schedule)
        self.state = self._get_state()
        self.step_count = 0
        self.resource_usage = None
        self.cached_makespan = None  # Khởi tạo lại caching
        self.cached_resource_usage = None  # Khởi tạo lại caching
        return self.state, {}  # Trả về (state, info) để tuân thủ Gym


# 2. Triển khai DQN bằng PyTorch
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),  # Lớp ẩn 1
            nn.ReLU(),
            nn.Linear(128, 64),  # Lớp ẩn 2
            nn.ReLU(),
            nn.Linear(64, output_dim)  # Output: Q-values cho mỗi action
        )

    def forward(self, x):
        return self.network(x)


# Agent DQN
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = deque(maxlen=5000)  # Replay buffer size 5000
        self.batch_size = 32
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Khởi tạo epsilon-greedy
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_every = 10  # Cập nhật target network mỗi 10 episode
        self.step_count = 0

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        current_q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# 3. Tích hợp RL với GA
def refine_with_rl(schedule, tasks, resources, episodes=100, steps=100):
    env = MSRCPSPEvn(schedule, tasks, resources)
    agent = DQNAgent(state_dim=7, action_dim=3)  # State 7 chiều, 3 action

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        #print(f"Episode {episode} starts with state: {state}")

        for step in range(steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()
            total_reward += reward

            #print(f"Step {step}, Action: {action}, Reward: {reward}, Total Reward: {total_reward}, New Makespan: {next_state[0]}")
            state = next_state

            if done:
                #print(f"Episode {episode} done with total reward: {total_reward}")
                break

    # Kiểm tra tính khả thi sau tinh chỉnh
    if not is_valid_order(env.schedule, tasks):
        env.schedule = repair_individual(env.schedule, tasks)

    # Đảm bảo lịch trình hợp lệ và tính lại makespan
    makespan = evaluate_schedule(env.schedule, tasks, resources)
    return env.schedule, makespan if makespan < float("inf") else evaluate_schedule(
        repair_individual(env.schedule, tasks), tasks, resources)


# Ví dụ thử nghiệm
"""if __name__ == "__main__":
    # Tải dữ liệu từ module data
    tasks, resources = parse_imopse_file("D:/NCKHSV/imopse_validator_pack/IMOPSE/def_small/10_3_5_3.def")
    sorted_tasks = {k: v for k, v in sorted(tasks.items()) if 1 <= k <= 100}

    # Giả sử lấy lịch trình tốt nhất từ GA (tích hợp từ ga.py)
    from ga import run_ga

    best_individual, best_makespan = run_ga(sorted_tasks, resources)
    print("Before RL - Best schedule:", best_individual)
    print("Before RL - Best makespan:", best_makespan)

    # Tinh chỉnh bằng RL
    refined_schedule, refined_makespan = refine_with_rl(best_individual, sorted_tasks, resources)
    print("After RL - Refined schedule:", refined_schedule)
    print("After RL - Refined makespan:", refined_makespan)"""