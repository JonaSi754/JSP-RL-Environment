from envs.JSP_v1 import env as JSP_ENV
import numpy as np

class Q_learning:
    
    action_space = 5
    
    def __init__(self, env: JSP_ENV, epsilon: np.float32, gamma: np.float32, learning_rate: np.float32) -> None:
        self.env = env
        self.state = self.state_mapping(self.env.reset())
        self.learning_rate = learning_rate  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        # 使用JSP-v1的环境，有四个全局状态，每个状态定义5个区间，共625个状态，5个动作
        self.q_table = np.zeros((625, Q_learning.action_space))
    
    def learn(self, total_timesteps: np.int32):
        time_step = 1
        while (time_step < total_timesteps):
            time_step += 1
            # 运行完成后重置状态
            if self.env.done:
                self.state = self.state_mapping(self.env.reset())
            
            # epsilon-greedy策略选择动作
            if np.random.uniform(0, 1) < self.epsilon:
                action = np.random.randint(0, Q_learning.action_space)
            else:
                action = np.argmax(self.q_table[self.state, :])
                
            # 更新Q表
            next_state, reward, done, _ = self.env.step(action)
            next_state = self.state_mapping(next_state)
            self.q_table[self.state, action] = self.q_table[self.state, action] + self.learning_rate * (reward + self.gamma * np.max(self.q_table[next_state, :]) - self.q_table[self.state, action])
            self.state = next_state
            
    def predict(self, curState: np.array):
        state = self.state_mapping(curState)
        return np.argmax(self.q_table[state, :])
    
    def save(self, fileDir):
        np.save(file=fileDir, arr=self.q_table)
    
    def load(self, model_path: str):
        self.q_table = np.load(model_path + '.npy')
    
    def state_mapping(self, state):
        # 按照五进制计算状态对应的序号
        for num in state:
            if num > 1:
                print('state out of range!')
        s = state // 0.2
        index = s[0] * 5 ** 3 + s[1] * 5 ** 2 + s[2] * 5 + s[3]
        return int(index)