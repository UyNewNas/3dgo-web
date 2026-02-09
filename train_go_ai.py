import asyncio
# from playwright.async_api import async_playwright
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import json
import argparse
from typing import Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GoAI")

# 奖励/惩罚配置：正常判负的奖励（负值），以及非法手惩罚是正常判负的倍数
NORMAL_LOSS_PENALTY = -1
ILLEGAL_PENALTY_MULTIPLIER = 2
ILLEGAL_PENALTY = NORMAL_LOSS_PENALTY * ILLEGAL_PENALTY_MULTIPLIER

# Playwright-based browser controller commented out because web interaction is not needed for local training.
# class PlaywrightGoController:
#     def __init__(self, html_file_path):
#         self.html_file_path = html_file_path
#         self.playwright = None
#         self.browser = None
#         self.page = None
#
#     async def setup(self):
#         ...
#
#     async def make_move(self, x, y, z):
#         ...
#
#     async def get_board_state(self):
#         ...
#
#     async def is_position_occupied(self, x, y, z):
#         ...
#
#     async def is_forbidden_point(self, x, y, z, color):
#         ...
#
#     async def reset_game(self):
#         ...
#
#     async def close(self):
#         ...

class Go3DEnv:
    def __init__(self, controller):
        self.controller = controller
        self.board_size = 7
        
        # 动作空间：343个位置 + Pass
        self.action_space = spaces.Discrete(self.board_size**3 + 1)
        
        # 状态空间：7x7x7的棋盘
        self.observation_space = spaces.Box(
            low=0, high=2, 
            shape=(self.board_size, self.board_size, self.board_size), 
            dtype=np.int8
        )
        
        self.current_player = 1  # 1:黑棋, 2:白棋
        
    async def reset(self):
        success = await self.controller.reset_game()
        if not success:
            logger.error("Failed to reset game")
        
        state = await self.controller.get_board_state()
        self.current_player = 1
        return state, {}
    
    async def step(self, action):
        # 处理Pass
        if action == self.board_size**3:
            # Pass逻辑
            reward = 0
            done = True
            next_state = await self.controller.get_board_state()
        else:
            # 将动作转换为坐标
            x = action // (self.board_size * self.board_size)
            y = (action % (self.board_size * self.board_size)) // self.board_size
            z = action % self.board_size
            # 检查页面上是否已有棋子
            try:
                occupied = await self.controller.is_position_occupied(x, y, z)
            except Exception:
                occupied = False

            if occupied:
                # 非法动作：位置已有棋子，直接判负（结束局面）
                logger.debug(f"Attempted illegal move to occupied position {x},{y},{z}")
                next_state = await self.controller.get_board_state()
                # 返回 done=True 表示本方直接判负，使用常量 ILLEGAL_PENALTY
                return next_state, ILLEGAL_PENALTY, True, False, {'illegal': True, 'reason': 'occupied'}

            # 检查是否为禁着点（自杀或打劫）
            color = 'black' if self.current_player == 1 else 'white'
            try:
                forbidden = await self.controller.is_forbidden_point(x, y, z, color)
            except Exception:
                forbidden = False

            if forbidden:
                # 非法动作：禁着点，直接判负（结束局面）
                logger.debug(f"Attempted illegal move to forbidden point {x},{y},{z} for {color}")
                next_state = await self.controller.get_board_state()
                return next_state, ILLEGAL_PENALTY, True, False, {'illegal': True, 'reason': 'forbidden'}
            
            # 执行落子
            done = await self.controller.make_move(x, y, z)
            
            # 简化奖励计算，使用常量 NORMAL_LOSS_PENALTY
            reward = 1 if not done else NORMAL_LOSS_PENALTY
            
            # 获取新状态
            next_state = await self.controller.get_board_state()
        
        # 切换玩家
        self.current_player = 3 - self.current_player
        
        return next_state, reward, done, False, {}
    
    async def render(self):
        # 通过Playwright自动渲染，不需要额外操作
        pass


class LocalGo3DEnv:
    """纯 Python 本地围棋引擎（7x7x7），提供与 Playwright 驱动的 env 相同的 async 接口，便于训练时使用本地模拟器以加速训练。"""
    def __init__(self, board_size=7):
        self.board_size = board_size
        self.action_space = spaces.Discrete(self.board_size ** 3 + 1)
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.board_size, self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1  # 1 black, 2 white

        # 内部状态
        self.pieces = {}  # key -> {x,y,z,color}
        self.move_history = []
        self.board_states = []
        self.isGameOver = False
        # 提子计数
        self.captures = {'black': 0, 'white': 0}

        # 6 个邻居方向
        self.directions = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        # 连续弃子计数（两次连续弃子结束棋局）
        self.consecutive_passes = 0
        # 记录最终得分与胜者（在局终时填充）
        self.scores = None
        self.winner = None

    def _key(self, x, y, z):
        return f"{x},{y},{z}"

    def reset_sync(self):
        self.pieces.clear()
        self.move_history = []
        self.board_states = []
        self.current_player = 1
        self.isGameOver = False
        self.consecutive_passes = 0
        self.scores = None
        self.winner = None
        self.captures = {'black': 0, 'white': 0}
        return self.get_board_state_sync()

    async def reset(self):
        state = self.reset_sync()
        return state, {}

    def get_board_state_sync(self):
        board = np.zeros((self.board_size, self.board_size, self.board_size), dtype=np.int8)
        for k, p in self.pieces.items():
            x, y, z = p['x'], p['y'], p['z']
            board[x, y, z] = 1 if p['color'] == 'black' else 2
        return board

    async def get_board_state(self):
        return self.get_board_state_sync()

    def is_position_occupied_sync(self, x, y, z):
        return self._key(x,y,z) in self.pieces

    async def is_position_occupied(self, x, y, z):
        return self.is_position_occupied_sync(x, y, z)

    def get_piece_color_sync(self, x, y, z):
        k = self._key(x,y,z)
        return self.pieces[k]['color'] if k in self.pieces else None

    def is_on_board(self, x, y, z):
        return 0 <= x < self.board_size and 0 <= y < self.board_size and 0 <= z < self.board_size

    def find_connected_group_sync(self, x, y, z):
        color = self.get_piece_color_sync(x, y, z)
        if not color:
            return []
        visited = set()
        q = [(x,y,z)]
        visited.add(self._key(x,y,z))
        res = []
        while q:
            cx, cy, cz = q.pop(0)
            res.append({'x':cx,'y':cy,'z':cz})
            for dx,dy,dz in self.directions:
                nx, ny, nz = cx+dx, cy+dy, cz+dz
                if self.is_on_board(nx, ny, nz):
                    k = self._key(nx, ny, nz)
                    if k not in visited and self.get_piece_color_sync(nx, ny, nz) == color:
                        visited.add(k)
                        q.append((nx, ny, nz))
        return res

    def calculate_liberties_sync(self, group):
        liberties = set()
        for pos in group:
            x, y, z = pos['x'], pos['y'], pos['z']
            for dx,dy,dz in self.directions:
                nx, ny, nz = x+dx, y+dy, z+dz
                if self.is_on_board(nx, ny, nz) and not self.get_piece_color_sync(nx, ny, nz):
                    liberties.add(self._key(nx, ny, nz))
        return len(liberties)

    def capture_pieces_sync(self, color):
        captured = []
        visited = set()
        for k, piece in list(self.pieces.items()):
            if piece['color'] == color and k not in visited:
                x,y,z = piece['x'], piece['y'], piece['z']
                group = self.find_connected_group_sync(x,y,z)
                for pos in group:
                    visited.add(self._key(pos['x'], pos['y'], pos['z']))
                if self.calculate_liberties_sync(group) == 0:
                    for pos in group:
                        pkey = self._key(pos['x'], pos['y'], pos['z'])
                        if pkey in self.pieces:
                            captured.append(pos)
                            del self.pieces[pkey]
        return captured

    def find_connected_empty_sync(self, x, y, z, visited):
        """Return list of empty positions connected to (x,y,z) and mark visited set."""
        group = []
        queue = [(x, y, z)]
        key = f"{x},{y},{z}"
        visited.add(key)
        group.append({'x': x, 'y': y, 'z': z})

        while queue:
            cx, cy, cz = queue.pop(0)
            for dx, dy, dz in self.directions:
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                if self.is_on_board(nx, ny, nz):
                    nkey = self._key(nx, ny, nz)
                    if nkey in visited:
                        continue
                    if not self.get_piece_color_sync(nx, ny, nz):
                        visited.add(nkey)
                        queue.append((nx, ny, nz))
                        group.append({'x': nx, 'y': ny, 'z': nz})
        return group

    def determine_territory_owner_sync(self, empty_group):
        """Determine owner of an empty group: 'black'|'white'|None (neutral)."""
        has_black = False
        has_white = False
        for pos in empty_group:
            x, y, z = pos['x'], pos['y'], pos['z']
            for dx, dy, dz in self.directions:
                nx, ny, nz = x + dx, y + dy, z + dz
                if self.is_on_board(nx, ny, nz):
                    color = self.get_piece_color_sync(nx, ny, nz)
                    if color == 'black':
                        has_black = True
                    elif color == 'white':
                        has_white = True
                    if has_black and has_white:
                        return None
        if has_black and not has_white:
            return 'black'
        if has_white and not has_black:
            return 'white'
        return None

    def calculate_territory_sync(self):
        """Calculate territory counts (empty points controlled by each side)."""
        territory = {'black': 0, 'white': 0, 'neutral': 0}
        visited = set()
        for x in range(self.board_size):
            for y in range(self.board_size):
                for z in range(self.board_size):
                    key = self._key(x, y, z)
                    if key in visited:
                        continue
                    if not self.get_piece_color_sync(x, y, z):
                        group = self.find_connected_empty_sync(x, y, z, visited)
                        owner = self.determine_territory_owner_sync(group)
                        if owner == 'black':
                            territory['black'] += len(group)
                        elif owner == 'white':
                            territory['white'] += len(group)
                        else:
                            territory['neutral'] += len(group)
        return territory

    def calculate_final_scores_sync(self):
        """Compute final scores: territory + captures."""
        territory = self.calculate_territory_sync()
        black_score = territory['black'] + self.captures.get('black', 0)
        white_score = territory['white'] + self.captures.get('white', 0)
        return {'black': black_score, 'white': white_score}

    def get_board_state_hash_sync(self):
        arr = [f"{k}:{v['color']}" for k,v in self.pieces.items()]
        return ';'.join(sorted(arr))

    def is_ko_sync(self, x, y, z, color):
        # 保存原始状态
        original_pieces = dict(self.pieces)
        original_hash = self.get_board_state_hash_sync()

        # 模拟落子
        temp_key = self._key(x,y,z)
        self.pieces[temp_key] = {'x':x,'y':y,'z':z,'color':color}

        # 模拟提子
        opponent_color = 'black' if color == 'white' else 'white'
        temp_captured = []
        visited = set()
        for k,piece in list(self.pieces.items()):
            if piece['color'] == opponent_color and k not in visited:
                px,py,pz = piece['x'], piece['y'], piece['z']
                opp_group = self.find_connected_group_sync(px,py,pz)
                for pos in opp_group:
                    visited.add(self._key(pos['x'], pos['y'], pos['z']))
                if self.calculate_liberties_sync(opp_group) == 0:
                    temp_captured.extend(opp_group)
                    for pos in opp_group:
                        del self.pieces[self._key(pos['x'], pos['y'], pos['z'])]

        new_hash = self.get_board_state_hash_sync()
        # 恢复
        self.pieces = original_pieces
        return new_hash in self.board_states

    def is_forbidden_point_sync(self, x, y, z, color):
        # 自杀判断：落子后自身无气且没有提子
        temp_key = self._key(x,y,z)
        self.pieces[temp_key] = {'x':x,'y':y,'z':z,'color':color}
        group = self.find_connected_group_sync(x,y,z)
        has_liberties = self.calculate_liberties_sync(group) > 0

        opponent_color = 'black' if color == 'white' else 'white'
        temp_captured = []
        visited = set()
        for k,piece in list(self.pieces.items()):
            if piece['color'] == opponent_color and k not in visited:
                px,py,pz = piece['x'], piece['y'], piece['z']
                opp_group = self.find_connected_group_sync(px,py,pz)
                for pos in opp_group:
                    visited.add(self._key(pos['x'], pos['y'], pos['z']))
                if self.calculate_liberties_sync(opp_group) == 0:
                    temp_captured.extend(opp_group)

        # 移除临时棋子
        del self.pieces[temp_key]

        # 打劫判断
        ko = self.is_ko_sync(x,y,z,color)

        return (not has_liberties and len(temp_captured) == 0) or ko

    async def is_forbidden_point(self, x, y, z, color):
        return self.is_forbidden_point_sync(x, y, z, color)

    def make_move_sync(self, x, y, z):
        # 假设调用前已经判断合法性
        key = self._key(x,y,z)
        color = 'black' if self.current_player == 1 else 'white'
        self.pieces[key] = {'x':x,'y':y,'z':z,'color':color}

        # 提走对方无气的棋子
        opponent_color = 'black' if color == 'white' else 'white'
        captured = self.capture_pieces_sync(opponent_color)

        # 更新提子计数：本方提走的对方子数累加到本方的 captures
        try:
            if captured:
                self.captures[color] = self.captures.get(color, 0) + len(captured)
        except Exception:
            # 保护性代码：若 self.captures 未初始化也不应导致崩溃
            self.captures = self.captures if hasattr(self, 'captures') else {'black': 0, 'white': 0}
            if captured:
                self.captures[color] = self.captures.get(color, 0) + len(captured)

        # 记录局面
        board_hash = self.get_board_state_hash_sync()
        move = {
            'step': len(self.move_history) + 1,
            'x': x, 'y': y, 'z': z,
            'color': color,
            'captured': [pos for pos in captured],
            'boardState': board_hash
        }
        self.move_history.append(move)
        self.board_states.append(board_hash)

        # 切换玩家
        self.current_player = 3 - self.current_player

        # 判断是否结束（棋盘已满）
        if len(self.move_history) >= self.board_size ** 3 + 1:
            self.isGameOver = True

        return self.isGameOver

    async def make_move(self, x, y, z):
        return self.make_move_sync(x, y, z)

    async def step(self, action):
        # 保持和 Go3DEnv.step 返回格式一致： next_state, reward, done, False, info
        # Pass 处理：只有双方连续 Pass 则结束游戏
        if action == self.board_size ** 3:
            acting_color = 'black' if self.current_player == 1 else 'white'
            # 记录弃子动作
            move = {
                'step': len(self.move_history) + 1,
                'x': None, 'y': None, 'z': None,
                'color': acting_color,
                'captured': [],
                'boardState': self.get_board_state_hash_sync(),
                'isPass': True
            }
            self.move_history.append(move)
            self.consecutive_passes += 1

            # 如果双方连续弃子，则结束并计算得分
            if self.consecutive_passes >= 2:
                self.isGameOver = True
                next_state = self.get_board_state_sync()
                # 终局：使用精确目数（地目 + 提子）计算最终得分
                final_scores = self.calculate_final_scores_sync()
                self.scores = final_scores
                if final_scores['black'] > final_scores['white']:
                    self.winner = 'black'
                elif final_scores['white'] > final_scores['black']:
                    self.winner = 'white'
                else:
                    self.winner = None

                # 奖励针对本次行动者
                if self.winner is None:
                    reward = 0
                else:
                    reward = 1 if acting_color == self.winner else NORMAL_LOSS_PENALTY

                # 切换玩家以保持一致性
                self.current_player = 3 - self.current_player
                return next_state, reward, True, False, {'winner': self.winner, 'scores': self.scores}

            # 非终局：切换玩家，连续弃子仍小于2，返回中间步 reward=0
            self.current_player = 3 - self.current_player
            next_state = self.get_board_state_sync()
            return next_state, 0, False, False, {}

        # 非 Pass：action -> coords
        x = action // (self.board_size * self.board_size)
        y = (action % (self.board_size * self.board_size)) // self.board_size
        z = action % self.board_size

        # 检查是否已有棋子
        if self.is_position_occupied_sync(x, y, z):
            move = {
                'step': len(self.move_history) + 1,
                'x': x, 'y': y, 'z': z,
                'color': 'black' if self.current_player == 1 else 'white',
                'captured': [],
                'boardState': self.get_board_state_hash_sync(),
                'illegal': True,
                'reason': 'occupied'
            }
            self.move_history.append(move)
            # 非法直接判负
            return self.get_board_state_sync(), ILLEGAL_PENALTY, True, False, {'illegal': True, 'reason': 'occupied'}

        # 检查禁着点
        color = 'black' if self.current_player == 1 else 'white'
        if self.is_forbidden_point_sync(x, y, z, color):
            move = {
                'step': len(self.move_history) + 1,
                'x': x, 'y': y, 'z': z,
                'color': color,
                'captured': [],
                'boardState': self.get_board_state_hash_sync(),
                'illegal': True,
                'reason': 'forbidden'
            }
            self.move_history.append(move)
            return self.get_board_state_sync(), ILLEGAL_PENALTY, True, False, {'illegal': True, 'reason': 'forbidden'}

        # 合法落子
        acting_color = 'black' if self.current_player == 1 else 'white'
        # 非弃子动作，重置连续弃子计数
        self.consecutive_passes = 0
        done = self.make_move_sync(x, y, z)
        next_state = self.get_board_state_sync()

        # 如果游戏结束（例如棋盘已满），按棋子数决定胜负；否则中间步奖励为0
        if done or self.isGameOver:
            # 使用精确目数（地目 + 提子）计算最终得分
            final_scores = self.calculate_final_scores_sync()
            self.scores = final_scores
            if final_scores['black'] > final_scores['white']:
                self.winner = 'black'
            elif final_scores['white'] > final_scores['black']:
                self.winner = 'white'
            else:
                self.winner = None

            if self.winner is None:
                reward = 0
            else:
                reward = 1 if acting_color == self.winner else NORMAL_LOSS_PENALTY
        else:
            reward = 0

        return next_state, reward, done, False, {'winner': self.winner, 'scores': self.scores} 

class DQN(nn.Module):
    """简单的DQN网络"""
    def __init__(self, input_shape, output_size):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        
        # 计算展平后的尺寸
        flat_size = input_shape[0] * input_shape[1] * input_shape[2]
        
        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class GoAITrainer:
    def __init__(self, html_file_path, device: Optional[str] = None, use_local_engine: bool = True, verbose: bool = False, save_games_dir: Optional[str] = None):
        self.html_file_path = html_file_path
        # controller is only used when not using local engine
        self.controller = None
        self.env = None
        self.use_local_engine = use_local_engine
        self.verbose = verbose
        # 保存棋谱的目录，默认 ./games
        self.save_games_dir = save_games_dir if save_games_dir is not None else os.path.join(os.getcwd(), 'games')
        # 设备（可通过参数或环境变量 TRAIN_DEVICE 指定）
        # device 参数示例: 'cuda' 或 'cpu'
        env_device = os.environ.get('TRAIN_DEVICE')
        if device is None:
            chosen = env_device if env_device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            chosen = device
        try:
            self.device = torch.device(chosen)
        except Exception:
            # 回退到 cpu
            logger.warning(f"Invalid device '{chosen}', falling back to cpu")
            self.device = torch.device('cpu')
        
        # DQN参数
        self.memory = deque(maxlen=50000)
        self.batch_size = 256
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
    async def setup(self):
        # Web/browser interaction removed: always use local engine for training to avoid per-step browser overhead.
        if not self.use_local_engine:
            logger.warning("Browser-driven environment requested but web interaction is disabled; falling back to LocalGo3DEnv.")
        logger.info("Using LocalGo3DEnv for training (no browser).")
        self.env = LocalGo3DEnv(board_size=7)
        
        # 初始化神经网络
        self.state_size = self.env.observation_space.shape
        # 确保 action_size 为 Python int（某些环境返回 numpy int）
        self.action_size = int(self.env.action_space.n)

        # 计算展平后的状态大小
        flat_state_size = self.state_size[0] * self.state_size[1] * self.state_size[2]

        # 打印设备/CUDA 诊断信息
        try:
            cuda_available = torch.cuda.is_available()
            # 使用 getattr 安全获取 torch.version.cuda
            cuda_version = None
            try:
                ver = getattr(torch, 'version', None)
                if ver is not None:
                    cuda_version = getattr(ver, 'cuda', None)
            except Exception:
                cuda_version = None
            cuda_device_count = torch.cuda.device_count()
        except Exception:
            cuda_available = False
            cuda_version = None
            cuda_device_count = 0
        logger.info(f"Using device: {self.device}")
        logger.info(f"torch.cuda.is_available(): {cuda_available}")
        logger.info(f"torch.version.cuda: {cuda_version}")
        logger.info(f"torch.cuda.device_count(): {cuda_device_count}")
        if cuda_available and cuda_device_count > 0:
            try:
                device_index = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(device_index)
                logger.info(f"CUDA device [{device_index}]: {device_name}")
            except Exception:
                logger.info("CUDA device available but failed to query device name")
        else:
            logger.info("CUDA not available. If you expect to use GPU, install a CUDA-enabled PyTorch build and ensure GPU drivers are installed. See https://pytorch.org for install instructions.")
        # 主网络与目标网络
        self.model = DQN(self.state_size, self.action_size).to(self.device)
        # 目标网络（用于稳定训练），初始与主网络同步
        self.target_model = DQN(self.state_size, self.action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        # 目标网络更新频率（以 replay 调用计数为基准）
        self.target_update_freq = 100
        self.train_steps = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """选择动作"""
        # 如果使用 epsilon 随机策略，随机选择一个合法动作（如果可用）
        if np.random.random() <= self.epsilon:
            # 对本地模拟器可以检查动作合法性并随机选择合法动作
            try:
                if isinstance(self.env, LocalGo3DEnv):
                    valid_actions = []
                    for a in range(int(self.action_size)):
                        if a == self.env.board_size ** 3:
                            valid_actions.append(a)
                            continue
                        x = a // (self.env.board_size * self.env.board_size)
                        y = (a % (self.env.board_size * self.env.board_size)) // self.env.board_size
                        z = a % self.env.board_size
                        # Occupied or forbidden -> invalid
                        if self.env.is_position_occupied_sync(x, y, z):
                            continue
                        color = 'black' if self.env.current_player == 1 else 'white'
                        if self.env.is_forbidden_point_sync(x, y, z, color):
                            continue
                        valid_actions.append(a)
                    if valid_actions:
                        return random.choice(valid_actions)
            except Exception:
                # 任何检查失败，回退为完全随机
                pass
            return random.randrange(int(self.action_size))

        # 否则使用网络选择：对非法动作进行屏蔽
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]  # (action_size,)

        # 屏蔽不可行动作（仅在本地引擎可用时）
        try:
            if isinstance(self.env, LocalGo3DEnv):
                for a in range(int(self.action_size)):
                    if a == self.env.board_size ** 3:
                        continue
                    x = a // (self.env.board_size * self.env.board_size)
                    y = (a % (self.env.board_size * self.env.board_size)) // self.env.board_size
                    z = a % self.env.board_size
                    if self.env.is_position_occupied_sync(x, y, z):
                        q_values[a] = -1e9
                        continue
                    color = 'black' if self.env.current_player == 1 else 'white'
                    if self.env.is_forbidden_point_sync(x, y, z, color):
                        q_values[a] = -1e9
        except Exception:
            # 屏蔽检查失败就不屏蔽
            pass

        return int(np.argmax(q_values))
    
    def replay(self):
        """经验回放"""
        if len(self.memory) < self.batch_size:
            return None
        # 批量采样并向量化计算，减少 Python 循环，提高 GPU 利用率
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([b[0] for b in batch], dtype=np.float32)  # (B,7,7,7)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.bool_)

        states_tensor = torch.from_numpy(states).float().to(self.device)
        next_states_tensor = torch.from_numpy(next_states).float().to(self.device)
        actions_tensor = torch.from_numpy(actions).long().to(self.device)
        rewards_tensor = torch.from_numpy(rewards).float().to(self.device)
        dones_tensor = torch.from_numpy(dones).to(self.device)

        # Q(s,a) and Q(s', a')
        q_values = self.model(states_tensor)  # (B, action_size)
        with torch.no_grad():
            # 使用目标网络估计下一步的 Q 值以提高稳定性
            q_next = self.target_model(next_states_tensor)  # (B, action_size)
            max_q_next, _ = torch.max(q_next, dim=1)

        # 计算目标值： 若 done 则 target = reward else reward + gamma * max_next
        not_done = (~dones_tensor).float()
        target_vals = rewards_tensor + self.gamma * max_q_next * not_done

        # 构建 target tensor
        target_q = q_values.clone().detach()
        batch_idx = torch.arange(self.batch_size, device=self.device)
        target_q[batch_idx, actions_tensor] = target_vals

        # 计算 loss 并优化
        loss = self.criterion(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 目标网络的周期性同步
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            try:
                self.target_model.load_state_dict(self.model.state_dict())
                logger.info(f"Target network updated at step {self.train_steps}")
            except Exception:
                logger.warning("Failed to update target network")

        # epsilon 衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 返回标量 loss 以便训练循环打印/记录
        return float(loss.detach().cpu().item())
    
    async def train(self, episodes=10):
        """训练循环"""
        logger.info("Starting training...")
        # ensure environment has been set up
        assert self.env is not None, "Environment not initialized. Call setup() before train()."
        
        for episode in range(episodes):
            state, _ = await self.env.reset()
            state = np.array(state)
            total_reward = 0
            done = False
            step_count = 0
            # per-episode stats
            illegal_moves = 0
            passes = 0
            total_captured = 0
            episode_losses = []
            while not done:
                action = self.act(state)
                # env.step returns: next_state, reward, done, _, info
                next_state, reward, done, _, info = await self.env.step(action)
                next_state = np.array(next_state)
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                step_count += 1

                # 统计非法/特殊动作：基于 env 返回的 info 字段判断是否非法
                if isinstance(info, dict) and info.get('illegal'):
                    illegal_moves += 1
                    illegal_reason = info.get('reason')
                    logger.debug(f"Illegal move detected: reason={illegal_reason}")
                # Pass 检测（action 为最后一个索引）
                if action == self.env.board_size ** 3:
                    passes += 1

                # 记录并打印每一步（可选）
                if self.verbose:
                    # 将动作索引转为坐标
                    if action == self.env.board_size ** 3:
                        coord = ('pass', 'pass', 'pass')
                    else:
                        x = action // (self.env.board_size * self.env.board_size)
                        y = (action % (self.env.board_size * self.env.board_size)) // self.env.board_size
                        z = action % self.env.board_size
                        coord = (x, y, z)
                    logger.info(f"Episode {episode+1} Step {step_count}: action={action} coord={coord} reward={reward} epsilon={self.epsilon:.4f}")
                
                if done:
                    # replay once more so terminal experiences are trained on
                    loss = self.replay()
                    if loss is not None:
                        episode_losses.append(loss)
                    # 如果是非法落子导致结束，打印原因（info 中可能包含 reason）
                    if isinstance(info, dict) and info.get('illegal'):
                        reason = info.get('reason', 'illegal')
                        logger.info(f"Episode: {episode+1} ended by illegal move ({reason}). Reward: {total_reward:.2f}, Steps: {step_count}, Illegal: {illegal_moves}, Passes: {passes}, AvgLoss: {np.mean(episode_losses) if episode_losses else 0:.6f}, Epsilon: {self.epsilon:.4f}")
                    else:
                        logger.info(f"Episode: {episode+1}, Reward: {total_reward:.2f}, Steps: {step_count}, Illegal: {illegal_moves}, Passes: {passes}, AvgLoss: {np.mean(episode_losses) if episode_losses else 0:.6f}, Epsilon: {self.epsilon:.4f}")
                    break
            
            # 在每局结束后进行一次回放训练（批量）并记录 loss
            loss = self.replay()
            if loss is not None:
                episode_losses.append(loss)

            # 记录每局统计（如果没有在 done 时记录）
            if not done:
                logger.info(f"Episode: {episode+1}, Reward: {total_reward:.2f}, Steps: {step_count}, Illegal: {illegal_moves}, Passes: {passes}, AvgLoss: {np.mean(episode_losses) if episode_losses else 0:.6f}, Epsilon: {self.epsilon:.4f}")
            
            # 每轮保存模型（改为每100局保存一次）
            if (episode + 1) % 100 == 0:
                model_path = f'go_ai_{episode+1}.pth'
                torch.save(self.model.state_dict(), model_path)
                logger.info(f"Model saved to {model_path}")

            # 保存棋谱（如果 env 有 move_history）并记录胜负与目数
            try:
                if hasattr(self.env, 'move_history') and self.env.move_history:
                    os.makedirs(self.save_games_dir, exist_ok=True)
                    import time
                    ts = int(time.time())
                    game_path = os.path.join(self.save_games_dir, f'game_episode_{episode+1}_{ts}.json')
                    game_record = {
                        'episode': episode+1,
                        'steps': step_count,
                        'total_reward': float(total_reward),
                        'illegal_moves': illegal_moves,
                        'passes': passes,
                        'avg_loss': float(np.mean(episode_losses)) if episode_losses else None,
                        'epsilon': float(self.epsilon),
                        'moves': self.env.move_history,
                        'winner': getattr(self.env, 'winner', None),
                        'scores': getattr(self.env, 'scores', None),
                    }
                    with open(game_path, 'w', encoding='utf-8') as f:
                        json.dump(game_record, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved game record to {game_path}")
            except Exception as e:
                logger.error(f"Failed to save game record: {e}")
        
        logger.info("Training completed")

async def main(device: Optional[str] = None, use_local: bool = True, verbose: bool = False, episodes: int = 1000):
    # 初始化训练器
    html_path = "3d-go-external-api.html"  # 替换为您的HTML文件路径
    trainer = GoAITrainer(html_path, device=device, use_local_engine=use_local, verbose=verbose)
    
    try:
        # 设置环境
        await trainer.setup()

        # 开始训练
        await trainer.train(episodes=episodes)
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        # No browser controller to close when using local engine.
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Go AI with optional GPU support")
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='Device to run on (cpu or cuda). Can also set TRAIN_DEVICE env var')
    parser.add_argument('--local', action='store_true', help='Use local Python engine instead of browser-driven env (much faster for training)')
    parser.add_argument('--verbose', action='store_true', help='Print per-move and per-episode logs')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train (default: 1000)')
    args = parser.parse_args()
    asyncio.run(main(device=args.device if args.device else None, use_local=args.local, verbose=args.verbose, episodes=args.episodes))