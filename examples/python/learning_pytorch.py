#!/usr/bin/env python3

# E. Culurciello, L. Mueller, Z. Boztoprak
# December 2020

import itertools as it
import os
import random
from collections import deque
from time import sleep, time

import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

import vizdoom as vzd
import matplotlib
import matplotlib.pyplot as plt

# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
train_epochs = 10
learning_steps_per_epoch = 100
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 10

model_savefile = "./model-doom.pth"
save_model = True
load_model = False
skip_learning = False

# Configuration file path
config_file_path = os.path.join(vzd.scenarios_path, "simpler_basic.cfg")
# config_file_path = os.path.join(vzd.scenarios_path, "rocket_basic.cfg")
# config_file_path = os.path.join(vzd.scenarios_path, "basic.cfg")

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("GPU available")
    # torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")
    print("GPU not available")

matplotlib.use('TkAgg')  # バックエンドを設定（必要に応じて 'TkAgg' を変更）


# Q値を可視化する関数
def visualize_q_values(agent, states, preprocess_screen):
    q_values = agent.q_net(states).cpu().data.numpy()  # Q値を計算
    # print(q_values.shape)
    num_actions = agent.action_size  # 行動の数
    num_states = states.shape[0]  # 状態の数

    # 状態を画像として表示
    plt.imshow(np.transpose(preprocess_screen, (1, 2, 0)))
    plt.show()
    plt.imsave("state.png", preprocess_screen[0], cmap="gray")
    # plt.imsave('state.png', np.transpose(preprocess_screen, (1, 2, 0)))

    # 各状態に対する行動ごとのQ値をヒートマップで表示
    for state_index in range(num_states):
        state_q_values = q_values[state_index]
        # リシェイプの形状を行動数に合わせる
        state_q_values = state_q_values.reshape((1, num_actions))  # (1, 8)にリシェイプ
        plt.figure(figsize=(8, 6))
        plt.imshow(state_q_values, cmap='viridis', interpolation='nearest')
        plt.title(f'Q-values for State {state_index}')
        plt.colorbar()
        plt.savefig(f'q_values_state_{state_index}.png')


def preprocess(img):
    """Down samples image to resolution"""
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img


def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")

    return game


def test(game, agent):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            game.make_action(actions[best_action_index], frame_repeat)
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print(
        "Results: mean: {:.1f} +/- {:.1f},".format(
            test_scores.mean(), test_scores.std()
        ),
        "min: %.1f" % test_scores.min(),
        "max: %.1f" % test_scores.max(),
    )


def run(game, agent, actions, num_episodes, frame_repeat, steps_per_episode=2000):
    """
    Run num epochs of training episodes.
    Skip frame_repeat number of frames after each action.
    """

    start_time = time()  # 訓練開始時刻を記録

    for episode in range(num_episodes):  # 指定されたエピソード数の訓練を繰り返す
        game.new_episode()  # 新しいゲームエピソードを開始
        train_scores = []  # エピソードごとのスコアを格納するリスト
        global_step = 0  # グローバルステップカウンターを初期化
        print("\nEpoch #" + str(episode + 1))  # 現在のエポックを表示

        for _ in trange(steps_per_episode, leave=False):  # 指定されたステップ数の間繰り返す
            state = preprocess(game.get_state().screen_buffer)  # 現在のゲーム画面を前処理
            action = agent.get_action(state)  # エージェントが行動を選択
            reward = game.make_action(actions[action], frame_repeat)  # 行動を実行し、報酬を受け取る
            done = game.is_episode_finished()  # エピソードが終了したかどうかを確認

            if not done:  # エピソードが終了していない場合
                next_state = preprocess(game.get_state().screen_buffer)  # 次のゲーム画面を前処理
            else:
                next_state = np.zeros((1, 30, 45)).astype(np.float32)  # 終了した場合は空の画面

            agent.append_memory(state, action, reward, next_state, done)  # 経験をメモリに追加

            if global_step > agent.batch_size:  # メモリがバッチサイズ以上の場合は訓練を開始
                agent.train()

            if done:  # エピソードが終了した場合
                train_scores.append(game.get_total_reward())  # エピソードの合計報酬を記録
                game.new_episode()  # 新しいゲームエピソードを開始

            global_step += 1  # グローバルステップを増やす

        agent.update_target_net()  # ターゲットネットワークを更新
        train_scores = np.array(train_scores)  # スコアをNumPy配列に変換

        # エピソードごとの訓練結果を表示
        print(
            "Results: mean: {:.1f} +/- {:.1f},".format(
                train_scores.mean(), train_scores.std()
            ),
            "min: %.1f," % train_scores.min(),
            "max: %.1f," % train_scores.max(),
        )

        # 各状態に対するQ値を可視化
        preprocess_screen = preprocess(game.get_state().screen_buffer)
        state = np.expand_dims(preprocess_screen, axis=0)
        state = torch.from_numpy(state).float().to(DEVICE)
        visualize_q_values(agent, state, preprocess_screen)

        # テスト関数を呼び出して訓練の途中結果を表示
        test(game, agent)

        if save_model:  # モデルの保存が有効な場合
            print("Saving the network weights to:", model_savefile)
            torch.save(agent.q_net, model_savefile)  # モデルの重みを保存

        # 経過時間を表示
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))

    game.close()  # ゲーム環境をクローズして訓練を終了
    return agent, game


class DuelQNet(nn.Module):
    """
    This is Duel DQN architecture.
    see https://arxiv.org/abs/1511.06581 for more information.
    """

    def __init__(self, available_actions_count):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.state_fc = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 1))

        self.advantage_fc = nn.Sequential(
            nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, available_actions_count)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 192)
        x1 = x[:, :96]  # input for the net to calculate the state value
        x2 = x[:, 96:]  # relative advantage of actions in the state
        state_value = self.state_fc(x1).reshape(-1, 1)
        advantage_values = self.advantage_fc(x2)
        x = state_value + (
            advantage_values - advantage_values.mean(dim=1).reshape(-1, 1)
        )

        return x


class DQNAgent:
    def __init__(
        self,
        action_size,
        memory_size,
        batch_size,
        discount_factor,
        lr,
        load_model,
        epsilon=1,
        epsilon_decay=0.9996,
        epsilon_min=0.1,
    ):
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount = discount_factor
        self.lr = lr
        self.memory = deque(maxlen=memory_size)
        self.criterion = nn.MSELoss()

        if load_model:
            print("Loading model from: ", model_savefile)
            self.q_net = torch.load(model_savefile)
            self.target_net = torch.load(model_savefile)
            self.epsilon = self.epsilon_min

        else:
            print("Initializing new model")
            self.q_net = DuelQNet(action_size).to(DEVICE)
            self.target_net = DuelQNet(action_size).to(DEVICE)

        self.opt = optim.SGD(self.q_net.parameters(), lr=self.lr)

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(DEVICE)
            action = torch.argmax(self.q_net(state)).item()
            return action

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        batch = random.sample(self.memory, self.batch_size)  # メモリからランダムにバッチをサンプリング
        batch = np.array(batch, dtype=object)  # バッチをNumPy配列に変換

        states = np.stack(batch[:, 0]).astype(float)  # バッチ内の状態（state）を取得し、前処理
        actions = batch[:, 1].astype(int)  # バッチ内の行動（action）を取得
        rewards = batch[:, 2].astype(float)  # バッチ内の報酬（reward）を取得
        next_states = np.stack(batch[:, 3]).astype(float)  # バッチ内の次の状態（next_state）を取得
        dones = batch[:, 4].astype(bool)  # バッチ内の終了状態（done）を取得
        not_dones = ~dones  # バッチ内の未終了状態を示すブールマスク

        row_idx = np.arange(self.batch_size)  # バッチ内のインデックスを生成

        # Double Q Learningのための次の状態の価値を計算
        with torch.no_grad():
            next_states = torch.from_numpy(next_states).float().to(DEVICE)
            # 次の状態における最適な行動のインデックスを取得
            idx = row_idx, np.argmax(self.q_net(next_states).cpu().data.numpy(), 1)
            # ターゲットネットワークで次の状態の価値を計算
            next_state_values = self.target_net(next_states).cpu().data.numpy()[idx]
            next_state_values = next_state_values[not_dones]  # 未終了状態のみを選択

        # TD誤差を計算して訓練用のターゲットQ値を生成
        q_targets = rewards.copy()  # 報酬をコピー
        q_targets[not_dones] += self.discount * next_state_values  # 未終了状態の場合は更新
        q_targets = torch.from_numpy(q_targets).float().to(DEVICE)

        # 行動を選択したときのQ値を取得
        idx = row_idx, actions
        states = torch.from_numpy(states).float().to(DEVICE)
        action_values = self.q_net(states)[idx].float().to(DEVICE)

        # 損失を計算してバックプロパゲーションを行い、重みを更新
        self.opt.zero_grad()
        td_error = self.criterion(q_targets, action_values)
        td_error.backward()
        self.opt.step()

        # ε-グリーディ法のε（探索確率）を更新
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


if __name__ == "__main__":
    # Initialize game and actions
    game = create_simple_game()
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Initialize our agent with the set parameters
    agent = DQNAgent(
        len(actions),
        lr=learning_rate,
        batch_size=batch_size,
        memory_size=replay_memory_size,
        discount_factor=discount_factor,
        load_model=load_model,
    )

    # Run the training for the set number of epochs
    if not skip_learning:
        agent, game = run(
            game,
            agent,
            actions,
            num_episodes=train_epochs,
            frame_repeat=frame_repeat,
            steps_per_episode=learning_steps_per_epoch,
        )

        print("======================================")
        print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.close()
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)
