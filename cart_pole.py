import gym
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# Random player
def random_player(game_env):
    episodes = 10
    for episode in range(1,episodes + 1):
        game_env.reset()
        done = False
        score = 0

        while not done:
            game_env.render()
            action = random.choice([0,1])
            n_state, reward, done, info = game_env.step(action)
            score += reward

        print('Episode:{} Score:{}'.format(episode,score))

# Deep learning model
def build_model(states,actions):
    deep_model = Sequential()
    deep_model.add(Flatten(input_shape=(1,states)))
    deep_model.add(Dense(24, activation='relu'))
    deep_model.add(Dense(24, activation='relu'))
    deep_model.add(Dense(actions, activation='linear'))
    return deep_model

def build_agent(deep_model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn_agent = DQNAgent(model=deep_model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn_agent

def run():
    game_env = gym.make('CartPole-v0')
    game_states = game_env.observation_space.shape[0]
    game_actions = game_env.action_space.n
    model = build_model(game_states,game_actions)

    dqn = build_agent(model,game_actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # dqn.fit(game_env, nb_steps=50000, visualize=False, verbose=1)
    # dqn.save_weights('models/stonks.h5f',overwrite=True)

    dqn.load_weights('models/stonks.h5f')
    dqn.test(game_env, nb_episodes=5, visualize=True)

    # random_player(game_env)

run()

