import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers.legacy import Adam
import os
import f1_env
import gym
from keras import __version__
tf.keras.__version__ = __version__
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import os


def build_model(states, actions) :
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, target_model_update=0.01)

    return dqn

if __name__ == "__main__" :
    file_dir = file_dir = os.path.dirname(os.path.realpath('__file__'))

    env = gym.make('f1_env_lap_simul', render_mode='None')

    states = env.observation_space.shape[0]
    print(states)
    actions = env.action_space.n

    model =build_model(states,actions)

    print(model.summary())

    dqn = build_agent(model, actions)
    dqn.compile(Adam(learning_rate=0.007), metrics=['mae'])
    #dqn.fit(env, nb_steps=40000, visualize=False, verbose=1)

    #dqn.save_weights(os.path.join(file_dir, 'saved_models\\dqn_6_new.h5f'), overwrite=True)
    dqn.load_weights(os.path.join(file_dir, 'saved_models\\dqn_6_new.h5f'))

    scores = dqn.test(env, nb_episodes=1, visualize=False, verbose=2)
    print(np.mean(scores.history['episode_reward']))
