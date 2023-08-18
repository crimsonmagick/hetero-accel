import numpy as np
import gym
import logging
import torch
import random
import os
from copy import deepcopy
from collections import OrderedDict, namedtuple
from time import time
from types import SimpleNamespace
from src import project_dir
from src.loggers import TBLogger, AccuracyEnvCSVLogger
from src.rl import HW_Metrics, Accuracy_Metrics
from src.rl.reward import DEFAULT_REWARD

logger = logging.getLogger(__name__)

# observation - state space
Observation = namedtuple('Observation',
                         ['num_of_layers', 
                          'last_pruning_action', 'last_quant_action',
                          ])

# action space (for verbosity)
Action = namedtuple('Action', ['pruning_ratio', 'quantization_bits'])


class PruningQuantizationEnvironment(gym.Env):
    """Representation of a DNN for pruning and quantization
       into a gym RL Environment
    """
    def __init__(self, compressor, data_loaders, state_args):
        super(PruningQuantizationEnvironment, self).__init__()
        self.args = state_args
        self.name = self.__class__.__name__

        # configure environment logging
        self.logdir = state_args.logdir
        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)
        self.tb_logger = TBLogger(log_dir=self.logdir)
        self.csv_logger = AccuracyEnvCSVLogger(log_dir=self.logdir)

        # wrapper for application neural network
        self.compressor = compressor
        _, self.valid_loader, self.test_loader = data_loaders

        # original accuracy statistics
        original_val_metrics = self.run_inference()
        self.original_val_metrics = Accuracy_Metrics(*original_val_metrics)
        logger.debug(f"{self.name}: Original validation metrics: {self.original_val_metrics}")

        # original compression metrics
        original_hw_metrics = self.compute_resources()

        # define action space
        self.action_space = gym.spaces.Box(low=0, high=1, dtype='float', seed=state_args.seed)
        # TODO: define observation space
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype='float', seed=state_args.seed)

        # prepare for the optimization to begin
        self.current_timesteps = 0
        self.episode = 0
        self.best_reward = float('-inf')
        #self.compressor.log_model(self.tb_logger.writer)
        self.reset(init_only=True)

    def reset(self, init_only=False, verbose=True):
        """Reset the environment at the start of episodes and return an observation
        """
        logger.info("{}: Resetting the environment at {}".format(
                    self.name,
                    'initialization' if init_only else f'episode {self.episode}'))
        self.start_episode = time()
        self.current_state_idx = 0

        self.compressor.reset()
        self.get_state_representation()
        self.csv_logger.reset_history()
        self.latest_hw_metrics = HW_Metrics(*[0 for _ in HW_Metrics._fields])
        self.latest_val_metrics = Accuracy_Metrics(*[0 for _ in Accuracy_Metrics._fields])
        self.latest_reward = DEFAULT_REWARD

        if init_only:
            self.raw_action_history = []
            self.action_history = []
        elif verbose:
            logger.info("+" + "-" * 50 + "+")
            logger.info(f"{self.name}: Episode {self.episode} is starting")
        return self.get_observation()

    def step(self, action):
        """Update the observation based on the agent's action
        """
        self.current_timesteps += 1
        info = {}

        # translate the action from numbers to discrete values
        translated_action = self.translate_action(action)
        self.raw_action_history.append(action)
        self.action_history.append(translated_action)
        info['action'] = translated_action
        logger.debug("{}: Episode {}.{} (timestep {}): action={}, translated action={}".format(
                     self.name, self.episode, self.current_state_idx, self.current_timesteps,
                     action, translated_action))

        self.prune_and_quantize(translated_action)
        self.compute_reward()
        obs = self.get_observation()
        self.finalize_episode()

        info['val_metrics'] = self.latest_val_metrics
        info['hw_metrics'] = self.latest_hw_metrics
        return obs, self.latest_reward, True, info

    def prune_and_quantize(self, action):
        """Prune and quantize the DNN based on the action
        """
        self.compressor.prune_and_quantize(action.pruning_ratio, action.quantization_bits)

    def translate_action(self, raw_action):
        """Translate the given action into meaningful parameters
        """
        pruning_action, quant_action = raw_action
        pruning_action = self.compressor.translate_pruning_action(pruning_action)
        quant_action = self.compressor.translate_quant_action(quant_action)
        return Action(pruning_action, quant_action) 

    def render(self):
        """Broadcast the state of the environment after its step
        """
        logger.info("Invoked rendering function")

    def run_inference(self, use_validation=True):
        """Run a forward pass of the dataset, on the validation on test set
        """
        if use_validation:
            return self.compressor.validate()
        return self.compressor.test()

    def compute_resources(self):
        """Compute the compression metrics related to pruning and quantization
        """
        raise NotImplementedError
        self.latest_hw_metrics = None

    def compute_reward(self):
        """Compute the reward based on the current state
        """
        # retrain for a given amount of epochs
        if self.args.retrain_epochs > 0:
            logger.info(f"{self.name}: Retraining for {self.args.retrain_epochs} epochs")
            self.compressor.train(self.args.retrain_epochs)

        # calculate accuracy and hw-related metrics
        val_metrics = self.run_inference()
        self.latest_val_metrics = Accuracy_Metrics(*val_metrics)
        hw_metrics = self.compute_resources()
        self.latest_hw_metrics = HW_Metrics(*hw_metrics)

        self.latest_reward = self.args.reward_fn(self.latest_val_metrics, self.latest_hw_metrics,
                                                 self.accuracy_constraints, self.hw_constraints)

        logger.debug("{}: Episode {}.{} (timestep {}): reward={}, top1={}, top5={}, loss={}".format(
                     self.name, self.episode, self.current_state_idx, self.current_timesteps,
                     self.latest_reward, self.latest_val_metrics.top1,
                     self.latest_val_metrics.top5, self.latest_val_metrics.loss))

    def get_state_representation(self):
        """Create a representation of the state space of the model
        """
        raise NotImplementedError
        self.state_representation = None
        logger.debug("{}: Episode {}.{} (timestep {}): State representation:\n{}".format(
                     self.name, self.episode, self.current_state_idx, self.current_timesteps,
                     self.state_representation))

    def get_observation(self):
        """Generate an observation of the environment
        """
        obs = self.state_representation[self.current_state_idx, :]
        logger.debug("{}: Episode {}.{} (timestep {}): obs={}".format(
                     self.name, self.episode, self.current_state_idx, self.current_timesteps,
                     obs))
        return obs

    def finalize_episode(self):
        """Finalize the episode and log its progress
        """
        reward = self.latest_reward
        is_best = False
        if reward > self.best_reward:
            is_best = True
            self.best_reward = reward
            self.best_episode = self.episode
        self.compressor.save_model(self.episode, is_best)
 
        # log the evaluation results of the the current episode
        logger.info("{}: Top1: {:.3f} - Top5: {:.3f} - Loss {:.3f}".format(
                    self.name, *self.latest_val_metrics))
        logger.info(f"{self.name}: Reward: {reward:.3e}")
        logger.info(f'{self.name}: Episode {self.episode} completed in {time() - self.start_episode:.3f}s')
        self.record_step(final=True)

        # increment episode counter
        self.episode += 1

    def record_step(self, final=False):
        """Record the latest statistics of the current step
        """
        stats['episode'] = self.episode
        stats['reward'] = self.latest_reward
        stats['timestep'] = self.current_timesteps

        # include the actions taken by the agent
        stats['action'] = list(self.action_history[-1]._asdict().values())

        raise NotImplementedError

        if final:
            # include the accuracy measurements
            val_stats = self.latest_val_metrics._asdict()
            stats.update(val_stats)

            # include accumulated HW metrics, using the 'total' notation to avoid
            #  overwriting the history of the same metrics
            for hw_metric, value in vars(self.accum_hw_metrics).items():
                stats['total_' + hw_metric] = value

            # output the statistics of the episode to csv
            self.csv_logger.record_and_log_stats(stats)
            # record the entire episode using Tensorboard
            self.tb_logger.log_episode(stats)
            return

        # add a record of the current statistics to csv logger history
        self.csv_logger.add_record(stats)
        # record the timestep using TensorBoard
        self.tb_logger.log_step(stats)

