import logging
import os
import stable_baselines3 as sb
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, StopTrainingOnNoModelImprovement
from sb3_contrib import QRDQN

logger = logging.getLogger(__name__)

__all__ = ['_Agent', 'A2C_Agent']


class _Agent():
    """Wrapper base class for stable-baselines-3 agent"""
    def __init__(self, environment, evaluation_environment, agent_args):
        self.args = agent_args
        self.prefix = getattr(agent_args, 'prefix', '')
        if self.prefix:
            self.prefix += '_'

        self.savedir = agent_args.logdir
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)

        # load agent from specified path
        if agent_args.load_from is not None:
            self.load(savepath=agent_args.load_from)

        self.eval_env = evaluation_environment
        self.callbacks = []

        if evaluation_environment is not None and agent_args.eval_frequency is not None \
                                              and agent_args.eval_frequency > 0:
            # create callbacks for the agent, starting with interleaving evaluation
            #  between episodes. Also, add an event callback after evaluations that 
            #  would stop the training after no improvement is shown
            stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=agent_args.no_improv_evals,
                                                                   min_evals=agent_args.min_evals,
                                                                   verbose=agent_args.verbose)
            callback = EvalCallback(eval_env=evaluation_environment,
                                    best_model_save_path=logging.getLogger().logdir,
                                    log_path=logging.getLogger().logdir,
                                    # this translates the eval frequency to episodes
                                    eval_freq=agent_args.eval_frequency,
                                    # epochs to run eval env on
                                    n_eval_episodes=1,
                                    callback_after_eval=stop_train_callback,
                                    deterministic=True,
                                    render=False)
            self.add_callback(callback)

        # include a timeout callback based on episodes instead of timesteps
        if agent_args.train_episodes is not None:
            callback = StopTrainingOnMaxEpisodes(max_episodes=agent_args.train_episodes,
                                                 verbose=agent_args.verbose)
            self.add_callback(callback)


    def add_callback(self, callback):
        """Save a callback to be passed during training
        """
        self.callbacks.append(callback)

    def learn(self, **kwargs):
        """Wrapper function for stable-baselines3 learn() function
        """
        self.agent.learn(total_timesteps=self.args.timesteps,
                         callback=self.callbacks,
                         **kwargs)

    def predict(self, observation):
        """Return a prediction from a given obsevation
        """
        return self.agent.predict(observation, deterministic=self.args.deterministic)

    def evaluate_policy(self, eval_env=None):
        """Evaluate the learned policy on the evaluation environment
        """
        mean_rewards, std_rewards = None, None
        if self.args.eval_episodes is not None and self.args.eval_episodes > 0:
            assert eval_env is not None or self.eval_env is not None, "Define an evaluation environment to evauate the " \
                                                                      "trained policy"

            mean_rewards, std_rewards = sb.common.evaluation.evaluate_policy(self.agent, eval_env,
                                                                             n_eval_episodes=self.args.eval_episodes)
        return mean_rewards, std_rewards

    def finalize(self):
        """Finalize experiment by saving agent and closing environments
        """
        self.save(True)
        self.agent.get_env().close()

    def save(self, save_replay_buffer=False):
        """Save the current agent
        """
        self.agent.save(os.path.join(self.savedir, self.prefix + 'agent_model'))
        if save_replay_buffer and hasattr(self.agent, 'replay_buffer'):
            self.agent.save_replay_buffer(os.path.join(self.savedir, self.prefix + 'agent_replay_buffer'))

    def load(self, environment=None, savepath=None):
        """Load a saved version of the agent and set a new instance of the environment, if needed
        """
        if savepath is None:
            savepath = os.path.join(self.savedir, self.prefix + 'agent_model')
        env = getattr(self.agent, 'env', None) if environment is None else environment
        self.agent = self.agent_class.load(savepath, env=env, verbose=self.args.verbose)


class A2C_Agent(_Agent):
    """Wrapper for SB3 A2C Agent"""
    def __init__(self, env, eval_env, agent_args):
        # TODO: Hyperparameter tuning? (Optuna for example)
        # main agent model: A2C
        self.agent_class = sb.A2C
        self.agent = sb.A2C('MlpPolicy',
                            env,
                            policy_kwargs=agent_args.policy_kwargs,
                            device=agent_args.device,
                            use_rms_prop=True,
                            rms_prop_eps=1e-5,
                            tensorboard_log=agent_args.logdir,
                            verbose=agent_args.verbose,
                            seed=agent_args.seed)
        super().__init__(env, eval_env, agent_args)
 

    def save(self, save_replay_buffer=False):
        if save_replay_buffer:
            logger.warning("Cannot save replay buffer for A2C agent: A2C uses multiple "
                           "workers to avoid the use of a replay buffer.")
        super().save(False)


class QRDQN_Agent(_Agent):
    """Wrapper for SB3-contrib QR-DQN Agent"""
    def __init__(self, env, eval_env, agent_args):
        # main agent model: QR-DQN
        self.agent_class = QRDQN
        self.agent = QRDQN('MlpPolicy',
                           env,
                           learning_starts=1,
                           policy_kwargs=agent_args.policy_kwargs,
                           device=agent_args.device,
                           tensorboard_log=agent_args.logdir,
                           verbose=agent_args.verbose,
                           seed=agent_args.seed)
        super().__init__(env, eval_env, agent_args)
