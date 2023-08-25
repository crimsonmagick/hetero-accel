import logging
import traceback
import os.path
import wandb
import torch
from copy import deepcopy
from collections import OrderedDict
from types import SimpleNamespace
from src import dataset_dirs, pretrained_checkpoint_paths
from src.utils import env_cfg, handle_model_subapps
from src.net_wrapper import TorchNetworkWrapper
from src.dataset import load_data
from src.rl.env import PruningQuantizationEnvironment
from src.rl.compressor import PruningQuantizationCompressor 
from src.rl import reward as rewards
from src.rl.agent import A2C_Agent
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv


logger = logging.getLogger(__name__)


def main():
    """Main function for executing RL training or other operations
    """
    #wandb.init(project='hetero-accel')

    args = env_cfg()
    args.logdir = logging.getLogger().logdir

    
    dnn_args = deepcopy(args)
    dnns = OrderedDict()
    datasets = OrderedDict()
    for idx, arch in enumerate(args.arch):
        arch = arch.lower()
        dnn_args.arch = arch
 
        # configure dataset
        if 'cifar100' in arch and 'cifar100' not in datasets:
            dataset = 'cifar100'
        elif 'cifar10' in arch and 'cifar10' not in datasets:
            dataset = 'cifar10'
        elif 'imagenet' in arch and 'imagenet' not in datasets:
            dataset = 'imagenet'
        elif 'mnist' in arch and 'mnist' not in datasets:
            dataset = 'mnist'
        data_loaders = load_data(
            dataset, dataset_dirs[dataset], arch,
            args.batch_size, args.workers,
            args.validation_split,
            args.effective_train_size,
            args.effective_valid_size,
            args.effective_test_size,
            args.evaluate_model_mode,
            True, #self.args.verbose
        )
        datasets[dataset] = data_loaders

        # configure checkpoint
        dnn_args.dataset = dataset
        dnn_args.resumed_checkpoint_path = None
        if len(args.resumed_checkpoint_path) > idx:
            dnn_args.resumed_checkpoint_path = args.resumed_checkpoint_path[idx]
        elif args.pretrained and dataset != 'imagenet':
            dnn_args.resumed_checkpoint_path = pretrained_checkpoint_paths[arch]
            dnn_args.pretrained = False

        # load DNN wrapper
        net_wrapper = TorchNetworkWrapper.from_args(dnn_args)
        dnns[arch] = net_wrapper

        do_exit = handle_model_subapps(net_wrapper, data_loaders, args)

    logger.info(f"{dnns}")
    logger.info(f"{datasets}")
    #wandb.finish()

    if do_exit:
        exit()



def rl():
    args = env_cfg()
    args.logdir = logging.getLogger().logdir

    dnn_args = deepcopy(args)
    dnns = OrderedDict()
    datasets = OrderedDict()
    for idx, arch in enumerate(args.arch):
        arch = arch.lower()
        dnn_args.arch = arch
 
        # configure dataset
        if 'cifar100' in arch and 'cifar100' not in datasets:
            dataset = 'cifar100'
        elif 'cifar10' in arch and 'cifar10' not in datasets:
            dataset = 'cifar10'
        elif 'imagenet' in arch and 'imagenet' not in datasets:
            dataset = 'imagenet'
        elif 'mnist' in arch and 'mnist' not in datasets:
            dataset = 'mnist'
        data_loaders = load_data(
            dataset, dataset_dirs[dataset], arch,
            args.batch_size, args.workers,
            args.validation_split,
            args.effective_train_size,
            args.effective_valid_size,
            args.effective_test_size,
            args.evaluate_model_mode,
            True, #self.args.verbose
        )
        datasets[dataset] = data_loaders

        # configure checkpoint
        dnn_args.dataset = dataset
        dnn_args.resumed_checkpoint_path = None
        if len(args.resumed_checkpoint_path) > idx:
            dnn_args.resumed_checkpoint_path = args.resumed_checkpoint_path[idx]
        elif args.pretrained and dataset != 'imagenet':
            dnn_args.resumed_checkpoint_path = pretrained_checkpoint_paths[arch]
            dnn_args.pretrained = False


        # initialize environment
        state_args = SimpleNamespace(logdir=args.logdir,
                                     env_seed=args.global_seed,
                                     model_save_frequency=args.rl_model_save_frequency,
                                     retrain_epochs=args.rl_retrain_epochs,
                                     use_validation_set=args.rl_use_validation_set,
                                     reward_fn=rewards.__dict__[args.rl_reward_type],
                                     accuracy_constraints=SimpleNamespace(top1=args.rl_top1_constraint,
                                                                          top5=args.rl_top5_constraint,
                                                                          loss=args.rl_loss_constraint),
                                     hw_constraints=SimpleNamespace(energy=args.rl_energy_constraint,
                                                                    sparsity=args.rl_sparsity_constraint,
                                                                    size=args.rl_size_constraint),
                                     )
        compression_args = SimpleNamespace(arch=dnn_args.arch,
                                           dataset=dnn_args.dataset,
                                           gpus=args.gpus,
                                           cpu=args.cpu,
                                           load_serialized=args.load_serialized,
                                           pretrained=args.pretrained,
                                           resumed_checkpoint_path=dnn_args.resumed_checkpoint_path,
                                           profile_model=args.use_profiler,
                                           print_frequency=args.batch_print_frequency,
                                           verbose=args.model_verbose,
                                           logdir=args.logdir,
                                           # compression arguments
                                           pruning_high=args.rl_pruning_high,
                                           pruning_low=args.rl_pruning_low,
                                           quant_high=args.rl_quant_high,
                                           quant_low=args.rl_quant_low,
                                           layer_type_whitelist=(torch.nn.Conv2d,),
                                           pruning_group_type=args.rl_pruning_group_type,
                                           )
        def make_env():
            env = PruningQuantizationEnvironment(data_loaders, state_args, compression_args)
            #check_env(env)
            return env
        # add a wrapper for single threaded execution
        #env = DummyVecEnv([lambda: make_env()])
        env = make_env()

        # initialize agent
        agent_args = SimpleNamespace(logdir=args.logdir,
                                     prefix=None,
                                     deterministic=args.rl_agent_deterministic,
                                     seed=args.global_seed,
                                     verbose=args.rl_agent_verbose,
                                     policy_kwargs=dict(),
                                     device=env.compressor.device,
                                     eval_frequency=args.rl_agent_eval_frequency,
                                     no_improv_evals=args.rl_agent_no_improv_evals,
                                     min_evals=args.rl_agent_min_evals,
                                     timesteps=args.rl_agent_total_timesteps,
                                     train_episodes=args.rl_agent_train_episodes,
                                     eval_episodes=args.rl_agent_eval_episodes,
                                     load_from=args.rl_agent_load_from_path)
        agent = A2C_Agent(agent_args, env, None)

        # train the agent
        agent.learn()

        # evaluate the agent
        eval_env = env
        mean_reward, std_reward = agent.evaluate_policy(eval_env)
        if mean_reward is not None and std_reward is not None:
            logger.info("Policy evaluation: mean rewards: {mean_reward:.3e}, std rewards: {std_reward:.3e}")

        # explicit episode execution to gather final actions and metrics
        obs_t0 = eval_env.env.reset()
        action, _ = agent.predict(obs)
        obs_t1, reward, done, info = env.step(action)

        agent.finalize()


        exit()


if __name__ == '__main__':
    try:
        rl()
        #main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if logger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the logger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = logger.handlers
            logger.handlers = [h for h in logger.handlers if type(h) != logging.StreamHandler]
            logger.error(traceback.format_exc())
            logger.handlers = handlers_bak
        raise
    finally:
        if logger is not None and hasattr(logging.getLogger(), 'log_filename'):
            logger.info('')
            logger.info('Log file for this run: ' + os.path.realpath(logging.getLogger().log_filename))
            exit()

