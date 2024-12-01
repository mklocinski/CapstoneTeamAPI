import os
import pickle
from XRAI_Environment import OutputWrapper
from CheckpointWrapper import learn_with_checkpoints
from mpi4py import MPI
import datetime
from deep_rl_for_swarms.common import logger, cmd_util
from deep_rl_for_swarms.common.act_wrapper import ActWrapper
from deep_rl_for_swarms.policies import mlp_mean_embedding_policy
from deep_rl_for_swarms.rl_algo.trpo_mpi import trpo_mpi
from deep_rl_for_swarms.ma_envs.envs.point_envs import rendezvous

# -------------------------------------------------------------------------- #
# ------------------------ Default Model Vars  ----------------------------- #
# -------------------------------------------------------------------------- #


def load_training_state(current_state, policy_fn):
    act_wrapper = ActWrapper.load(current_state, policy_fn)

    with open(current_state[0], "rb") as f:
        training_state = pickle.load(f)

    with open(current_state[1], "rb") as f:
        counts = pickle.load(f)

    episodes_so_far = counts['episodes_so_far']
    timesteps_so_far = counts['timesteps_so_far']
    iters_so_far = counts['iters_so_far']
    print("Training state loaded.")

    return act_wrapper, episodes_so_far, timesteps_so_far, iters_so_far


def TrainWrapper(
                 environment,
                 environment_params,
                 model_params,
                 map_object,
                 rai_params,
                 current_state):

        print("Session variables...")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        dstr = datetime.datetime.now().strftime('%Y%m%d_%H%M_%S')

        # Use os.path.join to handle file paths correctly across platforms
        log_dir = os.path.join(project_root, 'Output', f"{environment}_{dstr}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        import deep_rl_for_swarms.common.tf_util as U
        sess = U.single_threaded_session()
        sess.__enter__()

        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            logger.configure(format_strs=['csv'], dir=log_dir)
        else:
            logger.configure(format_strs=[])
            logger.set_level(logger.DISABLED)

        print("Define policy function")
        def policy_fn(name, ob_space, ac_space):
            return mlp_mean_embedding_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                       hid_size=[64], feat_size=[64])

        print("Define environment...")
        if environment == 'Rendezvous':
            env = rendezvous.RendezvousEnv(**environment_params)
        else:
            env = rendezvous.RendezvousEnv(**environment_params)


        print("Start training....")

        # if current_state[0] is None:
        #     print("Implement output wrapper...")
        #     output_logger = OutputWrapper(env,
        #                                   env_type=environment,
        #                                   environment_params=environment_params,
        #                                   model_params=model_params,
        #                                   map_object=map_object,
        #                                   rai_params=rai_params,
        #                                   log_file=os.path.join(log_dir, 'output.json'),
        #                                   param_file=os.path.join(log_dir, 'param.json'),
        #                                   map_file=os.path.join(log_dir, 'map.json'))
        #     learn_with_checkpoints(output_logger, policy_fn,
        #                max_timesteps=10,
        #                act_wrapper=None,
        #                 episodes_so_far=0,
        #                 timesteps_so_far=0,
        #                 iters_so_far=0,
        #                **model_params)
        # else:
        #     print("Implement output wrapper...")
        #     output_logger = OutputWrapper(env,
        #                                   env_type=environment,
        #                                   environment_params=environment_params,
        #                                   model_params=model_params,
        #                                   map_object=map_object,
        #                                   log_file=os.path.join(log_dir, 'output.json'),
        #                                   param_file=os.path.join(log_dir, 'param.json'),
        #                                   map_file=os.path.join(log_dir, 'map.json'))
        #     act_wrapper, episodes_so_far, timesteps_so_far, iters_so_far = load_training_state(current_state, policy_fn)
        #     learn_with_checkpoints(output_logger, policy_fn,
        #                    max_timesteps=10,
        #                    act_wrapper=act_wrapper,
        #                    episodes_so_far=episodes_so_far,
        #                    timesteps_so_far=timesteps_so_far,
        #                    iters_so_far=iters_so_far,
        #                    **model_params)
        print("Implement output wrapper...")
        output_logger = OutputWrapper(env,
                                          env_type=environment,
                                          environment_params=environment_params,
                                          model_params=model_params,
                                          map_object=map_object,
                                          rai_params=rai_params,
                                          log_file=os.path.join(log_dir, 'output.json'),
                                          param_file=os.path.join(log_dir, 'param.json'),
                                          map_file=os.path.join(log_dir, 'map.json'))
        learn_with_checkpoints(output_logger, policy_fn,
                                   max_timesteps=10,
                                   act_wrapper=None,
                                   episodes_so_far=0,
                                   timesteps_so_far=0,
                                   iters_so_far=0,
                                   **model_params)

        output_logger.close()
        env.close()


def main(input_environment, input_environment_params, input_model_params, map_object, input_rai_params, current_state):
    TrainWrapper(environment=input_environment,
                 environment_params=input_environment_params,
                 model_params=input_model_params,
                 map_object=map_object,
                 rai_params=input_rai_params,
                 current_state=current_state)


if __name__ == '__main__':
    main()
