from GetOutput import OutputWrapper, ModelOutput
import numpy as np
from mpi4py import MPI
import datetime
from drlss.deep_rl_for_swarms.common import logger, cmd_util
from drlss.deep_rl_for_swarms.policies import mlp_mean_embedding_policy
from drlss.deep_rl_for_swarms.rl_algo.trpo_mpi import trpo_mpi
from drlss.deep_rl_for_swarms.ma_envs.envs.point_envs import rendezvous


# -- Notes
# > Next test creation of database

# -------------------------------------------------------------------------- #
# ---------------------------- Description --------------------------------- #
# -------------------------------------------------------------------------- #
# This contains the same contents as the
# \models\base_model\run_multiagent_trpo.py file, but with an environment
# that is wrapped in the DRLOutput logger. Running this file is the
# same as running the original run_multiagent_trpo.py script, but with our
# specified output.

# -------------------------------------------------------------------------- #
# ------------------------------ Run Model --------------------------------- #
# -------------------------------------------------------------------------- #
environment='Rendezvous'
environment_params = dict(nr_agents=20,
                                       obs_mode='sum_obs_acc',
                                       comm_radius=100 * np.sqrt(2),
                                       world_size=100,
                                       distance_bins=8,
                                       bearing_bins=8,
                                       torus=False,
                                       dynamics='unicycle_acc')
model_params = dict(timesteps_per_batch=10,
                                      max_kl=0.01,
                                      cg_iters=10,
                                    cg_damping=0.1,
                                    gamma=0.99,
                                     lam=0.98,
                                     vf_iters=5,
                                     vf_stepsize=1e-3)
def TrainWrapper(num_timesteps,
                 log_dir,
                 environment=environment,
                 environment_params = environment_params,
                 model_params = model_params
                 ):
        import drlss.deep_rl_for_swarms.common.tf_util as U
        sess = U.single_threaded_session()
        sess.__enter__()

        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            logger.configure(format_strs=['csv'], dir=log_dir)
        else:
            logger.configure(format_strs=[])
            logger.set_level(logger.DISABLED)

        def policy_fn(name, ob_space, ac_space):
            return mlp_mean_embedding_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                       hid_size=[64], feat_size=[64])

        if environment == 'Rendezvous':
            env = rendezvous.RendezvousEnv(**environment_params)

        output_logger = OutputWrapper(env,
                                      env_type=environment,
                                      environment_params=environment_params,
                                      model_params=model_params,
                                      log_file= log_dir + '/output.json',
                                      param_file= log_dir + '/param.json')

        trpo_mpi.learn(output_logger, policy_fn,
                       max_timesteps=num_timesteps,
                       **model_params)

        output_logger.close()
        env.close()
        all_output = ModelOutput(output_logger.log_data, output_logger.param_data)
        all_output.generate_tables()
        all_output_counts = {}
        for tbl, df in all_output.tables.items():
            all_output_counts[tbl] = len(df)
        print(all_output_counts)
        return(all_output_counts)

def main():
    env_id = environment
    dstr = datetime.datetime.now().strftime('%Y%m%d_%H%M_%S')
    log_dir = 'C:/Users/mkloc/PycharmProjects/deep_rl_for_swarms/Output/' + env_id + '_' + dstr
    TrainWrapper(num_timesteps=10, log_dir=log_dir)

if __name__ == '__main__':
    main()


