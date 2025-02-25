import json

from deep_rl_for_swarms.common import explained_variance, zipsame, dataset
from deep_rl_for_swarms.common import logger
import deep_rl_for_swarms.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from deep_rl_for_swarms.common import colorize
from mpi4py import MPI
from collections import deque
from deep_rl_for_swarms.common.mpi_adam import MpiAdam
from deep_rl_for_swarms.common.cg import cg
from contextlib import contextmanager
from deep_rl_for_swarms.common.act_wrapper import ActWrapper
import pickle
import requests


# -------------------------------------------------------------------------- #
# ------------------------ Checkpoint Helper Functions --------------------- #
# -------------------------------------------------------------------------- #
# def check_model_status():
#     status = db.session.execute(text('SELECT state FROM status')).scalar()
#     return status
#
#
# def record_model_episode(current_episode):
#     update_query = text('''UPDATE status SET episode = :current_episode WHERE state = 'running' ''')
#     db.session.execute(update_query, {"current_episode": current_episode})
#     db.session.commit()

def check_model_status():
    print("Checking model status...")
    try:
        response = requests.get("http://app:8000/api/check_model_status", timeout=10)
        # https://xraiapi-ba66c372be3f.herokuapp.com
        response = json.loads(response)
        if 'status' in response.keys():
            return response["status"]
        else:
            return "running"
    except requests.Timeout:
        return "running"

def record_model_episode(data):
    response = requests.post("http://app:8000/api/record_model_episode", json=data)

def save_checkpoint(action_wrapper, episodes_so_far, timesteps_so_far, iters_so_far):
    print("Saving checkpoint...")
    action_wrapper.save("utils/pickles/checkpoint.pkl")

    counters = {
        'episodes_so_far': episodes_so_far,
        'timesteps_so_far': timesteps_so_far,
        'iters_so_far': iters_so_far
    }
    with open("utils/pickles/checkpoint_counters.pkl", "wb") as f:
        pickle.dump(counters, f)

    print("Checkpoint saved.")


def load_checkpoint(policy_fn, env):
    action_wrapper = ActWrapper.load("utils/pickles/checkpoint.pkl", policy_fn)

    with open('utils/pickles/checkpoint_counters.pkl', 'rb') as f:
        counters = pickle.load(f)

    return action_wrapper, counters["episodes_so_far"], counters["timesteps_so_far"], counters["iters_so_far"],


# -------------------------------------------------------------------------- #
# ------------------ trpo_mpi's original helper functions ------------------ #
# -------------------------------------------------------------------------- #

def traj_segment_generator(pi, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    n_agents = len(env.agents)
    ac = np.vstack([env.action_space.sample() for _ in range(n_agents)])
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []
    time_steps = []

    # Initialize history arrays
    sub_sample_thresh = 8
    if n_agents > sub_sample_thresh:
        sub_sample = True
        sub_sample_idx = np.random.choice(n_agents, sub_sample_thresh, replace=False)

        obs = np.array([[ob[ssi] for ssi in sub_sample_idx] for _ in range(horizon)])
        rews = np.zeros([horizon, sub_sample_thresh], 'float32')
        vpreds = np.zeros([horizon, sub_sample_thresh], 'float32')
        news = np.zeros([horizon, sub_sample_thresh], 'int32')
        acs = np.array([ac[sub_sample_idx] for _ in range(horizon)])
        prevacs = acs.copy()
    else:
        sub_sample = False
        obs = np.array([ob for _ in range(horizon)])
        rews = np.zeros([horizon, n_agents], 'float32')
        vpreds = np.zeros([horizon, n_agents], 'float32')
        news = np.zeros([horizon, n_agents], 'int32')
        acs = np.array([ac for _ in range(horizon)])
        prevacs = acs.copy()

    while True:
        prevac = ac[sub_sample_idx] if sub_sample else ac
        ac, vpred = pi.act(stochastic, np.vstack(ob))
        if t > 0 and t % horizon == 0:
            yield [
                dict(
                    ob=np.array(obs[:, na, :]),
                    rew=np.array(rews[:, na]),
                    vpred=np.array(vpreds[:, na]),
                    new=np.array(news[:, na]),
                    ac=np.array(acs[:, na, :]),
                    prevac=np.array(prevacs[:, na, :]),
                    nextvpred=vpred[na] * (1 - new) if not sub_sample else vpred[sub_sample_idx[na]] * (1 - new),
                    ep_rets=[epr[na] for epr in ep_rets],
                    ep_lens=ep_lens,
                    time_steps=np.array(time_steps)
                ) for na in range(min(n_agents, sub_sample_thresh))
            ]
            _, vpred = pi.act(stochastic, ob)
            ep_rets = []
            ep_lens = []
            time_steps = []
        i = t % horizon
        time_steps.append(t)
        obs[i] = [ob[ssi] for ssi in sub_sample_idx] if sub_sample else ob
        vpreds[i] = vpred[sub_sample_idx] if sub_sample else vpred
        news[i] = new
        acs[i] = ac[sub_sample_idx] if sub_sample else ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew[sub_sample_idx] if sub_sample else rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    new = [np.append(p["new"], 0) for p in seg]
    vpred = [np.append(p["vpred"], p["nextvpred"]) for p in seg]

    for i, p in enumerate(seg):
        T = len(p["rew"])
        p["adv"] = gaelam = np.empty(T, 'float32')
        rew = p["rew"]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - new[i][t + 1]
            delta = rew[t] + gamma * vpred[i][t + 1] * nonterminal - vpred[i][t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        p["tdlamret"] = p["adv"] + p["vpred"]


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


# -------------------------------------------------------------------------- #
# ------------------------ learn_with_checkpoints() ------------------------ #
# -------------------------------------------------------------------------- #

def learn_with_checkpoints(env, policy_fn, *,
                           timesteps_per_batch, max_kl, cg_iters,
                           gamma, lam, entcoeff=0.0, cg_damping=1e-2,
                           vf_stepsize=3e-4, vf_iters=3, max_timesteps=0, max_episodes=0, max_iters=0,
                           callback=None, act_wrapper=None, episodes_so_far=0, timesteps_so_far=0, iters_so_far=0):
    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)

    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)
    oldpi = policy_fn("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])
    ret = tf.placeholder(dtype=tf.float32, shape=[None])

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = entcoeff * meanent

    vferr = tf.reduce_mean(tf.square(pi.vpred - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    var_list.extend([v for v in all_var_list if v.name.split("/")[1].startswith("me")])
    vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start + sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g * tangent) for (g, tangent) in zipsame(klgrads, tangents)])
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in
                                                    zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    act_params = {
        'name': "pi",
        'ob_space': ob_space,
        'ac_space': ac_space,
    }

    # -------------------------------------------------------------------------- #
    # Checks if there is an ActWrapper pickle, if no creates new ActWrapper here #
    # -------------------------------------------------------------------------- #

    if act_wrapper is None:
        act_wrapper = ActWrapper(pi, act_params)
    # -------------------------------------------------------------------------- #

    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    seg_gen = traj_segment_generator(act_wrapper, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40)
    rewbuffer = deque(maxlen=40)

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1

    while True:
        # -------------------------------------------------------------------------- #
        # Checks if model has been paused if True then model is saved and then break #
        # -------------------------------------------------------------------------- #
        # if check_model_status() == "paused":
        #      save_checkpoint(act_wrapper, episodes_so_far, timesteps_so_far, iters_so_far)
        #      break
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************" % iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        ob = np.concatenate([s['ob'] for s in seg], axis=0)
        ac = np.concatenate([s['ac'] for s in seg], axis=0)
        atarg = np.concatenate([s['adv'] for s in seg], axis=0)
        tdlamret = np.concatenate([s['tdlamret'] for s in seg], axis=0)
        vpredbefore = np.concatenate([s["vpred"] for s in seg], axis=0)
        atarg = (atarg - atarg.mean()) / atarg.std()

        args = ob, ac, atarg
        fvpargs = [arr[::5] for arr in args]

        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        assign_old_eq_new()
        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*args)
        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)
        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank == 0)
            assert np.isfinite(stepdir).all()
            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                improve = surr - surrbefore
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum()))
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])


        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)

        with timed("vf"):
            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((ob, tdlamret), include_final_partial_batch=False, batch_size=64):
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg[0]["ep_lens"], seg[0]["ep_rets"])
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

    if rank == 0:
        logger.dump_tabular()
