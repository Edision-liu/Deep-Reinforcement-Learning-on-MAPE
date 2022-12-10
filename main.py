import os
import json
import datetime
import numpy as np
import torch
import utils
import random
from copy import deepcopy
from arguments import get_args
from tensorboardX import SummaryWriter
from eval import evaluate
from learner import setup_master
from pprint import pprint
import warnings                 #!!!!!!!!
warnings.filterwarnings("ignore", category=UserWarning)    #!!!!!!!!
np.set_printoptions(suppress=True, precision=4)


def train(args, return_early=False):
    print("Start train function")
    writer = SummaryWriter(args.log_dir)    #记录
    envs = utils.make_parallel_envs(args)    #创建环境
    master = setup_master(args)     #
    # used d    uring evaluation only
    eval_master, eval_env = setup_master(args, return_env=True)
    obs = envs.reset()  # shape - num_processes x num_agents x obs_dim
    master.initialize_obs(obs)
    n = len(master.all_agents)
    episode_rewards = torch.zeros([args.num_processes, n], device=args.device)
    final_rewards = torch.zeros([args.num_processes, n], device=args.device)

    # start simulations
    start = datetime.datetime.now()
    print(start)
    for j in range(args.num_updates):      # 12207
        for step in range(args.num_steps):    # 128  //episode
            with torch.no_grad():
                actions_list = master.act(step)     # step=0~127   ，此时的action_list为每个智能体即将要做的行为，
                                                # 此后智能体action属性就已经被更改了，但是期望不被更改，而是更改为更有利的action
            agent_actions = np.transpose(np.array(actions_list), (1, 0, 2))
            # print(agent_actions)
            obs, reward, done, info = envs.step(agent_actions)    # 每个返回的参数都是n个智能体的
            # print(reward)
            reward = torch.from_numpy(np.stack(reward)).float().to(args.device)    # ???????
            episode_rewards += reward
            masks = torch.FloatTensor(1 - 1.0 * done).to(args.device)  # ??????
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            master.update_rollout(obs, reward, masks)

        master.wrap_horizon()
        return_vals = master.update()
        value_loss = return_vals[:, 0]
        action_loss = return_vals[:, 1]
        dist_entropy = return_vals[:, 2]
        master.after_update()

        if j % args.save_interval == 0 and not args.test:
            savedict = {'models': [agent.actor_critic.state_dict() for agent in master.all_agents]}
            ob_rms = (None, None) if envs.ob_rms is None else (envs.ob_rms[0].mean, envs.ob_rms[0].var)
            savedict['ob_rms'] = ob_rms
            savedir = args.save_dir + '/ep' + str(j) + '.pt'
            torch.save(savedict, savedir)




        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0:
            end = datetime.datetime.now()
            seconds = (end - start).total_seconds()
            mean_reward = final_rewards.mean(dim=0).cpu().numpy()
            print(
                "Updates {} | Num timesteps {} | Time {} | FPS {}\nMean reward {}\nEntropy {:.4f} Value loss {:.4f} Policy loss {:.4f}\n".
                format(j, total_num_steps, str(end - start), int(total_num_steps / seconds),
                       mean_reward, dist_entropy[0], value_loss[0], action_loss[0]))
            if not args.test:
                for idx in range(n):
                    writer.add_scalar('agent' + str(idx) + '/training_reward', mean_reward[idx], j) # 第一个参数可以简单理解为保存图的名称，第二个参数是可以理解为Y轴数据，第三个参数可以理解为X轴数据。当Y轴数据不止一个时，可以使用writer.add_scalars().

                writer.add_scalar('all/value_loss', value_loss[0], j)
                writer.add_scalar('all/action_loss', action_loss[0], j)
                writer.add_scalar('all/dist_entropy', dist_entropy[0], j)

        if args.eval_interval is not None and j % args.eval_interval == 0:
            ob_rms = (None, None) if envs.ob_rms is None else (envs.ob_rms[0].mean, envs.ob_rms[0].var)
            print('===========================================================================================')
            _, eval_perstep_rewards, final_min_dists, num_success, eval_episode_len = evaluate(args, None,
                                                                                               master.all_policies,
                                                                                               ob_rms=ob_rms,
                                                                                               env=eval_env,
                                                                                               master=eval_master)
            print('Evaluation {:d} | Mean per-step reward {:.2f}'.format(j // args.eval_interval,
                                                                         eval_perstep_rewards.mean()))
            print('Num success {:d}/{:d} | Episode Length {:.2f}'.format(num_success, args.num_eval_episodes,
                                                                         eval_episode_len))
            if final_min_dists:
                print('Final_dists_mean {}'.format(np.stack(final_min_dists).mean(0)))
                print('Final_dists_var {}'.format(np.stack(final_min_dists).var(0)))
            print('===========================================================================================\n')

            if not args.test:
                writer.add_scalar('all/eval_success', 100.0 * num_success / args.num_eval_episodes, j)
                writer.add_scalar('all/episode_length', eval_episode_len, j)
                for idx in range(n):
                    writer.add_scalar('agent' + str(idx) + '/eval_per_step_reward', eval_perstep_rewards.mean(0)[idx],
                                      j)
                    if final_min_dists:
                        writer.add_scalar('agent' + str(idx) + '/eval_min_dist', np.stack(final_min_dists).mean(0)[idx],
                                          j)

            curriculum_success_thres = 0.1     # 0.9
            if return_early and num_success * 1. / args.num_eval_episodes > curriculum_success_thres:
                savedict = {'models': [agent.actor_critic.state_dict() for agent in master.all_agents]}
                ob_rms = (None, None) if envs.ob_rms is None else (envs.ob_rms[0].mean, envs.ob_rms[0].var)
                savedict['ob_rms'] = ob_rms
                savedir = args.save_dir + '/ep' + str(j) + '.pt'
                torch.save(savedict, savedir)
                print('===========================================================================================\n')
                print('{} agents: training complete. Breaking.\n'.format(args.num_agents))
                print('===========================================================================================\n')
                break

    writer.close()
    if return_early:
        return savedir


if __name__ == '__main__':
    args = get_args()
    if args.seed is None:
        args.seed = random.randint(0, 10000)
    args.num_updates = args.num_frames // args.num_steps // args.num_processes             # //来表示整数除法，返回不大于结果的一个最大的整数
    torch.manual_seed(args.seed)    #设置 (CPU) 生成随机数的种子，并返回一个torch.Generator对象。 设置种子的意思是一旦固定种子，每次生成随机数都将从这个种子开始搜寻。
    torch.set_num_threads(1)        #线程设置  default 1
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    pprint(vars(args))         # vars返回对象object的属性和属性值的字典对象
    if not args.test:
        with open(os.path.join(args.save_dir, 'params.json'), 'w') as f:
            params = deepcopy(vars(args))
            params.pop('device')
            json.dump(params, f)
    train(args)
