import json
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from matplotlib.font_manager import FontProperties  # 导入字体模块

def chinese_font():
    ''' 设置中文字体，注意需要根据自己电脑情况更改字体路径，否则还是默认的字体
    '''
    try:
        font = FontProperties(
        fname='C:\Windows\Fonts\AdobeHeitiStd-Regular.otf', size=15) # fname系统字体路径，此处是mac的
    except:
        font = None
    return font

def plot_rewards_cn(rewards, ma_rewards, plot_cfg, tag='train'):
    ''' 中文画图
    '''
    sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的学习曲线".format(plot_cfg.env_name,
              plot_cfg.algo_name), fontproperties=chinese_font())
    plt.xlabel(u'回合数', fontproperties=chinese_font())
    plt.plot(rewards)
    plt.plot(ma_rewards)
    plt.legend((u'奖励', u'滑动平均奖励',), loc="best", prop=chinese_font())
    if plot_cfg.save:
        plt.savefig(plot_cfg.result_path+f"{tag}_rewards_curve_cn")
    # plt.show()

    
def plot_rewards(rewards, plot_cfg, tag='train', instance=''):
    sns.set()
    plt.figure()  # create a figure instance, so that we can easily plot many pictures
    plt.title("learning curve on {} of {} for {}".format(
        plot_cfg.device, plot_cfg.algo_name, instance))
    plt.xlabel('epsiodes')
    data = [pd.DataFrame() for _ in range(plot_cfg.n_envs)]
    for i in range(plot_cfg.n_envs):
        data[i]['step'] = np.linspace(1, plot_cfg.n_sample, plot_cfg.n_sample)
        data[i]['reward'] = np.array(rewards[i])
    df = pd.concat(data, axis=0, ignore_index=True)
    df.to_csv(plot_cfg.data_path + 'reward_df.csv')
    sns.lineplot(data=df, x='step', y='reward')
    if plot_cfg.save_fig:
        plt.savefig(plot_cfg.result_path+"{}_rewards_curve".format(tag))
    plt.show()
    
    
def plot_makespans(makespans, plot_cfg, tag='train', instance=''):
    sns.set()
    plt.figure()  # create a figure instance, so that we can easily plot many pictures
    plt.title("makespan curve on {} of {} for {}".format(
        plot_cfg.device, plot_cfg.algo_name, instance))
    plt.xlabel('epsiodes')
    data = [pd.DataFrame() for _ in range(plot_cfg.n_envs)]
    for i in range(plot_cfg.n_envs):
        data[i]['step'] = np.linspace(1, plot_cfg.n_sample, plot_cfg.n_sample)
        data[i]['makespan'] = np.array(makespans[i])
    df = pd.concat(data, axis=0, ignore_index=True)
    df.to_csv(plot_cfg.data_path + 'makespan_df.csv')
    sns.lineplot(data=df, x='step', y='makespan')
    if plot_cfg.save_fig:
        plt.savefig(plot_cfg.result_path+"{}_makespans_curve".format(tag))
    plt.show()
    
    
def plot_losses(losses, algo="DQN", save=True, path='./'):
    sns.set()
    plt.figure()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('epsiodes')
    plt.plot(losses, label='rewards')
    plt.legend()
    if save:
        plt.savefig(path+"losses_curve")
    plt.show()
    
    
def save_results_1(dic, tag='train', path='./results'):
    ''' save rewards
    '''
    for key,value in dic.items():
        np.save(path+'{}_{}.npy'.format(tag,key),value)
    print('Results saved!')
    
    
def save_results(rewards, ma_rewards, tag='train', path='./results'):
    ''' 保存奖励
    '''
    np.save(path+'{}_rewards.npy'.format(tag), rewards)
    np.save(path+'{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('结果保存完毕！')
    
    
def save_args(args,path=None):
    # 保存参数   
    args_dict = vars(args)  
    Path(path).mkdir(parents=True, exist_ok=True) 
    with open(f"{path}/params.json", 'w') as fp:
        json.dump(args_dict, fp)   
    print("Parameters saved!")



def make_dir(*paths):
    ''' create a folder
    '''
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def del_empty_dir(*paths):
    ''' delete all empty folders under this dir
    '''
    for path in paths:
        dirs = os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path, dir)):
                os.removedirs(os.path.join(path, dir))
