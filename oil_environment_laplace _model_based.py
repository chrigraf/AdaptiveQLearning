import gym
import numpy as np
from adaptive_Agent import AdaptiveDiscretization
from eNet_model_Agent import eNetModelBased
from eNet_Agent import eNet
from adaptive_model_Agent import AdaptiveModelBasedDiscretization
from data_Agent import dataUpdateAgent
from src import environment
from src import experiment
from src import agent
import pickle


''' Defining parameters to be used in the experiment'''

epLen = 5
nEps = 5000
numIters = 25

lam = 1
starting_state = 0.5
env = environment.makeLaplaceOil(epLen, lam, starting_state)


##### PARAMETER TUNING FOR OIL ENVIRONMENT
scaling_list = [0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1, 1.25, 1.5, 1.75, 2, 5]
# scaling_list = [0.6, 0.5] # lam = 1
# scaling_list = [2, 0.6] # lam = 10
# scaling_list = [0.75, 0.25] # lam = 50
max_reward_adapt = 0
max_reward_e_net = 0
opt_adapt_scaling = 0.01
opt_e_net_scaling = 0.01

# Tries each of the scaling to find the optimal one
for scaling in scaling_list:

    #### RUNNING EXPERIMENTS ON ADAPTIVE ALGORITHM
    #
    # agent_list_adap = []
    # for _ in range(numIters):
    #     agent_list_adap.append(AdaptiveModelBasedDiscretization(epLen, nEps, scaling))

    dict = {'seed': 1, 'epFreq' : 1, 'targetPath': './tmp.csv', 'deBug' : False, 'nEps': nEps, 'recFreq' : 10, 'numIters' : numIters}
    #
    # exp = experiment.Experiment(env, agent_list_adap, dict)
    # adap_fig = exp.run()
    # dt_adapt_data = exp.save_data()
    #
    # if (dt_adapt_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0] >= max_reward_adapt:
    #     max_reward_adapt = (dt_adapt_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0]
    #     opt_adapt_scaling = scaling
    #     dt_adapt = dt_adapt_data
    #     opt_adapt_agent_list = agent_list_adap
    #
    #### RUNNING EXPERIMENTS ON UNIFORM MESH ALGORITHM

    epsilon = (nEps * epLen)**(-1 / 4)
    action_net = np.arange(start=0, stop=1, step=epsilon)
    state_net = np.arange(start=0, stop=1, step=epsilon)

    agent_list = []
    for _ in range(numIters):
        agent_list.append(eNet(action_net, state_net, epLen, scaling))

    exp = experiment.Experiment(env, agent_list, dict)
    exp.run()
    dt_net_data = exp.save_data()

    if (dt_net_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0] >= max_reward_e_net:
        max_reward_e_net = (dt_net_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0]
        opt_e_net_scaling = scaling
        dt_net = dt_net_data
        opt_e_net_agent = agent_list

print(opt_adapt_scaling)
print(opt_e_net_scaling)


##### SAVING DATA TO CSV FOR PLOTTING


dt_adapt.to_csv('oil_laplace_adapt_50.csv')
dt_net.to_csv('oil_laplace_net_50.csv')
agent = opt_adapt_agent_list[-1]
filehandler = open('oil_laplace_agent_50.obj', 'wb')
pickle.dump(agent, filehandler)
