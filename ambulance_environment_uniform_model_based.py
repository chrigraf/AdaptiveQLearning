import gym
import numpy as np
from adaptive_model_Agent import AdaptiveModelBasedDiscretization
from src import environment
from src import experiment
from src import agent
import pickle


''' Defining parameters to be used in the experiment'''

epLen = 5
nEps = 2000
numIters = 50
loc = 0.8+np.pi/60
scale = 2
def arrivals():
    return np.random.uniform(0,1)

alpha = 0.25
starting_state = 0.5

env = environment.make_ambulanceEnvMDP(epLen, arrivals, alpha, starting_state)


##### PARAMETER TUNING FOR AMBULANCE ENVIRONMENT

scaling_list = [0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1, 1.5, 2, 5]
# scaling_list = [0.25, .01] # alpha = 0.25
# scaling_list = [0.25, .1] # alpha = 1
# scaling_list = [0.1] # alpha = 0
max_reward_adapt = 0
max_reward_e_net = 0
opt_adapt_scaling = 0.01

# TRYING OUT EACH SCALING FACTOR FOR OPTIMAL ONE
for scaling in scaling_list:

    # RUNNING EXPERIMENT FOR ADAPTIVE ALGORITHM

    agent_list_adap = []
    for _ in range(numIters):
        agent_list_adap.append(AdaptiveModelBasedDiscretization(epLen, nEps, scaling))

    dict = {'seed': 1, 'epFreq' : 1, 'targetPath': './tmp.csv', 'deBug' : False, 'nEps': nEps, 'recFreq' : 10, 'numIters' : numIters}

    exp = experiment.Experiment(env, agent_list_adap, dict)
    adap_fig = exp.run()
    dt_adapt_data = exp.save_data()

    if (dt_adapt_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0] > max_reward_adapt:
        max_reward_adapt = (dt_adapt_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0]
        opt_adapt_scaling = scaling
        dt_adapt = dt_adapt_data
        opt_adapt_agent_list = agent_list_adap

print(opt_adapt_scaling)



# SAVING DATA TO CSV

dt_adapt.to_csv('ambulance_uniform_adapt_25.csv')
agent = opt_adapt_agent_list[-1]
filehandler = open('ambulance_uniform_agent_25.obj', 'wb')
pickle.dump(agent, filehandler)
