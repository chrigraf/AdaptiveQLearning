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

ambulance_list = ['uniform', 'beta']
param_list_ambulance = ['0', '1', '25']

for problem in ambulance_list:
    for param in param_list_ambulance:

        epLen = 5
        nEps = 2000
        numIters = 10
        if problem == 'beta':
            def arrivals(step):
                return np.random.beta(5,2)
        else:
            def arrivals(step):
                return np.random.uniform(0,1)
        if param == '1':
            alpha = 1
        elif param == '0':
            alpha = 0
        else:
            alpha = 0.25

        starting_state = 0.5

        env = environment.make_ambulanceEnvMDP(epLen, arrivals, alpha, starting_state)

        ##### PARAMETER TUNING FOR AMBULANCE ENVIRONMENT

        scaling_list = [0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1, 1.5, 2, 5]
        # scaling_list = [0.5, .01] # alpha = 1
        # scaling_list = [1, .4] # alpha = 0
        # scaling_list = [0.5, 0.01] # alpha = 0.25
        max_reward_adapt = 0
        max_reward_e_net = 0
        opt_adapt_scaling = 0.01
        opt_e_net_scaling = 0.01

        # TRYING OUT EACH SCALING FACTOR FOR OPTIMAL ONE
        for scaling in scaling_list:

            # RUNNING EXPERIMENT FOR ADAPTIVE ALGORITHM
            #
            #
            agent_list_adap = []
            for _ in range(numIters):
                agent_list_adap.append(AdaptiveModelBasedDiscretization(epLen, nEps, scaling))
            #
            dict = {'seed': 1, 'epFreq' : 1, 'targetPath': './tmp.csv', 'deBug' : False, 'nEps': nEps, 'recFreq' : 10, 'numIters' : numIters}
            #
            exp = experiment.Experiment(env, agent_list_adap, dict)
            adap_fig = exp.run()
            dt_adapt_data = exp.save_data()

            if (dt_adapt_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0] > max_reward_adapt:
                max_reward_adapt = (dt_adapt_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0]
                opt_adapt_scaling = scaling
                dt_adapt = dt_adapt_data
                opt_adapt_agent_list = agent_list_adap

            # RUNNING EXPERIMENT FOR EPSILON NET ALGORITHM

            epsilon = (nEps * epLen)**(-1 / 4)
            action_net = np.arange(start=0, stop=1, step=epsilon)
            state_net = np.arange(start=0, stop=1, step=epsilon)

            agent_list = []
            for _ in range(numIters):
                agent_list.append(eNetModelBased(action_net, state_net, epLen, scaling, 0))

            exp = experiment.Experiment(env, agent_list, dict)
            exp.run()
            dt_net_data = exp.save_data()

            if (dt_net_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0] > max_reward_e_net:
                max_reward_e_net = (dt_net_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0]
                opt_e_net_scaling = scaling
                dt_net = dt_net_data

        print(problem)
        print(param)
        print(opt_adapt_scaling)
        print(opt_e_net_scaling)




        # SAVING DATA TO CSV

        dt_adapt.to_csv('./data/ambulance_'+problem+'_adapt_model_'+param+'.csv')
        dt_net.to_csv('./data/ambulance_'+problem+'_net_model_'+param+'.csv')
