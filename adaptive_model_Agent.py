import numpy as np
from src import agent
from tree_model_based import Node, Tree


class AdaptiveModelBasedDiscretization(agent.FiniteHorizonAgent):

    def __init__(self, epLen, numIters, scaling):
        '''args:
            epLen - number of steps per episode
            numIters - total number of iterations
            scaling - scaling parameter for UCB term
        '''

        self.epLen = epLen
        self.numIters = numIters
        self.scaling = scaling

        # List of tree's, one for each step
        self.tree_list = []

        # Makes a new partition for each step and adds it to the list of trees
        for h in range(epLen):
            # print(h)
            tree = Tree(epLen)
            self.tree_list.append(tree)

    def reset(self):
        # Resets the agent by setting all parameters back to zero
        # List of tree's, one for each step
        self.tree_list = []

        # Makes a new partition for each step and adds it to the list of trees
        for h in range(self.epLen):
            tree = Tree(self.epLen)
            self.tree_list.append(tree)

        # Gets the number of arms for each tree and adds them together
    def get_num_arms(self):
        total_size = 0
        for tree in self.tree_list:
            total_size += tree.get_number_of_active_balls()
        return total_size

    def update_obs(self, obs, action, reward, newObs, timestep):
        '''Add observation to records'''
        print('Updating observations at step: ' + str(timestep))
        # print('Old state: ' + str(obs) + ' action: ' + str(action) + ' newState: ' + str(newObs))
        # Gets the active trees based on current timestep
        tree = self.tree_list[timestep]

        # Gets the active ball by finding the argmax of Q values of relevant
        active_node, _ = tree.get_active_ball(obs)

        active_node.num_visits += 1
        t = active_node.num_visits
        # print('Num visits: ' + str(t))
        # Update empirical estimate of average reward for that node
        active_node.rEst = ((t-1)*active_node.rEst + reward) / t


        if timestep == self.epLen - 1:
            vEst = 0

        else:
            next_tree = self.tree_list[timestep+1]
            # update transition kernel based off of new transition
            # print(next_tree.state_leaves)
            new_obs_loc = np.argmin(np.abs(np.asarray(next_tree.state_leaves) - newObs))
            active_node.pEst[new_obs_loc] += 1
            transition = np.asarray(active_node.pEst)
            # print(transition / np.sum(transition))
            # print(next_tree.vEst)
            # print(np.dot(np.asarray(next_tree.vEst), transition / np.sum(transition)))
            vEst = min(self.epLen, np.dot(np.asarray(next_tree.vEst), transition / np.sum(transition)))

        bonus = self.scaling*np.sqrt(1/t)
        active_node.qVal = vEst + active_node.rEst + bonus

        '''determines if it is time to split the current ball'''
        if t >= 4**active_node.num_splits:
            if timestep >= 1:
                children = tree.split_node(active_node, timestep, self.tree_list[timestep-1])
            else:
                children = tree.split_node(active_node, timestep, None)

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        self.greedy = self.greedy
        # Should update value function estimate for induced leaves at this step
        # for tree in self.tree_list:
        #     index = 0
        #     for state_val in tree.state_leaves:
        #         _, qMax = get_active_ball(state_val)
        #         tree.vEst[index] = min(self.epLen, qMax)
        #         index += 1

        pass

    def split_ball(self, node):
        children = self.node.split_ball()
        pass

    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to a greedy policy

        Args:
            state - int - current state
            timestep - int - timestep *within* episode

        Returns:
            action - int
        '''
        # Considers the partition of the space for the current timestep
        tree = self.tree_list[timestep]

        # Gets the selected ball
        active_node, qVal = tree.get_active_ball(state)

        # Picks an action uniformly in that ball
        action = np.random.uniform(active_node.action_val - active_node.radius, active_node.action_val + active_node.radius)

        return action

    def pick_action(self, state, timestep):
        action = self.greedy(state, timestep)
        return action
