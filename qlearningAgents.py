# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from backend import ReplayMemory

import backend
import gridworld


import random,util,math
import numpy as np
import copy

class QLearningAgent(ReinforcementAgent):
    def __init__(self, **args):
        """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
        ReinforcementAgent.__init__(self, **args)
        self.qValues = util.Counter()  # Dictionary-like structure for Q-values

    def getQValue(self, state, action):
        """
          Returns Q(state, action)
          Should return 0.0 if we have never seen a state-action pair
        """
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state, action)
          where the max is over legal actions.
          If there are no legal actions, return 0.0.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        return max(self.getQValue(state, action) for action in legalActions)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.
          If there are no legal actions, return None.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        bestValue = self.computeValueFromQValues(state)
        bestActions = [action for action in legalActions if self.getQValue(state, action) == bestValue]
        return random.choice(bestActions)  # Break ties randomly

    def getAction(self, state):
        """
          Compute the action to take in the current state.
          With probability epsilon, take a random action.
          Otherwise, take the best policy action.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward: float):
        """
          Perform Q-value update using the observed transition.
        """
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.qValues[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue and update.
       All other QLearningAgent functions should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()  # Weights for the features

    def getQValue(self, state, action):
        """
          Returns Q(state, action) = w * featureVector
          where * is the dot product operator.
        """
        features = self.featExtractor.getFeatures(state, action)
        return sum(self.weights[feature] * value for feature, value in features.items())

    def update(self, state, action, nextState, reward: float):
        """
          Update weights based on transition.
        """
        features = self.featExtractor.getFeatures(state, action)
        correction = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        
        for feature, value in features.items():
            self.weights[feature] += self.alpha * correction * value

    def final(self, state):
        """Called at the end of each game."""
        PacmanQAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            # Optionally print weights for debugging
            print("Final weights:", self.weights)
