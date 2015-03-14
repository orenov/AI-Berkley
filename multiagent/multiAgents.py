# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        currentPos = currentGameState.getPacmanPosition()
        currentFood = currentGameState.getFood()
        currentGhostStates = currentGameState.getGhostStates()
        currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]

        additional_score = 0

        current_distance_to_food = float("inf")
        current_distance_to_ghost = float("inf")
        new_distance_to_food = float("inf")
        new_distance_to_ghost = float("inf")

        for food in newFood.asList():
          d = manhattanDistance(food, newPos)
          new_distance_to_food = min([new_distance_to_food, d])
        

        pos_x, pos_y = newPos
        
        if currentFood[pos_x][pos_y]:
          additional_score += 10

        food_before_action = len(currentFood.asList())
        food_after_action = len(newFood.asList())
        
        for ghost in successorGameState.getGhostPositions():
          d = manhattanDistance(newPos, ghost)
          new_distance_to_ghost = min([new_distance_to_ghost, d])

        score = 1.0/new_distance_to_food - food_after_action + additional_score 

        if new_distance_to_ghost < 2:
          score -= 500

        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def maxvalue(self ,state, agentIndex, currentdepth):
      v = (float("-inf"), "Stop")
      for action in state.getLegalActions(agentIndex):
        v = max([v, (self.value(state.generateSuccessor(agentIndex, action), (currentdepth+1) % self.number_of_agents, currentdepth+1), action)], key=lambda item:item[0])
      return v

    def minvalue(self, state, agentIndex, currentdepth):
      v = (float("inf"), "Stop")
      for action in state.getLegalActions(agentIndex):
        v = min([v, (self.value(state.generateSuccessor(agentIndex, action), (currentdepth+1) % self.number_of_agents, currentdepth+1), action)], key=lambda item:item[0])
      return v

    def value(self, state, agentIndex, currentdepth):
      if state.isLose() or state.isWin() or currentdepth >= self.depth*self.number_of_agents:
        return self.evaluationFunction(state)
      if (agentIndex == 0):
        return self.maxvalue(state, agentIndex, currentdepth)[0]  
      else:
        return self.minvalue(state, agentIndex, currentdepth)[0]


    def getAction(self, gameState):
        self.number_of_agents = gameState.getNumAgents()
        path2 = self.maxvalue(gameState,0,0)
        return path2[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def maxvalue(self ,state, agentIndex, currentdepth, alpha, beta):
      v = (float("-inf"), "Stop")
      for action in state.getLegalActions(agentIndex):
        v = max([v, (self.value(state.generateSuccessor(agentIndex, action), (currentdepth+1) % self.number_of_agents, currentdepth+1, alpha, beta), action)], key=lambda item:item[0])
        if v[0] > beta:
          return v
        alpha = max(alpha, v[0])
      return v

    def minvalue(self, state, agentIndex, currentdepth, alpha, beta):
      v = (float("inf"), "Stop")
      for action in state.getLegalActions(agentIndex):
        v = min([v, (self.value(state.generateSuccessor(agentIndex, action), (currentdepth+1) % self.number_of_agents, currentdepth+1, alpha, beta), action)], key=lambda item:item[0])
        if v[0] < alpha:
          return v
        beta = min(beta, v[0])
      return v

    def value(self, state, agentIndex, currentdepth, alpha, beta):
      if state.isLose() or state.isWin() or currentdepth >= self.depth*self.number_of_agents:
        return self.evaluationFunction(state)
      if (agentIndex == 0):
        return self.maxvalue(state, agentIndex, currentdepth, alpha, beta)[0]  
      else:
        return self.minvalue(state, agentIndex, currentdepth, alpha, beta)[0]

    def getAction(self, gameState):
        self.number_of_agents = gameState.getNumAgents()
        alpha = float("-inf")
        beta  = float("inf")
        path2 = self.maxvalue(gameState,0,0,alpha,beta)
        return path2[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def maxvalue(self ,state, agentIndex, currentdepth):
      v = (float("-inf"), "Stop")
      for action in state.getLegalActions(agentIndex):
        v = max([v, (self.value(state.generateSuccessor(agentIndex, action), (currentdepth+1) % self.number_of_agents, currentdepth+1), action)], key=lambda item:item[0])
      return v

    def expectedvalue(self, state, agentIndex, currentdepth):
      v = [0.0]
      for action in state.getLegalActions(agentIndex):
        v.append(self.value(state.generateSuccessor(agentIndex, action), (currentdepth+1) % self.number_of_agents, currentdepth+1))
      return sum(v)/(len(v)-1)

    def value(self, state, agentIndex, currentdepth):
      if state.isLose() or state.isWin() or currentdepth >= self.depth*self.number_of_agents:
        return self.evaluationFunction(state)
      if (agentIndex == 0):
        return self.maxvalue(state, agentIndex, currentdepth)[0]  
      else:
        return self.expectedvalue(state, agentIndex, currentdepth)


    def getAction(self, gameState):
        self.number_of_agents = gameState.getNumAgents()
        path2 = self.maxvalue(gameState,0,0)
        return path2[1]

def betterEvaluationFunction(currentGameState):
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    capsulesPos = currentGameState.getCapsules()
    distance_to_food_list = []
    distance_to_ghost_list = []
    distance_to_capsules_list = []
    score = 0
    additional_score = 0
    for food in currentFood.asList():
        distance_to_food_list.append(manhattanDistance(food, currentPos))
    for food in capsulesPos:
      distance_to_capsules_list.append(manhattanDistance(food, currentPos))

    legalMoves = len(currentGameState.getLegalActions(0)) +1

    food_state = len(currentFood.asList())
        
    for ghost in currentGameState.getGhostPositions():
        distance_to_ghost_list.append(manhattanDistance(currentPos, ghost))

    if newScaredTimes[0] != 0:
      score += 40000.0/min(distance_to_ghost_list)
    elif min(distance_to_ghost_list) < 2:
      score -= 500
    elif min(distance_to_capsules_list +[float("inf")]) < 7:
      score += 50000.0/min(distance_to_capsules_list)
    mmax = max(distance_to_food_list + [0.0]) 
    mmin = min(distance_to_food_list + [float(10000)]) 
    harmonic = 0.001 + 2*mmax *mmin/(mmax + mmin)
    #score = currentGameState.getScore() - sum(distance_to_food_list) + 1.0/min([distance_to_food_list, float("inf")]) - food_state
    score = -100* len(capsulesPos) + currentGameState.getScore() - mmax - mmin + 50.0/harmonic + 90.0/sum(distance_to_food_list+[0.001]) - 100*food_state + 10.0/min([distance_to_food_list, float("inf")]) + legalMoves
    
    if food_state == 0:
      score += 2000

    return score

# Abbreviation
better = betterEvaluationFunction

