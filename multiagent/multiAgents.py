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
        #print successorGameState
        #print newPos
        #print newFood
        #print newGhostStates
        #print newScaredTimes
        "*** YOUR CODE HERE ***"
        oldFood=currentGameState.getFood()
        foodList=oldFood.asList()
        maxdistance=-1
        newPos=list(newPos)
        if action=='Stop':
            return -1
        for state in newGhostStates:
            if state.getPosition() == tuple(newPos) and (state.scaredTimer == 0):
                return -1
        for food in foodList:
            if manhattanDistance(food,newPos)!=0:
                distance=1.0/manhattanDistance(food,newPos)
                #print distance
                if distance>maxdistance:
                    maxdistance=distance
            else:
                maxdistance=2
        return maxdistance

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

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def minmax_decision(gameState,deepness,agentIndex):
            if agentIndex>=gameState.getNumAgents():
                agentIndex=0
                deepness+=1
            if (deepness==self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            elif agentIndex==0:
                return max_value(gameState,deepness)
            else:
                return min_value(gameState,deepness,agentIndex)
        def min_value(gamestate,deepness,agentIndex):
            output = ["meow", float("inf")]
            ghostactions=gamestate.getLegalActions(agentIndex)
            if not ghostactions:
                return self.evaluationFunction(gamestate)
            for action in ghostactions:
                newstate=gamestate.generateSuccessor(agentIndex,action)
                value=minmax_decision(newstate,deepness,agentIndex+1)
                if type(value) is list:
                    testval=value[1]
                else:
                    testval=value
                if testval<output[1]:
                    output=[action,testval]
            return output
        def max_value(gamestate,deepness):
            output = ["meow", -float("inf")]
            pactions=gamestate.getLegalActions(0)
            if not pactions:
                return self.evaluationFunction(gamestate)
            for action in pactions:
                newstate=gamestate.generateSuccessor(0,action)
                value=minmax_decision(newstate,deepness,1)
                if type(value) is list:
                    testval=value[1]
                else:
                    testval=value
                if testval>output[1]:
                    output=[action,testval]
            return output
        outputList = minmax_decision(gameState, 0, 0)
        return outputList[0]
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minmax_decision(gameState,deepness,agentIndex,alpha,beta):
            if agentIndex>=gameState.getNumAgents():
                agentIndex=0
                deepness+=1
            if (deepness==self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            elif agentIndex==0:
                return max_value(gameState,deepness,alpha,beta)
            else:
                return min_value(gameState,deepness,agentIndex,alpha,beta)
        def min_value(gamestate,deepness,agentIndex,alpha,beta):
            output = ["meow", float("inf")]
            ghostactions=gamestate.getLegalActions(agentIndex)
            if not ghostactions:
                return self.evaluationFunction(gamestate)
            for action in ghostactions:
                newstate=gamestate.generateSuccessor(agentIndex,action)
                value=minmax_decision(newstate,deepness,agentIndex+1,alpha,beta)
                if type(value) is list:
                    testval=value[1]
                else:
                    testval=value
                if testval<output[1]:
                    output=[action,testval]
                if testval<alpha:
                    return [action,testval]
                beta=min(beta,testval)#get the minimum of maxnode
            return output
        def max_value(gamestate,deepness,alpha,beta):
            output = ["meow", -float("inf")]
            pactions=gamestate.getLegalActions(0)
            if not pactions:
                return self.evaluationFunction(gamestate)
            for action in pactions:
                newstate=gamestate.generateSuccessor(0,action)
                value=minmax_decision(newstate,deepness,1,alpha,beta)
                if type(value) is list:
                    testval=value[1]
                else:
                    testval=value
                if testval>output[1]:
                    output=[action,testval]
                if testval>beta:
                    return [action,testval]
                alpha=max(alpha,testval)#get the maximum of minnode
            return output
        outputList = minmax_decision(gameState, 0, 0,-float("inf"),float("inf"))
        return outputList[0]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def minmax_decision(gameState,deepness,agentIndex):
            if agentIndex>=gameState.getNumAgents():
                agentIndex=0
                deepness+=1
            if (deepness==self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            elif agentIndex==0:
                return max_value(gameState,deepness)
            else:
                return min_value(gameState,deepness,agentIndex)
        def min_value(gamestate,deepness,agentIndex):
            output = ["meow", 0]
            ghostactions=gamestate.getLegalActions(agentIndex)
            if not ghostactions:
                return self.evaluationFunction(gamestate)
            probability = 1.0/len(ghostactions)
            for action in ghostactions:
                newstate=gamestate.generateSuccessor(agentIndex,action)
                value=minmax_decision(newstate,deepness,agentIndex+1)
                if type(value) is list:
                    testval=value[1]
                else:
                    testval=value
                output[0]=action
                output[1]+=testval*probability
            return output
        def max_value(gamestate,deepness):
            output = ["meow", -float("inf")]
            pactions=gamestate.getLegalActions(0)
            if not pactions:
                return self.evaluationFunction(gamestate)
            for action in pactions:
                newstate=gamestate.generateSuccessor(0,action)
                value=minmax_decision(newstate,deepness,1)
                if type(value) is list:
                    testval=value[1]
                else:
                    testval=value
                if testval>output[1]:
                    output=[action,testval]
            return output
        outputList = minmax_decision(gameState, 0, 0)
        return outputList[0]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    foodpos=currentGameState.getFood().asList()
    foodDist=[]
    ghoststates=currentGameState.getGhostStates()
    currentpos=list(currentGameState.getPacmanPosition())
    for food in foodpos:
        fpacdistance=manhattanDistance(food,currentpos)
        foodDist.append(-1*fpacdistance)
    if not foodDist:
        foodDist.append(0)
    return max(foodDist)+currentGameState.getScore()
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

