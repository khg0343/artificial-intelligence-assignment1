## ID: 20180373 NAME: Kim Hyeonji
######################################################################################
# Problem 2a
# minimax value of the root node: 6
# pruned edges: i, m, x, 2
######################################################################################

from util import manhattanDistance
from game import Directions
import random, util, math

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


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

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
      """
      Your minimax agent (problem 1)
      """
      def getAction(self, gameState):
            """
            Returns the minimax action from the current gameState using self.depth
            and self.evaluationFunction. Terminal states can be found by one of the following: 
            pacman won, pacman lost or there are no legal moves. 

            Here are some method calls that might be useful when implementing minimax.

            gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

            Directions.STOP:
            The stop direction, which is always legal

            gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

            gameState.getNumAgents():
            Returns the total number of agents in the game

            gameState.getScore():
            Returns the score corresponding to the current state of the game
            It corresponds to Utility(s)
      
            gameState.isWin():
            Returns True if it's a winning state
      
            gameState.isLose():
            Returns True if it's a losing state

            self.depth:
            The depth to which search should continue
            """

            # BEGIN_YOUR_ANSWER (our solution is 30 lines of code, but don't worry if you deviate from this)
            # raise NotImplementedError  # remove this line before writing code   
            def value(state, depth, agentIndex, isMinValue):
                  if isMinValue == 1:
                        v = min_value(state, depth, 1)
                  elif agentIndex == state.getNumAgents()-1 :
                        v = max_value(state, depth+1)
                  else :
                        v = min_value(state, depth, agentIndex+1)
                  return v
                                    
            def max_value(state, depth):
                  v = -math.inf
                  if depth == self.depth or state.isWin() or state.isLose():
                        v = self.evaluationFunction(state)
                  else :
                        for action in state.getLegalActions():
                              v = max(v, value(state.generateSuccessor(0, action), depth, 1, 1))
                  return v

            def min_value(state, depth, agentIndex):
                  v = math.inf   
                  if depth == self.depth or state.isWin() or state.isLose():
                        v = self.evaluationFunction(state)
                  else :                  
                        for action in state.getLegalActions(agentIndex):
                              v = min(v, value(state.generateSuccessor(agentIndex, action), depth, agentIndex, 0))
                  return v
            
            v = -math.inf
            move = Directions.STOP
            for action in gameState.getLegalActions():
                  temp = value(gameState.generateSuccessor(0, action), 0, 1, 1)
                  if temp > v :
                        v = temp
                        move = action

            return move
            
      # END_YOUR_ANSWER

######################################################################################
# Problem 2b: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
      """
      Your minimax agent with alpha-beta pruning (problem 2)
      """

      def getAction(self, gameState):
            """
            Returns the minimax action using self.depth and self.evaluationFunction
            """

            # BEGIN_YOUR_ANSWER (our solution is 42 lines of code, but don't worry if you deviate from this)
            # raise NotImplementedError  # remove this line before writing code
            def value(state, alpha, beta, depth, agentIndex, isMinValue):
                  if isMinValue == 1:
                        v = min_value(state, alpha, beta, depth, 1)
                  elif agentIndex == state.getNumAgents()-1 :
                        v = max_value(state, alpha, beta, depth+1)
                  else :
                        v = min_value(state, alpha, beta, depth, agentIndex+1)
                  return v
            
            def max_value(state, alpha, beta, depth):
                  v = -math.inf
                  if depth == self.depth or state.isWin() or state.isLose():
                        v = self.evaluationFunction(state)
                  else :
                        for action in state.getLegalActions():
                              v = max(v, value(state.generateSuccessor(0, action), alpha, beta, depth, 1, 1))
                              if v >= beta : return v
                              alpha = max(alpha, v)
                  return v

            def min_value(state, alpha, beta, depth, agentIndex):
                  v = math.inf
                  if depth == self.depth or state.isWin() or state.isLose():
                        v = self.evaluationFunction(state)
                  else :      
                        for action in state.getLegalActions(agentIndex):
                              v = min(v, value(state.generateSuccessor(agentIndex, action), alpha, beta, depth, agentIndex, 0)) 
                              if v <= alpha: return v
                              beta = min(beta, v)
                  return v

            alpha = -math.inf
            beta = -math.inf
            v = -math.inf
            move = Directions.STOP
            for action in gameState.getLegalActions():
                  temp = value(gameState.generateSuccessor(0, action), alpha, beta, 0, 1, 1)
                  if temp > v :
                        v = temp
                        move = action
                  alpha = max(alpha, v)

            return move
            # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
      """
      Your expectimax agent (problem 3)
      """

      def getAction(self, gameState):
            """
            Returns the expectimax action using self.depth and self.evaluationFunction

            All ghosts should be modeled as choosing uniformly at random from their
            legal moves.
            """

            # BEGIN_YOUR_ANSWER (our solution is 30 lines of code, but don't worry if you deviate from this)
            # raise NotImplementedError  # remove this line before writing code
            def value(state, depth, agentIndex, isExpValue):
                  if isExpValue == 1:                         
                        v = exp_value(state, depth, 1)
                  elif agentIndex == state.getNumAgents()-1 : 
                        v = max_value(state, depth+1)
                  else :                                      
                        v = exp_value(state, depth, agentIndex+1)
                  return v
            
            def max_value(state, depth):
                  v = -math.inf
                  if depth == self.depth or state.isWin() or state.isLose():
                        v = self.evaluationFunction(state)
                  else : 
                        for action in state.getLegalActions():
                              v = max(v, value(state.generateSuccessor(0, action), depth, 1, 1))

                  return v

            def exp_value(state, depth, agentIndex):
                  v = 0
                  if depth == self.depth or state.isWin() or state.isLose():
                        v = self.evaluationFunction(state)
                  else :
                        p = 1/len(state.getLegalActions(agentIndex))
                        for action in state.getLegalActions(agentIndex):
                              v += p * value(state.generateSuccessor(agentIndex, action), depth, agentIndex, 0)

                  return v

            move = Directions.STOP
            v = -math.inf
            for action in gameState.getLegalActions():
                  temp = value(gameState.generateSuccessor(0, action), 0, 1, 1)
                  if temp > v :
                        v = temp
                        move = action
            return move
      
            # END_YOUR_ANSWER

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
      """
      Your extreme, unstoppable evaluation function (problem 4).
      """

      # BEGIN_YOUR_ANSWER (our solution is 60 lines of code, but don't worry if you deviate from this)
      # raise NotImplementedError  # remove this line before writing code
      currentPos = currentGameState.getPacmanPosition()
      currentFood = currentGameState.getFood()
      capsulePos = currentGameState.getCapsules()

      fooddistance = []
      capsuledistance = []

      for food in currentFood.asList():
            fooddistance.append(manhattanDistance(currentPos, food))

      for capsule in capsulePos:
            capsuledistance.append(manhattanDistance(currentPos, capsule))

      score = 0
      x = currentPos[0]
      y = currentPos[1]

      for ghostState in currentGameState.getGhostStates():
            gd = manhattanDistance(currentPos, ghostState.configuration.getPosition())

            if gd < 2:
                  if ghostState.scaredTimer != 0:
                        score += 1000.0/(gd+1)
                  else : 
                        score -= 1000.0/(gd+1)

      if min(capsuledistance+[float(100)])<5:
            score += 500.0/(min(capsuledistance))

      for capsule in capsulePos:
            if (capsule[0]==x) & (capsule[1]==y):
                  score += 600.0

      minfooddistance = min(fooddistance+[float(100)])

      return score + 1.0/minfooddistance - len(fooddistance)*10.0 

      # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction

