# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    features['successorScore'] = self.getScore(successor)

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
      
    capsuleList = self.getCapsules(successor)
    if len(capsuleList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsuleList])
      features['distanceToCapsule'] = minDistance
    
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
    val = 10
    if(dists):
        val = min(dists)
    if val<2:
        if myState.scaredTimer>5:
            features['ghost'] = -.1
        else:
            features['ghost'] = 1
    else:
        features['ghost'] = 0
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1, 'distanceToCapsule': -3, 'ghost': -500}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    timer = 0
    # Compute distance to the nearest food
    foodList = self.getFoodYouAreDefending(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    
    #capsuleList = self.getFoodYouAreDefending(successor).asList()
    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    capsuleList = self.getCapsulesYouAreDefending(successor)
    if len(capsuleList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsuleList])
      features['distanceToCapsule'] = minDistance
        
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None and myState.scaredTimer==0]
    features['numInvaders'] = len(invaders)
    '''
    if len(invaders) > 0 and myState.scaredTimer==0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      if myState.scaredTimer==0:
          features['invaderDistance'] = min(dists)
      else:
          features['invaderDistance'] = min(dists)*-1
    '''
    dists = [] #FIXME Particle filter - if we know the exact location, move the particles to the spot it's at


    particleFilter = ParticleFilter()
    

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'distanceToFood': -2, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'distanceToCapsule': -5}

'''
# The JointParticleFilter and getObservationDistribution method (and associated constants) were copied from project4
class JointParticleFilter:
  "JointParticleFilter tracks a joint distribution over tuples of all ghost positions."

  def __init__(self, numParticles=600):
     self.setNumParticles(numParticles)

  def setNumParticles(self, numParticles):
    self.numParticles = numParticles

  def initialize(self, gameState, legalPositions):
    "Stores information about the game, then initializes particles."
    self.numGhosts = gameState.getNumAgents() - 1
    self.ghostAgents = []
    self.legalPositions = legalPositions
    self.initializeParticles()

  def initializeParticles(self):
    "Initializes particles randomly.  Each particle is a tuple of ghost positions. Use self.numParticles for the number of particles"
    self.particles = [tuple([random.choice(self.legalPositions) for _ in range(self.numGhosts)] + [1]) for _ in range(self.numParticles)]

  def addGhostAgent(self, agent):
    "Each ghost agent is registered separately and stored (in case they are different)."
    self.ghostAgents.append(agent)

  def elapseTime(self, gameState):
    """
    Samples each particle's next state based on its current state and the gameState.
    """
    newParticles = []
    for oldParticle in self.particles:
      newParticle = list(oldParticle) # A list of ghost positions
      "*** YOUR CODE HERE ***"
      for i in range(self.numGhosts):
        prevGhostPositions = newParticle[:-1]
        newPosDist = getPositionDistributionForGhost(setGhostPositions(gameState, prevGhostPositions),
                                                     i, self.ghostAgents[i])
        newParticle[i] = util.sampleFromCounter(newPosDist)
      newParticles.append(tuple(newParticle))
    self.particles = newParticles

  def getJailPosition(self, i):
    return (2 * i + 1, 1);

  def observeState(self, gameState):
    """
    Resamples the set of particles using the likelihood of the noisy observations.
    """
    pacmanPosition = gameState.getPacmanPosition()
    noisyDistances = gameState.getNoisyGhostDistances()
    if len(noisyDistances) < self.numGhosts: return
    emissionModels = [busters.getObservationDistribution(dist) for dist in noisyDistances]

    "*** YOUR CODE HERE ***"
    WEIGHT = -1
    self.particles = self.sendToJail(noisyDistances)
    newParticles = []
    for par in self.particles:
        newParticle = list(par)
        newParticle[WEIGHT] = reduce(mul, [emissionModels[i][util.manhattanDistance(pacmanPosition, newParticle[i])]
                                       for i in range(len(emissionModels)) if noisyDistances[i] != None])
        newParticles.append(tuple(newParticle))
    newDistribution = util.Counter()
    for par in newParticles:
        newDistribution[par[:WEIGHT]] = sum(newParticle[WEIGHT] for newParticle in newParticles if newParticle[:WEIGHT] == par[:WEIGHT])
    if len([par for par in newParticles if newDistribution[par[:WEIGHT]] > 0]) == 0:
        self.initializeParticles()
        self.particles = self.sendToJail(noisyDistances)
        newDistribution = util.Counter()
        for par in self.particles:
            newDistribution[par[:WEIGHT]] = sum(newParticle[WEIGHT] for newParticle in self.particles if newParticle[:WEIGHT] == par[:WEIGHT])
    else:
        self.particles = [tuple(list(util.sampleFromCounter(newDistribution)) + [1]) for _  in range(self.numParticles)]

  #If certain ghosts are meant to be in jail, update the particles accordingly
  def sendToJail(self, noisyDistances):
    newParticles = [list(par) for par in self.particles]
    if any(noisyDistance == None for noisyDistance in noisyDistances):
        for i in range(len(self.ghostAgents)):
            if noisyDistances[i] == None:
                for j in range(len(newParticles)):
                    newParticles[j][i] = self.getJailPosition(i)
        return [tuple(par) for par in newParticles]
    else:
        return self.particles

  def getBeliefDistribution(self):
    dist = util.Counter()
    for part in self.particles: dist[part] += 1
    dist.normalize()
    return dist

SONAR_NOISE_RANGE = 15 # Must be odd
SONAR_MAX = (SONAR_NOISE_RANGE - 1)/2
SONAR_NOISE_VALUES = [i - SONAR_MAX for i in range(SONAR_NOISE_RANGE)]
SONAR_DENOMINATOR = 2 ** SONAR_MAX  + 2 ** (SONAR_MAX + 1) - 2.0
SONAR_NOISE_PROBS = [2 ** (SONAR_MAX-abs(v)) / SONAR_DENOMINATOR  for v in SONAR_NOISE_VALUES]

def getNoisyDistance(pos1, pos2):
  if pos2[1] == 1: return None
  distance = util.manhattanDistance(pos1, pos2)
  return max(0, distance + util.sample(SONAR_NOISE_PROBS, SONAR_NOISE_VALUES))

observationDistributions = {}
def getObservationDistribution(noisyDistance):
  """
  Returns the factor P( noisyDistance | TrueDistances ), the likelihood of the provided noisyDistance
  conditioned upon all the possible true distances that could have generated it.
  """
  global observationDistributions
  if noisyDistance == None:
    return util.Counter()
  if noisyDistance not in observationDistributions:
    distribution = util.Counter()
    for error , prob in zip(SONAR_NOISE_VALUES, SONAR_NOISE_PROBS):
      distribution[max(1, noisyDistance - error)] += prob
    observationDistributions[noisyDistance] = distribution
  return observationDistributions[noisyDistance]
'''

