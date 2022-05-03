# myTeam.py
# ---------
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


from copy import deepcopy
import json
from os import path
from tkinter import E
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from baselineTeam import DefensiveReflexAgent

# Constants
class Constants:
    TRAINING = False
    WEIGHTS_FILE = "weights_sussy_pacman.json"
    EMPTY_WEIGHTS = {"offense": {}, "defense": {}}
    FEATURES = {
        "offense": [
            "ateFood",
            "nearestFood",
            "avoidStop",
            "closerToEnemy",
            "distanceHome",
            "successorScore",
        ],
        "defense": [
            "atePacman",
            "onZoneDefense",
            "toZoneDefense",
            "invaderDistance",
            "onDefense",
            "movingInZone",
        ],
    }
    DISCOUNT = 0.1
    ALPHA = 0.15
    EPSILON = 0.1


#################
# Team creation #
#################


def createTeam(
    firstIndex,
    secondIndex,
    isRed,
    first="OffensiveAgent",
    second="DefensiveAgent",
):
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

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########


class ApproxQLearningAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).
        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)
        IMPORTANT: This method may run for at most 15 seconds.
        """

        """
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        """
        CaptureAgent.registerInitialState(self, gameState)
        self.start_state = gameState.getAgentPosition(self.index)
        self.weights = self.initialize_weights()
        # print("-----------weights-----------")
        # print(self.weights)

        self.lastAction = Directions.STOP
        self.collectedFood = 0
        self.totalFood = len(self.getFood(gameState).asList())
        self.rev = -1 if gameState.isOnRedTeam(self.index) else 1
        self.safePositions = [
            ((gameState.data.layout.width / 2) - 1, y)
            for y in range(gameState.data.layout.height)
            if not gameState.hasWall((gameState.data.layout.width / 2) - 1, y)
        ]
        self.defenseZone = [
            ((gameState.data.layout.width / 2) + (3 * self.rev) - 1, y)
            for y in range(gameState.data.layout.height)
            if not gameState.hasWall(
                (gameState.data.layout.width / 2) + (3 * self.rev) - 1, y
            )
        ]
        self.defenseZone += [
            ((gameState.data.layout.width / 2) + (2 * self.rev) - 1, y)
            for y in range(gameState.data.layout.height)
            if not gameState.hasWall(
                (gameState.data.layout.width / 2) + (2 * self.rev) - 1, y
            )
        ]
        self.defenseZone += [
            ((gameState.data.layout.width / 2) + (5 * self.rev) - 1, y)
            for y in range(gameState.data.layout.height)
            if not gameState.hasWall(
                (gameState.data.layout.width / 2) + (5 * self.rev) - 1, y
            )
        ]
        self.defenseZone += [
            ((gameState.data.layout.width / 2) + (4 * self.rev) - 1, y)
            for y in range(gameState.data.layout.height)
            if not gameState.hasWall(
                (gameState.data.layout.width / 2) + (4 * self.rev) - 1, y
            )
        ]
        self.topZone = (
            (
                (gameState.data.layout.width / 2) + (2 * self.rev),
                1,
            )
            if not gameState.isOnRedTeam(self.index)
            else (
                (gameState.data.layout.width / 2) + (3 * self.rev),
                1,
            )
        )
        self.bottomZone = (
            (
                (gameState.data.layout.width / 2) + (2 * self.rev),
                gameState.data.layout.height - 2,
            )
            if not gameState.isOnRedTeam(self.index)
            else (
                (
                    (gameState.data.layout.width / 2) + (3 * self.rev),
                    gameState.data.layout.height - 2,
                )
            )
        )
        self.toTop = False

    def computeActionFromQValues(self, gameState):
        """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        actions = gameState.getLegalActions(self.index)
        if not actions:
            return None
        highest_val = float("-inf")
        best_actions = []
        for action in actions:
            q_val = self.getQValue(gameState, action)
            if q_val > highest_val:
                best_actions = []
                best_actions.append(action)
                highest_val = q_val
            elif q_val == highest_val:
                best_actions.append(action)
        action_choice = random.choice(best_actions)
        self.lastAction = action_choice
        self.prevState = gameState
        return action_choice

    # Choose best action do same as hw2
    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        legal_actions = gameState.getLegalActions(self.index)
        current_state = self.getCurrentObservation()
        action = self.lastAction
        prev_state = self.getPreviousObservation()
        if prev_state:
            reward = self.getReward(prev_state, action, current_state)
            if Constants.TRAINING:
                self.updateWeights(prev_state, action, current_state, reward)
        if not legal_actions:
            return None

        eps_greedy = Constants.EPSILON
        if not Constants.TRAINING:
            eps_greedy = 0.0
        if util.flipCoin(eps_greedy):
            action_choice = random.choice(legal_actions)
            self.lastAction = action_choice
            self.prevState = gameState
            return action_choice

        return self.computeActionFromQValues(gameState)

    def initialize_weights(self):
        if path.exists(Constants.WEIGHTS_FILE):
            file = open(Constants.WEIGHTS_FILE, "r")
            fileDict = json.load(file)[
                "offense" if self.isOffensiveAgent() else "defense"
            ]
            file.close()
            return self.convertDictToCounter(fileDict)
        weights = util.Counter()
        for feature in Constants.FEATURES[
            "offense" if self.isOffensiveAgent() else "defense"
        ]:
            weights[feature] = 0.0
        return weights

    def save_weights(self):
        weights = deepcopy(Constants.EMPTY_WEIGHTS)
        file = open(Constants.WEIGHTS_FILE, "w")
        print(self.weights)
        weights["offense" if self.isOffensiveAgent() else "defense"] = self.weights
        json.dump(weights, file)
        file.close()

    def final(self, gameState):
        print("-------------END OF GAME--------------")
        # Change based on if you want offense to save or defense to save
        if Constants.TRAINING and self.isOffensiveAgent():
            self.save_weights()

    def getQValue(self, state, action):
        """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        state_action_features = self.getFeatures(state, action).items()
        return sum(
            [
                feature_val * self.weights[feature]
                for feature, feature_val in state_action_features
            ]
        )

    def get_q_value(self, state, action):
        weights = self.getWeights()
        features = self.getFeatures(state, action)
        return weights * features

    def getWeights(self):
        return self.weights

    def updateWeights(self, prev_state, action, state, reward):
        features = self.getFeatures(prev_state, action)
        print("UPDATING WEIGHTS WITH THESE FEATURES: ")
        q_val = self.computeValueFromQValues(state)
        difference = (reward + (Constants.DISCOUNT * q_val)) - self.getQValue(
            prev_state, action
        )
        for feature, feature_val in features.items():
            self.weights[feature] += Constants.ALPHA * difference * feature_val
        print(self.weights)

    def computeValueFromQValues(self, state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """
        legal_actions = state.getLegalActions(self.index)
        return (
            max([self.getQValue(state, action) for action in legal_actions])
            if legal_actions
            else 0
        )

    def convertDictToCounter(self, d):
        result = util.Counter()
        for key, value in d.items():
            result[key] = value
        return result

    def getSuccessor(self, gameState, action):
        """
        baselineTeam.py
        Finds the next successor which is a grid position (location tuple).
        The gameState that this method returns is the same as the current one
        EXCEPT that the agent would have taken the specified action.
        We use this method for considering potential actions before choosing one.
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def isOffensiveAgent(self):
        return isinstance(self, OffensiveAgent)

    def getDistanceToNearestEnemy(self, state):
        myPos = state.getAgentState(self.index).getPosition()
        opponentIndices = self.getOpponents(state)
        distanceToNearestEnemy = 100
        nearestEnemyIndex = None
        for opponentIndex in opponentIndices:
            if state.getAgentPosition(opponentIndex) == None:
                continue
            opponentPos = state.getAgentPosition(opponentIndex)
            current_distance = self.distancer.getDistance(myPos, opponentPos)
            if current_distance < distanceToNearestEnemy:
                distanceToNearestEnemy = current_distance
                nearestEnemyIndex = opponentIndex

        return (nearestEnemyIndex, distanceToNearestEnemy)


class OffensiveAgent(ApproxQLearningAgent):
    """
    A QLearningAgent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, state, action):
        features = util.Counter()
        successor = self.getSuccessor(state, action)
        foodList1 = self.getFood(state).asList()
        foodList2 = self.getFood(successor).asList()
        features["ateFood"] = 0 if len(foodList1) <= len(foodList2) else 300

        stateScore = state.getScore()
        successorScore = successor.getScore()
        if state.isOnRedTeam(self.index):
            if successorScore == stateScore:
                features["successorScore"] = 0
            elif successorScore > stateScore:
                features["successorScore"] = 300
        else:
            if successorScore == stateScore:
                features["successorScore"] = 0
            elif successorScore < stateScore:
                features["successorScore"] = 300

        myPos1 = state.getAgentState(self.index).getPosition()
        minDistance1 = min([self.getMazeDistance(myPos1, food) for food in foodList1])
        myPos2 = successor.getAgentState(self.index).getPosition()
        minDistance2 = min([self.getMazeDistance(myPos2, food) for food in foodList2])
        myState = successor.getAgentState(self.index)
        if myState.isPacman:
            (enemyIndex1, distanceToEnemy1) = self.getDistanceToNearestEnemy(state)
            (enemyIndex2, distanceToEnemy2) = self.getDistanceToNearestEnemy(successor)
            scared = (
                state.getAgentState(enemyIndex2).scaredTimer > 0.0
                if enemyIndex2 is not None
                else False
            )
            if scared or (distanceToEnemy2 > 5):
                features["closerToEnemy"] = 0
            else:
                if distanceToEnemy2 > distanceToEnemy1:
                    features["closerToEnemy"] = 40
                else:
                    features["closerToEnemy"] = -40
        if (
            features["ateFood"] < 100
            and features["closerToEnemy"] == 0
            and self.collectedFood == 0
        ):
            if minDistance1 > minDistance2:
                features["nearestFood"] = 5
            else:
                features["nearestFood"] = -5

        if self.collectedFood > 0 and features["successorScore"] == 0:
            distanceHome1 = self.getDistanceToHome(state)
            distanceHome2 = self.getDistanceToHome(successor)
            if distanceHome2 < distanceHome1:
                features["distanceHome"] = 20
            else:
                features["distanceHome"] = -20

        if action == "stop":
            features["avoidStop"] = -500
        else:
            features["avoidStop"] = 1.0
        features.divideAll(100)
        return features

    def getReward(self, prev_state, action, current_state):
        reward = 0
        features = self.getFeatures(prev_state, action)
        currentFoodList = self.getFood(current_state).asList()
        previousFoodList = self.getFood(prev_state).asList()
        successor = prev_state.generateSuccessor(self.index, action)
        foodList = self.getFood(current_state).asList()

        if (
            self.distancer.getDistance(
                prev_state.getAgentState(self.index).getPosition(),
                current_state.getAgentState(self.index).getPosition(),
            )
            > 1
        ):
            self.collectedFood = 0

        if (
            prev_state.getAgentState(self.index).isPacman
            and not successor.getAgentState(self.index).isPacman
            and self.collectedFood > 0
        ):
            self.collectedFood = 0

        if features["ateFood"] >= 1:
            self.collectedFood += 1
        reward += features["ateFood"]
        reward += features["avoidStop"] * 3
        reward += features["nearestFood"]
        reward += features["closerToEnemy"]
        reward += features["distanceHome"]

        return reward

    def getDistanceToHome(self, state):
        currentPos = state.getAgentState(self.index).getPosition()
        distancesToSafe = [
            self.distancer.getDistance(currentPos, safePos)
            for safePos in self.safePositions
        ]
        return min(distancesToSafe)


class DefensiveAgent(ApproxQLearningAgent):
    """
    A QLearningAgent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, state, action):
        features = util.Counter()
        successor = self.getSuccessor(state, action)

        myState1 = state.getAgentState(self.index)
        myPos1 = myState1.getPosition()

        myState2 = successor.getAgentState(self.index)
        myPos2 = myState2.getPosition()

        scared = state.getAgentState(self.index).scaredTimer > 0
        # state.getAgentState(myPos2).scaredTimer > 0.0

        features["onDefense"] = 1.0
        if myState2.isPacman:
            features["onDefense"] = 0.0

        enemiesCurrent = [state.getAgentState(i) for i in self.getOpponents(state)]
        invadersCurrent = [
            a for a in enemiesCurrent if a.isPacman and a.getPosition() != None
        ]
        enemiesSuccessor = [
            successor.getAgentState(i) for i in self.getOpponents(successor)
        ]
        invadersSuccessor = [
            a for a in enemiesSuccessor if a.isPacman and a.getPosition() != None
        ]

        if len(invadersSuccessor) < len(invadersCurrent):
            features["atePacman"] = 300
        else:
            features["atePacman"] = 0

        if len(invadersSuccessor) > 0 and len(invadersSuccessor) == len(
            invadersCurrent
        ):
            dist1 = min(
                [self.getMazeDistance(myPos1, a.getPosition()) for a in invadersCurrent]
            )
            dist2 = min(
                [
                    self.getMazeDistance(myPos2, a.getPosition())
                    for a in invadersSuccessor
                ]
            )
            if dist2 < dist1:
                features["invaderDistance"] = -10 if scared else 10
            else:
                features["invaderDistance"] = 10 if scared else -10

        # Staying in Zone Defense
        if features["invaderDistance"] == 0 and features["atePacman"] == 0:
            prevPos = state.getAgentState(self.index).getPosition()
            currentPos = successor.getAgentState(self.index).getPosition()
            distanceToTopZone1 = self.distancer.getDistance(prevPos, self.topZone)
            distanceToBottomZone1 = self.distancer.getDistance(prevPos, self.bottomZone)
            distanceToTopZone2 = self.distancer.getDistance(currentPos, self.topZone)
            distanceToBottomZone2 = self.distancer.getDistance(
                currentPos, self.bottomZone
            )
            if currentPos in self.defenseZone:
                features["onZoneDefense"] = 10
                if self.toTop and currentPos == self.topZone:
                    self.toTop = not self.toTop
                elif not self.toTop and currentPos == self.bottomZone:
                    self.toTop = not self.toTop
                if self.toTop:
                    if distanceToTopZone1 > distanceToTopZone2:
                        features["movingInZone"] = 10
                    else:
                        features["movingInZone"] = -10
                else:
                    if distanceToBottomZone1 > distanceToBottomZone2:
                        features["movingInZone"] = 10
                    else:
                        features["movingInZone"] = -10
            else:
                features["onZoneDefense"] = 0

        # Move Closer to Zone Defense
        if (
            features["atePacman"] == 0
            and features["invaderDistance"] == 0
            and features["onZoneDefense"] == 0
        ):
            distanceZone1 = self.getDistanceToZone(state)
            distanceZone2 = self.getDistanceToZone(successor)
            if distanceZone2 < distanceZone1:
                features["toZoneDefense"] = 10
            else:
                features["toZoneDefense"] = -10

        if action == "stop":
            features["avoidStop"] = -500
        else:
            features["avoidStop"] = 1.0
        features.divideAll(100)
        return features

    def getReward(self, prev_state, action, current_state):
        reward = 0
        features = self.getFeatures(prev_state, action)
        currentFoodList = self.getFood(current_state).asList()
        previousFoodList = self.getFood(prev_state).asList()
        successor = prev_state.generateSuccessor(self.index, action)
        foodList = self.getFood(current_state).asList()
        reward += features["toZoneDefense"]
        reward += features["avoidStop"] * 3
        reward += features["zoneDefense"]
        reward += features["invaderDistance"]
        reward += features["atePacman"]
        reward += features["movingInZone"]

        return reward

    def getDistanceToZone(self, state):
        currentPos = state.getAgentState(self.index).getPosition()
        distancesToZone = [
            self.distancer.getDistance(currentPos, zone) for zone in self.defenseZone
        ]
        return min(distancesToZone)
