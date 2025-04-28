import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from util import manhattanDistance
from game import Directions
import random, util
import os
from game import Agent
import matplotlib.pyplot as plt

# Constants
FOOD_REWARD = 15
CAPSULE_REWARD = 100
GHOST_PENALTY = -50
PATTERN_PENALTY = -50
MAX_PATTERN_LENGTH = 8
MAX_RECENT_POSITIONS = 5
TARGET_UPDATE_FREQUENCY = 10
BATCH_SIZE = 128
REPLAY_BUFFER_CAPACITY = 100000
LEARNING_RATE = 0.002
DISCOUNT_FACTOR = 0.95

def scoreEvaluationFunction(currentGameState):
    """
    Default evaluation function that returns the score of the state.
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all multi-agent searchers.
    """
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """

        def minimax(agentIndex, depth, state):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Pacman (maximizing player)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, state)

            # Ghosts (minimizing bots)
            else:
                return minValue(agentIndex, depth, state)

        def maxValue(agentIndex, depth, state):
            legalMoves = state.getLegalActions(agentIndex)
            if not legalMoves:
                return self.evaluationFunction(state)

            bestValue = float('-inf')
            for action in legalMoves:
                successor = state.generateSuccessor(agentIndex, action)
                bestValue = max(bestValue, minimax(1, depth, successor))
            return bestValue

        def minValue(agentIndex, depth, state):
            legalMoves = state.getLegalActions(agentIndex)
            if not legalMoves:
                return self.evaluationFunction(state)

            bestValue = float('inf')
            nextAgent = agentIndex + 1
            if nextAgent == state.getNumAgents():
                nextAgent = 0
                depth += 1

            for action in legalMoves:
                successor = state.generateSuccessor(agentIndex, action)
                bestValue = min(bestValue, minimax(nextAgent, depth, successor))
            return bestValue

        # Start the minimax algorithm
        legalMoves = gameState.getLegalActions(0)
        bestAction = None
        bestScore = float('-inf')
        for action in legalMoves:
            successor = gameState.generateSuccessor(0, action)
            score = minimax(1, 0, successor)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

class StrategicAgentWithEvolution(Agent):
    """
    An agent that uses deep learning with evolution and auto-restart to improve its decision-making.
    """

    def __init__(self):
        # Device configuration (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        self.criterion = nn.MSELoss()

        # Replay buffer for experience replay
        self.replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY)

        # Training parameters
        self.batch_size = BATCH_SIZE
        self.gamma = DISCOUNT_FACTOR
        self.target_update_frequency = TARGET_UPDATE_FREQUENCY

        # Game tracking
        self.recent_positions = []
        self.max_recent_positions = MAX_RECENT_POSITIONS
        self.recent_patterns = []
        self.max_pattern_length = MAX_PATTERN_LENGTH
        self.generation_played = 0
        self.current_score = 0
        self.best_score = self.load_best_score()

        # Load or initialize model
        if os.path.exists("pacman_model.pth"):
            print("Loading existing model...")
            try:
                self.model.load_state_dict(torch.load("pacman_model.pth"))
                self.model.eval()
            except RuntimeError as e:
                print(f"Error loading model: {e}. Creating a new model...")
                self.save_model()
        else:
            print("No existing model found. Creating a new model...")
            self.save_model()

    def _build_model(self):
        """
        Builds the neural network model with additional layers, dropout, and batch normalization.
        Input size: 9 (features), Output size: 5 (actions).
        """
        return nn.Sequential(
            nn.Linear(9, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def save_model(self):
        """
        Saves the current model to a file and pushes changes to Git if the score improves.
        """
        os.system('git pull')
        if self.current_score >= self.best_score:
            torch.save(self.model.state_dict(), "pacman_model.pth")
            print("New model saved as 'pacman_model.pth'.")
            self.save_best_score()
            time.sleep(2.5)
            os.system('git add *')
            os.system('git commit -m "auto-update: model updated"')
            os.system('git push')

    def load_best_score(self):
        """
        Loads the best score from a file. If the file does not exist, sets the best score to -inf.
        """
        if os.path.exists("best_score.txt"):
            try:
                with open("best_score.txt", "r") as f:
                    return float(f.read().strip())
            except ValueError:
                print("Invalid value in best_score.txt. Resetting best score to -inf.")
        return float('-inf')

    def save_best_score(self):
        """
        Saves the current best score to a file.
        """
        with open("best_score.txt", "w") as f:
            f.write(str(self.best_score))
        print(f"Best score saved: {self.best_score}")

    def extract_features(self, gameState):
        """
        Extracts features from the game state for input to the neural network.
        """
        pacmanPosition = gameState.getPacmanPosition()
        ghostPositions = gameState.getGhostPositions()
        foodPositions = gameState.getFood().asList()
        capsules = gameState.getCapsules()
        walls = gameState.getWalls()

        def min_distance(positions):
            return min([manhattanDistance(pacmanPosition, pos) for pos in positions], default=0)

        # Feature calculations
        nearest_ghost_distance = min_distance(ghostPositions)
        nearest_food_distance = min_distance(foodPositions)
        nearest_capsule_distance = min_distance(capsules)
        scared_ghosts = [ghost for ghost in gameState.getGhostStates() if ghost.scaredTimer > 0]
        nearest_scared_ghost_distance = min_distance([ghost.getPosition() for ghost in scared_ghosts])
        food_density = sum(1 for food in foodPositions if manhattanDistance(pacmanPosition, food) <= 3)
        wall_distances = [
            manhattanDistance(pacmanPosition, (x, y))
            for x in range(walls.width)
            for y in range(walls.height)
            if walls[x][y]
        ]
        nearest_wall_distance = min(wall_distances, default=0)

        return np.array([
            nearest_ghost_distance,
            nearest_food_distance,
            nearest_capsule_distance,
            nearest_scared_ghost_distance,
            food_density,
            nearest_wall_distance,
            len(foodPositions),
            len(capsules),
            gameState.getScore()
        ], dtype=np.float32)

    def getAction(self, gameState):
        """
        Chooses an action using the neural network model.
        """
        self.model.eval()
        legalMoves = gameState.getLegalActions()
        features = self.extract_features(gameState)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Predict action scores
        with torch.no_grad():
            action_scores = self.model(features_tensor).squeeze(0).cpu().numpy()

        # Map actions to indices
        action_to_index = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2, Directions.WEST: 3, Directions.STOP: 4}
        best_action = max(legalMoves, key=lambda action: action_scores[action_to_index[action]])

        # Avoid cycles by penalizing repeated positions
        pacmanPosition = gameState.getPacmanPosition()
        if pacmanPosition in self.recent_positions:
            legalMoves.remove(best_action)
            if legalMoves:
                best_action = random.choice(legalMoves)

        # Update recent positions
        self.recent_positions.append(pacmanPosition)
        if len(self.recent_positions) > self.max_recent_positions:
            self.recent_positions.pop(0)

        return best_action

    def train(self, epochs=10):
        """
        Trains the model using mini-batches from the replay buffer.
        """
        if len(self.replay_buffer.buffer) < BATCH_SIZE:
            print("Not enough data in replay buffer to train.")
            return

        for epoch in range(epochs):
            batch = self.replay_buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert data to tensors
            states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
            next_states = torch.tensor(
                np.array([ns if ns is not None else np.zeros_like(states[0]) for ns in next_states]),
                dtype=torch.float32
            ).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

            # Map actions to indices
            action_to_index = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2, Directions.WEST: 3, Directions.STOP: 4}
            actions = torch.tensor([action_to_index[action] for action in actions], dtype=torch.long).to(self.device)

            # Compute target Q-values
            with torch.no_grad():
                next_q_values = self.target_model(next_states).max(dim=1)[0]
                targets = rewards + DISCOUNT_FACTOR * next_q_values * (1 - dones)

            # Compute loss and update model
            predictions = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
            loss = self.criterion(predictions, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        # Update target network periodically
        if self.generation_played % TARGET_UPDATE_FREQUENCY == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            print("Target model updated.")

class ReplayBuffer:
    def __init__(self, capacity=REPLAY_BUFFER_CAPACITY):
        self.buffer = []
        self.capacity = capacity

    def add(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        priorities = [abs(exp[2]) for exp in self.buffer]
        probabilities = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probabilities)
        return [self.buffer[i] for i in indices]

    def size(self):
        return len(self.buffer)
