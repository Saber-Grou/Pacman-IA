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
REPLAY_BUFFER_CAPACITY = 100000000
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99

class MultiAgentSearchAgent(Agent):

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
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
            # Terminal state: win, lose, or depth limit reached
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
            if nextAgent == state.getNumAgents():  # Last ghost, go to next depth
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)  # Add target network
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # Reduced learning rate for finer updates
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)  # Learning rate scheduler
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY)  # Initialize replay buffer
        self.batch_size = BATCH_SIZE  # Increased batch size for more stable training
        self.gamma = DISCOUNT_FACTOR  # Slightly reduced discount factor to prioritize immediate rewards
        self.target_update_frequency = TARGET_UPDATE_FREQUENCY  # Update target network every 10 episodes
        self.training_data = []
        self.recent_positions = []  # Memory of recent positions
        self.max_recent_positions = MAX_RECENT_POSITIONS  # Limit for recent positions
        self.generation_played = 0  # Track the number of games played
        self.load_best_score()  # Initialize the best score as negative infinity
        self.recent_patterns = []  # Track recent patterns of positions and actions
        self.max_pattern_length = MAX_PATTERN_LENGTH  # Length of the pattern to track
        self.pattern_penalty = PATTERN_PENALTY  # Penalty for repeating a pattern
        self.current_score = self.load_best_score()  # Current score of the game

        # Check if the model file exists
        if os.path.exists("pacman_model.pth"):
            print("Loading existing model...")
            try:
                self.model.load_state_dict(torch.load("pacman_model.pth"))
                self.model.eval()
            except RuntimeError as e:
                print(f"Error loading model: {e}")
                print("Creating a new model...")
                self.save_model()
        else:
            print("No existing model found. Creating a new model...")
            self.save_model()

        # Initialize the target model with the same weights as the main model
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def save_model(self):
        """
        Saves the current model to a file and pushes the changes to Git.
        """
        os.system('git pull')
        self.load_best_score()
        if self.current_score > self.best_score:
            torch.save(self.model.state_dict(), "pacman_model.pth")
            print("New model saved as 'pacman_model.pth'.")
            self.save_best_score()
            # Delay to allow for file system updates
            time.sleep(2.5)
            # Execute Git commands
            os.system('git add *')
            os.system('git commit -m "auto-update: model updated"')
            os.system('git push')

    def _build_model(self):
        """
        Builds an improved neural network model.
        """
        return nn.Sequential(
            nn.Linear(9, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout to prevent overfitting
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # Output layer for 5 possible actions
        )

    def extract_features(self, gameState):
        pacmanPosition = gameState.getPacmanPosition()
        ghostPositions = gameState.getGhostPositions()
        foodPositions = gameState.getFood().asList()
        capsules = gameState.getCapsules()
        walls = gameState.getWalls()

        def min_distance(positions):
            return min([manhattanDistance(pacmanPosition, pos) for pos in positions], default=0)

        # Calculate features
        nearest_ghost_distance = min_distance(ghostPositions)
        nearest_food_distance = min_distance(foodPositions)
        nearest_capsule_distance = min_distance(capsules)
        scared_ghosts = [ghost for ghost in gameState.getGhostStates() if ghost.scaredTimer > 0]
        nearest_scared_ghost_distance = min_distance([ghost.getPosition() for ghost in scared_ghosts])

        # Food clustering (density of food within a radius)
        food_density = sum(1 for food in foodPositions if manhattanDistance(pacmanPosition, food) <= 3)

        # Distance to nearest wall
        wall_distances = [
            manhattanDistance(pacmanPosition, (x, y))
            for x in range(walls.width)
            for y in range(walls.height)
            if walls[x][y]
        ]
        nearest_wall_distance = min(wall_distances, default=0)

        features = [
            nearest_ghost_distance,
            nearest_food_distance,
            nearest_capsule_distance,
            nearest_scared_ghost_distance,
            food_density,
            nearest_wall_distance,
            len(foodPositions),
            len(capsules),
            gameState.getScore()
        ]
        return np.array(features, dtype=np.float32)

    def getAction(self, gameState):
        """
        Chooses an action using the neural network model.
        """
        self.model.eval()  # Désactiver BatchNorm et Dropout en mode prédiction
        legalMoves = gameState.getLegalActions()
        features = self.extract_features(gameState)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Predict action scores
        with torch.no_grad():
            action_scores = self.model(features_tensor).squeeze(0).cpu().numpy()

        # Map actions to indices
        action_to_index = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2, Directions.WEST: 3, Directions.STOP: 4}
        index_to_action = {v: k for k, v in action_to_index.items()}

        # Filter scores for legal moves only
        best_action = None
        best_score = float('-inf')
        for action in legalMoves:
            index = action_to_index[action]
            if action_scores[index] > best_score:
                best_score = action_scores[index]
                best_action = action

        # Vérifier si Pacman est bloqué dans un cycle
        pacmanPosition = gameState.getPacmanPosition()
        if pacmanPosition in self.recent_positions:
            # Forcer une action différente en cas de cycle
            legalMoves.remove(best_action)  # Retirer l'action choisie
            if legalMoves:  # Si d'autres actions sont disponibles
                best_action = random.choice(legalMoves)
        else:
            gameState.setScore(gameState.getScore() + 1)  # Increment score for non-cyclic action

        # Mettre à jour la mémoire des positions récentes
        self.recent_positions.append(pacmanPosition)
        if len(self.recent_positions) > self.max_recent_positions:
            self.recent_positions.pop(0)

        # Collect training data
        self.collect_training_data(gameState, best_action, gameState.getScore(), None, False)

        return best_action

    def collect_training_data(self, gameState, action, reward, next_state, done):
        """
        Collects training data and stores it in the replay buffer.
        """
        pacman_position = gameState.getPacmanPosition()
        ghost_positions = gameState.getGhostPositions()
        food_positions = gameState.getFood().asList()

        # Reward for eating food
        if pacman_position in food_positions:
            reward += FOOD_REWARD

        # Reward for eating capsules
        if pacman_position in gameState.getCapsules():
            reward += CAPSULE_REWARD

        # Dynamic ghost penalty
        ghost_distances = [manhattanDistance(pacman_position, ghost_pos) for ghost_pos in ghost_positions]
        if ghost_distances:
            nearest_ghost_distance = min(ghost_distances)
            if nearest_ghost_distance <= 4:
                reward += GHOST_PENALTY * 2  # Higher penalty for being very close
            elif nearest_ghost_distance <= 8:
                reward += GHOST_PENALTY

        # Reward for eating scared ghosts
        scared_ghosts = [ghost for ghost in gameState.getGhostStates() if ghost.scaredTimer > 0]
        for ghost in scared_ghosts:
            if manhattanDistance(pacman_position, ghost.getPosition()) <= 1:
                reward += 200  # Bonus for eating scared ghosts

        # Time penalty
        reward -= 1  # Penalize each step to encourage faster completion

        # Penalize repeated patterns
        current_pattern = (pacman_position, action)
        self.recent_patterns.append(current_pattern)
        if len(self.recent_patterns) > MAX_PATTERN_LENGTH:
            self.recent_patterns.pop(0)
        if self.recent_patterns.count(current_pattern) > 1:
            reward += PATTERN_PENALTY

        # Add experience to replay buffer
        features = self.extract_features(gameState)
        next_features = self.extract_features(next_state) if next_state else None
        self.replay_buffer.add((features, action, reward, next_features, done))

    def train(self, epochs=10):
        """
        Trains the model using mini-batches from the replay buffer.
        """
        if len(self.replay_buffer.buffer) < BATCH_SIZE:
            print("Not enough data in replay buffer to train.")
            return

        for epoch in range(epochs):
            # Sample a batch from the replay buffer
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

            # Compute predicted Q-values
            predictions = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

            # Compute loss and update model
            loss = self.criterion(predictions, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Print loss periodically
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        # Update target network periodically
        if self.generation_played % TARGET_UPDATE_FREQUENCY == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            print("Target model updated.")

    def onGameEnd(self, gameState):
        """
        Called at the end of each game to train the model and restart the game.
        """
        self.generation_played += 1
        self.current_score = gameState.getScore()
        self.load_best_score()
        # Update best score if the current score is higher
        if self.current_score > self.best_score:
            print(f"New high score achieved: {self.current_score} (previous best: {self.best_score}).")
            self.save_model()
            self.best_score = self.current_score
        else:
            print(f"Game {self.generation_played} ended with score {self.current_score}. No improvement.")

        print(f"Replay buffer size: {self.replay_buffer.size()}")
        print(f"Best score so far: {self.best_score}")

        # Train the model every 10 games
        if self.generation_played % 10 == 0:
            self.train()
    
    def save_best_score(self):
        with open("best_score.txt", "w") as f:
            f.write(str(self.best_score))

    def load_best_score(self):
        if os.path.exists("best_score.txt"):
            with open("best_score.txt", "r") as f:
                self.best_score = float(f.read())
        else:
            self.best_score = float('-inf')

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
