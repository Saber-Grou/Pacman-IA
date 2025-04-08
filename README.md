# Project Description

## Features

- **Minimax Agent**: Implements the Minimax algorithm for adversarial search.
- **Strategic Agent with Evolution**: Uses deep learning and evolutionary strategies for decision-making.
- **Replay Buffer**: Stores experiences for training the Strategic Agent.
- **Custom Evaluation Functions**: Enhances Pacman's decision-making by evaluating game states.

## Requirements

This code runs with Python 3. You just need a recent version of Python 3 installed and to clone this repository.

### Troubleshooting

This code was translated from Python 2 to 3. In case of errors, you might need to install the package `future`:

```zsh
pip install future
```

## Usage

### Running Pacman with Different Agents

1. **Strategic Agent with Evolution**:
   ```zsh
   python pacman.py -p StrategicAgentWithEvolution -l mediumClassic -n 20 --frameTime 0
   ```
   Add `-q` to execute in the background:
   ```zsh
   python pacman.py -p StrategicAgentWithEvolution -l mediumClassic -n 10000 --frameTime 0 -q
   ```

2. **Minimax Agent**:
   ```zsh
   python pacman.py -p MinimaxAgent -a depth=4 --frameTime 0
   ```

### Layouts

The project includes several layouts for testing:
- `smallClassic`
- `mediumClassic`
- `originalClassic`

You can specify layouts using the `-l` flag:
```zsh
python pacman.py -l smallClassic -p MinimaxAgent
```

### Credit

Kaber-Grou (Me)
