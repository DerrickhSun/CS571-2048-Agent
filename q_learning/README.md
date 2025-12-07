# 2048 Reinforcement Learning Agent

This project implements the 2048 game along with a Q-learning agent that learns to play through trial and error.

It includes:
- A Tkinter GUI version of 2048  
- A tabular Q-learning training script  
- A Gym-style environment wrapper  
- An agent autoplay mode  

---

## 1. Train the Agent

Run the Q-learning trainer:

```bash
python q_learning.py
```

This will:
- Simulate thousands of 2048 games
- Update the Q-table based on rewards received
- Save the learned values to `q_table.pkl`

## 2. Play manually
Run the GUI version of 2048:

```bash
python puzzle.py
```

Controls:
- Arrow keys to move tiles

## 3. Autoplay with the trained agent
Run the autoplay script:

```bash
python puzzle.py agent
```

In agent mode:
- The AI selects moves continuously
- Moves update roughly every 50ms
- The game runs until win or loss
- No keyboard input is needed

## 4. Q-learning Environment
```
Q[s][a] = Q[s][a] + α * (reward + γ * max_a' Q[s'][a'] − Q[s][a])

# 2048 DQN Agent

## Train Q Learning Enivornment:

python train_dqn.py 
Use the parameter "--help" for more information about training hyper parameters
After training, the saved model can be used for testing using test_dqn.py

python test_dqn.py
Use the parameter "--help" for more information about testing and epsilon tuning.


