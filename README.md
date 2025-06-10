# CS175 - Malmo Zombie RL

Reinforcement learning agents to fight zombies in Minecraft using Project Malmo.

## Setup

### Docker (Recommended)
```bash
docker run -it -p 5901:5901 -p 6901:6901 -p 8888:8888 -e VNC_PW=password -v $(pwd):/home/malmo/cs175 andkram/malmo
```

Connect to:
- Browser: http://localhost:6901/?password=password
- Jupyter: Check terminal for token URL

### Manual Setup
```bash
pip install -r requirements.txt
# Install Project Malmo separately
```

## Usage

### Test Environment
```bash
python single_zombie.py
```

### Train with Stable Baselines
```bash
python wrapper.py
```

### Train Custom PPO
```bash
python ppo_agent.py
```

## Files
- `wrapper.py` - Reward wrapper and Stable Baselines training
- `stable_baseline_env.py` - Malmo environment
- `ppo_agent.py` - Custom PPO implementation
- `single_zombie.py` - Simple environment test