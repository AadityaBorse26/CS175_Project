# Statement of Collaboration
# Worked with Mukund Nair and Aaditya Borse

import random

items=['pumpkin', 'sugar', 'egg', 'egg', 'red_mushroom', 'planks', 'planks']

food_recipes = {'pumpkin_pie': ['pumpkin', 'egg', 'sugar'],
                'pumpkin_seeds': ['pumpkin'],
               'bowl': ['plank', 'planks'],
               'mushroom_stew': ['bowl', 'red_mushroom']}

rewards_map = {'pumpkin': -5, 'egg': -25, 'sugar': -10,
               'pumpkin_pie': 100, 'pumpkin_seeds': -50,
              'red_mushroom': 5, 'planks': 5, 'bowl': 1,
              'mushroom_stew': 100}

def is_solution(reward):
    return reward == 205

def get_curr_state(items):
    return tuple(sorted(items))

def choose_action(curr_state, possible_actions, eps, q_table):
    rnd = random.random()
    # Do random epsilon action
    if rnd < eps:
        a = random.randint(0, len(possible_actions) - 1)
        return possible_actions[a]
    # Do greedy action
    else:
        state_qs = q_table.get(curr_state, {})
    
        max_val = float('-inf')
        best_actions = []

        for action in possible_actions:
            val = state_qs.get(action, 0)
            if val > max_val:
                max_val = val
                best_actions = [action]
            elif val == max_val:
                best_actions.append(action)
        return random.choice(best_actions) if best_actions else random.choice(possible_actions)