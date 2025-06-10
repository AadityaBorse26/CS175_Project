import gym
from gym import spaces
from malmo import MalmoPython
import numpy as np
import time
import json
import math

MISSION_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>Agent watches a zombie</Summary>
              </About>
              
              <ServerSection>
                <ServerInitialConditions>
                  <Time>
                      <StartTime>14000</StartTime>
                      <AllowPassageOfTime>true</AllowPassageOfTime>
                  </Time>
                  <AllowSpawning>false</AllowSpawning>
                </ServerInitialConditions>
                
                <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;7,2;1;"/>
                  <ServerQuitFromTimeUp timeLimitMs="10000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Adventure">
                <Name>MalmoTutorialBot</Name>    
                <AgentStart>
                  <Placement x="0.5" y="3" z="7.5" yaw="180"/>
                  <Inventory>
                    <InventoryItem slot="0" type="iron_sword"/>
                  </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromFullStats/>
                  
                  <ObservationFromNearbyEntities>
                     <Range name="entities" xrange="10" yrange="2" zrange="10" />
                  </ObservationFromNearbyEntities>
                  
                  <ContinuousMovementCommands turnSpeedDegs="180"/>
                  <ChatCommands/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

# Constants
MAX_EPISODE_STEPS = 200
ZOMBIE_SPAWN_DISTANCE = 8
ACTION_DELAY = 0.2
MISSION_RETRY_ATTEMPTS = 3
MISSION_START_TIMEOUT = 10
OBSERVATION_WAIT_TIMEOUT = 10

class MalmoZombieEnv(gym.Env):
    def __init__(self):
        super(MalmoZombieEnv, self).__init__()
        
        self.action_space = spaces.Box(
            low=np.array([-1, 0, 0]), 
            high=np.array([1, 1, 1]), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(5,), 
            dtype=np.float32
        )
        
        self.agent_host = MalmoPython.AgentHost()
        self.mission_xml = MISSION_XML
        self.step_count = 0
        self.zombie_spawn_attempted = False
        
    def spawn_zombie_in_front(self, distance=None):
        if distance is None:
            distance = ZOMBIE_SPAWN_DISTANCE
            
        world_state = self.agent_host.getWorldState()
        attempts = 0
        while not world_state.observations and attempts < OBSERVATION_WAIT_TIMEOUT:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            attempts += 1

        self.agent_host.sendCommand("chat /kill @e[type=minecraft:zombie]")
        time.sleep(0.5)
            
        try:
            obs = json.loads(world_state.observations[-1].text)
            x = obs.get("XPos", 0)
            y = obs.get("YPos", 0)
            z = obs.get("ZPos", 0)
            yaw = obs.get("Yaw", 0)

            rad = math.radians(yaw)
            dx = -math.sin(rad) * distance
            dz = math.cos(rad) * distance

            spawn_x = round(x + dx) + 0.5
            spawn_y = y
            spawn_z = round(z + dz) + 0.5

            summon_cmd = "chat /summon minecraft:zombie %.1f %.1f %.1f" % (spawn_x, spawn_y, spawn_z)
            print("Spawning zombie at (%.1f, %.1f, %.1f)" % (spawn_x, spawn_y, spawn_z))

            self.agent_host.sendCommand(summon_cmd)
            self.zombie_spawn_attempted = True
            
        except Exception as e:
            print(f"Error spawning zombie: {e}")
    
    def _init_mission(self):
        print("Checking for previous missions...")
        for _ in range(MISSION_START_TIMEOUT): 
            world_state = self.agent_host.getWorldState()
            if not world_state.is_mission_running:
                break
            print("Previous mission running. Sending 'quit'...")
            self.agent_host.sendCommand("quit")
            time.sleep(1)
        else:
            print("Timeout waiting for previous mission to end.")
        
        mission_spec = MalmoPython.MissionSpec(self.mission_xml, True)
        mission_record = MalmoPython.MissionRecordSpec()
        
        max_retries = MISSION_RETRY_ATTEMPTS
        for retry in range(max_retries):
            try:
                self.agent_host.startMission(mission_spec, mission_record)
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    raise e
                else:
                    time.sleep(2)
                    
        print("Waiting for the mission to start...")
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
        print("Mission started!")
        
        self.spawn_zombie_in_front()

    def step(self, action):
        turn = float(action[0])
        move = float(action[1])
        attack = float(action[2])

        self.agent_host.sendCommand("turn %f" % turn)
        self.agent_host.sendCommand("move %f" % move)
        self.agent_host.sendCommand("attack %d" % (1 if attack > 0.5 else 0))

        time.sleep(ACTION_DELAY)

        self.agent_host.sendCommand("turn 0")
        self.agent_host.sendCommand("move 0")
        self.agent_host.sendCommand("attack 0")

        world_state = self.agent_host.getWorldState()
        obs = self._get_observation(world_state)

        reward = 0.0
        done = not world_state.is_mission_running or self.step_count >= MAX_EPISODE_STEPS
        
        self.step_count += 1

        return obs, reward, done, {}
    
    def reset(self):
        self.step_count = 0
        self.zombie_spawn_attempted = False
        
        self._init_mission()
        return self._get_observation(self.agent_host.getWorldState())
    
    def _get_observation(self, world_state):
        obs = np.zeros(5, dtype=np.float32)

        if world_state.observations:
            obs_text = world_state.observations[-1].text
            obs_dict = json.loads(obs_text)

            obs[0] = obs_dict.get("Life", 0)

            entities = obs_dict.get("entities", [])
            for ent in entities:
                if ent.get("name") == "Zombie":
                    obs[1] = ent.get("x", 0)
                    obs[2] = ent.get("y", 0)
                    obs[3] = ent.get("z", 0)
                    obs[4] = 1
                    break
        
        print(obs)
        return obs
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass
