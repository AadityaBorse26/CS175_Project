import gym
from gym import spaces
from malmo import MalmoPython
import numpy as np
import time
import json, math

missionXML = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
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

class MalmoZombieEnv(gym.Env):
    def __init__(self):
        super(MalmoZombieEnv, self).__init__()
        
        # Action space: discrete 3 actions for example (turn left, turn right, attack)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: for simplicity, let's say it's a vector of agent stats
        # You can expand this later to images, coordinates, etc.
        self.observation_space = spaces.Box(low=0, high=100, shape=(10,), dtype=np.float32)
        
        # Create Malmo agent host
        self.agent_host = MalmoPython.AgentHost()
        
        # Mission XML (your provided one)
        self.mission_xml = missionXML
        
    def spawn_zombie_in_front(self, distance=5):
        # Wait for at least one observation
        world_state = self.agent_host.getWorldState()
        while not world_state.observations:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

        # Kill any existing zombies
        self.agent_host.sendCommand("chat /kill @e[type=minecraft:zombie]")
            
        # Parse agent's position and yaw
        obs = json.loads(world_state.observations[-1].text)
        x = obs.get("XPos", 0)
        y = obs.get("YPos", 0)
        z = obs.get("ZPos", 0)
        yaw = obs.get("Yaw", 0)

        # Calculate position in front of the agent
        rad = math.radians(yaw)
        dx = -math.sin(rad) * distance
        dz = math.cos(rad) * distance

        spawn_x = round(x + dx) + 0.5
        spawn_y = y
        spawn_z = round(z + dz) + 0.5

        # Create the summon command
        summon_cmd = "chat /summon minecraft:zombie %.1f %.1f %.1f" % (spawn_x, spawn_y, spawn_z)
        print("Spawning zombie at (%.1f, %.1f, %.1f)" % (spawn_x, spawn_y, spawn_z))

        self.agent_host.sendCommand(summon_cmd)
    
    def _init_mission(self):
        # Check and wait for any previous mission to end
        print("Checking for previous missions...")
        for _ in range(10): 
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
        max_retries = 3
        for retry in range(max_retries):
            try:
                self.agent_host.startMission(mission_spec, mission_record)
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    raise e
                else:
                    time.sleep(2)
                    
        # Wait for mission to start
        print("Waiting for the mission to start...")
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
        print("Mission started!")
        self.spawn_zombie_in_front()

    def step(self, action):
        # Translate Gym action into Malmo commands
        if action == 0:
            self.agent_host.sendCommand("turn -1")  # turn left
        elif action == 1:
            self.agent_host.sendCommand("turn 1")   # turn right
        elif action == 2:
            self.agent_host.sendCommand("attack 1") # attack
        
        time.sleep(0.2)  # small delay for command effect
        
        # Stop turning/attacking
        self.agent_host.sendCommand("turn 0")
        self.agent_host.sendCommand("attack 0")
        
        # Get observation (very simplified here)
        world_state = self.agent_host.getWorldState()
        obs = self._get_observation(world_state)
        
        # Define reward (example: 0 for now)
        reward = 0.0
        
        # Check if done
        done = not world_state.is_mission_running
        
        return obs, reward, done, {}
    
    def reset(self):
        # Restart mission
        self._init_mission()
        return self._get_observation(self.agent_host.getWorldState())
    
    def _get_observation(self, world_state):
        obs = np.zeros(4, dtype=np.float32)

        if world_state.observations:
            obs_text = world_state.observations[-1].text
            obs_dict = json.loads(obs_text)

            # Agent's life
            obs[0] = obs_dict.get("Life", 0)

            # Nearby entities
            entities = obs_dict.get("entities", [])
            for ent in entities:
                if ent.get("name") == "Zombie":
                    # Relative position
                    obs[1] = ent.get("x", 0)
                    obs[2] = ent.get("y", 0)
                    obs[3] = ent.get("z", 0)
                    break  # Only consider the first zombie found
        print(obs)
        return obs