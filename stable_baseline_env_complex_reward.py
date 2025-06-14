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
                  <ServerQuitFromTimeUp timeLimitMs="15000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Adventure">
                <Name>ZombieFighter</Name>    
                <AgentStart>
                  <Placement x="0.5" y="3" z="7.5" yaw="180"/>
                  <Inventory>
                    <InventoryItem slot="0" type="iron_sword"/>
                  </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromFullStats/>
                  <ObservationFromRay/>
                  
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
        
        # Action space: turn, move, attack
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]), 
            high=np.array([1.0, 1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Enhanced observation space:
        # [0-2]: Zombie relative position (x, y, z)
        # [3]: Zombie present flag (0 or 1)
        # [4]: Ray hit zombie (0 or 1)
        # [5]: Distance to zombie
        # [6]: Angle to zombie (normalized between -1 and 1)
        # [7]: Ray distance (how far away the zombie is when looking at it)
        # [8]: Time since last attack (normalized)
        # [9]: Zombie health (normalized 0-1, where 1 is full health)
        # [10]: Agent was hit recently (0 or 1)
        self.observation_space = spaces.Box(
            low=np.array([-10.0, -10.0, -10.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]), 
            high=np.array([10.0, 10.0, 10.0, 1.0, 1.0, 10.0, 1.0, 10.0, 1.0, 1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Create Malmo agent host
        self.agent_host = MalmoPython.AgentHost()
        
        # Mission XML
        self.mission_xml = missionXML
        
        # State tracking
        self.zombie_was_present = False
        self.zombie_killed = False
        self.prev_zombie_distance = None
        self.zombie_spawn_attempted = False 
        self.step_count = 0
        self.max_episode_steps = 200
        self.reset_phase = False
        self.last_attack_step = -10  # For tracking attack cooldown
        self.was_facing_zombie = False  # Track if agent was previously facing zombie
        self.last_zombie_health = None
        self.facing_duration = 0
        self.consecutive_hits = 0
        self.backing_up_steps = 0
        
        # Simplified tracking for kill detection
        self.zombie_present_history = []
        self.zombie_attack_history = []
        self.zombie_distance_history = []
        self.history_length = 5
        
        # Add tracking for agent damage
        self.agent_health = 20.0
        self.last_agent_health = 20.0
        self.was_hit_recently = False
        self.optimal_combat_distance = 2.5  # The ideal distance to fight from
        
        # Rewards
        self.zombie_kill_reward = 100.0
        
    def clear_all_zombies(self):
        """Kill all existing zombies before spawning a new one"""
        self.reset_phase = True
        self.agent_host.sendCommand("chat /kill @e[type=minecraft:zombie]")
        time.sleep(0.5)
        self.reset_phase = False
        
    def spawn_zombie_in_front(self, distance=5):
        """Spawn zombie in front of the agent with improved reliability"""  
        if self.zombie_spawn_attempted:
            return
            
        # Wait for observation
        world_state = self.agent_host.getWorldState()
        attempts = 0
        while not world_state.observations and attempts < 10:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            attempts += 1
            if not world_state.is_mission_running:
                return

        try:
            # Get agent position
            obs = json.loads(world_state.observations[-1].text)
            x = obs.get("XPos", 0)
            y = obs.get("YPos", 0)
            z = obs.get("ZPos", 0)
            yaw = obs.get("Yaw", 0)

            # Calculate spawn position
            rad = math.radians(yaw)
            dx = -math.sin(rad) * distance
            dz = math.cos(rad) * distance

            spawn_x = round(x + dx) + 0.5
            spawn_y = y
            spawn_z = round(z + dz) + 0.5

            # Spawn adult zombie with NoAI:0 to ensure it can move
            summon_cmd = "chat /summon minecraft:zombie %.1f %.1f %.1f {IsBaby:0,NoAI:0,Health:20,Attributes:[{Name:\"generic.follow_range\",Base:40}]}" % (spawn_x, spawn_y, spawn_z)
            self.agent_host.sendCommand(summon_cmd)
            self.zombie_spawn_attempted = True
            
            # Give a clear indication that zombie was spawned
            self.agent_host.sendCommand("chat Zombie spawned at distance %.1f" % distance)
            time.sleep(0.5)
            
        except Exception as e:
            print("Error spawning zombie: %s" % str(e))
    
    def _init_mission(self):
        # End any previous mission
        print("Checking for previous missions...")
        for _ in range(10): 
            world_state = self.agent_host.getWorldState()
            if not world_state.is_mission_running:
                break
            self.agent_host.sendCommand("quit")
            time.sleep(1)
        
        # Start new mission
        mission_spec = MalmoPython.MissionSpec(self.mission_xml, True)
        mission_record = MalmoPython.MissionRecordSpec()
        
        # Try to start mission
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
        print("Waiting for mission to start...")
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
        print("Mission started!")
        
        # Clear zombies and spawn new one
        self.clear_all_zombies()
        time.sleep(0.5)
        
        # Spawn zombie with more reliable parameters
        self.spawn_zombie_in_front(distance=5)  # Slightly closer for better visibility
        time.sleep(0.5)

    def _detect_zombie_kill(self, obs, attack_action):
        """Detect if zombie was killed"""
        zombie_present_now = obs[3] > 0
        was_attacking = attack_action > 0.5
        
        # Update histories
        self.zombie_present_history.append(zombie_present_now)
        self.zombie_attack_history.append(was_attacking)
        
        if zombie_present_now:
            distance = obs[5]  # Use the distance value from observation
            self.zombie_distance_history.append(distance)
        else:
            self.zombie_distance_history.append(None)
        
        # Keep only recent history
        if len(self.zombie_present_history) > self.history_length:
            self.zombie_present_history.pop(0)
            self.zombie_attack_history.pop(0)
            self.zombie_distance_history.pop(0)
        
        # Need enough history
        if len(self.zombie_present_history) < 3:
            return False
        
        # Check kill pattern
        recent_zombie_present = any(self.zombie_present_history[-3:-1])
        current_zombie_absent = not zombie_present_now
        was_attacking_recently = any(self.zombie_attack_history[-3:])
        
        # Check if zombie was close when attacking
        close_attack = False
        for i in range(max(0, len(self.zombie_distance_history) - 3), len(self.zombie_distance_history)):
            if (self.zombie_distance_history[i] is not None and 
                self.zombie_distance_history[i] < 3.0 and 
                i < len(self.zombie_attack_history) and
                self.zombie_attack_history[i]):
                close_attack = True
                break
        
        # Kill detected
        kill_detected = (recent_zombie_present and 
                        current_zombie_absent and 
                        was_attacking_recently and 
                        close_attack and 
                        not self.reset_phase)
        
        if kill_detected:
            self.agent_host.sendCommand("chat ZOMBIE DEFEATED! Victory!")
            print("Zombie killed!")
        
        return kill_detected

    def step(self, action):
        # Process actions
        turn = np.clip(float(action[0]), -1.0, 1.0)
        move = np.clip(float(action[1]), -1.0, 1.0)
        attack = np.clip(float(action[2]), 0.0, 1.0)

        # Send commands
        self.agent_host.sendCommand("turn %f" % turn)
        self.agent_host.sendCommand("move %f" % move)
        if attack > 0.5:
            self.agent_host.sendCommand("attack 1")
            self.last_attack_step = self.step_count
        
        time.sleep(0.15)

        # Stop actions
        self.agent_host.sendCommand("turn 0")
        self.agent_host.sendCommand("move 0")
        self.agent_host.sendCommand("attack 0")

        # Get world state
        world_state = self.agent_host.getWorldState()
        
        # Get observation
        obs = self._get_enhanced_observation(world_state)
        
        # Check for zombie kill
        kill_reward = 0.0
        if not self.zombie_killed and self._detect_zombie_kill(obs, attack):
            self.zombie_killed = True
            kill_reward = self.zombie_kill_reward
        
        # Calculate reward
        reward = self._calculate_enhanced_reward(obs, action) + kill_reward
        
        # Check if done
        done = (not world_state.is_mission_running or 
                self.zombie_killed or 
                self.step_count >= self.max_episode_steps)
        
        self.step_count += 1
        
        # Info dict
        info = {
            'zombie_present': obs[3] > 0,
            'zombie_killed': self.zombie_killed,
            'facing_zombie': obs[4] > 0,
            'distance': obs[5] if obs[3] > 0 else -1,
            'angle': obs[6] if obs[3] > 0 else -1
        }

        return obs, reward, done, info
    
    def _calculate_enhanced_reward(self, obs, action):
        """Enhanced reward function with balanced backing up behavior"""
        reward = 0.0

        # Skip during reset
        if self.reset_phase:
            return 0.0

        # If zombie is present
        if obs[3] > 0:  # Zombie present
            self.zombie_was_present = True

            # Get distance and angle
            distance = obs[5]
            angle = obs[6]
            zombie_health = obs[9] * 20.0  # Convert back to raw health value
            was_hit = obs[10] > 0  # Check if agent was hit recently
            attack_cooldown = obs[8]  # Move this line up here, outside the if block

            # Track consecutive backing up steps
            if not hasattr(self, 'backing_up_steps'):
                self.backing_up_steps = 0

            if action[1] < -0.3:  # Moving backward significantly
                self.backing_up_steps += 1
            else:
                self.backing_up_steps = 0

            # HIGHEST PRIORITY: Track health changes (successful hits)
            if hasattr(self, 'last_zombie_health') and self.last_zombie_health is not None:
                health_change = self.last_zombie_health - zombie_health
                if health_change > 0:
                    # Zombie took damage! This is the most important outcome
                    hit_reward = health_change * 15.0  # INCREASED reward for damaging zombie
                    reward += hit_reward
                    self.agent_host.sendCommand("chat Hit landed! Damage: %.1f (Reward: +%.1f)" % (health_change, hit_reward))

                    # Additional bonus for consecutive hits
                    if hasattr(self, 'consecutive_hits'):
                        self.consecutive_hits += 1
                        combo_bonus = min(8.0, self.consecutive_hits * 1.5)  # INCREASED combo bonus
                        reward += combo_bonus
                        if self.consecutive_hits > 1:
                            self.agent_host.sendCommand("chat Combo x%d! (Bonus: +%.1f)" % (self.consecutive_hits, combo_bonus))
                    else:
                        self.consecutive_hits = 1
                else:
                    # Reset combo if no damage dealt
                    self.consecutive_hits = 0

            # Store current health for next comparison
            self.last_zombie_health = zombie_health

            # SECOND PRIORITY: Distance management with BALANCED backing up behavior
            too_close = distance < 1.5  # REDUCED threshold (was 1.8)
            too_far = distance > 3.5    # Zombie is too far

            # Calculate distance from optimal
            distance_from_optimal = abs(distance - self.optimal_combat_distance)

            # Basic distance reward/penalty
            if distance_from_optimal < 0.7:  # Within good combat range
                reward += 1.0
            elif too_close:
                reward -= 0.3  # REDUCED penalty for being too close (was 0.5)

                # BACKING UP REWARD: REDUCED reward for moving backward when too close
                if action[1] < -0.3:  # Moving backward significantly
                    reward += 1.0  # REDUCED from 2.0
                    if obs[4] > 0:  # Still facing the zombie while backing up
                        reward += 0.5  # REDUCED from 1.0

                    # PENALTY FOR EXCESSIVE BACKING UP
                    if self.backing_up_steps > 5:
                        reward -= 0.2 * (self.backing_up_steps - 5)  # Increasing penalty for backing up too long
            elif too_far:
                # INCREASED reward for moving forward when too far
                if action[1] > 0.3:  # Moving forward significantly
                    reward += 1.0  # INCREASED from 0.5

            # BACKING UP WHEN HIT: REDUCED reward for backing up when taking damage
            if was_hit:
                # Reduced reward for backing up when being hit
                if action[1] < -0.3:  # Moving backward significantly
                    reward += 1.5  # REDUCED from 3.0
                    if obs[4] > 0:  # Still facing the zombie while backing up
                        reward += 1.0  # REDUCED from 2.0
                else:
                    # REDUCED penalty for not backing up when hit
                    reward -= 0.5  # REDUCED from 1.0

            # THIRD PRIORITY: Facing the zombie (ray hit)
            if obs[4] > 0:  # Ray hit zombie
                # Calculate facing reward based on distance
                facing_reward = 8.0  # Base reward for facing zombie

                # Extra reward based on ray distance - more reward for closer targeting
                ray_distance = obs[7]
                if ray_distance < 2.0:
                    facing_reward += 3.0  # Reward for close-range targeting
                elif ray_distance < 3.5:
                    facing_reward += 1.5  # Medium range targeting

                reward += facing_reward

                # Track how long the agent maintains line of sight
                if hasattr(self, 'facing_duration'):
                    self.facing_duration += 1
                    # Additional reward for maintaining line of sight
                    if self.facing_duration > 5:  # After maintaining for 5+ steps
                        stability_bonus = min(3.0, (self.facing_duration - 5) * 0.2)  # Up to +3.0
                        reward += stability_bonus
                else:
                    self.facing_duration = 1

                # Announce first target acquisition or when agent first faces zombie
                if not self.was_facing_zombie:
                    self.agent_host.sendCommand("chat Target acquired!")
                    self.was_facing_zombie = True
            else:
                # Penalty for not facing zombie
                reward -= 2.0
                self.was_facing_zombie = False
                self.facing_duration = 0  # Reset facing duration

                # But give partial reward for being close to the right angle
                if abs(angle) < 0.2:  # Tightened angle requirement (~11 degrees)
                    reward += 0.5  # Reward for almost facing
                elif abs(angle) < 0.4:  # (~23 degrees)
                    reward += 0.1  # Small reward for being somewhat close

            # FOURTH PRIORITY: Attack timing and positioning - INCREASED REWARDS
            if action[2] > 0.5:  # Attacking
                # Perfect attack conditions: facing zombie at optimal distance
                if obs[4] > 0:  # Ray hit zombie
                    if abs(distance - self.optimal_combat_distance) < 1.0:  # Within 1 block of optimal distance
                        # Perfect attack positioning - INCREASED
                        reward += 6.0  # INCREASED from 4.0
                    elif distance < 3.5:  # Still close enough to potentially hit
                        reward += 2.0  # INCREASED from 1.0
                    else:
                        # Too far away
                        reward -= 1.0
                else:
                    # Not facing zombie
                    reward -= 2.0

                # Penalize attack spamming
                if attack_cooldown < 0.3:  # If attacked recently
                    reward -= 1.0

            # NEW: Penalty for not attacking when in perfect position
            elif obs[4] > 0 and abs(distance - self.optimal_combat_distance) < 0.8 and attack_cooldown > 0.5:
                # In perfect position but not attacking
                reward -= 0.5  # Mild penalty for missing attack opportunity
        else:
            # No zombie visible
            if self.zombie_was_present and not self.zombie_killed:
                reward -= 1.0  # Penalty for losing sight of zombie

            # Reset all tracking variables
            self.prev_zombie_distance = None
            self.was_facing_zombie = False
            self.last_zombie_health = None
            self.facing_duration = 0
            self.consecutive_hits = 0
            self.backing_up_steps = 0

        # Small time penalty
        reward -= 0.01

        return reward
    
    def _get_enhanced_observation(self, world_state):
        """Get enhanced observation with additional useful information"""
        # [0-2]: Zombie relative position (x, y, z)
        # [3]: Zombie present flag (0 or 1)
        # [4]: Ray hit zombie (0 or 1)
        # [5]: Distance to zombie
        # [6]: Angle to zombie (normalized between -1 and 1)
        # [7]: Ray distance (how far away the zombie is when looking at it)
        # [8]: Time since last attack (normalized)
        # [9]: Zombie health (normalized 0-1, where 1 is full health)
        # [10]: Agent was hit recently (0 or 1)
        obs = np.zeros(11, dtype=np.float32)  # Added one more dimension

        if world_state.observations:
            try:
                obs_text = world_state.observations[-1].text
                obs_dict = json.loads(obs_text)

                # Track agent health to detect hits
                current_health = obs_dict.get("Life", 20.0)
                if self.last_agent_health > current_health:
                    self.was_hit_recently = True
                    # Set a flag in observation that agent was hit
                    obs[10] = 1.0
                else:
                    self.was_hit_recently = False
                    obs[10] = 0.0

                # Update agent health tracking
                self.last_agent_health = current_health
                self.agent_health = current_health

                # Get agent yaw for angle calculations
                agent_yaw = obs_dict.get("Yaw", 0)
                agent_yaw_rad = math.radians(agent_yaw)

                # Find zombie in entities
                zombie_found = False
                entities = obs_dict.get("entities", [])
                for ent in entities:
                    if ent.get("name") == "Zombie":
                        zombie_found = True
                        # Relative position
                        zombie_x = ent.get("x", 0)
                        zombie_z = ent.get("z", 0)
                        zombie_y = ent.get("y", 0)

                        obs[0] = zombie_x
                        obs[1] = zombie_y
                        obs[2] = zombie_z
                        obs[3] = 1.0  # Zombie present

                        # Calculate distance to zombie
                        distance = math.sqrt(zombie_x**2 + zombie_y**2 + zombie_z**2)
                        obs[5] = distance

                        # Calculate angle to zombie (in agent's reference frame)
                        # Convert to radians and normalize to [-1, 1]
                        angle_to_zombie = math.atan2(-zombie_x, zombie_z)
                        rel_angle = (angle_to_zombie - agent_yaw_rad) % (2 * math.pi)
                        if rel_angle > math.pi:
                            rel_angle -= 2 * math.pi

                        # Normalize to [-1, 1]
                        obs[6] = rel_angle / math.pi

                        # Extract zombie health (if available)
                        # Try different possible field names for health
                        zombie_health = None
                        for health_field in ["life", "health", "Health"]:
                            if health_field in ent:
                                zombie_health = float(ent[health_field])
                                break

                        # Normalize health (zombies typically have 20 health in Minecraft)
                        if zombie_health is not None:
                            obs[9] = zombie_health / 20.0
                        else:
                            obs[9] = 1.0  # Default to full health if not available

                        break

                # Check if ray hit zombie
                if "LineOfSight" in obs_dict:
                    los = obs_dict["LineOfSight"]
                    if los.get("type") == "Zombie":
                        obs[4] = 1.0  # Ray hit zombie
                        obs[7] = los.get("distance", 0)  # Distance from ray

                        if obs[3] == 0:  # If zombie wasn't found in entities but ray hit it
                            # This can happen if the zombie is at edge of entity range but in line of sight
                            zombie_found = True
                            obs[3] = 1.0
                            # Estimate distance from ray
                            obs[5] = obs[7]

                        # If we have line of sight but no health from entity list, try to get it from LineOfSight
                        if obs[9] == 0:
                            for health_field in ["hitboxHealth", "health", "life", "Health"]:
                                if health_field in los:
                                    obs[9] = float(los[health_field]) / 20.0
                                    break

                # Debug output if no zombie found
                if not zombie_found and self.zombie_spawn_attempted and not self.reset_phase:
                    # Only print this occasionally to avoid spam
                    if self.step_count % 10 == 0:
                        print("No zombie found in observation. Entities:", entities)

                # Calculate attack cooldown
                attack_cooldown = min(1.0, max(0.0, (self.step_count - self.last_attack_step) / 10.0))
                obs[8] = attack_cooldown

            except Exception as e:
                print("Error processing observation: %s" % str(e))

        return obs
    
    def reset(self):
        # Reset state
        self.zombie_was_present = False
        self.zombie_killed = False
        self.prev_zombie_distance = None
        self.zombie_spawn_attempted = False
        self.step_count = 0
        self.reset_phase = False
        self.last_attack_step = -10
        self.was_facing_zombie = False
        self.last_zombie_health = None
        self.facing_duration = 0
        self.consecutive_hits = 0
        self.agent_health = 20.0
        self.last_agent_health = 20.0
        self.was_hit_recently = False
        self.backing_up_steps = 0

        # Reset tracking
        self.zombie_present_history = []
        self.zombie_attack_history = []
        self.zombie_distance_history = []

        # Start new mission
        self._init_mission()

        # Wait longer for initial stabilization
        time.sleep(0.2)  # Increased from 0.5 to 1.0

        # Wait for zombie to spawn and stabilize
        max_wait_steps = 15  # Increased from 10 to 15
        zombie_present = False

        for attempt in range(max_wait_steps):
            # Get observation
            world_state = self.agent_host.getWorldState()
            if not world_state.is_mission_running:
                # Mission failed to start properly, try again
                self._init_mission()
                world_state = self.agent_host.getWorldState()

            if world_state.observations:
                obs = self._get_enhanced_observation(world_state)
                if obs[3] > 0:  # Zombie is present
                    zombie_present = True
                    self.agent_host.sendCommand("chat Zombie detected, starting episode.")
                    break


        # Get initial observation after ensuring zombie is present
        initial_obs = self._get_enhanced_observation(self.agent_host.getWorldState())
        while initial_obs[3] == 0:
            initial_obs = self._get_enhanced_observation(self.agent_host.getWorldState())

        return initial_obs
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass