from __future__ import print_function
import MalmoPython
import time
import sys
import os

# For Python 3: ensure print flushes immediately
if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
else:
    import functools
    print = functools.partial(print, flush=True)

# XML defining the mission
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
                  
                  <DrawingDecorator>
                    <DrawEntity x="0.5" y="3" z="0.5" type="Zombie"/>
                  </DrawingDecorator>
                  
                  <ServerQuitFromTimeUp timeLimitMs="10000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Adventure">
                <Name>MalmoTutorialBot</Name>    
                <AgentStart>
                  <Placement x="0.5" y="3" z="7.5" yaw="180"/>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromFullStats/>
                  <ContinuousMovementCommands turnSpeedDegs="180"/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

# Create agent host
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

# Start the mission
my_mission = MalmoPython.MissionSpec(missionXML, True)
my_mission_record = MalmoPython.MissionRecordSpec()

print("Starting mission...")
max_retries = 5
for retry in range(max_retries):
    try:
        agent_host.startMission(my_mission, my_mission_record)
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print("Error starting mission:", e)
            exit(1)
        else:
            time.sleep(2)

# Wait for mission to start
print("Waiting for mission to start ", end='')
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end='')
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

print()
print("Mission running ", end=' ')

# Just wait while the agent does nothing
while world_state.is_mission_running:
    time.sleep(0.5)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

print()
print("Mission ended.")