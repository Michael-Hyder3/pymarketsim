from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
from marketsim.simulator.sampled_arrival_simulator_custom import SimulatorSampledArrivalCustom

surpluses = []
valueAgents = []

for i in tqdm(range(1000)):
    sim = SimulatorSampledArrivalCustom(num_background_agents=25,
                                       sim_time=1000,
                                       lam=5e-3,
                                       mean=1e5,
                                       r=0.05,
                                       shock_var=1e5,
                                       q_max=10,
                                       pv_var=5e6,
                                       shade=[250,500],
                                       hbl_agent=True)
    sim.run()
    fundamental_val = sim.markets[0].get_final_fundamental()
    values = []
    for agent_id in sim.agents:
        agent = sim.agents[agent_id]
        value = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
        # print(agent.cash, agent.position, agent.get_pos_value(), value)
        values.append(value)
    valueAgents.append(values)
    # diagnostic: print agent counts occasionally
    if i % 100 == 0:
        print(f'iter={i}, agent_count={len(sim.agents)}')
        # print mean values shape if available
        if len(valueAgents) > 0:
            try:
                print('mean valueAgents shape:', np.mean(valueAgents, axis=0).shape)
            except Exception:
                pass

    if len(values) > 0:
        surpluses.append(sum(values)/len(values))
    else:
        # diagnostic when values empty
        print(f'iter={i} produced empty values; agent ids: {list(sim.agents.keys())}')

valueAgents = np.mean(valueAgents, axis = 0)
print(valueAgents)
num_agents = 26


input(valueAgents)
fig, ax = plt.subplots()
plt.scatter([0]*num_agents, valueAgents)  # Placing all points along the same x-axis position (0)
if num_agents == 26:
    plt.scatter([0], valueAgents[-1], color='red')
plt.xlabel('Ignore')
plt.show()


if len(surpluses) > 0:
    avg_surplus = sum(surpluses) / len(surpluses)
    print(avg_surplus * num_agents)
else:
    print('No surpluses collected; nothing to print.')