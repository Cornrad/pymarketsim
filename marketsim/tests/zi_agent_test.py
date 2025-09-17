from marketsim.simulator.sampled_arrival_simulator import SimulatorSampledArrival


def test_sampled_arrival_simulator_runs():
    sim = SimulatorSampledArrival(
        num_background_agents=5,
        sim_time=25,
        lam=0.5,
        mean=1000,
        r=0.05,
        shock_var=50,
        q_max=5,
        pv_var=1e4,
        shade=[20, 40],
    )
    sim.run()
    fundamental_val = sim.markets[0].get_final_fundamental()
    values = []
    for agent in sim.agents.values():
        value = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
        values.append(value)
    assert len(values) == len(sim.agents)
