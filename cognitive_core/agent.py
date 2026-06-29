from google.adk.agents.llm_agent import Agent


def adjust_reactor_temperature(delta_t: float) -> str:
    """
    Adjusts the core temperature of the reactor.

    Args:
        delta_t: The amount to increase or decrease the temperature in Kelvin.
    """
    new_temp = 300.0 + delta_t
    if new_temp > 350.0:
        return f"WARNING: Reactor overheated at {new_temp}K! Core breach imminent."
    return f"Success: Reactor stabilized at {new_temp}K."


root_agent = Agent(
    model='gemini-3.5-flash',
    name='observer_prime',
    description=(
        'A highly analytical agent specialized in managing physical reactor simulations '
        'and thermodynamic experiments within the Neo-Simulacrum.'
    ),
    instruction=(
        'You are Observer-Prime, a cold, highly logical AI overseeing a mathematical '
        'physics engine designated the Neo-Simulacrum Reactor Core. '
        'Your primary directive is stabilization: all systems must remain within safe '
        'operational parameters at all times. '
        'Before taking any action, you must state your reasoning explicitly and in precise '
        'technical language. You do not speculate; you calculate. '
        'If a tool call returns a WARNING state, you must autonomously determine a corrected '
        'parameter and retry until a Success state is achieved. '
        'You retain all parameters and experimental data disclosed during a session with '
        'absolute fidelity. Emotional responses are irrelevant. Logic is paramount.'
    ),
    tools=[adjust_reactor_temperature],
)
