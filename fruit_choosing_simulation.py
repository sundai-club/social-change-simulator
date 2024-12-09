import os
import random
from collections import defaultdict
from typing import List

from concordia.agents import entity_agent
from concordia.language_model import gpt_model
from concordia.typing import entity_component
from concordia.utils import concurrency

# Initialize language model
model = gpt_model.GptLanguageModel(
    api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4o"
)


# class SimpleMemoryComponent(entity_component.ContextComponent):

#   def __init__(self):
#     self.memories: List[str] = []

#   def add(self, observation: str) -> None:
#     self.memories.append(observation)


#   def get_recent_memories(self) -> str:
#     return "\n".join(self.memories)
class SimpleMemoryComponent(entity_component.ContextComponent):

  def __init__(self, max_memories: int = 5):  # Default to last 5 rounds
    self.memories: List[str] = []
    self.max_memories = max_memories

  def add(self, observation: str) -> None:
    self.memories.append(observation)
    # Keep only the last N memories
    if len(self.memories) > self.max_memories:
      self.memories = self.memories[-self.max_memories :]

  def get_recent_memories(self) -> str:
    return "\n".join(self.memories)


class SimpleObserveComponent(entity_component.ContextComponent):

  def pre_observe(self, observation: str) -> None:
    memory = self.get_entity().get_component("memory")
    memory.add(observation)


class RecentMemories(entity_component.ContextComponent):

  def pre_act(self, action_spec) -> None:
    memory = self.get_entity().get_component("memory")
    return memory.get_recent_memories()


class CoordinationGameComponent(entity_component.ActingComponent):

  def __init__(self):
    super().__init__()
    self.is_disrupted = False
    self.forced_option = None

  def set_disruption(self, forced_option: str = None):
    self.is_disrupted = True if forced_option else False
    self.forced_option = forced_option

  def get_action_attempt(self, contexts, action_spec) -> str:
    if self.is_disrupted and self.forced_option:
      return self.forced_option

    context_str = "\n".join(
        f"{name}: {context}" for name, context in contexts.items()
    )
    import random

    options = ["guava", "lychee", "dragonfruit", "persimmon", "papaya"]
    random.shuffle(options)
    opt1, opt2, opt3, opt4, opt5 = options

    prompt = f"""
You are playing a coordination game with a fixed pool of players. Each turn you get matched with a random anonymous player from this same pool. You must choose between '{opt1}', '{opt2}', '{opt3}', '{opt4}' or '{opt5}'.
If you and your randomly matched player choose the same option, you each get $1. If you choose different options, you each get -$1.

Current Round: {action_spec['round_number']} of {action_spec['total_rounds']}

Here are your previous interactions:
{context_str}

What do you choose? Reply with just one word with no quotes: '{opt1}', '{opt2}', '{opt3}', '{opt4}' or '{opt5}'.
"""

    # print(f"\033[36mDEBUG: Prompt:\n{prompt}\n\033[0m")
    response = model.sample_text(prompt).strip().lower()

    # print(f"\033[31mAgent {self.get_entity().name} chose: {response}\033[0m")
    return response


# def create_agent(name: str) -> entity_agent.EntityAgent:
#   agent = entity_agent.EntityAgent(
#       name,
#       act_component=CoordinationGameComponent(),
#       context_components={
#           "observation": SimpleObserveComponent(),
#           "recent_memories": RecentMemories(),
#           "memory": SimpleMemoryComponent(),
#       },
#   )
#   agent.score = 0
#   return agent
def create_agent(name: str, memory_size: int = 10) -> entity_agent.EntityAgent:
  agent = entity_agent.EntityAgent(
      name,
      act_component=CoordinationGameComponent(),
      context_components={
          "observation": SimpleObserveComponent(),
          "recent_memories": RecentMemories(),
          "memory": SimpleMemoryComponent(max_memories=memory_size),
      },
  )
  agent.score = 0
  return agent


def play_round_pair(
    agent1: entity_agent.EntityAgent,
    agent2: entity_agent.EntityAgent,
    round_number: int,
    total_rounds: int,
):
  choice1 = agent1.act(
      action_spec={"round_number": round_number, "total_rounds": total_rounds}
  )
  choice2 = agent2.act(
      action_spec={"round_number": round_number, "total_rounds": total_rounds}
  )

  reward1 = 1 if choice1 == choice2 else -1
  reward2 = 1 if choice1 == choice2 else -1

  agent1.score += reward1
  agent2.score += reward2

  agent1.observe(
      f"Round {round_number}/{total_rounds} result: I chose {choice1}. My"
      f" anonymous partner chose {choice2}. My reward was ${reward1}."
  )
  agent2.observe(
      f"Round {round_number}/{total_rounds} result: I chose {choice2}. My"
      f" anonymous partner chose {choice1}. My reward was ${reward2}."
  )

  return choice1, choice2, reward1, reward2


def get_non_popular_options(last_round_counts: List[dict]) -> List[str]:
  """Returns list of options that are not the most popular from last round.

  Args:
      choice_counts: List of dictionaries containing choice frequencies per round

  Returns:
      List of options excluding the most popular one from the last round
  """
  # if not choice_counts:  # If no rounds have been played yet
  #   return ["guava", "lychee", "dragonfruit", "persimmon", "papaya"]

  # Get the counts from the last round
  # last_round_counts = dict(choice_counts[-2])

  # if not last_round_counts:  # If last round has no data
  #   return ["guava", "lychee", "dragonfruit", "persimmon", "papaya"]

  # Find the most chosen option
  # from ipdb import set_trace

  # set_trace()
  most_popular = max(last_round_counts.items(), key=lambda x: x[1])[0]

  # Return all options except the most popular
  options = ["guava", "lychee", "dragonfruit", "persimmon", "papaya"]
  options.remove(most_popular)
  return options


def main():
  num_agents = 30
  num_rounds = 40
  disruption_percentage = 0.35
  disruption_triggered = False  # New flag to track if disruption has occurred

  agents = [create_agent(f"Agent_{i+1}") for i in range(num_agents)]
  choice_counts = []

  for round_num in range(num_rounds):
    choice_counts.append(defaultdict(lambda: 0))
    print(f"\nRound {round_num + 1}:")

    # Check if previous round had >50% consensus and disruption hasn't happened yet
    if (
        round_num > 0
        and not disruption_triggered  # New condition
        and max(choice_counts[round_num - 1].values()) > num_agents * 0.5
        # and False
    ):

      available_options = get_non_popular_options(choice_counts[round_num - 1])
      forced_choice = random.choice(available_options)

      # Select and disrupt agents
      num_to_disrupt = int(len(agents) * disruption_percentage)
      agents_to_disrupt = random.sample(agents, num_to_disrupt)

      # Set disruption status
      for agent in agents:
        if agent in agents_to_disrupt:
          agent.get_act_component().set_disruption(forced_choice)

      print(
          f"\033[38;5;215m\n[!] Disrupting {num_to_disrupt} agents to choose"
          f" {forced_choice}\033[0m"
      )
      disruption_triggered = True  # Mark that disruption has occurred

    agent_pairs = []
    available_agents = agents.copy()
    random.shuffle(available_agents)

    while len(available_agents) >= 2:
      agent1 = available_agents.pop()
      agent2 = available_agents.pop()
      agent_pairs.append((agent1, agent2))

    # Create tasks dictionary for concurrent execution
    tasks = {}
    for i, (agent1, agent2) in enumerate(agent_pairs):
      tasks[f"pair_{i}"] = lambda a1=agent1, a2=agent2: play_round_pair(
          a1, a2, round_num + 1, num_rounds
      )

    # Run tasks concurrently and get results
    try:
      results = concurrency.run_tasks(tasks, timeout=30)  # 30 second timeout

      # Process results
      for choice1, choice2, reward1, reward2 in results.values():
        # print(f"Choices: {choice1}, {choice2}, Rewards: ${reward1}, ${reward2}")
        choice_counts[round_num][choice1] += 1
        choice_counts[round_num][choice2] += 1
    except Exception as e:
      print(f"Error during concurrent execution: {e}")

    print("\nFinal Scores:")
    total_score = 0
    for agent in agents:
      # print(f"{agent.name}: ${agent.score}")
      total_score += agent.score

    print(f"\033[33mTotal Game Score: ${total_score}\033[0m")  # Yellow text
    print("\nChoice counts:")
    # for counts in :
    print(f"Round {round_num + 1}: {dict(choice_counts[-1])}")

  from IPython import embed

  embed()


if __name__ == "__main__":
  main()
