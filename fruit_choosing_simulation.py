import os
import random
from typing import List

from concordia.agents import entity_agent
from concordia.language_model import gpt_model
from concordia.typing import entity_component

# Initialize language model
model = gpt_model.GptLanguageModel(
    api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4o"
)


class SimpleMemoryComponent(entity_component.ContextComponent):

  def __init__(self):
    self.memories: List[str] = []

  def add(self, observation: str) -> None:
    self.memories.append(observation)

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

  def get_action_attempt(self, contexts, action_spec) -> str:
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

    print(f"\033[36mDEBUG: Prompt:\n{prompt}\n\033[0m")
    response = model.sample_text(prompt).strip().lower()

    print(f"\033[31mAgent {self.get_entity().name} chose: {response}\033[0m")
    return response


def create_agent(name: str) -> entity_agent.EntityAgent:
  agent = entity_agent.EntityAgent(
      name,
      act_component=CoordinationGameComponent(),
      context_components={
          "observation": SimpleObserveComponent(),
          "recent_memories": RecentMemories(),
          "memory": SimpleMemoryComponent(),
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


def main():
  num_agents = 4
  num_rounds = 30

  agents = [create_agent(f"Agent_{i+1}") for i in range(num_agents)]

  for round_num in range(num_rounds):
    print(f"\nRound {round_num + 1}:")

    agent_pairs = []
    available_agents = agents.copy()
    random.shuffle(available_agents)

    while len(available_agents) >= 2:
      agent1 = available_agents.pop()
      agent2 = available_agents.pop()
      agent_pairs.append((agent1, agent2))

    for agent1, agent2 in agent_pairs:
      choice1, choice2, reward1, reward2 = play_round_pair(
          agent1, agent2, round_num + 1, num_rounds
      )
      print(f"{agent1.name} chose: {choice1}, got reward: ${reward1}")
      print(f"{agent2.name} chose: {choice2}, got reward: ${reward2}")

    print("\nAgent memories:")
    for agent in agents:
      print(f"\n{agent.name}'s memories:")
      print(agent.get_component("memory").get_recent_memories())

    print("\nFinal Scores:")
    total_score = 0
    for agent in agents:
      print(f"{agent.name}: ${agent.score}")
      total_score += agent.score

    print(f"\033[33mTotal Game Score: ${total_score}\033[0m")  # Yellow text


if __name__ == "__main__":
  main()