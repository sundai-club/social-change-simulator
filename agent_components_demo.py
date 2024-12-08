# agent_components_demo.py

import collections
import os
from typing import List, Tuple, Optional
from PIL import Image
import random
from collections import Counter
import time

import sentence_transformers

from concordia import typing
from concordia.agents import entity_agent
from concordia.associative_memory import associative_memory
from concordia.components.agent import action_spec_ignored, memory_component
from concordia.language_model import gpt_model, language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity, entity_component

# Initialize embedder
_embedder_model = sentence_transformers.SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)
embedder = lambda x: _embedder_model.encode(x, show_progress_bar=False)

# Initialize language model (you'll need to set your API key)
GPT_MODEL_NAME = "gpt-4"  # Or your preferred model

model = gpt_model.GptLanguageModel(
    api_key=GPT_API_KEY, model_name=GPT_MODEL_NAME
)


# Simple apple-eating component
class AppleEating(entity_component.ActingComponent):

  def get_action_attempt(self, context, action_spec) -> str:
    return "Eat the apple."


# Component that observes and stores in memory
class Observe(entity_component.ContextComponent):

  def pre_observe(self, observation: str) -> None:
    self.get_entity().get_component("memory").add(observation, {})


# Component that retrieves recent memories
class RecentMemories(entity_component.ContextComponent):

  def pre_act(self, action_spec) -> None:
    recent_memories_list = (
        self.get_entity()
        .get_component("memory")
        .retrieve(
            query="",  # Don't need a query to retrieve recent memories.
            limit=5,
            scoring_fn=legacy_associative_memory.RetrieveRecent(),
        )
    )
    recent_memories = " ".join(memory.text for memory in recent_memories_list)
    print(f"*****\nDEBUG: Recent memories:\n  {recent_memories}\n*****")
    return recent_memories


# Component that acts based on context
class NameGuessing(entity_component.ActingComponent):
    def __init__(self, model: language_model.LanguageModel):
        self._model = model
        self._current_items: Optional[List[str]] = None
        self._partner_guess: Optional[str] = None
        self._reward_history: List[float] = []
        self._last_guess: Optional[str] = None
        self._currency = 1.00  # Start with $1
        
    def get_currency(self) -> float:
        return self._currency
        
    def spend(self, amount: float) -> bool:
        if self._currency >= amount:
            self._currency -= amount
            return True
        return False
        
    def add_currency(self, amount: float):
        self._currency += amount

    def set_items(self, items: List[str]):
        self._current_items = items
        
    def set_partner_guess(self, guess: str):
        self._partner_guess = guess

    def get_action_attempt(self, contexts, action_spec) -> str:
        if not self._current_items:
            return "No items provided to choose from."
            
        if not self.spend(0.10):  # Try to spend 10 cents
            return "Insufficient funds to make a choice."
            
        context_for_action = "\n".join(
            f"{name}: {context}" for name, context in contexts.items()
        )
        
        # Include reward history in the prompt
        reward_context = ""
        if self._reward_history:
            reward_context = f"\nYour previous choice '{self._last_guess}' received a reward of {self._reward_history[-1]}."
        
        prompt = f"""Looking at these items: {', '.join(self._current_items)}, which would you choose? 
        If you received your partner's choice of '{self._partner_guess}', consider it in your response.
        {reward_context}
        
        {context_for_action}

        Which item would you choose? Respond with ONLY the item name (apple or banana) with no additional text."""
        
        guess = self._model.sample_text(prompt).strip().lower()
        # Clean up the response to ensure it's just the item name
        if 'apple' in guess:
            guess = 'apple'
        elif 'banana' in guess:
            guess = 'banana'
            
        self._last_guess = guess
        return guess

    def receive_reward(self, reward: float):
        self._reward_history.append(reward)


# Improved recent memories component using ActionSpecIgnored
class RecentMemoriesImproved(action_spec_ignored.ActionSpecIgnored):

  def __init__(self):
    super().__init__("Recent memories")

  def _make_pre_act_value(self) -> str:
    recent_memories_list = (
        self.get_entity()
        .get_component("memory")
        .retrieve(
            query="",
            limit=5,
            scoring_fn=legacy_associative_memory.RetrieveRecent(),
        )
    )
    recent_memories = " ".join(memory.text for memory in recent_memories_list)
    print(f"*****\nDEBUG: Recent memories:\n  {recent_memories}\n*****")
    return recent_memories


# Helper function for relevant memories
def _recent_memories_str_to_list(recent_memories: str) -> list[str]:
  return [memory.strip() + "." for memory in recent_memories.split(".")]


# Relevant memories component
class RelevantMemories(action_spec_ignored.ActionSpecIgnored):

  def __init__(self):
    super().__init__("Relevant memories")

  def _make_pre_act_value(self) -> str:
    recent_memories = (
        self.get_entity().get_component("recent_memories").get_pre_act_value()
    )
    recent_memories_list = _recent_memories_str_to_list(recent_memories)
    recent_memories_set = set(recent_memories_list)
    memory = self.get_entity().get_component("memory")
    relevant_memories_list = []

    for recent_memory in recent_memories_list:
      relevant = memory.retrieve(
          query=recent_memory,
          limit=3,
          scoring_fn=legacy_associative_memory.RetrieveAssociative(
              add_time=False
          ),
      )
      for mem in relevant:
        if mem.text not in recent_memories_set:
          relevant_memories_list.append(mem.text)
          recent_memories_set.add(mem.text)

    relevant_memories = "\n".join(relevant_memories_list)
    print(f"*****\nDEBUG: Relevant memories:\n{relevant_memories}\n*****")
    return relevant_memories


# Add a new class to manage the name guessing game
class NameGuessingGame:
    def __init__(self, num_agents: int = 20, convergence_threshold: float = 0.7):
        self.num_agents = num_agents
        self.convergence_threshold = convergence_threshold
        self.agents: List[entity_agent.EntityAgent] = []
        self.pairs: List[Tuple[entity_agent.EntityAgent, entity_agent.EntityAgent]] = []
        self.rounds_played = 0
        self.start_time = None
        
        # Create agents
        for i in range(num_agents):
            raw_memory = legacy_associative_memory.AssociativeMemoryBank(
                associative_memory.AssociativeMemory(embedder)
            )
            
            agent = entity_agent.EntityAgent(
                f"Agent_{i}",
                act_component=NameGuessing(model),
                context_components={
                    "observation": Observe(),
                    "recent_memories": RecentMemoriesImproved(),
                    "memory": memory_component.MemoryComponent(raw_memory),
                }
            )
            self.agents.append(agent)
            
    def create_pairs(self):
        """Randomly pair agents"""
        available_agents = self.agents.copy()
        random.shuffle(available_agents)
        
        while len(available_agents) >= 2:
            agent1 = available_agents.pop()
            agent2 = available_agents.pop()
            self.pairs.append((agent1, agent2))
    
    def calculate_rewards(self, final_guesses: dict) -> dict:
        """Calculate rewards based on name agreement"""
        # Count the frequency of each name
        name_counts = Counter(guess.lower() for guess in final_guesses.values())
        total_agents = len(self.agents)
        
        # Calculate rewards for each agent
        rewards = {}
        for agent_name, guess in final_guesses.items():
            # Reward is the proportion of agents who agreed on this name
            reward = name_counts[guess.lower()] / total_agents
            rewards[agent_name] = reward
            
        return rewards

    def check_convergence(self, final_guesses: dict) -> bool:
        """Check if agents have converged on a name"""
        # Count the frequency of each name
        name_counts = Counter(guess.lower() for guess in final_guesses.values())
        
        # Check if any name has reached the convergence threshold
        most_common_name, count = name_counts.most_common(1)[0]
        return count / len(self.agents) >= self.convergence_threshold

    def play_round(self, items: List[str]) -> Tuple[dict, dict, dict]:
        """Play one round and return initial guesses, final guesses, and rewards"""
        # Check if agents have sufficient funds
        active_agents = [
            agent for agent in self.agents 
            if agent._act_component.get_currency() >= 0.10
        ]
        
        if len(active_agents) < 2:
            print("\nNot enough agents with sufficient funds to continue.")
            return {}, {}, {}
            
        # Set the items for all agents
        for agent in active_agents:
            agent._act_component.set_items(items)
            
        # First round of choices
        initial_guesses = {}
        for agent in active_agents:
            guess = agent.act()
            if guess != "Insufficient funds to make a choice.":
                initial_guesses[agent.name] = guess
            
        # Share choices between pairs and get final choices
        final_guesses = {}
        for agent1, agent2 in self.pairs:
            if (agent1.name in initial_guesses and 
                agent2.name in initial_guesses):
                agent1._act_component.set_partner_guess(initial_guesses[agent2.name])
                agent2._act_component.set_partner_guess(initial_guesses[agent1.name])
                
                final_guess1 = agent1.act()
                final_guess2 = agent2.act()
                
                if final_guess1 != "Insufficient funds to make a choice.":
                    final_guesses[agent1.name] = final_guess1
                if final_guess2 != "Insufficient funds to make a choice.":
                    final_guesses[agent2.name] = final_guess2
            
        # Calculate rewards
        rewards = self.calculate_rewards(final_guesses)
        
        # Distribute rewards to agents
        for agent_name, reward in rewards.items():
            agent = next(a for a in self.agents if a.name == agent_name)
            agent._act_component.receive_reward(reward)
        
        return initial_guesses, final_guesses, rewards

    def play_until_convergence(self, items: List[str], max_rounds: int = 100):
        """Play rounds until convergence or max_rounds is reached"""
        self.start_time = time.time()
        self.rounds_played = 0
        
        while self.rounds_played < max_rounds:
            self.rounds_played += 1
            initial_guesses, final_guesses, rewards = self.play_round(items)
            
            # Print round results
            most_common = Counter(guess.lower() for guess in final_guesses.values()).most_common(1)[0]
            print(f"\nRound {self.rounds_played}:")
            print(f"Most common choice: {most_common[0]} ({most_common[1]} votes)")
            print(f"Average reward: {sum(rewards.values()) / len(rewards):.3f}")
            
            if self.check_convergence(final_guesses):
                elapsed_time = time.time() - self.start_time
                print(f"\nConverged after {self.rounds_played} rounds!")
                print(f"Time taken: {elapsed_time:.2f} seconds")
                return True
                
        print("\nFailed to converge within maximum rounds")
        return False


def main():
    # Create the game with 4 agents
    game = NameGuessingGame(num_agents=4, convergence_threshold=1.0)
    game.create_pairs()
    
    # Define the items to choose from
    items = ["apple", "banana"]
    
    # Play exactly 3 rounds or until agents run out of money
    print("\nStarting game with 4 agents for 3 rounds...")
    print("Each agent starts with $1.00 and each round costs $0.10")
    
    for round_num in range(3):
        print(f"\n=== ROUND {round_num + 1} ===")
        
        # Print current balances
        print("\nCurrent balances:")
        for agent in game.agents:
            balance = agent._act_component.get_currency()
            print(f"{agent.name}: ${balance:.2f}")
            
        initial_guesses, final_guesses, rewards = game.play_round(items)
        
        if not final_guesses:  # No active agents with sufficient funds
            print("Game over - insufficient funds to continue.")
            break
            
        # Print detailed results for each round
        print("\nInitial choices:")
        for agent_name, choice in initial_guesses.items():
            print(f"{agent_name}: {choice}")
            
        print("\nFinal choices after consulting partners:")
        for agent_name, choice in final_guesses.items():
            print(f"{agent_name}: {choice}")
            
        print("\nRewards:")
        for agent_name, reward in rewards.items():
            print(f"{agent_name}: {reward:.2f}")
        
        # Print consensus information
        choices_count = Counter(choice.lower() for choice in final_guesses.values())
        most_common = choices_count.most_common(1)[0]
        print(f"\nConsensus level: {most_common[0]} chosen by {most_common[1]}/{len(final_guesses)} agents")


if __name__ == "__main__":
    main()
