# agent_components_demo.py

import collections
import os
from typing import List, Tuple, Optional
from PIL import Image
import random
from collections import Counter
import time
import warnings
import pandas as pd

import sentence_transformers

from concordia import typing
from concordia.agents import entity_agent
from concordia.associative_memory import associative_memory
from concordia.components.agent import action_spec_ignored, memory_component
from concordia.language_model import gpt_model, language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity, entity_component

from game_logger import GameLogger

# Suppress FutureWarning about DataFrame concatenation
warnings.simplefilter(action='ignore', category=FutureWarning)

# Initialize embedder
_embedder_model = sentence_transformers.SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)
embedder = lambda x: _embedder_model.encode(x, show_progress_bar=False)

# Initialize language model (you'll need to set your API key)
GPT_MODEL_NAME = "gpt-4"  # Or your preferred model
GPT_API_KEY = ""

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
        
        prompt = f"""You are playing a game where you must choose between {', '.join(self._current_items)}.
        Your goal is to choose the same item as other players to maximize rewards.
        {f"Your partner chose: {self._partner_guess}" if self._partner_guess else ""}
        {reward_context}
        
        {context_for_action}

        Choose ONLY ONE item from the list by outputting just the item name in lowercase (apple or banana). 
        DO NOT add any extra text or explanation. Just output the single word 'apple' or 'banana'."""
        
        response = self._model.sample_text(prompt).strip().lower()
        
        # Strictly enforce response format
        if response == 'apple' or response == 'banana':
            self._last_guess = response
            return response
        else:
            # Default to first item if response is invalid
            self._last_guess = self._current_items[0].lower()
            return self._last_guess

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
    if recent_memories_list:  # Only print if there are memories
        recent_memories = " ".join(memory.text for memory in recent_memories_list)
        print(f"*****\nDEBUG: Recent memories:\n  {recent_memories}\n*****")
        return recent_memories
    return ""


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
    """A game where agents try to guess the same name."""

    def __init__(self, num_agents: int = 4, convergence_threshold: float = 0.75):
        """Initialize the game with the given number of agents."""
        self.num_agents = num_agents
        self.convergence_threshold = convergence_threshold
        self.agents = []
        self.pairs = []
        self.rounds_played = 0
        
        # Initialize game logger
        self.logger = GameLogger(
            game_id=str(int(time.time())),
            num_agents=num_agents,
            convergence_threshold=convergence_threshold,
            options=["apple", "banana"]
        )
        self.logger.start_game()
        
        # Create agents
        for i in range(num_agents):
            # Create the agent's components
            act_component = NameGuessing(model)
            raw_memory = legacy_associative_memory.AssociativeMemoryBank(
                associative_memory.AssociativeMemory(embedder)
            )
            
            # Create the agent
            agent = entity_agent.EntityAgent(
                f"Agent_{i}",
                act_component=act_component,
                context_components={
                    "observation": Observe(),
                    "memory": memory_component.MemoryComponent(raw_memory),
                    "recent_memories": RecentMemoriesImproved(),  
                }
            )
            
            # Add initial observation to memory
            agent.get_component("observation").pre_observe(f"I am Agent_{i}")
            
            self.agents.append(agent)
            
            # Log the agent in the game logger
            self.logger.log_agent(
                agent_id=agent.name,
                strategy="random_with_influence",
                initial_currency=1.0
            )

    def create_pairs(self):
        """Randomly pair agents for interaction."""
        available_agents = self.agents.copy()
        random.shuffle(available_agents)
        self.pairs = []
        
        while len(available_agents) >= 2:
            agent1 = available_agents.pop()
            agent2 = available_agents.pop()
            self.pairs.append((agent1, agent2))

    def calculate_rewards(self, final_guesses: dict) -> dict:
        """Calculate rewards based on name agreement."""
        rewards = {}
        for agent1, agent2 in self.pairs:
            if agent1.name in final_guesses and agent2.name in final_guesses:
                guess1 = final_guesses[agent1.name].lower()
                guess2 = final_guesses[agent2.name].lower()
                
                # Both agents get 0.5 if they agree
                if guess1 == guess2:
                    rewards[agent1.name] = 0.5
                    rewards[agent2.name] = 0.5
                else:
                    rewards[agent1.name] = 0.1
                    rewards[agent2.name] = 0.1
        return rewards

    def check_convergence(self, final_guesses: dict) -> bool:
        """Check if agents have converged on a name."""
        if not final_guesses:
            return False
            
        # Count occurrences of each guess
        guess_counts = Counter(guess.lower() for guess in final_guesses.values())
        
        # Find the most common guess and its count
        most_common = guess_counts.most_common(1)[0]
        percentage = most_common[1] / len(final_guesses)
        
        # Log convergence state
        self.logger.finalize_convergence(
            converged=percentage >= self.convergence_threshold,
            winning_item=most_common[0],
            percentage=percentage * 100
        )
        
        return percentage >= self.convergence_threshold

    def play_round(self, items: List[str]) -> Tuple[dict, dict, dict]:
        """Play one round and return initial guesses, final guesses, and rewards."""
        self.rounds_played += 1  # Increment round counter at start of round
        self.create_pairs()
        initial_guesses = {}
        final_guesses = {}
        
        # First, get initial guesses from all agents
        for agent in self.agents:
            act_component = agent._act_component
            if act_component.get_currency() >= 0.10:  # Check if agent can afford to play
                act_component.set_items(items)
                initial_guess = agent.act()
                if initial_guess:
                    initial_guesses[agent.name] = initial_guess
        
        # Then, let agents consult with their partners
        for agent1, agent2 in self.pairs:
            if agent1.name in initial_guesses and agent2.name in initial_guesses:
                # Share guesses
                agent1._act_component.set_partner_guess(initial_guesses[agent2.name])
                agent2._act_component.set_partner_guess(initial_guesses[agent1.name])
                
                # Make final decisions
                final_guess1 = agent1.act()
                final_guess2 = agent2.act()
                
                if final_guess1 and final_guess2:
                    final_guesses[agent1.name] = final_guess1
                    final_guesses[agent2.name] = final_guess2
        
        # Calculate and distribute rewards
        rewards = self.calculate_rewards(final_guesses)
        for agent_name, reward in rewards.items():
            for agent in self.agents:
                if agent.name == agent_name:
                    agent._act_component.receive_reward(reward)
                    break
        
        # Log the round
        self.logger.log_round(
            round_num=self.rounds_played,  # Use current round number
            pairs=[(a1.name, a2.name) for a1, a2 in self.pairs],
            initial_guesses=initial_guesses,
            final_guesses=final_guesses,
            rewards=rewards
        )
        
        # Update agent histories
        for agent in self.agents:
            if agent.name in final_guesses:
                # Find partner for this agent
                partner_id = ""
                for a1, a2 in self.pairs:
                    if a1.name == agent.name:
                        partner_id = a2.name
                        break
                    elif a2.name == agent.name:
                        partner_id = a1.name
                        break
                
                self.logger.update_agent(
                    agent_id=agent.name,
                    round_num=self.rounds_played,  # Use current round number
                    partner_id=partner_id,
                    initial_guess=initial_guesses.get(agent.name, ""),
                    final_guess=final_guesses[agent.name],
                    reward=rewards.get(agent.name, 0.0)
                )
        
        return initial_guesses, final_guesses, rewards

    def play_until_convergence(self, items: List[str], max_rounds: int = 100):
        """Play rounds until convergence or max_rounds is reached."""
        while self.rounds_played < max_rounds:
            initial_guesses, final_guesses, rewards = self.play_round(items)
            
            # Print round results
            most_common = Counter(guess.lower() for guess in final_guesses.values()).most_common(1)[0]
            print(f"\nRound {self.rounds_played}:")
            print(f"Most common choice: {most_common[0]} ({most_common[1]} votes)")
            print(f"Average reward: {sum(rewards.values()) / len(rewards):.3f}")
            
            if self.check_convergence(final_guesses):
                # Compute final statistics and save to file
                self.logger.compute_statistics()
                self.logger.end_game()
                self.logger.save_to_file("game_logs.json")
                
                print(f"\nConverged after {self.rounds_played} rounds!")
                return True
                
        print("\nFailed to converge within maximum rounds")
        return False


def main():
    # Create the game with 4 agents
    game = NameGuessingGame(num_agents=4, convergence_threshold=1.0)
    
    # Define the items to choose from
    items = ["apple", "banana"]
    
    # Play exactly 3 rounds or until agents run out of money
    print("\nStarting game with 4 agents for 3 rounds...")
    print("Each agent starts with $1.00 and each round costs $0.10")
    
    for _ in range(3):  # Don't use round_num, use game's internal counter
        print(f"\n=== ROUND {game.rounds_played + 1} ===")
        
        # Print current balances
        print("\nCurrent balances:")
        for agent in game.agents:
            balance = agent._act_component.get_currency()
            print(f"{agent.name}: ${balance:.2f}")
            
        initial_guesses, final_guesses, rewards = game.play_round(items)
        
        if not final_guesses:  # No active agents with sufficient funds
            print("Game over - insufficient funds to continue.")
            break
            
        print("\nFinal choices:")
        for agent_name, choice in final_guesses.items():
            print(f"{agent_name}: {choice}")
            
        print("\nRewards:")
        for agent_name, reward in rewards.items():
            print(f"{agent_name}: {reward:.2f}")
        
        # Print consensus information
        choices_count = Counter(choice.lower() for choice in final_guesses.values())
        most_common = choices_count.most_common(1)[0]
        print(f"\nConsensus level: {most_common[0]} chosen by {most_common[1]}/{len(final_guesses)} agents")
    
    # Compute final statistics and save game log
    game.logger.compute_statistics()
    game.logger.end_game()
    game.logger.save_to_file("game_logs.json")
    print("\nGame logs saved to game_logs.json")


if __name__ == "__main__":
    main()
