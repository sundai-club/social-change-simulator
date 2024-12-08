# agent_components_demo.py

import collections
import os

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
GPT_API_KEY = os.getenv("OPENAI_API_KEY")
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
class SimpleActing(entity_component.ActingComponent):

  def __init__(self, model: language_model.LanguageModel):
    self._model = model

  def get_action_attempt(self, contexts, action_spec) -> str:
    context_for_action = "\n".join(
        f"{name}: {context}" for name, context in contexts.items()
    )
    print(f"*****\nDEBUG:\n  context_for_action:\n{context_for_action}\n*****")
    call_to_action = action_spec.call_to_action.format(
        name=self.get_entity().name, timedelta="2 minutes"
    )
    sampled_text = self._model.sample_text(
        f"{context_for_action}\n\n{call_to_action}\n",
    )
    return sampled_text


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


def main():
  # Test the simple apple-eating agent
  print("\nTesting simple apple-eating agent:")
  agent = entity_agent.EntityAgent("Alice", act_component=AppleEating())
  print(agent.act())

  # Test agent with memory components
  print("\nTesting agent with memory components:")
  raw_memory = legacy_associative_memory.AssociativeMemoryBank(
      associative_memory.AssociativeMemory(embedder)
  )

  agent = entity_agent.EntityAgent(
      "Alice",
      act_component=SimpleActing(model),
      context_components={
          "observation": Observe(),
          "recent_memories": RecentMemories(),
          "memory": memory_component.MemoryComponent(raw_memory),
      },
  )

  # Add observations
  agent.observe(
      "You absolutely hate apples and would never willingly eat them."
  )
  agent.observe("You don't particularly like bananas.")
  agent.observe("You are in a room.")
  agent.observe("The room has only a table in it.")
  agent.observe("On the table there are two fruits: an apple and a banana.")
  agent.observe("The apple is shinny red and looks absolutely irresistible!")
  agent.observe("The banana is slightly past its prime.")

  print(agent.act())

  # Test agent with improved memory components
  print("\nTesting agent with improved memory components:")
  raw_memory = legacy_associative_memory.AssociativeMemoryBank(
      associative_memory.AssociativeMemory(embedder)
  )

  agent = entity_agent.EntityAgent(
      "Alice",
      act_component=SimpleActing(model),
      context_components={
          "observation": Observe(),
          "relevant_memories": RelevantMemories(),
          "recent_memories": RecentMemoriesImproved(),
          "memory": memory_component.MemoryComponent(raw_memory),
      },
  )

  # Add same observations
  agent.observe(
      "You absolutely hate apples and would never willingly eat them."
  )
  agent.observe("You don't particularly like bananas.")
  agent.observe("You are in a room.")
  agent.observe("The room has only a table in it.")
  agent.observe("On the table there are two fruits: an apple and a banana.")
  agent.observe("The apple is shinny red and looks absolutely irresistible!")
  agent.observe("The banana is slightly past its prime.")

  print(agent.act())


if __name__ == "__main__":
  main()
