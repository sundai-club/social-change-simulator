import json
import time
from collections import Counter
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Any, Optional


class GameLogger:
    def __init__(self, game_id: str, num_agents: int, convergence_threshold: float, options: List[str]):
        """Initialize a new game logger.

        Args:
            game_id: Unique identifier for the game
            num_agents: Number of agents participating in the game
            convergence_threshold: Threshold for determining convergence
            options: List of possible choices for agents
        """
        self.game_data = {
            "game_id": game_id,
            "num_agents": num_agents,
            "convergence_threshold": convergence_threshold,
            "options": options,
            "rounds_played": 0,
            "start_time_utc": None,
            "end_time_utc": None,
            "final_convergence": {},
            "agents": [],
            "rounds": [],
            "statistics": {}
        }
        self.agent_histories: Dict[str, List[Dict[str, Any]]] = {}
        self.current_round_data: Dict[str, Any] = {}

    def start_game(self) -> None:
        """Record the start time of the game."""
        self.game_data["start_time_utc"] = datetime.now(timezone.utc).isoformat()

    def log_agent(self, agent_id: str, strategy: str, initial_currency: float) -> None:
        """Log a new agent's initial state.

        Args:
            agent_id: Unique identifier for the agent
            strategy: Strategy type of the agent
            initial_currency: Starting currency amount
        """
        agent_data = {
            "agent_id": agent_id,
            "strategy": strategy,
            "initial_currency": initial_currency,
            "final_currency": initial_currency,  # Will be updated later
            "history": []
        }
        self.game_data["agents"].append(agent_data)
        self.agent_histories[agent_id] = []

    def update_agent(self, agent_id: str, round_num: int, partner_id: str,
                    initial_guess: str, final_guess: str, reward: float) -> None:
        """Update an agent's history with round results.

        Args:
            agent_id: Agent's identifier
            round_num: Current round number
            partner_id: Partner agent's identifier
            initial_guess: Agent's initial guess
            final_guess: Agent's final guess after consultation
            reward: Reward received for this round
        """
        round_data = {
            "round": round_num,
            "partner_id": partner_id,
            "initial_guess": initial_guess,
            "final_guess": final_guess,
            "reward": reward
        }
        self.agent_histories[agent_id].append(round_data)

        # Update the agent's final currency
        for agent in self.game_data["agents"]:
            if agent["agent_id"] == agent_id:
                agent["final_currency"] = agent["final_currency"] + reward
                agent["history"] = self.agent_histories[agent_id]
                break

    def log_round(self, round_num: int, pairs: List[Tuple[str, str]],
                 initial_guesses: Dict[str, str], final_guesses: Dict[str, str],
                 rewards: Dict[str, float]) -> None:
        """Log the results of a complete round.

        Args:
            round_num: Current round number
            pairs: List of agent pairs for this round
            initial_guesses: Dictionary of initial guesses by agent
            final_guesses: Dictionary of final guesses by agent
            rewards: Dictionary of rewards by agent
        """
        round_data = {
            "round_number": round_num,
            "pairs": []
        }

        for agent1_id, agent2_id in pairs:
            pair_data = {
                "agent_1_id": agent1_id,
                "agent_2_id": agent2_id,
                "initial_guesses": {
                    agent1_id: initial_guesses.get(agent1_id, ""),
                    agent2_id: initial_guesses.get(agent2_id, "")
                },
                "final_guesses": {
                    agent1_id: final_guesses.get(agent1_id, ""),
                    agent2_id: final_guesses.get(agent2_id, "")
                },
                "rewards": {
                    agent1_id: rewards.get(agent1_id, 0.0),
                    agent2_id: rewards.get(agent2_id, 0.0)
                }
            }
            round_data["pairs"].append(pair_data)

        self.game_data["rounds"].append(round_data)
        self.game_data["rounds_played"] = round_num

    def finalize_convergence(self, converged: bool, winning_item: str, percentage: float) -> None:
        """Record the final convergence state.

        Args:
            converged: Whether the game converged
            winning_item: The item that won (most chosen)
            percentage: Percentage of agents that chose the winning item
        """
        self.game_data["final_convergence"] = {
            "converged": converged,
            "winning_item": winning_item,
            "percentage": percentage
        }

    def compute_statistics(self) -> None:
        """Compute and record final game statistics."""
        # Calculate total rewards
        total_rewards = sum(
            reward
            for round_data in self.game_data["rounds"]
            for pair in round_data["pairs"]
            for reward in pair["rewards"].values()
        )

        # Find most popular item per round
        most_popular_items = {}
        for round_data in self.game_data["rounds"]:
            round_num = round_data["round_number"]
            all_guesses = []
            for pair in round_data["pairs"]:
                all_guesses.extend(pair["final_guesses"].values())
            if all_guesses:
                most_common = Counter(all_guesses).most_common(1)[0][0]
                most_popular_items[str(round_num)] = most_common

        self.game_data["statistics"] = {
            "total_rewards_distributed": total_rewards,
            "average_reward_per_agent": round(total_rewards / self.game_data["num_agents"], 2),
            "most_popular_item_per_round": most_popular_items,
            "rounds_until_convergence": (
                self.game_data["rounds_played"]
                if self.game_data["final_convergence"].get("converged", False)
                else None
            )
        }

    def end_game(self) -> None:
        """Record the end time of the game."""
        self.game_data["end_time_utc"] = datetime.now(timezone.utc).isoformat()

    def save_to_file(self, filename: str) -> None:
        """Save the game data to a JSON file. If the file exists, append the new game data,
        if not, create a new file with a list of games.

        Args:
            filename: Name of the file to save to
        """
        try:
            # Try to read existing file
            with open(filename, 'r') as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        # Convert single game to list if necessary
                        existing_data = [existing_data] if existing_data else []
                except json.JSONDecodeError:
                    # If file is empty or invalid JSON, start fresh
                    existing_data = []
        except FileNotFoundError:
            # If file doesn't exist, start with empty list
            existing_data = []

        # Append new game data
        existing_data.append(self.game_data)

        # Write back to file
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=4)
