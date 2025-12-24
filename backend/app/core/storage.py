import asyncio
from typing import Dict
from .schemas import GameState

# In-memory storage for active games
games: Dict[str, GameState] = {}

def get_game(game_id: str) -> GameState:
    return games.get(game_id)

def save_game(game_id: str, state: GameState):
    games[game_id] = state
