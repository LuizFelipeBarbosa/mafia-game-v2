import asyncio
from typing import Dict

# Asyncio locks per game to prevent race conditions during phase advancement
game_locks: Dict[str, asyncio.Lock] = {}

def get_game_lock(game_id: str) -> asyncio.Lock:
    if game_id not in game_locks:
        game_locks[game_id] = asyncio.Lock()
    return game_locks[game_id]
