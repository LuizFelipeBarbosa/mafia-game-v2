from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class Phase(str, Enum):
    DISCUSSION = "Discussion"
    VOTING = "Voting"
    DEFENSE = "Defense"
    JUDGMENT = "Judgment"
    LAST_WORDS = "Last Words"
    NIGHT = "Night"
    GAME_OVER = "Game Over"

class Role(str, Enum):
    VILLAGER = "Villager"
    MAFIA = "Mafia"
    DETECTIVE = "Detective"

class GameConfig(BaseModel):
    num_players: int = 5
    num_mafia: int = 1
    has_detective: bool = True
    phase_durations: Dict[Phase, int] = {
        Phase.DISCUSSION: 45,
        Phase.VOTING: 30,
        Phase.DEFENSE: 20,
        Phase.JUDGMENT: 20,
        Phase.LAST_WORDS: 5,
        Phase.NIGHT: 35
    }
    model: str = "openai/gpt-3.5-turbo" # Default model

class Player(BaseModel):
    id: str
    name: str
    role: Role
    is_alive: bool = True
    private_memory: List[str] = []
    role_info: str = "" # Private info (e.g. mafia teammates)

class GameState(BaseModel):
    game_id: str
    config: GameConfig
    day: int = 1
    phase: Phase = Phase.DISCUSSION
    seconds_remaining: int = 45
    players: List[Player] = []
    transcript: List[Dict[str, Any]] = []
    is_paused: bool = True
    accused_id: Optional[str] = None
    last_phase_remaining_time: Optional[int] = None # For resuming voting phase
    pending_kills: List[str] = [] # List of player IDs killed during the night
    winner: Optional[str] = None
    voting_complete: bool = False  # Track if voting has been done for this phase
    mafia_votes_collected: bool = False  # Track if mafia structured voting is done for this night
    mafia_discussion_done: bool = False  # Track if mafia discussion is done for this night
