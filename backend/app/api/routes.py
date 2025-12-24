from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, status
from typing import List, Optional
import uuid
import asyncio
import time
import random
from .ws import manager
from ..core.schemas import GameState, GameConfig, Player, Role, Phase
from ..core.storage import save_game, get_game, games
from ..core.locks import get_game_lock
from ..core.engine_graph import create_game_graph, AgentState, collect_day_votes, collect_mafia_votes, handle_detective_investigation, handle_judgment, handle_last_words

router = APIRouter()
graph = create_game_graph()

async def check_win_condition(game: GameState, game_id: str) -> bool:
    """Check if game is over and broadcast result. Returns True if game ended."""
    mafia_count = len([p for p in game.players if p.role == Role.MAFIA and p.is_alive])
    town_count = len([p for p in game.players if p.role != Role.MAFIA and p.is_alive])
    
    if mafia_count == 0:
        game.winner = "TOWN"
        game.phase = Phase.GAME_OVER
        event = {
            "type": "game_over",
            "game_id": game_id,
            "ts": int(time.time()),
            "day": game.day,
            "phase": game.phase.value,
            "payload": {"winner": "TOWN", "message": "🎉 TOWN WINS! All mafia have been eliminated!"}
        }
        game.transcript.append(event)
        await manager.broadcast(game_id, event)
        save_game(game_id, game)
        return True
    
    if mafia_count >= town_count:
        game.winner = "MAFIA"
        game.phase = Phase.GAME_OVER
        event = {
            "type": "game_over",
            "game_id": game_id,
            "ts": int(time.time()),
            "day": game.day,
            "phase": game.phase.value,
            "payload": {"winner": "MAFIA", "message": "💀 MAFIA WINS! They now control the town!"}
        }
        game.transcript.append(event)
        await manager.broadcast(game_id, event)
        save_game(game_id, game)
        return True
    
    return False

async def game_loop(game_id: str):
    """Background task to manage phase timers and auto-advancement."""
    while True:
        game = get_game(game_id)
        if not game or game.phase == Phase.GAME_OVER:
            break
        
        if not game.is_paused:
            async with get_game_lock(game_id):
                if game.seconds_remaining > 0:
                    # Every 5 seconds, perform a step for discussion phases ONLY
                    # Voting happens at END of timer, not during the phase
                    if game.phase == Phase.DISCUSSION:
                        if game.seconds_remaining % 5 == 0:
                            await advance_game(game_id, is_timer_tick=False)
                    elif game.phase == Phase.NIGHT:
                        # During night, mafia discuss - also every 5 seconds
                        if game.seconds_remaining % 5 == 0:
                            await advance_game(game_id, is_timer_tick=False)
                    # Defense phase - accused speaks
                    elif game.phase == Phase.DEFENSE:
                        if game.seconds_remaining % 5 == 0:
                            await advance_game(game_id, is_timer_tick=False)
                    
                    game.seconds_remaining -= 1
                    # Broadcast tick
                    await manager.broadcast(game_id, {
                        "type": "phase_tick",
                        "game_id": game_id,
                        "day": game.day,
                        "phase": game.phase.value,
                        "payload": {"seconds_remaining": game.seconds_remaining}
                    })
                else:
                    # Phase time up - this is when we trigger end-of-round voting!
                    await advance_game(game_id, is_timer_tick=True)
        
        await asyncio.sleep(1)

async def advance_game(game_id: str, is_timer_tick: bool = False):
    """Advances the game to the next phase or step."""
    game = get_game(game_id)
    if not game: return

    # Track transcript length to find new events
    old_transcript_len = len(game.transcript)

    # === DURING PHASE (is_timer_tick=False): Just run discussion ===
    if not is_timer_tick:
        phase_to_node = {
            Phase.DISCUSSION: "discussion",
            Phase.DEFENSE: "defense",
            Phase.NIGHT: "night",
        }
        node = phase_to_node.get(game.phase)
        if node:
            state: AgentState = {"game_state": game, "last_event": None, "next_step": ""}
            result = await graph.ainvoke(state)
            game = result["game_state"]
            
            # Broadcast new events
            new_events = game.transcript[old_transcript_len:]
            for event in new_events:
                await manager.broadcast(game_id, event)
        
        save_game(game_id, game)
        return

    # === TIMER EXPIRED (is_timer_tick=True): Trigger end-of-round voting ===
    
    if game.phase == Phase.DISCUSSION:
        if game.day == 1:
            # Day 1: No voting, go straight to night
            game.phase = Phase.NIGHT
            game.seconds_remaining = game.config.phase_durations[Phase.NIGHT]
            game.mafia_discussion_done = False
            game.mafia_votes_collected = False
        else:
            # Day 2+: Collect votes from each player individually
            announce_event = {
                "type": "chat",
                "game_id": game_id,
                "ts": int(time.time()),
                "day": game.day,
                "phase": game.phase.value,
                "payload": {"speaker": "System", "text": "🗳️ VOTING TIME! Each player will now be asked who they want to put on trial..."}
            }
            game.transcript.append(announce_event)
            await manager.broadcast(game_id, announce_event)
            
            # Collect votes
            old_len = len(game.transcript)
            accused = await collect_day_votes(game)
            
            # Broadcast new events from voting
            for event in game.transcript[old_len:]:
                await manager.broadcast(game_id, event)
            
            if accused:
                # Someone nominated - go to defense
                game.accused_id = accused.id
                game.phase = Phase.DEFENSE
                game.seconds_remaining = game.config.phase_durations[Phase.DEFENSE]
            else:
                # No nomination - go to night
                game.phase = Phase.NIGHT
                game.seconds_remaining = game.config.phase_durations[Phase.NIGHT]
                game.mafia_discussion_done = False
                game.mafia_votes_collected = False
    
    elif game.phase == Phase.DEFENSE:
        # Defense time up - run judgment immediately (no separate JUDGMENT phase in game loop)
        announce_event = {
            "type": "chat",
            "game_id": game_id,
            "ts": int(time.time()),
            "day": game.day,
            "phase": "Judgment",
            "payload": {"speaker": "System", "text": "⚖️ JUDGMENT TIME! Each player will now vote on the accused's fate..."}
        }
        game.transcript.append(announce_event)
        await manager.broadcast(game_id, announce_event)
        
        # Set phase to JUDGMENT
        game.phase = Phase.JUDGMENT
        
        # Run judgment directly (not via graph which has wrong entry point)
        state: AgentState = {"game_state": game, "last_event": None, "next_step": ""}
        old_len = len(game.transcript)
        result = await handle_judgment(state)  # Direct call
        game = result["game_state"]
        
        # Broadcast judgment events
        for event in game.transcript[old_len:]:
            await manager.broadcast(game_id, event)
        
        # If guilty (phase is now LAST_WORDS), run execution immediately
        if game.phase == Phase.LAST_WORDS:
            # Run last words and execution immediately
            state = {"game_state": game, "last_event": None, "next_step": ""}
            old_len = len(game.transcript)
            result = await handle_last_words(state)  # Direct call
            game = result["game_state"]
            
            for event in game.transcript[old_len:]:
                await manager.broadcast(game_id, event)
            
            # Check win condition after execution
            if await check_win_condition(game, game_id):
                return
        
        # If innocent verdict sent back to VOTING, we need to go to night instead
        if game.phase == Phase.VOTING:
            game.phase = Phase.NIGHT
            game.seconds_remaining = game.config.phase_durations[Phase.NIGHT]
            game.accused_id = None
        
        # If phase is still JUDGMENT (shouldn't happen), fail safe to night
        if game.phase == Phase.JUDGMENT:
            game.phase = Phase.NIGHT
            game.seconds_remaining = game.config.phase_durations[Phase.NIGHT]
            game.accused_id = None
    
    elif game.phase == Phase.JUDGMENT:
        # Shouldn't reach here normally - fail safe to night
        game.phase = Phase.NIGHT
        game.seconds_remaining = game.config.phase_durations[Phase.NIGHT]
        game.accused_id = None
    
    elif game.phase == Phase.LAST_WORDS:
        # Run last words (execution happens inside the node)
        state: AgentState = {"game_state": game, "last_event": None, "next_step": ""}
        old_len = len(game.transcript)
        result = await graph.ainvoke(state)
        game = result["game_state"]
        
        for event in game.transcript[old_len:]:
            await manager.broadcast(game_id, event)
        
        # Check win condition after execution
        if await check_win_condition(game, game_id):
            return
    
    elif game.phase == Phase.NIGHT:
        # Night timer expired - NOW collect mafia votes
        # This announcement is private to mafia only
        announce_event = {
            "type": "mafia_chat",
            "game_id": game_id,
            "ts": int(time.time()),
            "day": game.day,
            "phase": game.phase.value,
            "payload": {"speaker": "System", "text": "⚰️ The Mafia now votes on their victim...", "private": True}
        }
        # Store in mafia_transcript (NOT the main transcript - town cannot see this)
        game.mafia_transcript.append(announce_event)
        # Do NOT broadcast to all clients - mafia messages are private
        
        # Collect mafia votes (also writes to mafia_transcript)
        await collect_mafia_votes(game)
        
        # No broadcasting of mafia votes - they are private
        
        # Detective investigation
        old_len = len(game.transcript)
        await handle_detective_investigation(game)
        for event in game.transcript[old_len:]:
            await manager.broadcast(game_id, event)
        
        # Advance to next day
        game.day += 1
        game.phase = Phase.DISCUSSION
        game.seconds_remaining = game.config.phase_durations[Phase.DISCUSSION]
        game.voting_complete = False
        game.mafia_votes_collected = False
        game.mafia_discussion_done = False
        
        # Process morning announcements (kills)
        if game.pending_kills:
            for killed_id in game.pending_kills:
                killed_player = next((p for p in game.players if p.id == killed_id), None)
                if killed_player:
                    killed_player.is_alive = False
                    event = {
                        "type": "night_result",
                        "game_id": game_id,
                        "ts": int(time.time()),
                        "day": game.day,
                        "phase": game.phase.value,
                        "payload": {"killed": killed_player.name, "role_revealed": killed_player.role.value}
                    }
                    game.transcript.append(event)
                    await manager.broadcast(game_id, event)
            game.pending_kills = []
        
        # Check win condition after night kills
        if await check_win_condition(game, game_id):
            return
    
    # Broadcast phase start if we transitioned
    await manager.broadcast(game_id, {
        "type": "phase_start",
        "game_id": game_id,
        "day": game.day,
        "phase": game.phase.value,
        "payload": {"duration_sec": game.seconds_remaining}
    })

    save_game(game_id, game)

@router.post("/games")
async def create_new_game(config: GameConfig):
    game_id = str(uuid.uuid4())
    players = []
    # Assign roles
    roles = [Role.MAFIA] * config.num_mafia
    if config.has_detective:
        roles.append(Role.DETECTIVE)
    roles += [Role.VILLAGER] * (config.num_players - len(roles))
    random.shuffle(roles)
    
    for i in range(config.num_players):
        role = roles[i]
        p = Player(
            id=f"Player_{i}",
            name=f"Player_{i}",
            role=role,
            role_info=f"Your name is {f'Player_{i}'}. You are {role.value}."
        )
        # Mafia knows teammates
        if role == Role.MAFIA:
            p.role_info += " Your teammates: " + ", ".join([f"Player_{j}" for j, r in enumerate(roles) if r == Role.MAFIA and i != j])
        players.append(p)

    game_state = GameState(
        game_id=game_id,
        config=config,
        players=players,
        seconds_remaining=10 if config.phase_durations[Phase.DISCUSSION] > 10 else config.phase_durations[Phase.DISCUSSION]
    )
    # Actually, let's just force 10s for Day 1 as per requirement
    game_state.seconds_remaining = 10
    save_game(game_id, game_state)
    return {"game_id": game_id, "ws_url": f"/games/{game_id}/stream"}

@router.post("/games/{game_id}/start")
async def start_game(game_id: str, background_tasks: BackgroundTasks):
    game = get_game(game_id)
    if not game: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    game.is_paused = False
    background_tasks.add_task(game_loop, game_id)
    return {"status": "started"}

@router.post("/games/{game_id}/pause")
async def pause_game(game_id: str):
    game = get_game(game_id)
    if not game: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    game.is_paused = True
    return {"status": "paused"}

@router.post("/games/{game_id}/resume")
async def resume_game(game_id: str):
    game = get_game(game_id)
    if not game: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    game.is_paused = False
    return {"status": "resumed"}

@router.post("/games/{game_id}/step")
async def step_game(game_id: str):
    async with get_game_lock(game_id):
        await advance_game(game_id)
    return {"status": "stepped"}

@router.get("/games/{game_id}")
async def get_game_status(game_id: str):
    game = get_game(game_id)
    if not game: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    # Exclude mafia_transcript from the response - it's private to mafia only
    data = game.dict()
    data.pop("mafia_transcript", None)
    return data

@router.get("/games/{game_id}/export")
async def export_game(game_id: str, debug: bool = False):
    game = get_game(game_id)
    if not game: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    data = game.dict()
    if not debug:
        # Strip private info
        for p in data["players"]:
            del p["private_memory"]
            del p["role_info"]
    return data
