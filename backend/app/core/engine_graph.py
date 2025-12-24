import asyncio
import random
import time
from typing import Annotated, Dict, Any, List, TypedDict, Union, Optional
from langgraph.graph import StateGraph, END
from .schemas import GameState, Phase, Role, Player
from .llm_openrouter import call_openrouter, call_structured_vote
from .prompts import SYSTEM_PROMPT, get_role_prompt, MAFIA_NIGHT_PROMPT

class AgentState(TypedDict):
    game_state: GameState
    last_event: Optional[Dict[str, Any]]
    next_step: str # Hint for the next node if needed


# ==================== HELPER FUNCTIONS ====================

def find_mentioned_players(
    text: str,
    players: List[Player],
    exclude_names: Optional[List[str]] = None
) -> List[Player]:
    """Find all players mentioned in text, excluding specified names."""
    exclude = set(name.lower() for name in (exclude_names or []))
    text_lower = text.lower()
    return [
        p for p in players
        if p.name.lower() in text_lower and p.name.lower() not in exclude
    ]


def get_last_speaker_and_mentions(
    chats: List[Dict[str, Any]],
    players: List[Player]
) -> tuple[Optional[str], List[Player]]:
    """Extract last speaker name and mentioned players from recent chats."""
    if not chats:
        return None, []
    
    last_chat = chats[-1]
    last_speaker = last_chat.get("payload", {}).get("speaker", "")
    last_text = last_chat.get("payload", {}).get("text", "")
    mentioned = find_mentioned_players(last_text, players, exclude_names=[last_speaker])
    
    return last_speaker, mentioned


def build_transcript_summary(transcript: List[Dict[str, Any]], limit: int = 10) -> str:
    """Build a formatted transcript summary string."""
    return "\n".join([
        f"{t['payload'].get('speaker', 'System')}: {t['payload'].get('text', t['type'])}"
        for t in transcript[-limit:]
    ])


def select_speaker_with_mentions(
    players: List[Player],
    mentioned: List[Player],
    last_speaker: Optional[str],
    mention_response_chance: float = 0.7
) -> Player:
    """Select a speaker, prioritizing mentioned players, avoiding repeat speakers."""
    # Try mentioned players first
    if mentioned:
        eligible_mentioned = [p for p in mentioned if p.name != last_speaker]
        if eligible_mentioned and random.random() < mention_response_chance:
            return random.choice(eligible_mentioned)
    
    # Fall back to any eligible player
    eligible = [p for p in players if p.name != last_speaker]
    if not eligible:
        eligible = players
    
    return random.choice(eligible)


def create_chat_event(
    game_state: 'GameState',
    speaker: str,
    text: str,
    event_type: str = "chat",
    **extra_payload
) -> Dict[str, Any]:
    """Create a standardized chat/event dictionary."""
    return {
        "type": event_type,
        "game_id": game_state.game_id,
        "ts": int(time.time()),
        "day": game_state.day,
        "phase": game_state.phase.value,
        "payload": {"speaker": speaker, "text": text, **extra_payload}
    }


def create_system_event(
    game_state: 'GameState',
    text: str,
    event_type: str = "chat",
    **extra_payload
) -> Dict[str, Any]:
    """Create a system announcement event."""
    return create_chat_event(game_state, speaker="System", text=text, event_type=event_type, **extra_payload)


def get_filtered_transcript(
    game_state: 'GameState',
    player: Optional[Player] = None,
    limit: int = 10,
    public_only: bool = False,
    phase_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get filtered transcript with privacy and phase filtering.
    
    Args:
        game_state: The current game state
        player: If provided, filter private messages based on player's role
        limit: Maximum number of messages to return
        public_only: If True, only return public messages
        phase_filter: If provided, only return messages from this phase
    """
    raw_transcript = game_state.transcript[-(limit + 5):]  # Fetch extra to filter down
    filtered = []
    
    for t in raw_transcript:
        payload = t.get("payload", {})
        is_private = payload.get("private", False)
        
        # Phase filter
        if phase_filter and t.get("phase") != phase_filter:
            continue
        
        # Public messages always included
        if not is_private:
            filtered.append(t)
            continue
        
        # Public only mode - skip private
        if public_only:
            continue
        
        # No player context - skip private messages
        if not player:
            continue
        
        # Handle private messages based on player role
        speaker = payload.get("speaker", "")
        text = payload.get("text", "")
        
        # Mafia Chat - messages from [Mafia] tagged speakers
        if "[Mafia]" in speaker and player.role == Role.MAFIA:
            filtered.append(t)
        # Mafia-related System messages
        elif speaker == "System" and player.role == Role.MAFIA and ("🗳️" in text or "🎯" in text or "💤" in text):
            filtered.append(t)
        # Detective Investigation Result
        elif speaker == "System" and "🕵️" in text and player.role == Role.DETECTIVE:
            filtered.append(t)
    
    return filtered[-limit:]


def get_mafia_chats(game_state: 'GameState', limit: int = 15) -> List[Dict[str, Any]]:
    """Get private mafia chat messages from recent transcript."""
    return [
        t for t in game_state.transcript[-limit:]
        if t.get("payload", {}).get("private") and "[Mafia]" in t.get("payload", {}).get("speaker", "")
    ]


def tally_votes(votes: List[str], abstain_value: str = "abstain") -> Dict[str, int]:
    """
    Count votes excluding abstentions.
    
    Args:
        votes: List of vote targets (player names or abstain)
        abstain_value: Value to treat as abstention
    
    Returns:
        Dictionary mapping target names to vote counts (excluding abstains)
    """
    tally: Dict[str, int] = {}
    for vote in votes:
        if vote.lower() == abstain_value.lower():
            continue
        if vote not in tally:
            tally[vote] = 0
        tally[vote] += 1
    return tally


def get_majority_winner(
    tally: Dict[str, int],
    total_voters: int
) -> tuple[Optional[str], int, int]:
    """
    Determine if there's a majority winner.
    
    Args:
        tally: Vote counts by candidate name
        total_voters: Total number of voters (for calculating majority threshold)
    
    Returns:
        Tuple of (winner_name or None, votes_received, votes_needed)
    """
    votes_needed = (total_voters + 1) // 2
    
    if not tally:
        return None, 0, votes_needed
    
    most_voted = max(tally.items(), key=lambda x: x[1])
    winner_name, most_votes = most_voted
    
    if most_votes >= votes_needed:
        return winner_name, most_votes, votes_needed
    
    return None, most_votes, votes_needed


def format_vote_summary(tally: Dict[str, int]) -> str:
    """Format vote tally as a summary string."""
    if not tally:
        return "No votes cast"
    sorted_votes = sorted(tally.items(), key=lambda x: x[1], reverse=True)
    return ", ".join([f"{name}: {votes} vote(s)" for name, votes in sorted_votes])


# ==================== END HELPER FUNCTIONS ====================

async def get_player_action(player: Player, game_state: GameState) -> Dict[str, Any]:
    """Wait for player action (LLM call)."""
    # Get filtered transcript based on player's role (handles privacy)
    recent_transcript = get_filtered_transcript(game_state, player=player, limit=10)
    transcript_summary = build_transcript_summary(recent_transcript, limit=10)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n" + get_role_prompt(player.role.value)},
        {"role": "user", "content": f"""
Current Day: {game_state.day}
Phase: {game_state.phase.value}
Seconds Remaining: {game_state.seconds_remaining}
Alive Players: {', '.join([p.name for p in game_state.players if p.is_alive])}
Accused: {next((p.name for p in game_state.players if p.id == game_state.accused_id), 'None')}

Recent Transcript:
{transcript_summary}

Your Roles: {player.role.value}
Private Role Info: {player.role_info}
Your Memory: {player.private_memory[-5:] if player.private_memory else "None"}
"""}
    ]
    response = await call_openrouter(messages, model=game_state.config.model)
    return response

async def handle_discussion(state: AgentState) -> AgentState:
    """Handle day discussion phase - one player speaks per call."""
    game_state = state["game_state"]
    alive_players = [p for p in game_state.players if p.is_alive]
    if not alive_players:
        return state

    # Get recent public chats for speaker selection
    recent_chats = [
        t for t in game_state.transcript[-5:]
        if t.get("type") == "chat" and not t.get("payload", {}).get("private")
    ]
    
    # Use helper to get last speaker and mentioned players
    last_speaker, mentioned = get_last_speaker_and_mentions(recent_chats, alive_players)
    
    # Use helper to select speaker with mention priority
    speaker = select_speaker_with_mentions(
        alive_players, mentioned, last_speaker, mention_response_chance=0.7
    )
    
    response = await get_player_action(speaker, game_state)
    
    if response["public_text"]:
        event = create_chat_event(game_state, speaker=speaker.name, text=response["public_text"])
        game_state.transcript.append(event)
        state["last_event"] = event

    if response["private_thought"]:
        speaker.private_memory.append(response["private_thought"])
    return state

async def handle_voting(state: AgentState) -> AgentState:
    """
    Voting Phase: ALL players vote on who to put on trial.
    Each vote is broadcast to chat. Majority votes sends someone to the stand.
    """
    game_state = state["game_state"]
    alive_players = [p for p in game_state.players if p.is_alive]
    
    if not alive_players or len(alive_players) < 2:
        return state
    
    # Skip if already have an accused or voting is complete
    if game_state.accused_id or game_state.voting_complete:
        return state
    
    print(f"DEBUG: Starting voting phase - collecting votes from {len(alive_players)} players")
    
    # Announce voting has started
    start_event = create_system_event(
        game_state,
        "🗳️ VOTING PHASE: Each player will now be asked individually who they want to put on trial. You may vote for a player or ABSTAIN based on the discussion."
    )
    game_state.transcript.append(start_event)
    
    # Build context from recent public discussion
    public_transcript = get_filtered_transcript(game_state, public_only=True, limit=15)
    discussion_context = build_transcript_summary(public_transcript, limit=15)
    
    # Collect votes from all players
    votes: List[str] = []  # Store raw vote targets
    vote_tally: Dict[str, int] = {}  # player_name -> vote count
    
    for voter in alive_players:
        valid_targets = [p.name for p in alive_players if p.id != voter.id]
        valid_targets.append("abstain")
        
        context = f"""DAY {game_state.day} - VOTING PHASE

RECENT DISCUSSION:
{discussion_context if discussion_context else "(No discussion)"}

ALIVE PLAYERS: {', '.join([p.name for p in alive_players])}

Based on the discussion above, who do you think is MAFIA and should be put on trial?
You may vote for any player to nominate them for trial.
Or you may choose "ABSTAIN" if you are unsure or don't want to vote based on the conversation."""
        
        vote = await call_structured_vote(
            player_name=voter.name,
            context=context,
            valid_targets=valid_targets,
            vote_type="nominate",
            model=game_state.config.model
        )
        
        print(f"DEBUG: Nomination vote from {voter.name}: {vote}")
        
        # Process and broadcast the vote
        if vote.lower() != "abstain":
            target = next((p for p in game_state.players 
                          if p.name.lower() == vote.lower() and p.is_alive and p.id != voter.id), None)
            
            if target:
                # Track vote
                if target.name not in vote_tally:
                    vote_tally[target.name] = 0
                vote_tally[target.name] += 1
                votes.append(target.name)
                
                # Broadcast
                event = create_system_event(game_state, f"🗳️ {voter.name} votes to put {target.name} on trial!")
                game_state.transcript.append(event)
            else:
                # Invalid target - treat as skip
                event = create_system_event(game_state, f"⏭️ {voter.name} skips voting.")
                game_state.transcript.append(event)
        else:
            event = create_system_event(game_state, f"⏭️ {voter.name} skips voting.")
            game_state.transcript.append(event)
    
    # Determine winner using helper
    winner_name, most_votes, votes_needed = get_majority_winner(vote_tally, len(alive_players))
    
    # Announce results
    summary = format_vote_summary(vote_tally)
    results_event = create_system_event(
        game_state, 
        f"📊 VOTE RESULTS: {summary}. Votes needed (half): {votes_needed}"
    )
    game_state.transcript.append(results_event)
    
    game_state.voting_complete = True
    
    if winner_name:
        accused = next((p for p in game_state.players if p.name == winner_name and p.is_alive), None)
        if accused:
            game_state.accused_id = accused.id
            game_state.last_phase_remaining_time = game_state.seconds_remaining
            game_state.voting_complete = False  # Reset for next time
            
            trial_event = create_chat_event(
                game_state,
                speaker="System",
                text=f"⚖️ {accused.name} has received {most_votes} votes and is being sent to the STAND! They will now plead their case.",
                event_type="trial_started",
                accused=accused.name,
                votes=most_votes
            )
            game_state.transcript.append(trial_event)
            state["last_event"] = trial_event
            
            game_state.phase = Phase.DEFENSE
            game_state.seconds_remaining = game_state.config.phase_durations[Phase.DEFENSE]
            print(f"DEBUG: {accused.name} sent to trial with {most_votes} votes")
    else:
        no_trial_event = create_system_event(
            game_state,
            "❌ No player received enough votes for trial. The town could not reach a consensus. Night will fall soon..."
        )
        game_state.transcript.append(no_trial_event)
    
    return state

async def handle_defense(state: AgentState) -> AgentState:
    """
    Defense Phase: Accused opens with a statement, then discussion follows.
    Each call: One player speaks - accused first, then alternating discussion.
    """
    game_state = state["game_state"]
    accused = _get_accused(game_state)
    
    if not accused or not accused.is_alive:
        game_state.phase = Phase.VOTING
        return state
    
    alive_players = [p for p in game_state.players if p.is_alive]
    other_players = [p for p in alive_players if p.id != accused.id]
    
    # Check if accused has already given opening statement
    defense_chats = _get_defense_chats(game_state)
    accused_has_opened = any(t.get("type") == "defense_speech" for t in defense_chats)
    
    # STEP 1: Accused MUST open with a defense statement
    if not accused_has_opened:
        return await _handle_opening_defense(state, accused, game_state)
    
    # STEP 2: Discussion - alternate between accused and challengers
    speaker, is_accused_speaking = _select_defense_speaker(
        accused, other_players, alive_players, defense_chats
    )
    
    return await _handle_defense_discussion(
        state, game_state, accused, speaker, is_accused_speaking
    )


def _get_accused(game_state: 'GameState') -> Optional[Player]:
    """Get the currently accused player."""
    return next(
        (p for p in game_state.players if p.id == game_state.accused_id),
        None
    )


def _get_defense_chats(game_state: 'GameState') -> List[Dict[str, Any]]:
    """Get all chat messages from the Defense phase."""
    return [
        t for t in game_state.transcript
        if t.get("phase") == "Defense" and t.get("type") in ("defense_speech", "chat")
    ]


def _select_defense_speaker(
    accused: Player,
    other_players: List[Player],
    alive_players: List[Player],
    defense_chats: List[Dict[str, Any]]
) -> tuple[Player, bool]:
    """
    Select who speaks next in the defense phase.
    Returns (speaker, is_accused_speaking).
    """
    last_speaker, mentioned_players = get_last_speaker_and_mentions(
        defense_chats, alive_players
    )
    
    # If accused was mentioned, they respond (70% chance)
    if accused in mentioned_players and random.random() < 0.7:
        return accused, True
    
    # Otherwise, a challenger speaks
    if other_players:
        # Prioritize mentioned players (excluding accused)
        challenger_mentions = [p for p in mentioned_players if p.id != accused.id]
        speaker = select_speaker_with_mentions(
            other_players, challenger_mentions, last_speaker
        )
        return speaker, False
    
    # Fallback to accused
    return accused, True


async def _handle_opening_defense(
    state: AgentState,
    accused: Player,
    game_state: 'GameState'
) -> AgentState:
    """Handle the accused's opening defense statement."""
    transcript_summary = build_transcript_summary(game_state.transcript)
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + "\n" + get_role_prompt(accused.role.value) + _DEFENSE_OPENING_PROMPT
        },
        {
            "role": "user",
            "content": f"""Day {game_state.day} - YOU ARE ON TRIAL

Recent Events:
{transcript_summary}

Your Role: {accused.role.value}
Your Memory: {accused.private_memory[-5:] if accused.private_memory else "None"}

⚠️ GIVE YOUR OPENING DEFENSE! Explain why you are innocent!"""
        }
    ]
    
    response = await call_openrouter(messages, model=game_state.config.model)
    defense_text = response["public_text"] or "I am innocent! Please believe me!"
    
    event = create_chat_event(
        game_state,
        speaker=accused.name,
        text=defense_text,
        event_type="defense_speech",
        is_defense=True
    )
    game_state.transcript.append(event)
    state["last_event"] = event
    
    if response["private_thought"]:
        accused.private_memory.append(response["private_thought"])
    
    return state


async def _handle_defense_discussion(
    state: AgentState,
    game_state: 'GameState',
    accused: Player,
    speaker: Player,
    is_accused_speaking: bool
) -> AgentState:
    """Handle discussion during defense phase (after opening statement)."""
    transcript_summary = build_transcript_summary(game_state.transcript)
    
    if is_accused_speaking:
        messages = _build_accused_response_messages(accused, game_state.day, transcript_summary)
    else:
        messages = _build_challenger_messages(speaker, accused, game_state.day, transcript_summary)
    
    response = await call_openrouter(messages, model=game_state.config.model)
    
    if response["public_text"]:
        event = create_chat_event(
            game_state,
            speaker=speaker.name,
            text=response["public_text"],
            event_type="defense_speech" if is_accused_speaking else "chat",
            **(({"is_defense": True}) if is_accused_speaking else {})
        )
        game_state.transcript.append(event)
        state["last_event"] = event
    
    if response["private_thought"]:
        speaker.private_memory.append(response["private_thought"])
    
    return state


def _build_accused_response_messages(accused: Player, day: int, transcript: str) -> List[Dict]:
    """Build LLM messages for accused responding to challenges."""
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + "\n" + get_role_prompt(accused.role.value) + _DEFENSE_RESPONSE_PROMPT
        },
        {
            "role": "user",
            "content": f"""Day {day} - Defense Phase

Recent Discussion:
{transcript}

Your Role: {accused.role.value}

Respond to the challenges. Defend yourself!"""
        }
    ]


def _build_challenger_messages(
    speaker: Player,
    accused: Player,
    day: int,
    transcript: str
) -> List[Dict]:
    """Build LLM messages for a player challenging the accused."""
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + "\n" + get_role_prompt(speaker.role.value) + _CHALLENGER_PROMPT
        },
        {
            "role": "user",
            "content": f"""Day {day} - {accused.name} is on trial

Recent Discussion:
{transcript}

Accused: {accused.name}
Your Role: {speaker.role.value}

Challenge their defense, ask questions, or share your opinion. Be brief!"""
        }
    ]


# === Defense Phase Prompt Constants ===
_DEFENSE_OPENING_PROMPT = """

🔴 YOU ARE ON TRIAL FOR YOUR LIFE! 🔴

You MUST defend yourself NOW. The town has voted to put you on trial.
If you are found GUILTY, you will be EXECUTED immediately.
Convince the town to vote INNOCENT. Your survival depends on this!
"""

_DEFENSE_RESPONSE_PROMPT = """
🔴 YOU ARE ON TRIAL! Respond to challenges and defend yourself!
"""

_CHALLENGER_PROMPT = """
⚖️ TRIAL IN PROGRESS - Challenge the accused's defense or express your opinion.
"""

async def handle_judgment(state: AgentState) -> AgentState:
    """
    Judgment Phase: All alive players (except accused) vote Guilty, Innocent, or Abstain.
    Majority rules - Guilty > Innocent means execution.
    """
    game_state = state["game_state"]
    accused = next((p for p in game_state.players if p.id == game_state.accused_id), None)
    
    if not accused:
        game_state.phase = Phase.VOTING
        return state
    
    voters = [p for p in game_state.players if p.is_alive and p.id != game_state.accused_id]
    if not voters:
        return state
    
    tally = {"guilty": 0, "innocent": 0, "abstain": 0}
    
    # Announce voting has begun
    start_event = create_system_event(
        game_state,
        f"⚖️ JUDGMENT TIME: The town will now vote on {accused.name}'s fate. Vote GUILTY, INNOCENT, or ABSTAIN."
    )
    game_state.transcript.append(start_event)
    
    # Build context from defense phase
    defense_transcript = get_filtered_transcript(game_state, phase_filter="Defense", limit=10)
    defense_context = build_transcript_summary(defense_transcript, limit=10)
    
    # Each voter casts their vote
    for voter in voters:
        context = f"""ACCUSED: {accused.name}

DEFENSE STATEMENT:
{defense_context if defense_context else "(No defense given)"}

You must now decide: Is {accused.name} guilty of being Mafia?"""
        
        vote = await call_structured_vote(
            player_name=voter.name,
            context=context,
            valid_targets=["guilty", "innocent", "abstain"],
            vote_type="verdict",
            model=game_state.config.model
        )
        
        print(f"DEBUG: Judgment vote from {voter.name}: {vote}")
        
        tally[vote] += 1
        
        # Announce the vote
        vote_emoji = {"guilty": "🔴", "innocent": "🟢", "abstain": "⚪"}.get(vote, "⚪")
        event = create_system_event(game_state, f"{vote_emoji} {voter.name} votes {vote.upper()}")
        game_state.transcript.append(event)
    
    # Determine verdict
    verdict = "guilty" if tally["guilty"] > tally["innocent"] else "innocent"
    
    # Announce the verdict
    verdict_event = create_chat_event(
        game_state,
        speaker="System",
        text=f"📜 VERDICT: {accused.name} has been found {verdict.upper()}! (Guilty: {tally['guilty']}, Innocent: {tally['innocent']}, Abstain: {tally['abstain']})",
        event_type="trial_verdict",
        accused=accused.name,
        verdict=verdict,
        tally=tally
    )
    game_state.transcript.append(verdict_event)
    state["last_event"] = verdict_event
    
    if verdict == "guilty":
        game_state.phase = Phase.LAST_WORDS
        game_state.seconds_remaining = game_state.config.phase_durations[Phase.LAST_WORDS]
        print(f"DEBUG: {accused.name} found GUILTY - proceeding to Last Words")
    else:
        # Innocent - return to Voting phase with remaining time
        game_state.phase = Phase.VOTING
        remaining_time = game_state.last_phase_remaining_time or 0
        game_state.seconds_remaining = max(remaining_time - 10, 5)
        game_state.accused_id = None
        
        innocent_event = create_system_event(
            game_state,
            f"✅ {accused.name} has been found INNOCENT and returns to the town. Voting continues with {game_state.seconds_remaining} seconds remaining."
        )
        game_state.transcript.append(innocent_event)
        print(f"DEBUG: {accused.name} found INNOCENT - returning to Voting")
    
    return state

async def handle_last_words(state: AgentState) -> AgentState:
    """Handle convicted player's last words before execution."""
    game_state = state["game_state"]
    accused = next((p for p in game_state.players if p.id == game_state.accused_id), None)
    
    if accused:
        response = await get_player_action(accused, game_state)
        if response["public_text"]:
            event = create_chat_event(game_state, speaker=accused.name, text=response["public_text"])
            game_state.transcript.append(event)
            state["last_event"] = event
        
        # Execute
        accused.is_alive = False
        execution_event = create_chat_event(
            game_state,
            speaker="System",
            text="",  # Not a chat event really
            event_type="execution",
            player=accused.name,
            role_revealed=accused.role.value
        )
        game_state.transcript.append(execution_event)
        
        # Announce death and reveal role publicly
        death_event = create_system_event(
            game_state,
            f"☠️ {accused.name} has been executed! They were a **{accused.role.value}**."
        )
        game_state.transcript.append(death_event)
        
        # Update all players' memories with the revealed role
        for player in game_state.players:
            player.private_memory.append(f"REVEALED: {accused.name} was {accused.role.value} (executed Day {game_state.day})")
        
    game_state.accused_id = None
    game_state.phase = Phase.NIGHT
    game_state.seconds_remaining = game_state.config.phase_durations[Phase.NIGHT]
    return state

async def handle_night(state: AgentState) -> AgentState:
    """
    Night Phase Handler - RESPONSIVE DISCUSSION.
    Each call: ONE mafia member speaks. First initiates, others respond when mentioned.
    Creates natural back-and-forth conversation. Actual voting happens at end of timer.
    """
    game_state = state["game_state"]
    
    mafia_members = [p for p in game_state.players if p.role == Role.MAFIA and p.is_alive]
    if not mafia_members:
        return state
    
    # Get valid targets for context
    valid_targets = [p.name for p in game_state.players if p.is_alive and p.role != Role.MAFIA]
    if not valid_targets:
        return state
    
    # Get context using helpers
    public_transcript = get_filtered_transcript(game_state, public_only=True, limit=15)
    day_context = build_transcript_summary(public_transcript, limit=15)
    
    mafia_chats = get_mafia_chats(game_state, limit=15)
    mafia_chat = build_transcript_summary(mafia_chats, limit=15)
    
    # RESPONSIVE speaker selection
    # Extract last speaker name (strip [Mafia] prefix)
    last_mafia_speaker = None
    mentioned_mafia: List[Player] = []
    
    if mafia_chats:
        last_chat = mafia_chats[-1]
        last_speaker_full = last_chat.get("payload", {}).get("speaker", "")
        last_mafia_speaker = last_speaker_full.replace("[Mafia] ", "")
        last_text = last_chat.get("payload", {}).get("text", "").lower()
        
        # Find mafia members mentioned in the last message
        mentioned_mafia = find_mentioned_players(last_text, mafia_members, exclude_names=[last_mafia_speaker])
    
    # Use helper for speaker selection
    speaker = select_speaker_with_mentions(
        mafia_members, mentioned_mafia, last_mafia_speaker, mention_response_chance=0.7
    )
    
    # Build prompt based on whether this is initiating or responding
    teammates = [m.name for m in mafia_members if m.id != speaker.id]
    is_first_message = len(mafia_chats) == 0
    
    if is_first_message:
        prompt_context = f"""NIGHT {game_state.day} - MAFIA DISCUSSION (You are INITIATING)

YOUR MAFIA TEAMMATES: {', '.join(teammates) if teammates else "(You are the only Mafia left)"}

YOU ARE {speaker.name}.

RECENT DAY EVENTS:
{day_context if day_context else "(No significant events)"}

ALIVE PLAYERS (potential targets): {', '.join(valid_targets)}

You are starting the mafia discussion. Suggest a target and explain your reasoning.
Consider who might be the Detective or who is dangerous to keep alive."""
    else:
        prompt_context = f"""NIGHT {game_state.day} - MAFIA DISCUSSION (RESPONDING)

YOUR MAFIA TEAMMATES: {', '.join(teammates) if teammates else "(You are the only Mafia left)"}

YOU ARE {speaker.name}.

MAFIA CHAT SO FAR:
{mafia_chat}

RECENT DAY EVENTS:
{day_context if day_context else "(No significant events)"}

ALIVE PLAYERS (potential targets): {', '.join(valid_targets)}

Respond to your teammates' suggestions. Agree, disagree, or propose a different target.
Keep the discussion moving toward a decision. If agreement is reached, strategize for future nights."""
    
    discussion_messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n" + MAFIA_NIGHT_PROMPT},
        {"role": "user", "content": prompt_context}
    ]
    
    response = await call_openrouter(discussion_messages, model=game_state.config.model)
    
    if response["public_text"]:
        event = create_chat_event(
            game_state,
            speaker=f"[Mafia] {speaker.name}",
            text=response["public_text"],
            private=True
        )
        game_state.transcript.append(event)
        print(f"DEBUG: Mafia discussion from {speaker.name}")
    
    if response["private_thought"]:
        speaker.private_memory.append(response["private_thought"])
    
    return state


def check_game_over(state: AgentState) -> bool:
    game_state = state["game_state"]
    mafia_count = len([p for p in game_state.players if p.role == Role.MAFIA and p.is_alive])
    town_count = len([p for p in game_state.players if p.role != Role.MAFIA and p.is_alive])
    
    if mafia_count == 0:
        game_state.winner = "TOWN"
        game_state.phase = Phase.GAME_OVER
        return True
    if mafia_count >= town_count:
        game_state.winner = "MAFIA"
        game_state.phase = Phase.GAME_OVER
        return True
    return False


# ============================================================================
# STANDALONE VOTING FUNCTIONS - Called from routes.py at end of timer
# ============================================================================

async def collect_day_votes(game_state: GameState) -> Optional[Player]:
    """
    Collect votes from each alive player individually.
    Called at the END of Discussion phase when timer expires.
    Returns the accused player if someone got majority, None otherwise.
    """
    alive_players = [p for p in game_state.players if p.is_alive]
    
    if not alive_players or len(alive_players) < 2:
        return None
    
    print(f"DEBUG: Collecting day votes from {len(alive_players)} players")
    
    # Build context from recent public discussion
    public_transcript = get_filtered_transcript(game_state, public_only=True, limit=15)
    discussion_context = build_transcript_summary(public_transcript, limit=15)
    
    vote_tally: Dict[str, int] = {}
    
    for voter in alive_players:
        valid_targets = [p.name for p in alive_players if p.id != voter.id]
        valid_targets.append("abstain")
        
        context = f"""DAY {game_state.day} - VOTING TIME

DISCUSSION SUMMARY:
{discussion_context if discussion_context else "(No discussion)"}

ALIVE PLAYERS: {', '.join([p.name for p in alive_players])}

Based on the discussion, who do you think is MAFIA and should be put on trial?
You may vote for any player to nominate them.
Or you may choose "ABSTAIN" if the conversation didn't reveal clear suspects."""
        
        vote = await call_structured_vote(
            player_name=voter.name,
            context=context,
            valid_targets=valid_targets,
            vote_type="nominate",
            model=game_state.config.model
        )
        
        print(f"DEBUG: {voter.name} votes: {vote}")
        
        # Process and broadcast the vote
        if vote.lower() != "abstain":
            target = next((p for p in game_state.players 
                          if p.name.lower() == vote.lower() and p.is_alive and p.id != voter.id), None)
            
            if target:
                if target.name not in vote_tally:
                    vote_tally[target.name] = 0
                vote_tally[target.name] += 1
                
                event = create_system_event(game_state, f"🗳️ {voter.name} votes for {target.name}")
                game_state.transcript.append(event)
            else:
                event = create_system_event(game_state, f"⏭️ {voter.name} abstains")
                game_state.transcript.append(event)
        else:
            event = create_system_event(game_state, f"⏭️ {voter.name} abstains")
            game_state.transcript.append(event)
    
    # Determine winner using helper
    winner_name, most_votes, votes_needed = get_majority_winner(vote_tally, len(alive_players))
    
    # Announce results
    summary = format_vote_summary(vote_tally)
    results_event = create_system_event(
        game_state,
        f"📊 RESULTS: {summary}. Need {votes_needed} to nominate."
    )
    game_state.transcript.append(results_event)
    
    if winner_name:
        accused = next((p for p in game_state.players if p.name == winner_name and p.is_alive), None)
        if accused:
            trial_event = create_system_event(
                game_state,
                f"⚖️ {accused.name} is PUT ON TRIAL with {most_votes} votes!"
            )
            game_state.transcript.append(trial_event)
            print(f"DEBUG: {accused.name} nominated for trial with {most_votes} votes")
            return accused
    else:
        no_trial_event = create_system_event(game_state, "❌ No majority reached. Night falls...")
        game_state.transcript.append(no_trial_event)
    
    return None


async def collect_mafia_votes(game_state: GameState) -> None:
    """
    Collect kill votes from each mafia member individually.
    Called at the END of Night phase when timer expires.
    Sets game_state.pending_kills with the chosen target.
    """
    mafia_members = [p for p in game_state.players if p.role == Role.MAFIA and p.is_alive]
    
    if not mafia_members:
        return
    
    valid_targets = [p.name for p in game_state.players if p.is_alive and p.role != Role.MAFIA]
    if not valid_targets:
        return
    
    valid_targets.append("abstain")
    
    # Build context using helpers
    mafia_chats = get_mafia_chats(game_state, limit=15)
    mafia_chat = build_transcript_summary(mafia_chats, limit=15)
    
    public_transcript = get_filtered_transcript(game_state, public_only=True, limit=20)
    day_context = build_transcript_summary(public_transcript, limit=20)
    
    context = f"""MAFIA DISCUSSION:
{mafia_chat if mafia_chat else "(No discussion)"}

DAY EVENTS:
{day_context if day_context else "(No events)"}

VALID TARGETS: {', '.join(valid_targets[:-1])}
You may also "abstain" to not kill.

WHO DO YOU VOTE TO KILL?"""
    
    vote_tally: Dict[str, int] = {}
    
    for mafia in mafia_members:
        vote = await call_structured_vote(
            player_name=mafia.name,
            context=context,
            valid_targets=valid_targets,
            vote_type="kill",
            model=game_state.config.model
        )
        
        print(f"DEBUG: Mafia {mafia.name} votes: {vote}")
        
        if vote not in vote_tally:
            vote_tally[vote] = 0
        vote_tally[vote] += 1
        
        event = create_chat_event(
            game_state,
            speaker="System",
            text=f"🗳️ {mafia.name} votes: {vote}",
            private=True
        )
        game_state.transcript.append(event)
    
    # Resolve votes (exclude abstains)
    player_votes = {k: v for k, v in vote_tally.items() if k.lower() != "abstain"}
    
    if player_votes:
        max_votes = max(player_votes.values())
        top_targets = [name for name, count in player_votes.items() if count == max_votes]
        chosen_name = sorted(top_targets)[0]  # Alphabetical tiebreaker
        
        target = next((p for p in game_state.players 
                      if p.name.lower() == chosen_name.lower() and p.is_alive and p.role != Role.MAFIA), None)
        
        if target:
            game_state.pending_kills = [target.id]
            event = create_chat_event(
                game_state,
                speaker="System",
                text=f"🎯 Target: {target.name}",
                private=True
            )
            game_state.transcript.append(event)
            print(f"DEBUG: Mafia chose to kill {target.name}")
    else:
        event = create_chat_event(
            game_state,
            speaker="System",
            text="💤 Mafia chose not to kill tonight.",
            private=True
        )
        game_state.transcript.append(event)
        print("DEBUG: Mafia abstained")


async def handle_detective_investigation(game_state: GameState) -> None:
    """
    Handle detective's investigation at end of night.
    Called from routes.py after mafia votes.
    """
    detective = next((p for p in game_state.players if p.role == Role.DETECTIVE and p.is_alive), None)
    
    if not detective:
        return
    
    # Valid targets (alive, not self, not about to die)
    valid_targets = [p for p in game_state.players 
                     if p.is_alive and p.id != detective.id and p.id not in game_state.pending_kills]
    
    if not valid_targets:
        return
    
    d_messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n" + get_role_prompt("Detective") + "\nYou must investigate someone. Output action with type 'investigate' and target name."},
        {"role": "user", "content": f"Alive Players: {', '.join([p.name for p in valid_targets])}\nWho do you investigate?"}
    ]
    
    response = await call_openrouter(d_messages, model=game_state.config.model)
    print(f"DEBUG: Detective response: {response}")
    
    target_name = response["action"].get("target")
    target = None
    
    if target_name:
        clean_name = str(target_name).strip().lower()
        target = next((p for p in valid_targets if clean_name in p.name.lower() or p.name.lower() == clean_name), None)
    
    if target:
        result = "MAFIA" if target.role == Role.MAFIA else "NOT MAFIA"
        detective.private_memory.append(f"Investigation: {target.name} is {result}")
        
        event = create_chat_event(
            game_state,
            speaker="System",
            text=f"🕵️ {target.name} is {result}",
            private=True
        )
        game_state.transcript.append(event)
        print(f"DEBUG: Detective investigated {target.name}: {result}")

def create_game_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("discussion", handle_discussion)
    workflow.add_node("voting", handle_voting)
    workflow.add_node("defense", handle_defense)
    workflow.add_node("judgment", handle_judgment)
    workflow.add_node("last_words", handle_last_words)
    workflow.add_node("night", handle_night)
    
    workflow.set_entry_point("discussion")
    
    # Logic for transitions will be handled by the engine loop in main.py or routes.py
    # but we can define the edges here if we want the graph to run autonomously.
    # However, the requirement is "advances by exactly one atomic step" via /step.
    # So the graph might not need complex edge logic if we just call specific nodes.
    # But let's define them for completeness.
    
    workflow.add_edge("discussion", END) # We control transitions outside
    workflow.add_edge("voting", END)
    workflow.add_edge("defense", END)
    workflow.add_edge("judgment", END)
    workflow.add_edge("last_words", END)
    workflow.add_edge("night", END)
    
    return workflow.compile()
