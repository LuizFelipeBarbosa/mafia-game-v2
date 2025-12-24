import os
import json
import httpx
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

logger = logging.getLogger(__name__)

async def call_openrouter(
    messages: list,
    model: str = "openai/gpt-3.5-turbo",
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Calls OpenRouter and returns the parsed JSON response."""
    
    if os.getenv("DUMMY_LLM") == "true" or not OPENROUTER_API_KEY:
        # Fallback for testing or missing key
        return create_dummy_response(messages[-1]["content"])

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/mafia-game-llm",
        "X-Title": "Mafia Game LLM"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"}
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return parse_and_coerce_llm_output(content)
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Error calling OpenRouter: {e.response.status_code} - {e.response.text}")
        return create_dummy_response(f"HTTP Error: {e.response.status_code}")
    except Exception as e:
        import traceback
        logger.error(f"Error calling OpenRouter: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        return create_dummy_response(f"Error: {e}")

def parse_and_coerce_llm_output(raw_output: str) -> Dict[str, Any]:
    """Parses JSON output from LLM and ensures it follows the required schema."""
    try:
        data = json.loads(raw_output)
    except Exception:
        # Try to find JSON in the string if it's not pure JSON
        import re
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except:
                data = {}
        else:
            data = {}

    # Coerce to schema
    return {
        "private_thought": data.get("private_thought", "No thoughts."),
        "public_text": data.get("public_text", ""),
        "action": {
            "type": data.get("action", {}).get("type", "none"),
            "target": data.get("action", {}).get("target"),
            "value": data.get("action", {}).get("value")
        }
    }

def create_dummy_response(last_message: str) -> Dict[str, Any]:
    """Creates a no-op response for dummy mode."""
    return {
        "private_thought": "Thinking about the game...",
        "public_text": "I am a simple townsperson.",
        "action": {"type": "none", "target": None, "value": None}
    }


async def call_structured_vote(
    player_name: str,
    context: str,
    valid_targets: list,
    vote_type: str = "kill",  # "kill", "nominate", "verdict"
    model: str = "openai/gpt-3.5-turbo"
) -> str:
    """
    Forces a structured vote from the LLM. 
    Returns exactly one of valid_targets, or "abstain" if validation fails.
    
    Args:
        player_name: Name of the player who is voting
        context: Game context (discussion history, etc.)
        valid_targets: List of valid vote options (player names + "abstain")
        vote_type: Type of vote ("kill", "nominate", "verdict")
        model: LLM model to use
    
    Returns:
        A single string that is exactly one of valid_targets
    """
    
    # Build strict voting prompt based on type
    if vote_type == "kill":
        instruction = f"""You are {player_name}, a Mafia member. You MUST now vote on who to kill tonight.

VALID TARGETS (you MUST choose exactly one):
{chr(10).join([f'- "{t}"' for t in valid_targets])}

Based on the discussion, who should die tonight? Consider:
- Who is suspicious of the Mafia?
- Who might be the Detective?
- Who is leading the town?

OUTPUT ONLY the exact name of your target from the list above. Nothing else.
Example output: Player_2"""

    elif vote_type == "nominate":
        instruction = f"""You are {player_name}. You MUST vote on who to put on trial.

VALID TARGETS (you MUST choose exactly one):
{chr(10).join([f'- "{t}"' for t in valid_targets])}

Based on the discussion, who do you think is Mafia and should be put on trial?
If you are UNSURE or the discussion didn't reveal clear suspects, you may choose "abstain".

OUTPUT ONLY the exact name or "abstain" from the list above. Nothing else.
Example output: Player_3"""

    elif vote_type == "verdict":
        instruction = f"""You are {player_name}. You MUST vote on the accused player's fate.

VALID OPTIONS (you MUST choose exactly one):
- "guilty" - Execute the accused
- "innocent" - Spare the accused  
- "abstain" - No vote

Based on their defense, is the accused Mafia?

OUTPUT ONLY: guilty, innocent, or abstain. Nothing else.
Example output: guilty"""

    else:
        instruction = f"Choose one of: {', '.join(valid_targets)}"

    if os.getenv("DUMMY_LLM") == "true" or not OPENROUTER_API_KEY:
        # For testing: return first valid target or abstain
        return valid_targets[0] if valid_targets else "abstain"

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"Context:\n{context}\n\nYour vote:"}
    ]

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/mafia-game-llm",
        "X-Title": "Mafia Game LLM"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,  # Lower temperature for more deterministic output
        "max_tokens": 50  # We only need a short response
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            raw_vote = data["choices"][0]["message"]["content"].strip()
            
            # Clean up the vote - remove quotes, extra whitespace, punctuation
            clean_vote = raw_vote.strip('"\'.,!? ').lower()
            
            # Try to match against valid targets (case-insensitive)
            for target in valid_targets:
                if target.lower() == clean_vote or clean_vote in target.lower() or target.lower() in clean_vote:
                    logger.info(f"Structured vote from {player_name}: {target}")
                    return target
            
            # If no exact match, try partial matching
            for target in valid_targets:
                if target.lower() != "abstain" and (target.lower() in clean_vote or clean_vote in target.lower()):
                    logger.info(f"Structured vote (partial match) from {player_name}: {target}")
                    return target
            
            # Default to abstain if we can't parse the response
            logger.warning(f"Could not parse vote from {player_name}: '{raw_vote}' - defaulting to abstain")
            return "abstain"
    
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Error in structured vote for {player_name}: {e.response.status_code} - {e.response.text}")
        return "abstain"
    except Exception as e:
        import traceback
        logger.error(f"Error in structured vote call for {player_name}: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        return "abstain"
