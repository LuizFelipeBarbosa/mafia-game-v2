# Mafia Game LLM

A minimal Mafia/Town-of-Salem-style web app where LLM agents play against each other, orchestrated by LangGraph and powered by OpenRouter.

## Architecture

- **Backend**: FastAPI + LangGraph. The game state is managed as a state machine.
- **LLM**: OpenRouter (GPT-3.5-Turbo by default). Agents think privately and speak publicly.
- **Frontend**: React + Vite + TypeScript. A simple text-based transcript with phase timers.

## Setup

### Prerequisites

- Python 3.9+
- Node.js 18+
- OpenRouter API Key

### Backend Setup

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment using `uv`:
   ```bash
   uv venv
   source .venv/bin/activate # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```
4. Set up environment variables:
   Create a `.env` file in the `backend` directory:
   ```env
   OPENROUTER_API_KEY=your_api_key_here
   DUMMY_LLM=false # Set to true to run without API calls (mock mode)
   ```
5. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend Setup

1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```

## Rules and Phases

1. **Discussion** (45s): Public chat.
2. **Voting** (30s): Nominate a player for trial. If someone is accused, transition to Defense.
3. **Defense** (20s): The accused speaks.
4. **Judgment** (20s): Town votes Guilty/Innocent.
5. **Last Words** (5s): Final message if Guilty.
6. **Night** (35s): Mafia kills, Detective investigates.

*Critial Rule*: If a trial ends in an "Innocent" verdict, the Voting phase resumes with the remaining time.

## Limitations & Next Steps

- **In-Memory Storage**: Current state is lost on server restart.
- **Simple UI**: No avatars or complex animations as per requirements.
- **Agent Memory**: Currently limited to the last few messages.
- **Role Scoping**: Only Mafia and Detective roles are implemented in the MVP.
