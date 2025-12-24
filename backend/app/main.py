from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router as api_router
from .api.ws import manager
from .core.storage import get_game

app = FastAPI(title="Mafia Game LLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.websocket("/games/{game_id}/stream")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    await manager.connect(game_id, websocket)
    try:
        # Send initial state snapshot
        game = get_game(game_id)
        if game:
            # Exclude mafia_transcript - it's private to mafia only
            payload = game.dict()
            payload.pop("mafia_transcript", None)
            await websocket.send_json({
                "type": "snapshot",
                "game_id": game_id,
                "payload": payload
            })
        
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(game_id, websocket)
    except Exception:
        manager.disconnect(game_id, websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
