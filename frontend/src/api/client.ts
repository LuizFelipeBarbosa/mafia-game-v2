const API_BASE = "http://localhost:8000";
const WS_BASE = "ws://localhost:8000";

export const createGame = async (config: any) => {
    const response = await fetch(`${API_BASE}/games`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
    });
    return response.json();
};

export const startGame = async (gameId: string) => {
    await fetch(`${API_BASE}/games/${gameId}/start`, { method: "POST" });
};

export const pauseGame = async (gameId: string) => {
    await fetch(`${API_BASE}/games/${gameId}/pause`, { method: "POST" });
};

export const resumeGame = async (gameId: string) => {
    await fetch(`${API_BASE}/games/${gameId}/resume`, { method: "POST" });
};

export const stepGame = async (gameId: string) => {
    await fetch(`${API_BASE}/games/${gameId}/step`, { method: "POST" });
};

export const exportGame = async (gameId: string, debug: boolean = false) => {
    const response = await fetch(`${API_BASE}/games/${gameId}/export?debug=${debug}`);
    return response.json();
};

export const getGameStatus = async (gameId: string) => {
    const response = await fetch(`${API_BASE}/games/${gameId}`);
    return response.json();
};

export const connectStream = (gameId: string, onMessage: (msg: any) => void) => {
    const ws = new WebSocket(`${WS_BASE}/games/${gameId}/stream`);
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        onMessage(data);
    };
    return ws;
};
