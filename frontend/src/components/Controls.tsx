import React from 'react';
import { startGame, pauseGame, resumeGame, stepGame, exportGame } from '../api/client';

interface ControlsProps {
    gameId: string;
    isPaused: boolean;
    onExport: (data: any) => void;
}

const Controls: React.FC<ControlsProps> = ({ gameId, isPaused, onExport }) => {
    const handleExport = async () => {
        const data = await exportGame(gameId);
        onExport(data);
    };

    const handleExportDebug = async () => {
        const data = await exportGame(gameId, true);
        onExport(data);
    };

    return (
        <div style={{ padding: '1rem', borderTop: '1px solid #ccc', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            <button onClick={() => startGame(gameId)}>Start</button>
            {isPaused ? (
                <button onClick={() => resumeGame(gameId)}>Resume</button>
            ) : (
                <button onClick={() => pauseGame(gameId)}>Pause</button>
            )}
            <button onClick={() => stepGame(gameId)}>Step</button>
            <button onClick={handleExport}>Export</button>
            <button onClick={handleExportDebug}>Export (Debug)</button>
        </div>
    );
};

export default Controls;
