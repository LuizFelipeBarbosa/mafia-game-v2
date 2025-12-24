import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { connectStream, getGameStatus } from '../api/client';
import PhaseHeader from '../components/PhaseHeader';
import Transcript from '../components/Transcript';
import Controls from '../components/Controls';
import PlayerList from '../components/PlayerList';
import VictoryScreen from '../components/VictoryScreen';

const Game: React.FC = () => {
    const { id } = useParams<{ id: string }>();
    const [day, setDay] = useState(1);
    const [phase, setPhase] = useState("Discussion");
    const [secondsRemaining, setSecondsRemaining] = useState(0);
    const [events, setEvents] = useState<any[]>([]);
    const [players, setPlayers] = useState<any[]>([]);
    const [isPaused, setIsPaused] = useState(true);
    const [winner, setWinner] = useState<string | null>(null);

    useEffect(() => {
        if (!id) return;

        // Fetch initial status
        getGameStatus(id).then(game => {
            setDay(game.day);
            setPhase(game.phase);
            setSecondsRemaining(game.seconds_remaining);
            setEvents(game.transcript);
            setPlayers(game.players);
            setIsPaused(game.is_paused);
            if (game.winner) setWinner(game.winner);
        });

        const ws = connectStream(id, (msg) => {
            if (msg.type === 'snapshot') {
                const game = msg.payload;
                setDay(game.day);
                setPhase(game.phase);
                setSecondsRemaining(game.seconds_remaining);
                setEvents(game.transcript);
                setPlayers(game.players);
                setIsPaused(game.is_paused);
                if (game.winner) setWinner(game.winner);
            } else if (msg.type === 'phase_tick') {
                setSecondsRemaining(msg.payload.seconds_remaining);
                setDay(msg.day);
                setPhase(msg.phase);
            } else if (msg.type === 'phase_start') {
                setSecondsRemaining(msg.payload.duration_sec);
                setDay(msg.day);
                setPhase(msg.phase);
                setEvents(prev => [...prev, msg]);
                // Refresh full state on phase start to get updated player status (e.g. deaths)
                getGameStatus(id).then(g => setPlayers(g.players));
            } else if (msg.type === 'game_over') {
                // Game ended!
                setPhase('game_over');
                setWinner(msg.payload.winner);
                setEvents(prev => [...prev, msg]);
                // Refresh to get final player states
                getGameStatus(id).then(g => setPlayers(g.players));
            } else {
                // Other events (chat, trial, execution, etc.)
                setEvents(prev => [...prev, msg]);
            }
        });

        return () => ws.close();
    }, [id]);

    const handleExport = (data: any) => {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `game_${id}.json`;
        a.click();
    };

    if (!id) return <div>Invalid Game ID</div>;

    // Show victory screen when game is over
    if (phase === 'game_over' && winner) {
        return <VictoryScreen winner={winner} players={players} />;
    }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100%' }}>
            <PhaseHeader day={day} phase={phase} secondsRemaining={secondsRemaining} />
            <div style={{ display: 'flex', flex: 1, overflow: 'hidden', gap: '1rem', padding: '1rem' }}>
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                    <Transcript events={events} />
                </div>
                <PlayerList players={players} />
            </div>
            <Controls gameId={id} isPaused={isPaused} onExport={handleExport} />
        </div>
    );
};

export default Game;

