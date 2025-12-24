import React, { useEffect, useRef } from 'react';

interface TranscriptEvent {
    type: string;
    day?: number;
    phase?: string;
    payload: any;
}

interface TranscriptProps {
    events: TranscriptEvent[];
}

const Transcript: React.FC<TranscriptProps> = ({ events }) => {
    const endRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [events]);

    const renderEvent = (event: TranscriptEvent, index: number) => {
        switch (event.type) {
            case 'chat':
                const isPrivate = event.payload.private;
                return (
                    <div key={index} style={{
                        marginBottom: '0.5rem',
                        fontStyle: isPrivate ? 'italic' : 'normal',
                        color: isPrivate ? '#8e44ad' : 'inherit'
                    }}>
                        <strong>{event.payload.speaker}:</strong> {event.payload.text}
                    </div>
                );
            case 'trial_started':
                return (
                    <div key={index} style={{ color: 'orange', fontWeight: 'bold', margin: '1rem 0' }}>
                        ⚠️ Trial Started: {event.payload.accused} is on the stand!
                    </div>
                );
            case 'trial_verdict':
                return (
                    <div key={index} style={{ color: event.payload.verdict === 'guilty' ? 'red' : 'green', fontWeight: 'bold', margin: '1rem 0' }}>
                        ⚖️ Verdict: {event.payload.verdict.toUpperCase()} (Guilty: {event.payload.tally.guilty}, Innocent: {event.payload.tally.innocent})
                    </div>
                );
            case 'execution':
                return (
                    <div key={index} style={{ color: 'red', fontWeight: 'bold', margin: '1rem 0' }}>
                        💀 Execution: {event.payload.player} was executed! They were a {event.payload.role_revealed}.
                    </div>
                );
            case 'night_result':
                return (
                    <div key={index} style={{ color: 'darkred', fontWeight: 'bold', margin: '1rem 0' }}>
                        🌙 Night Result: {event.payload.killed} was found dead! They were a {event.payload.role_revealed}.
                    </div>
                );
            case 'phase_start':
                return (
                    <div key={index} style={{ borderTop: '1px solid #eee', marginTop: '1rem', paddingTop: '1rem', fontStyle: 'italic', color: '#666' }}>
                        --- {event.phase} Phase Started (Day {event.day}) ---
                    </div>
                );
            case 'game_end':
                return (
                    <div key={index} style={{ backgroundColor: '#f0f0f0', padding: '1rem', marginTop: '1rem', textAlign: 'center', border: '2px solid' }}>
                        <h1>🏆 Game Over: {event.payload.winner} Wins!</h1>
                        <p>{event.payload.reason}</p>
                    </div>
                );
            default:
                return null;
        }
    };

    return (
        <div style={{ flex: 1, overflowY: 'auto', padding: '1rem', backgroundColor: '#fafafa', borderRadius: '8px' }}>
            {events.map((e, i) => renderEvent(e, i))}
            <div ref={endRef} />
        </div>
    );
};

export default Transcript;
