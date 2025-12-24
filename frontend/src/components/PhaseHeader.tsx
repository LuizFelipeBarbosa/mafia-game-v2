import React from 'react';

interface PhaseHeaderProps {
    day: number;
    phase: string;
    secondsRemaining: number;
}

const PhaseHeader: React.FC<PhaseHeaderProps> = ({ day, phase, secondsRemaining }) => {
    return (
        <div style={{ padding: '1rem', borderBottom: '1px solid #ccc' }}>
            <h2>Day {day} - {phase}</h2>
            <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: secondsRemaining < 10 ? 'red' : 'inherit' }}>
                Time Remaining: {secondsRemaining}s
            </div>
        </div>
    );
};

export default PhaseHeader;
