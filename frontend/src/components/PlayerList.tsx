import React from 'react';

interface Player {
    id: string;
    name: string;
    role: string;
    is_alive: boolean;
}

interface PlayerListProps {
    players: Player[];
}

// Mafia roles have red names, Town roles have green names
const MAFIA_ROLES = ['Mafia', 'Godfather', 'Mafioso', 'Consort', 'Consigliere', 'Janitor', 'Blackmailer', 'Framer'];

const getNameColor = (role: string): string => {
    if (MAFIA_ROLES.some(r => role.toLowerCase().includes(r.toLowerCase()))) {
        return '#dc2626'; // Red for Mafia
    }
    return '#16a34a'; // Green for Town
};

const PlayerList: React.FC<PlayerListProps> = ({ players }) => {
    const aliveCount = players.filter(p => p.is_alive).length;

    return (
        <div style={{
            position: 'absolute',
            top: '1rem',
            right: '1rem',
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            padding: '1rem',
            borderRadius: '8px',
            boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
            zIndex: 1000,
            maxWidth: '250px',
            border: '1px solid #eee'
        }}>
            <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '1.1rem', borderBottom: '1px solid #ddd', paddingBottom: '0.25rem' }}>
                Players ({aliveCount} alive)
            </h3>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0, maxHeight: '70vh', overflowY: 'auto' }}>
                {players.map(player => (
                    <li key={player.id} style={{
                        padding: '0.25rem 0',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        fontSize: '0.9rem',
                        opacity: player.is_alive ? 1 : 0.6
                    }}>
                        <span style={{
                            fontWeight: 500,
                            color: getNameColor(player.role),
                            textDecoration: player.is_alive ? 'none' : 'line-through'
                        }}>
                            {player.name}
                        </span>
                        <span style={{
                            fontSize: '0.8rem',
                            color: '#666',
                            backgroundColor: '#f0f0f0',
                            padding: '2px 6px',
                            borderRadius: '4px',
                            textDecoration: player.is_alive ? 'none' : 'line-through'
                        }}>
                            {player.role}
                        </span>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default PlayerList;
