import React from 'react';
import { useNavigate } from 'react-router-dom';

interface VictoryScreenProps {
    winner: 'TOWN' | 'MAFIA' | string;
    players: Array<{
        name: string;
        role: string;
        is_alive: boolean;
    }>;
}

const VictoryScreen: React.FC<VictoryScreenProps> = ({ winner, players }) => {
    const navigate = useNavigate();
    const isTownWin = winner === 'TOWN';

    const containerStyle: React.CSSProperties = {
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: isTownWin
            ? 'linear-gradient(135deg, #1a4d2e 0%, #2e7d32 50%, #1a4d2e 100%)'
            : 'linear-gradient(135deg, #2d1f1f 0%, #4a1c1c 50%, #2d1f1f 100%)',
        color: '#fff',
        zIndex: 1000,
        animation: 'fadeIn 0.8s ease-out'
    };

    const titleStyle: React.CSSProperties = {
        fontSize: '4rem',
        fontWeight: 'bold',
        textShadow: '0 4px 20px rgba(0,0,0,0.5)',
        marginBottom: '0.5rem',
        animation: 'scaleIn 0.6s ease-out'
    };

    const subtitleStyle: React.CSSProperties = {
        fontSize: '1.5rem',
        opacity: 0.9,
        marginBottom: '2rem'
    };

    const emojiStyle: React.CSSProperties = {
        fontSize: '6rem',
        marginBottom: '1rem',
        animation: 'bounce 1s ease-in-out infinite'
    };

    const playersContainerStyle: React.CSSProperties = {
        display: 'flex',
        flexWrap: 'wrap',
        justifyContent: 'center',
        gap: '1rem',
        maxWidth: '600px',
        marginBottom: '2rem'
    };

    const playerCardStyle = (isAlive: boolean, role: string): React.CSSProperties => ({
        padding: '0.75rem 1rem',
        borderRadius: '12px',
        backgroundColor: 'rgba(255,255,255,0.15)',
        backdropFilter: 'blur(10px)',
        border: `2px solid ${role === 'Mafia' ? '#ff6b6b' : '#69db7c'}`,
        opacity: isAlive ? 1 : 0.5,
        textDecoration: isAlive ? 'none' : 'line-through'
    });

    const buttonStyle: React.CSSProperties = {
        marginTop: '2rem',
        padding: '1rem 3rem',
        fontSize: '1.25rem',
        fontWeight: 'bold',
        borderRadius: '50px',
        border: 'none',
        background: isTownWin
            ? 'linear-gradient(90deg, #43a047, #66bb6a)'
            : 'linear-gradient(90deg, #c62828, #ef5350)',
        color: '#fff',
        cursor: 'pointer',
        boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
        transition: 'transform 0.2s, box-shadow 0.2s'
    };

    return (
        <div style={containerStyle}>
            <style>
                {`
                    @keyframes fadeIn {
                        from { opacity: 0; }
                        to { opacity: 1; }
                    }
                    @keyframes scaleIn {
                        from { transform: scale(0.5); opacity: 0; }
                        to { transform: scale(1); opacity: 1; }
                    }
                    @keyframes bounce {
                        0%, 100% { transform: translateY(0); }
                        50% { transform: translateY(-20px); }
                    }
                `}
            </style>

            <div style={emojiStyle}>
                {isTownWin ? '🎉' : '💀'}
            </div>

            <h1 style={titleStyle}>
                {isTownWin ? 'TOWN WINS!' : 'MAFIA WINS!'}
            </h1>

            <p style={subtitleStyle}>
                {isTownWin
                    ? 'The Mafia has been eliminated. Peace returns to the town.'
                    : 'The Mafia has taken control. The town falls into darkness.'}
            </p>

            <div style={playersContainerStyle}>
                {players.map((player, idx) => (
                    <div key={idx} style={playerCardStyle(player.is_alive, player.role)}>
                        <span>{player.is_alive ? '' : '☠️ '}{player.name}</span>
                        <span style={{
                            marginLeft: '0.5rem',
                            padding: '0.2rem 0.5rem',
                            borderRadius: '4px',
                            backgroundColor: player.role === 'Mafia' ? 'rgba(255,107,107,0.3)' :
                                player.role === 'Detective' ? 'rgba(100,149,237,0.3)' :
                                    'rgba(105,219,124,0.3)',
                            fontSize: '0.85rem'
                        }}>
                            {player.role}
                        </span>
                    </div>
                ))}
            </div>

            <button
                style={buttonStyle}
                onClick={() => navigate('/')}
                onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'scale(1.05)';
                    e.currentTarget.style.boxShadow = '0 6px 30px rgba(0,0,0,0.4)';
                }}
                onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'scale(1)';
                    e.currentTarget.style.boxShadow = '0 4px 20px rgba(0,0,0,0.3)';
                }}
            >
                🔄 Play Again
            </button>
        </div>
    );
};

export default VictoryScreen;
