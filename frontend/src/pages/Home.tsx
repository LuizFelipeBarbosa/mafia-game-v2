import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { createGame } from '../api/client';

const Home: React.FC = () => {
    const navigate = useNavigate();
    const [numPlayers, setNumPlayers] = useState(5);
    const [numMafia, setNumMafia] = useState(1);
    const [hasDetective, setHasDetective] = useState(true);
    const [hasDoctor, setHasDoctor] = useState(true);
    const [hasVigilante, setHasVigilante] = useState(true);
    const [model, setModel] = useState("openai/gpt-3.5-turbo");

    const handleCreate = async (e: React.FormEvent) => {
        e.preventDefault();
        const config = {
            num_players: numPlayers,
            num_mafia: numMafia,
            has_detective: hasDetective,
            has_doctor: hasDoctor,
            has_vigilante: hasVigilante,
            model: model,
            phase_durations: {
                "Discussion": 45,
                "Voting": 30,
                "Defense": 20,
                "Judgment": 20,
                "Last Words": 5,
                "Night": 35
            }
        };
        const { game_id } = await createGame(config);
        navigate(`/game/${game_id}`);
    };

    return (
        <div style={{ padding: '2rem', maxWidth: '600px', margin: '0 auto' }}>
            <h1>Mafia Game LLM</h1>
            <form onSubmit={handleCreate} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <label>
                    Players:
                    <input type="number" value={numPlayers} onChange={e => setNumPlayers(parseInt(e.target.value))} min={3} />
                </label>
                <label>
                    Mafia:
                    <input type="number" value={numMafia} onChange={e => setNumMafia(parseInt(e.target.value))} min={1} />
                </label>
                <label>
                    Include Detective:
                    <input type="checkbox" checked={hasDetective} onChange={e => setHasDetective(e.target.checked)} />
                </label>
                <label>
                    Include Doctor:
                    <input type="checkbox" checked={hasDoctor} onChange={e => setHasDoctor(e.target.checked)} />
                </label>
                <label>
                    Include Vigilante:
                    <input type="checkbox" checked={hasVigilante} onChange={e => setHasVigilante(e.target.checked)} />
                </label>
                <label>
                    Model (OpenRouter):
                    <input type="text" value={model} onChange={e => setModel(e.target.value)} />
                </label>
                <button type="submit" style={{ padding: '0.5rem', cursor: 'pointer' }}>Create Game</button>
            </form>
        </div>
    );
};

export default Home;
