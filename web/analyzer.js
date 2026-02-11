const API_BASE = "http://localhost:8000";

document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-debug-btn');
    const content = document.getElementById('analyzer-content');

    runBtn.addEventListener('click', async () => {
        runBtn.disabled = true;
        runBtn.innerHTML = '<span class="loading-spinner"></span> Running...';

        try {
            const response = await fetch(`${API_BASE}/analyze/episode`);
            const data = await response.json();

            if (data.error) {
                content.innerHTML = `<div class="error-badge" style="padding: 20px; text-align: center;">Error: ${data.error}</div>`;
            } else {
                renderEpisode(data);
            }
        } catch (err) {
            content.innerHTML = `<div class="error-badge" style="padding: 20px; text-align: center;">Connection Error: ${err.message}</div>`;
        } finally {
            runBtn.disabled = false;
            runBtn.innerHTML = 'Run Debug Episode';
        }
    });

    function renderEpisode(data) {
        const { trace, final_rewards } = data;

        let html = `
            <div class="final-summary">
                <h2>Episode Complete</h2>
                <div style="display: flex; justify-content: center; gap: 40px; margin-top: 10px; font-weight: 800; font-size: 1.2rem;">
                    <span>P0 Reward: ${final_rewards[0] > 0 ? '+' : ''}${final_rewards[0]}</span>
                    <span>P1 Reward: ${final_rewards[1] > 0 ? '+' : ''}${final_rewards[1]}</span>
                </div>
            </div>
            <div class="episode-container">
        `;

        trace.forEach((step, index) => {
            const obs = step.observation;
            const isP0 = step.player_id === 0;

            html += `
                <div class="step-card">
                    <div class="step-header">
                        <div class="step-info">
                            <span class="round-label">Step ${index + 1}</span>
                            <span class="action-label ${isP0 ? 'call' : 'raise'}">Player ${step.player_id}</span>
                            <span class="round-label">${obs.current_round === 0 ? 'Pre-Flop' : 'Flop'}</span>
                        </div>
                        <div class="step-info">
                            <span class="reward-badge">True Value: ${step.true_value}</span>
                            <span class="error-badge">L2 Error: ${step.prediction_error.toFixed(4)}</span>
                        </div>
                    </div>
                    
                    <div class="state-preview" style="background: rgba(255,255,255,0.02); margin-top:10px;">
                        <div class="preview-cards">
                            <div class="preview-card" title="Player Hand">${obs.player_hand}</div>
                            <div style="width: 20px;"></div>
                            <div class="preview-card board" title="Board Card">${obs.board || '?'}</div>
                        </div>
                        <div class="preview-pot">
                            Pot: <span>${obs.pot[0] + obs.pot[1]}</span> (${obs.pot[0]} / ${obs.pot[1]})
                        </div>
                    </div>

                <div style="margin-top: 20px;">
                    <div class="stat-label" style="margin-bottom: 10px;">Mental Simulation (V-Values)</div>
                    <div class="eval-grid" style="display: grid; grid-template-columns: 1fr; gap: 20px;">
                        ${step.evaluations.map(e => `
                            <div class="eval-card ${e.action === step.selected_action ? 'chosen' : ''}" style="padding: 15px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                    <div style="display: flex; align-items: center; gap: 10px;">
                                        <span class="action-badge ${e.action.toLowerCase()}">${e.action}</span>
                                        ${e.action === step.selected_action ? '<span class="chosen-badge">CHOSEN</span>' : ''}
                                    </div>
                                    <div class="stat-value" style="font-size: 1.2rem; color: ${e.value >= 0 ? 'var(--accent-gold)' : '#e74c3c'}">
                                        ${e.value.toFixed(4)}
                                    </div>
                                </div>
                                
                                <div style="margin-bottom: 5px; font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;">
                                    Simulated State Vector (Network Input)
                                </div>
                                <div style="overflow-x: auto; background: rgba(0,0,0,0.2); border-radius: 6px;">
                                    <table style="width: 100%; border-collapse: collapse; font-size: 0.65rem; text-align: center;">
                                        <tr style="background: rgba(255,255,255,0.03); color: rgba(255,255,255,0.4);">
                                            <th colspan="3" style="padding: 3px; border: 1px solid rgba(255,255,255,0.05);">Hand</th>
                                            <th colspan="4" style="padding: 3px; border: 1px solid rgba(255,255,255,0.05);">Board</th>
                                            <th colspan="2" style="padding: 3px; border: 1px solid rgba(255,255,255,0.05);">Pot</th>
                                            <th style="padding: 3px; border: 1px solid rgba(255,255,255,0.05);">Turn</th>
                                            <th style="padding: 3px; border: 1px solid rgba(255,255,255,0.05);">Pos</th>
                                            <th style="padding: 3px; border: 1px solid rgba(255,255,255,0.05);">Rnd</th>
                                            <th style="padding: 3px; border: 1px solid rgba(255,255,255,0.05);">Term</th>
                                            <th style="padding: 3px; border: 1px solid rgba(255,255,255,0.05);">Pair</th>
                                            <th style="padding: 3px; border: 1px solid rgba(255,255,255,0.05);">Raises</th>
                                        </tr>
                                        <tr style="color: var(--accent-gold); opacity: 0.8; font-family: monospace; font-size: 0.6rem;">
                                            <td style="padding: 2px; border: 1px solid rgba(255,255,255,0.05);">J</td>
                                            <td style="padding: 2px; border: 1px solid rgba(255,255,255,0.05);">Q</td>
                                            <td style="padding: 2px; border: 1px solid rgba(255,255,255,0.05);">K</td>
                                            <td style="padding: 2px; border: 1px solid rgba(255,255,255,0.05);">J</td>
                                            <td style="padding: 2px; border: 1px solid rgba(255,255,255,0.05);">Q</td>
                                            <td style="padding: 2px; border: 1px solid rgba(255,255,255,0.05);">K</td>
                                            <td style="padding: 2px; border: 1px solid rgba(255,255,255,0.05);">None</td>
                                            <td style="padding: 2px; border: 1px solid rgba(255,255,255,0.05);">Me</td>
                                            <td style="padding: 2px; border: 1px solid rgba(255,255,255,0.05);">Opp</td>
                                            <td style="padding: 2px; border: 1px solid rgba(255,255,255,0.05);">?</td>
                                            <td style="padding: 2px; border: 1px solid rgba(255,255,255,0.05);">#</td>
                                            <td style="padding: 2px; border: 1px solid rgba(255,255,255,0.05);">#</td>
                                            <td style="padding: 2px; border: 1px solid rgba(255,255,255,0.05);">!</td>
                                            <td style="padding: 2px; border: 1px solid rgba(255,255,255,0.05);">=</td>
                                            <td style="padding: 2px; border: 1px solid rgba(255,255,255,0.05);">R</td>
                                        </tr>
                                        <tr style="font-family: monospace; font-size: 0.7rem;">
                                            ${e.encoded_state.map(val => `
                                                <td style="padding: 4px 2px; border: 1px solid rgba(255,255,255,0.05); background: ${val > 0 ? 'rgba(244, 208, 63, 0.08)' : 'transparent'};">
                                                    ${val.toFixed(2)}
                                                </td>
                                            `).join('')}
                                        </tr>
                                    </table>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
                    </div>
                </div>
            `;
        });

        html += `</div>`;
        content.innerHTML = html;

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
});
