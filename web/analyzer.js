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
                        <div class="eval-grid">
            `;

            step.evaluations.forEach(ev => {
                const isSelected = ev.action === step.selected_action;
                html += `
                    <div class="eval-item ${isSelected ? 'selected' : ''}">
                        <span class="action-label" style="background: transparent; padding: 0;">${ev.action}</span>
                        <span class="eval-val ${ev.value >= 0 ? 'value-pos' : 'value-neg'}">${ev.value.toFixed(4)}</span>
                        ${isSelected ? '<span class="best-badge">CHOSEN</span>' : ''}
                    </div>
                `;
            });

            html += `
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
