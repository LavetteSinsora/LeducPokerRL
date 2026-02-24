const API_BASE = window.location.port === '8001'
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : window.location.origin;

document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-debug-btn');
    const content = document.getElementById('analyzer-content');
    const agentSelect = document.getElementById('analyzer-agent-select');

    // Load analyzable agents into dropdown
    loadAnalyzableAgents();

    runBtn.addEventListener('click', async () => {
        const agentId = agentSelect.value;
        runBtn.disabled = true;
        runBtn.innerHTML = '<span class="loading-spinner"></span> Running...';

        try {
            const response = await fetch(`${API_BASE}/analyze/episode?agent_id=${agentId}`);
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

    async function loadAnalyzableAgents() {
        try {
            const res = await fetch(`${API_BASE}/api/agents`);
            const data = await res.json();
            const trainable = data.agents.filter(a => a.isTrainable);
            agentSelect.innerHTML = trainable.map(a =>
                `<option value="${a.id}">${a.displayName}</option>`
            ).join('');
        } catch (err) {
            console.error('Error loading agents:', err);
        }
    }

    // ── Dispatch ──────────────────────────────────────────────

    function renderEpisode(data) {
        const { trace, final_rewards, eval_type } = data;

        let html = renderSummary(final_rewards);
        html += `<div class="episode-container">`;

        trace.forEach((step, index) => {
            html += renderStepHeader(step, index);

            if (eval_type === 'policy') {
                html += renderPolicyEvaluations(step);
            } else {
                html += renderValueEvaluations(step);
            }

            html += renderEncodedState(step);
            html += `</div></div>`;  // close step-card
        });

        html += `</div>`;
        content.innerHTML = html;
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // ── Shared renderers ─────────────────────────────────────

    function renderSummary(final_rewards) {
        return `
            <div class="final-summary">
                <h2>Episode Complete</h2>
                <div style="display: flex; justify-content: center; gap: 40px; margin-top: 10px; font-weight: 800; font-size: 1.2rem;">
                    <span>P0 Reward: ${final_rewards[0] > 0 ? '+' : ''}${final_rewards[0]}</span>
                    <span>P1 Reward: ${final_rewards[1] > 0 ? '+' : ''}${final_rewards[1]}</span>
                </div>
            </div>`;
    }

    function renderStepHeader(step, index) {
        const obs = step.observation;
        const isP0 = step.player_id === 0;

        // Build right-side badges (true value always shown; prediction_error only for value agents)
        let rightBadges = `<span class="reward-badge">True Value: ${step.true_value}</span>`;
        if (step.prediction_error !== undefined) {
            rightBadges += `<span class="error-badge">L2 Error: ${step.prediction_error.toFixed(4)}</span>`;
        }

        return `
            <div class="step-card">
                <div class="step-header">
                    <div class="step-info">
                        <span class="round-label">Step ${index + 1}</span>
                        <span class="action-label ${isP0 ? 'call' : 'raise'}">Player ${step.player_id}</span>
                        <span class="round-label">${obs.current_round === 0 ? 'Pre-Flop' : 'Flop'}</span>
                    </div>
                    <div class="step-info">
                        ${rightBadges}
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
                </div>`;
    }

    // ── Value-based renderer ─────────────────────────────────

    function renderValueEvaluations(step) {
        return `
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
            </div>`;
    }

    // ── Policy gradient renderer ─────────────────────────────

    function renderPolicyEvaluations(step) {
        // Sort by probability descending for visual clarity
        const sorted = [...step.evaluations].sort((a, b) => b.probability - a.probability);

        return `
            <div style="margin-top: 20px;">
                <div class="stat-label" style="margin-bottom: 10px;">Policy Network Output (Action Probabilities)</div>
                <div style="background: rgba(0,0,0,0.2); border-radius: 12px; padding: 15px;">
                    ${sorted.map(e => {
                        const pct = (e.probability * 100).toFixed(1);
                        const rawPct = (e.raw_probability * 100).toFixed(1);
                        const isChosen = e.action === step.selected_action;
                        const actionClass = e.action.toLowerCase();
                        return `
                            <div class="prob-bar-container">
                                <div class="prob-bar-label">
                                    <span class="action-badge ${actionClass}">${e.action}</span>
                                </div>
                                <div class="prob-bar-track">
                                    <div class="prob-bar-fill ${actionClass} ${isChosen ? 'chosen' : ''}" style="width: ${Math.max(pct, 3)}%;">
                                        ${pct}%
                                    </div>
                                </div>
                                ${isChosen ? '<span class="chosen-badge">CHOSEN</span>' : '<span style="min-width: 55px;"></span>'}
                            </div>
                            <div style="text-align: right; font-size: 0.65rem; color: var(--text-muted); margin-bottom: 10px; margin-top: -4px;">
                                raw: ${rawPct}%
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>`;
    }

    // ── Encoded state vector (shared) ────────────────────────

    function renderEncodedState(step) {
        // Use step-level encoded_state (policy) or skip if not present
        const encoded = step.encoded_state;
        if (!encoded) return '';

        const FEATURE_HEADERS = ['J', 'Q', 'K', 'J', 'Q', 'K', 'None', 'Me', 'Opp', '?', '#', '#', '!', '=', 'R'];
        const GROUP_HEADERS = [
            { label: 'Hand', span: 3 },
            { label: 'Board', span: 4 },
            { label: 'Pot', span: 2 },
            { label: 'Turn', span: 1 },
            { label: 'Pos', span: 1 },
            { label: 'Rnd', span: 1 },
            { label: 'Term', span: 1 },
            { label: 'Pair', span: 1 },
            { label: 'Raises', span: 1 },
        ];

        return `
            <div style="margin-top: 15px;">
                <div style="margin-bottom: 5px; font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;">
                    State Vector (Network Input)
                </div>
                <div style="overflow-x: auto; background: rgba(0,0,0,0.2); border-radius: 6px;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.65rem; text-align: center;">
                        <tr style="background: rgba(255,255,255,0.03); color: rgba(255,255,255,0.4);">
                            ${GROUP_HEADERS.map(g => `<th colspan="${g.span}" style="padding: 3px; border: 1px solid rgba(255,255,255,0.05);">${g.label}</th>`).join('')}
                        </tr>
                        <tr style="color: var(--accent-gold); opacity: 0.8; font-family: monospace; font-size: 0.6rem;">
                            ${FEATURE_HEADERS.map(h => `<td style="padding: 2px; border: 1px solid rgba(255,255,255,0.05);">${h}</td>`).join('')}
                        </tr>
                        <tr style="font-family: monospace; font-size: 0.7rem;">
                            ${encoded.map(val => `
                                <td style="padding: 4px 2px; border: 1px solid rgba(255,255,255,0.05); background: ${val > 0 ? 'rgba(244, 208, 63, 0.08)' : 'transparent'};">
                                    ${val.toFixed(2)}
                                </td>
                            `).join('')}
                        </tr>
                    </table>
                </div>
            </div>`;
    }
});
