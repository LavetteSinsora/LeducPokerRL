const API_BASE = window.location.port === '8001'
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : window.location.origin;

let tournamentResults = null;
let chipRaceChart = null;
let allAgentsMap = {};
let pollInterval = null;

const AGENT_COLORS = [
    '#f4d03f', '#4ade80', '#60a5fa', '#f87171', '#a78bfa',
    '#fb923c', '#2dd4bf', '#e879f9', '#fbbf24', '#34d399',
];

// --- Tab Navigation ---

function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanels.forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`tab-${tabId}`).classList.add('active');

            if (tabId === 'chiprace' && chipRaceChart) {
                requestAnimationFrame(() => chipRaceChart.resize());
            }
        });
    });
}

// --- Tournament Controls ---

async function runTournament() {
    const numRounds = parseInt(document.getElementById('num-rounds').value) || 5000;
    const btn = document.getElementById('btn-run-tournament');
    btn.disabled = true;
    btn.textContent = 'Running...';

    try {
        await fetch(`${API_BASE}/api/tournament/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ num_rounds: numRounds })
        });
        startPolling();
    } catch (err) {
        console.error('Error starting tournament:', err);
        btn.disabled = false;
        btn.textContent = 'Run Tournament';
    }
}

function startPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(pollTournamentStatus, 2000);
    pollTournamentStatus();
}

async function pollTournamentStatus() {
    try {
        const resp = await fetch(`${API_BASE}/api/tournament/status`);
        const status = await resp.json();
        const statusDot = document.getElementById('status-dot');
        const statusText = document.getElementById('status-text');
        const btn = document.getElementById('btn-run-tournament');

        if (status.is_running) {
            statusDot.classList.add('running');
            statusText.textContent = status.progress || 'Running...';
            btn.disabled = true;
            btn.textContent = 'Running...';
        } else {
            statusDot.classList.remove('running');
            statusText.textContent = status.has_results ? 'Complete' : 'Ready';
            btn.disabled = false;
            btn.textContent = 'Run Tournament';

            if (pollInterval) {
                clearInterval(pollInterval);
                pollInterval = null;
            }

            if (status.has_results) {
                await loadResults();
            }
        }
    } catch (err) {
        console.error('Poll error:', err);
    }
}

async function loadResults() {
    try {
        const resp = await fetch(`${API_BASE}/api/tournament/results`);
        const results = await resp.json();
        if (results.error) return;

        tournamentResults = results;
        renderLeaderboard();
        renderChipRace();
    } catch (err) {
        console.error('Error loading results:', err);
    }
}

async function loadAgents() {
    try {
        const resp = await fetch(`${API_BASE}/api/agents`);
        const data = await resp.json();
        data.agents.forEach(a => {
            if (a.id !== 'human') {
                allAgentsMap[a.id] = a;
            }
        });
    } catch (err) {
        console.error('Error loading agents:', err);
    }
}

// --- Leaderboard ---

function renderLeaderboard() {
    if (!tournamentResults || !tournamentResults.rankings) return;

    const container = document.getElementById('leaderboard-content');
    const rankings = tournamentResults.rankings;

    let html = `
        <table class="leaderboard-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Agent</th>
                    <th>Avg Chips/Round</th>
                    <th>Robustness</th>
                    <th>Win Rate</th>
                </tr>
            </thead>
            <tbody>
    `;

    rankings.forEach((r, i) => {
        const rank = i + 1;
        const rankClass = rank <= 3 ? `rank-${rank}` : '';
        const agent = allAgentsMap[r.agent_id] || {};
        const displayName = agent.displayName || r.display_name || r.agent_id;
        const avgColor = r.avg >= 0 ? '#4ade80' : '#f87171';
        const robColor = r.robustness >= 0 ? '#4ade80' : '#f87171';
        const winPct = r.win_rate.toFixed(0);

        html += `
            <tr onclick="showDrilldown('${r.agent_id}')">
                <td><span class="rank-badge ${rankClass}">${rank}</span></td>
                <td>
                    <span class="agent-name-link">${displayName}</span>
                </td>
                <td style="color: ${avgColor}; font-weight: 600;">
                    ${r.avg >= 0 ? '+' : ''}${r.avg.toFixed(2)}
                </td>
                <td style="color: ${robColor}; font-weight: 600;">
                    ${r.robustness >= 0 ? '+' : ''}${r.robustness.toFixed(2)}
                </td>
                <td>${winPct}%</td>
            </tr>
        `;
    });

    html += `</tbody></table>`;

    if (tournamentResults.timestamp) {
        html += `<p style="opacity: 0.4; font-size: 0.8rem; margin-top: 15px; text-align: right;">
            Last run: ${new Date(tournamentResults.timestamp).toLocaleString()}
            &mdash; ${tournamentResults.num_rounds} rounds/matchup
        </p>`;
    }

    container.innerHTML = html;
}

// --- Drill-Down ---

function showDrilldown(agentId) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelector('[data-tab="drilldown"]').classList.add('active');
    document.getElementById('tab-drilldown').classList.add('active');

    renderDrilldown(agentId);
}

function renderDrilldown(agentId) {
    if (!tournamentResults || !tournamentResults.head_to_head) return;

    const container = document.getElementById('drilldown-content');
    const agent = allAgentsMap[agentId] || {};
    const displayName = agent.displayName || agentId;
    const h2h = tournamentResults.head_to_head;

    const matchups = [];
    const agentH2H = h2h[agentId] || {};
    Object.entries(agentH2H).forEach(([oppId, chips]) => {
        matchups.push({ opponent: oppId, chips: chips });
    });

    matchups.sort((a, b) => b.chips - a.chips);

    const totalChips = matchups.reduce((s, m) => s + m.chips, 0);
    const avgChips = matchups.length > 0 ? totalChips / matchups.length : 0;
    const bestMatchup = matchups.length > 0 ? matchups[0] : null;
    const worstMatchup = matchups.length > 0 ? matchups[matchups.length - 1] : null;
    const winRate = matchups.length > 0 ? matchups.filter(m => m.chips > 0).length / matchups.length : 0;

    let html = `
        <button class="drilldown-back" onclick="backToLeaderboard()">Back to Leaderboard</button>
        <h2 style="margin-bottom: 20px;">${displayName}</h2>

        <div class="drilldown-stats">
            <div class="stat-item">
                <span class="stat-value" style="color: ${avgChips >= 0 ? '#4ade80' : '#f87171'};">
                    ${avgChips >= 0 ? '+' : ''}${avgChips.toFixed(2)}
                </span>
                <span class="stat-label">Avg Chips/Round</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${(winRate * 100).toFixed(0)}%</span>
                <span class="stat-label">Win Rate</span>
            </div>
            <div class="stat-item">
                <span class="stat-value" style="font-size: 1rem;">
                    ${bestMatchup ? (allAgentsMap[bestMatchup.opponent]?.displayName || bestMatchup.opponent) : '-'}
                </span>
                <span class="stat-label">Best Matchup</span>
            </div>
        </div>

        <h3 style="margin-bottom: 15px; opacity: 0.8;">Matchup Details</h3>
    `;

    const maxAbs = Math.max(...matchups.map(m => Math.abs(m.chips)), 0.01);

    matchups.forEach(m => {
        const oppAgent = allAgentsMap[m.opponent] || {};
        const oppName = oppAgent.displayName || m.opponent;
        const pct = (Math.abs(m.chips) / maxAbs) * 50;
        const isPositive = m.chips >= 0;

        html += `
            <div class="matchup-bar-container">
                <div class="matchup-bar-label">
                    <span class="agent-name-link" onclick="renderDrilldown('${m.opponent}')" style="cursor: pointer;">
                        ${oppName}
                    </span>
                    <span style="color: ${isPositive ? '#4ade80' : '#f87171'}; font-weight: 600;">
                        ${isPositive ? '+' : ''}${m.chips.toFixed(2)}
                    </span>
                </div>
                <div class="matchup-bar-track">
                    <div class="matchup-bar-center"></div>
                    <div class="matchup-bar-fill ${isPositive ? 'positive' : 'negative'}"
                         style="width: ${pct}%;">
                    </div>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

function backToLeaderboard() {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelector('[data-tab="leaderboard"]').classList.add('active');
    document.getElementById('tab-leaderboard').classList.add('active');
}

// --- Chip Race ---

function renderChipRace() {
    if (!tournamentResults || !tournamentResults.chip_race) return;

    const container = document.getElementById('chiprace-content');
    const chipRace = tournamentResults.chip_race;

    container.innerHTML = '<div class="chip-race-container"><canvas id="chipRaceChart"></canvas></div>';

    const agents = chipRace.agents;
    const history = chipRace.history;

    const datasets = agents.map((agentId, i) => {
        const agent = allAgentsMap[agentId] || {};
        const displayName = agent.displayName || agentId;
        const color = AGENT_COLORS[i % AGENT_COLORS.length];

        return {
            label: displayName,
            data: history.map(h => ({
                x: h.epoch,
                y: h[agentId]
            })),
            borderColor: color,
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            tension: 0.2,
            pointRadius: 0,
            pointHoverRadius: 4,
        };
    });

    const ctx = document.getElementById('chipRaceChart').getContext('2d');

    if (chipRaceChart) chipRaceChart.destroy();

    chipRaceChart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Tournament Progress (%)', color: '#fff' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.6)',
                        stepSize: 1,
                        callback: function (value) {
                            if (chipRace.epochs && value <= chipRace.epochs) {
                                return (value / chipRace.epochs * 100).toFixed(0) + '%';
                            }
                            return value;
                        }
                    }
                },
                y: {
                    type: 'linear',
                    title: { display: true, text: 'Cumulative Chips', color: '#fff' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: 'rgba(255, 255, 255, 0.6)' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#fff', font: { family: 'Inter' } }
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(0)} chips`;
                        }
                    }
                }
            }
        }
    });
}

// --- Init ---

document.getElementById('btn-run-tournament').addEventListener('click', runTournament);

async function init() {
    initTabs();
    await loadAgents();

    try {
        const resp = await fetch(`${API_BASE}/api/tournament/status`);
        const status = await resp.json();
        if (status.is_running) {
            startPolling();
        } else if (status.has_results) {
            await loadResults();
        }
    } catch (err) {
        // No results yet
    }
}

init();
