// wiki.js — Research Wiki logic

// ── Agent metadata (loaded from graph_data.json) ──
let GRAPH_DATA = null;

// ── State ──
let currentView = 'tree';
let currentAgent = null;

// ── Init ──
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const resp = await fetch('wiki/graph_data.json');
        if (!resp.ok) throw new Error('Failed to load graph data: ' + resp.status);
        GRAPH_DATA = await resp.json();
    } catch (e) {
        console.error('Could not load wiki/graph_data.json:', e);
        // Provide empty fallback so the page doesn't crash
        GRAPH_DATA = { nodes: [], edges: [] };
        const treeEl = document.getElementById('evolution-tree');
        if (treeEl) treeEl.innerHTML =
            '<div style="display:flex;align-items:center;justify-content:center;height:100%;opacity:0.4;min-height:200px;">' +
            '<p>Could not load graph data. Place <code>graph_data.json</code> in <code>web/wiki/</code>.</p></div>';
    }

    initTabs();
    initTree();
    initSidebar();
    handleHashRoute();

    window.addEventListener('hashchange', handleHashRoute);
});

// ── Tab switching ──
function initTabs() {
    document.querySelectorAll('.wiki-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const view = tab.dataset.view;
            switchView(view);
        });
    });
}

function switchView(view) {
    currentView = view;
    document.querySelectorAll('.wiki-tab').forEach(t =>
        t.classList.toggle('active', t.dataset.view === view)
    );
    document.querySelectorAll('.wiki-view').forEach(v => v.classList.remove('active'));
    const viewEl = document.getElementById(view + '-view');
    if (viewEl) viewEl.classList.add('active');

    if (view === 'overview') loadOverview();
    if (view === 'tree') redrawTreeConnections();
}

// ── Hash routing ──
function handleHashRoute() {
    const hash = window.location.hash.slice(1); // remove #
    if (hash.startsWith('agent/')) {
        const agentId = hash.split('/')[1];
        switchView('agents');
        loadAgentPage(agentId);
    } else if (hash === 'overview') {
        switchView('overview');
    }
}

// ── Evolution tree (dagre layout + HTML cards + SVG connections) ──
let treeContainer = null;
let treeInner = null;
let treeSvg = null;
let treeNodePositions = {}; // { id: { x, y, width, height } }

function initTree() {
    treeContainer = document.getElementById('evolution-tree');
    if (!treeContainer || !GRAPH_DATA || GRAPH_DATA.nodes.length === 0) return;

    // Run dagre layout
    const g = new dagre.graphlib.Graph();
    g.setGraph({ rankdir: 'TB', nodesep: 50, ranksep: 90, edgesep: 30, marginx: 40, marginy: 40 });
    g.setDefaultEdgeLabel(() => ({}));

    const cardW = 200;
    const cardH = 56;

    GRAPH_DATA.nodes.forEach(n => {
        g.setNode(n.id, { width: cardW, height: cardH });
    });
    GRAPH_DATA.edges.forEach(e => {
        g.setEdge(e.from, e.to);
    });

    dagre.layout(g);

    // Build node lookup
    const nodeMap = {};
    GRAPH_DATA.nodes.forEach(n => nodeMap[n.id] = n);

    // Get graph dimensions
    const graphInfo = g.graph();
    const totalW = graphInfo.width + 80;
    const totalH = graphInfo.height + 80;

    // Create inner container
    treeInner = document.createElement('div');
    treeInner.className = 'tree-inner';
    treeInner.style.width = totalW + 'px';
    treeInner.style.height = totalH + 'px';

    // Create SVG for connections
    treeSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    treeSvg.setAttribute('class', 'tree-svg');
    treeSvg.setAttribute('width', totalW);
    treeSvg.setAttribute('height', totalH);
    treeInner.appendChild(treeSvg);

    // Create cards at dagre-computed positions
    treeNodePositions = {};
    g.nodes().forEach(id => {
        const pos = g.node(id);
        const node = nodeMap[id];
        if (!node) return;

        const left = pos.x - cardW / 2;
        const top = pos.y - cardH / 2;
        treeNodePositions[id] = { x: pos.x, y: pos.y, width: cardW, height: cardH };

        const card = document.createElement('div');
        card.className = 'tree-card' + (node.category === 'game_theory' ? ' game-theory' : '');
        card.style.left = left + 'px';
        card.style.top = top + 'px';
        card.dataset.agent = id;

        const avgScore = typeof node.avg_score === 'number' ? node.avg_score : null;
        const scoreColor = avgScore === null ? 'rgba(255,255,255,0.3)' : (avgScore >= 0 ? '#7daa8c' : '#aa7d7d');
        const scoreText = avgScore === null ? '—' : ((avgScore >= 0 ? '+' : '') + avgScore.toFixed(2));

        // Short description — first clause before ' — '
        const shortDesc = (node.description || '').split(' — ')[0];

        card.innerHTML = `
            <div class="tree-card-header">
                <span class="tree-card-name">${node.display_name}</span>
                <span class="tree-card-score" style="color:${scoreColor}">${scoreText}</span>
            </div>
            <div class="tree-card-desc">${shortDesc}</div>
        `;

        card.addEventListener('click', () => {
            window.location.hash = 'agent/' + id;
        });

        treeInner.appendChild(card);
    });

    treeContainer.appendChild(treeInner);

    // Draw connections
    drawTreeConnections();

    // Redraw on resize
    window.addEventListener('resize', redrawTreeConnections);
}

function drawTreeConnections() {
    if (!treeSvg) return;
    // Clear existing paths
    while (treeSvg.firstChild) treeSvg.removeChild(treeSvg.firstChild);

    GRAPH_DATA.edges.forEach(edge => {
        const from = treeNodePositions[edge.from];
        const to = treeNodePositions[edge.to];
        if (!from || !to) return;

        // From center-bottom of parent to center-top of child
        const x1 = from.x;
        const y1 = from.y + from.height / 2;
        const x2 = to.x;
        const y2 = to.y - to.height / 2;

        // Control points for smooth S-curve
        const midY = (y1 + y2) / 2;

        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`);
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', 'rgba(255,255,255,0.12)');
        path.setAttribute('stroke-width', '1');
        treeSvg.appendChild(path);
    });
}

function redrawTreeConnections() {
    drawTreeConnections();
}

// ── Sidebar ──
function initSidebar() {
    const sidebar = document.getElementById('wiki-sidebar');
    if (!sidebar || !GRAPH_DATA || !GRAPH_DATA.nodes) return;

    // Build parent lookup from edges
    const childToParents = {};
    GRAPH_DATA.edges.forEach(e => {
        if (!childToParents[e.to]) childToParents[e.to] = [];
        childToParents[e.to].push(e.from);
    });

    // Define lineage groups by root ancestor
    const groups = [
        {
            label: 'Value Lineage',
            description: 'TD(0) self-play family',
            ids: ['value_based', 'nstep_value', 'target_value', 'td_variant', 'aux_value']
        },
        {
            label: 'Adaptive Lineage',
            description: 'Opponent-modeling family',
            ids: ['adaptive_value', 'decay_adaptive', 'pop_adaptive', 'modulated_value',
                  'extended_adaptive', 'curriculum']
        },
        {
            label: 'History Lineage',
            description: 'Action-history family',
            ids: ['history_value', 'adaptive_history', 'pruned_history']
        },
        {
            label: 'Policy Lineage',
            description: 'Actor-critic family',
            ids: ['actor_critic', 'entropy_ac']
        },
        {
            label: 'Baselines',
            description: 'Non-learned references',
            ids: ['heuristic', 'cfr']
        }
    ];

    const nodeMap = {};
    GRAPH_DATA.nodes.forEach(n => nodeMap[n.id] = n);

    groups.forEach(group => {
        const section = document.createElement('div');
        section.className = 'sidebar-section';

        const header = document.createElement('h3');
        header.textContent = group.label;
        header.title = group.description;
        section.appendChild(header);

        const agentsDiv = document.createElement('div');
        agentsDiv.className = 'sidebar-agents';

        group.ids.forEach(id => {
            const node = nodeMap[id];
            if (!node) return;

            const btn = document.createElement('button');
            btn.className = 'sidebar-agent-btn';
            btn.dataset.agent = node.id;

            const avgScore = typeof node.avg_score === 'number' ? node.avg_score : 0;
            const scoreColor = avgScore >= 0 ? '#7daa8c' : '#aa7d7d';
            const scorePrefix = avgScore >= 0 ? '+' : '';
            const roundBadge = node.category === 'game_theory' ? 'GT' : 'R' + node.round;

            btn.innerHTML = `
                <span class="agent-name">
                    <span class="round-badge">${roundBadge}</span>
                    ${node.display_name}
                </span>
                <span class="agent-score" style="color: ${scoreColor}">${scorePrefix}${avgScore.toFixed(2)}</span>
            `;

            btn.addEventListener('click', () => {
                window.location.hash = 'agent/' + node.id;
            });

            agentsDiv.appendChild(btn);
        });

        section.appendChild(agentsDiv);
        sidebar.appendChild(section);
    });
}

// ── Load agent wiki page ──
async function loadAgentPage(agentId) {
    currentAgent = agentId;

    // Update sidebar active state
    document.querySelectorAll('.sidebar-agent-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.agent === agentId);
    });

    const contentEl = document.getElementById('wiki-markdown');
    contentEl.innerHTML = '<div class="loading-spinner"></div> Loading...';

    try {
        const resp = await fetch('wiki/' + encodeURIComponent(agentId) + '.md');
        if (!resp.ok) throw new Error('Not found');
        const md = await resp.text();
        contentEl.innerHTML = marked.parse(md);
        colorizeScores(contentEl);
    } catch (e) {
        contentEl.innerHTML = '<div class="placeholder-text" style="padding: 40px; text-align: center;">' +
            '<h2 style="opacity: 0.5;">Page Not Found</h2>' +
            '<p style="opacity: 0.3;">No wiki page for agent "' + agentId + '"</p>' +
            '</div>';
    }
}

// ── Load overview ──
let overviewLoaded = false;
async function loadOverview() {
    if (overviewLoaded) return;
    const el = document.getElementById('overview-markdown');
    try {
        const resp = await fetch('wiki/_overview.md');
        if (!resp.ok) throw new Error('Not found');
        const md = await resp.text();
        el.innerHTML = marked.parse(md);
        colorizeScores(el);
        overviewLoaded = true;
    } catch (e) {
        el.innerHTML = '<p style="opacity: 0.5;">Failed to load overview. Place <code>_overview.md</code> in <code>web/wiki/</code>.</p>';
    }
}

// ── Colorize scores in rendered markdown ──
function colorizeScores(container) {
    // Find table cells and colorize positive/negative numbers
    container.querySelectorAll('td').forEach(td => {
        const text = td.textContent.trim();
        const match = text.match(/^([+-]?\d+\.\d+)$/);
        if (match) {
            const val = parseFloat(match[1]);
            if (val > 0) {
                td.style.color = '#4ade80';
                td.style.fontWeight = '600';
            } else if (val < 0) {
                td.style.color = '#f87171';
                td.style.fontWeight = '600';
            }
        }
    });
}
