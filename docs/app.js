/* MicroRTS LLM Competition - Leaderboard App */

(function () {
    'use strict';

    var DATA_URL = 'data/tournament_results.json';
    var ANCHORS = ['RandomBiasedAI', 'HeavyRush', 'LightRush', 'WorkerRush', 'Tiamat', 'CoacAI'];

    // Tab switching
    document.querySelectorAll('.tab').forEach(function (tab) {
        tab.addEventListener('click', function (e) {
            e.preventDefault();
            var target = this.getAttribute('data-tab');
            if (!target) return;

            document.querySelectorAll('.tab').forEach(function (t) {
                t.classList.remove('active');
            });
            this.classList.add('active');

            document.querySelectorAll('.tab-content').forEach(function (c) {
                c.classList.remove('active');
            });
            var el = document.getElementById(target);
            if (el) el.classList.add('active');
        });
    });

    // Grade badge HTML
    function gradeBadge(grade) {
        var cls = 'grade-f';
        if (grade === 'A+') cls = 'grade-aplus';
        else if (grade === 'A') cls = 'grade-a';
        else if (grade === 'B') cls = 'grade-b';
        else if (grade === 'C') cls = 'grade-c';
        else if (grade === 'D') cls = 'grade-d';
        return '<span class="grade ' + cls + '">' + escapeHtml(grade) + '</span>';
    }

    function escapeHtml(str) {
        if (str == null) return '';
        return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    // Format opponent cell
    function opponentCell(opponents, anchorName) {
        var data = opponents ? opponents[anchorName] : null;
        if (!data) return '<td class="result-skip">--</td>';

        var w = data.wins || 0;
        var d = data.draws || 0;
        var l = data.losses || 0;
        var pts = data.weighted_points != null ? data.weighted_points : '?';

        var cls = 'result-skip';
        if (w > 0) cls = 'result-win';
        else if (l > 0) cls = 'result-loss';
        else if (d > 0) cls = 'result-draw';

        return '<td class="' + cls + '">' + w + 'W/' + d + 'D/' + l + 'L<br><small>' + pts + ' pts</small></td>';
    }

    // Format date
    function formatDate(dateStr) {
        if (!dateStr) return '--';
        try {
            var d = new Date(dateStr);
            return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
        } catch (e) {
            return dateStr.substring(0, 10);
        }
    }

    // Render leaderboard
    function renderLeaderboard(data) {
        var tbody = document.querySelector('#leaderboard-table tbody');
        if (!tbody) return;
        tbody.innerHTML = '';

        var entries = data.leaderboard || [];
        entries.forEach(function (entry, i) {
            var row = document.createElement('tr');
            var rank = i + 1;
            var html = '<td class="rank-cell">' + rank + '</td>';
            html += '<td>' + escapeHtml(entry.display_name) + '</td>';
            html += '<td class="score-cell">' + entry.score + '</td>';
            html += '<td>' + gradeBadge(entry.grade) + '</td>';
            html += '<td class="date-cell">' + formatDate(entry.date) + '</td>';

            ANCHORS.forEach(function (anchor) {
                html += opponentCell(entry.opponents, anchor);
            });

            row.innerHTML = html;
            tbody.appendChild(row);
        });
    }

    // Render benchmarks tab
    function renderBenchmarks(data) {
        var tbody = document.querySelector('#benchmarks-table tbody');
        if (!tbody) return;
        tbody.innerHTML = '';

        var entries = data.history || [];
        // Deduplicate: show best per name
        var seen = {};
        var best = [];
        entries.forEach(function (entry) {
            var key = entry.display_name;
            if (!seen[key] || entry.score > seen[key].score) {
                seen[key] = entry;
            }
        });
        // Sort by score descending
        best = Object.values(seen);
        best.sort(function (a, b) { return b.score - a.score; });

        best.forEach(function (entry) {
            var row = document.createElement('tr');
            var src = entry.source || 'unknown';
            row.innerHTML =
                '<td>' + escapeHtml(entry.display_name) + '</td>' +
                '<td class="score-cell">' + entry.score + '</td>' +
                '<td>' + gradeBadge(entry.grade) + '</td>' +
                '<td><span class="source-badge">' + escapeHtml(src) + '</span></td>' +
                '<td>' + formatDate(entry.date) + '</td>';
            tbody.appendChild(row);
        });
    }

    // Render head-to-head matrix
    function renderH2H(data) {
        var h2h = data.head_to_head || {};
        var players = Object.keys(h2h).sort();

        if (players.length === 0) {
            document.getElementById('h2h-empty').style.display = 'block';
            document.querySelector('#h2h .table-container').style.display = 'none';
            return;
        }

        var thead = document.querySelector('#h2h-table thead');
        var tbody = document.querySelector('#h2h-table tbody');

        // Header row
        var headerHtml = '<tr><th></th>';
        players.forEach(function (p) {
            headerHtml += '<th>' + escapeHtml(p.split(' (')[0]) + '</th>';
        });
        headerHtml += '</tr>';
        thead.innerHTML = headerHtml;

        // Body rows
        tbody.innerHTML = '';
        players.forEach(function (p1) {
            var row = document.createElement('tr');
            var html = '<td>' + escapeHtml(p1) + '</td>';

            players.forEach(function (p2) {
                if (p1 === p2) {
                    html += '<td class="h2h-self">-</td>';
                } else {
                    var record = h2h[p1] && h2h[p1][p2];
                    if (record) {
                        var cls = 'result-skip';
                        if (record.wins > record.losses) cls = 'result-win';
                        else if (record.losses > record.wins) cls = 'result-loss';
                        else if (record.draws > 0) cls = 'result-draw';
                        html += '<td class="' + cls + '">' + record.wins + 'W/' + record.draws + 'D/' + record.losses + 'L</td>';
                    } else {
                        html += '<td class="result-skip">--</td>';
                    }
                }
            });

            row.innerHTML = html;
            tbody.appendChild(row);
        });
    }

    // Render history tab
    function renderHistory(data) {
        var tbody = document.querySelector('#history-table tbody');
        if (!tbody) return;
        tbody.innerHTML = '';

        var entries = data.history || [];
        entries.forEach(function (entry) {
            var row = document.createElement('tr');
            var src = entry.source || 'unknown';
            row.innerHTML =
                '<td>' + formatDate(entry.date) + '</td>' +
                '<td>' + escapeHtml(entry.display_name) + '</td>' +
                '<td class="score-cell">' + entry.score + '</td>' +
                '<td>' + gradeBadge(entry.grade) + '</td>' +
                '<td><span class="source-badge">' + escapeHtml(src) + '</span></td>' +
                '<td>' + escapeHtml(entry.map || '--') + '</td>';
            tbody.appendChild(row);
        });
    }

    // Load data and render
    fetch(DATA_URL)
        .then(function (response) {
            if (!response.ok) throw new Error('Failed to load data: ' + response.status);
            return response.json();
        })
        .then(function (data) {
            renderLeaderboard(data);
            renderBenchmarks(data);
            renderH2H(data);
            renderHistory(data);

            var genEl = document.getElementById('generated-at');
            if (genEl && data.generated) {
                genEl.textContent = 'Data updated: ' + formatDate(data.generated);
            }
        })
        .catch(function (err) {
            console.error('Error loading tournament data:', err);
            var main = document.querySelector('main');
            if (main) {
                main.innerHTML = '<p class="empty-msg">Failed to load tournament data. ' +
                    'If viewing locally, serve with a local HTTP server (e.g. <code>python3 -m http.server</code> in the docs/ directory).</p>';
            }
        });
})();
