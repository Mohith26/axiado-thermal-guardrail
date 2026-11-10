/* Page logic: runs the seeded pipeline in the browser and renders it in the webui-vue shell. */
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const C = window.CARBON;
  const STATUS = { success: ['checkmark--filled', '#0a7d06'], warning: ['warning--filled', '#efc100'], danger: ['misuse', '#da1416'], info: ['information--filled', '#2d60e5'], secondary: ['information--filled', '#3f3f3f'] };
  function statusIcon(kind) {
    const [name, color] = STATUS[kind];
    // filled Carbon icons: white glyph inner path over a coloured disc (matches webui-vue StatusIcon)
    return `<svg class="status-ic" viewBox="0 0 32 32" style="fill:${color}"><circle cx="16" cy="16" r="10" fill="#fff"/>${C[name]}</svg>`;
  }
  function paintIcons(root) {
    root.querySelectorAll('svg[data-icon]').forEach(s => { s.innerHTML = C[s.dataset.icon] || ''; });
    root.querySelectorAll('span.status-ic[data-status]').forEach(s => { s.outerHTML = statusIcon(s.dataset.status); });
  }
  paintIcons(document);

  const DAY = '2026-11-09';
  const hms = (t) => { t = Math.round(t); const h = Math.floor(t / 3600), m = Math.floor(t % 3600 / 60), s = t % 60; return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`; };
  const hm = (t) => hms(t).slice(0, 5);
  const f1 = (x) => (Math.round(x * 10) / 10).toFixed(1);
  const pct = (x, d = 1) => (100 * x).toFixed(d) + '%';

  let prep = null, base = null, res = null, runMs = 0;

  function run(alpha) {
    const t0 = performance.now();
    if (!prep) prep = TG.prepare();
    if (!base) base = TG.evaluate(prep, alpha, { trace: true, only: ['stock', 'agent'] });
    const r = TG.evaluate(prep, alpha, { trace: true, only: ['flat', 'guard'] });
    res = { stock: base.stock, agent: base.agent, flat: r.flat, guard: r.guard, flatMargin: r.flatMargin, alpha };
    runMs = performance.now() - t0;
  }

  // ------------------------------------------------------------ cards + alert
  function renderCards() {
    const { stock, agent, guard, alpha } = res;
    const sG = TG.savings(stock, guard), sA = TG.savings(stock, agent);
    const al = guard.driftAlarms[0];
    const cards = [
      { title: 'Fan energy', big: `${guard.fanKWh.toFixed(2)} kWh`, sub: `Last 24 h, ${f1(sG)}% below the stock fan table. The agent alone saves ${f1(sA)}%, so the guardrail keeps ${Math.round(100 * sG / sA)}% of it.` },
      { title: 'Limit crossings', big: `${guard.crossingEvents}`, sub: `Agent alone: ${agent.crossingEvents} crossings and ${agent.throttleGpuS} GPU-seconds in slowdown. Stock table: ${stock.crossingEvents}.` },
      { title: 'Observed risk', big: pct(guard.overallMiss), sub: `of ${guard.decisions.toLocaleString()} fan decisions exceeded their 30 s bound (target ${pct(alpha, 0)}).` },
      al ? { title: 'Forecast drift', big: al.sensor, sub: `Reading ${f1(al.biasAfter)} \u00b0C above its model since ${hm(al.t)} (median of steady-load errors). Bounds widened automatically.` }
         : { title: 'Forecast drift', big: 'None', sub: 'Every sensor is tracking its model within 1 \u00b0C.' }
    ];
    $('cards').innerHTML = cards.map(c => `<div class="col"><div class="card"><div class="head"><h3 class="h5">${c.title}</h3><a href="#iv">View more</a></div><dl><dd class="big mono">${c.big}</dd><dd class="sub">${c.sub}</dd></dl></div></div>`).join('');
    $('alert-slot').innerHTML = al ? `<div class="alert alert-warning">${statusIcon('warning')}<p><strong>${al.sensor} is running hotter than its thermal model predicts.</strong> Median steady-load forecast error is +${f1(al.biasAfter)} \u00b0C since ${DAY} ${hms(al.t)}, which points to a cooling-path change (thermal interface, heatsink or airflow) on that GPU. Thermal guardrail widened its bounds; schedule an inspection of ${al.sensor}.</p></div>` : '';
    $('hdr-health').innerHTML = (al ? statusIcon('warning') : statusIcon('success')) + 'Health';
    $('replay-note').textContent = `Replay of ${DAY}: 24 h of 1 s telemetry, same workload for every policy.`;
  }

  // ------------------------------------------------------------ chart
  function windows() {
    const al = res.guard.driftAlarms[0];
    const w = [];
    if (al) { const a = Math.max(0, Math.floor((al.t - 2700) / 300) * 300); w.push({ id: 'drift', label: `${al.sensor} drift (${hm(a)} to ${hm(a + 9000)})`, a, b: a + 9000, step: 20 }); }
    const hv = prep.cfg.hvac; const ha = Math.floor((hv.start - 1200) / 300) * 300;
    w.push({ id: 'hvac', label: `Cooling derate (${hm(ha)} to ${hm(ha + 6600)})`, a: ha, b: ha + 6600, step: 20 });
    w.push({ id: 'day', label: 'Full day (2 min max)', a: 0, b: prep.dTest.N, step: 120 });
    return w;
  }
  function seriesHot(sim, a, b, step) { // hottest GPU reading
    const N = prep.dTest.N, S = prep.plat.sensors, out = [];
    for (let t = a; t < b; t += step) { let mx = -1e9; for (let k = t; k < Math.min(b, t + step); k++) for (let i = 2; i < S.length; i++) { const v = sim.read[i * N + k]; if (v > mx) mx = v; } out.push([t, mx]); }
    return out;
  }
  function seriesFan(sim, a, b, step) { const out = []; for (let t = a; t < b; t += step) { let s = 0, n = 0; for (let k = t; k < Math.min(b, t + step); k++) { s += sim.fan[k]; n++; } out.push([t, 100 * s / n]); } return out; }
  function renderChart() {
    const W = windows(); const sel = $('win');
    if (!sel.options.length) { sel.innerHTML = W.map(w => `<option value="${w.id}">${w.label}</option>`).join(''); }
    const w = W.find(x => x.id === sel.value) || W[0];
    const { guard, agent, stock } = res;
    const X0 = 56, X1 = 1046, T0 = 10, T1 = 240, F0 = 290, F1 = 400;
    const x = (t) => X0 + (t - w.a) / (w.b - w.a) * (X1 - X0);
    const gH = seriesHot(guard, w.a, w.b, w.step), aH = seriesHot(agent, w.a, w.b, w.step);
    let lo = 1e9, hi = -1e9; for (const p of gH.concat(aH)) { lo = Math.min(lo, p[1]); hi = Math.max(hi, p[1]); }
    lo = Math.floor((Math.min(lo, 80) - 2) / 5) * 5; hi = Math.max(hi, 90) + 1;
    const yT = (v) => T1 - (v - lo) / (hi - lo) * (T1 - T0);
    const yF = (v) => F1 - v / 100 * (F1 - F0);
    const path = (pts, y) => pts.map((p, i) => (i ? 'L' : 'M') + x(p[0]).toFixed(1) + ',' + y(p[1]).toFixed(1)).join('');
    let g = '';
    // grid + axes
    for (let v = Math.ceil(lo / 5) * 5; v <= hi; v += 5) g += `<line x1="${X0}" x2="${X1}" y1="${yT(v)}" y2="${yT(v)}" stroke="#e6e6e6"/><text x="${X0 - 8}" y="${yT(v) + 4}" font-size="12" fill="#666" text-anchor="end">${v}</text>`;
    for (let v = 0; v <= 100; v += 25) g += `<line x1="${X0}" x2="${X1}" y1="${yF(v)}" y2="${yF(v)}" stroke="#e6e6e6"/><text x="${X0 - 8}" y="${yF(v) + 4}" font-size="12" fill="#666" text-anchor="end">${v}</text>`;
    const span = w.b - w.a, tick = span > 30000 ? 3 * 3600 : span > 7200 ? 1800 : 900;
    for (let t = Math.ceil(w.a / tick) * tick; t <= w.b; t += tick) g += `<line x1="${x(t)}" x2="${x(t)}" y1="${T0}" y2="${F1}" stroke="#f0f0f0"/><text x="${x(t)}" y="${F1 + 18}" font-size="12" fill="#666" text-anchor="middle">${hm(t)}</text>`;
    g += `<text x="${X0}" y="${T0 - 0}" font-size="12" fill="#3f3f3f" dy="-0">\u00b0C</text><text x="${X0}" y="${F0 - 8}" font-size="12" fill="#3f3f3f">Fan duty %</text>`;
    // guard bound (from decision log): limit + bound of the applied decision
    const gb = guard.decisionLog.filter(d => d.t >= w.a && d.t < w.b && d.bound != null && isFinite(d.bound));
    if (w.step <= 30 && gb.length) {
      const lim = 87; let top = '';
      const agg = new Map(); for (const d of gb) { const k = d.t - (d.t - w.a) % w.step; agg.set(k, Math.max(agg.get(k) ?? -1e9, lim + d.bound)); }
      const pts = [...agg.entries()].sort((p, q) => p[0] - q[0]).map(([t, v]) => [t, Math.min(hi, v)]);
      top = pts.map((p, i) => (i ? 'L' : 'M') + x(p[0]).toFixed(1) + ',' + yT(p[1]).toFixed(1)).join('');
      g += `<path d="${top}" fill="none" stroke="#2d60e5" stroke-opacity=".35" stroke-width="1"/>`;
    }
    // limit lines
    g += `<line x1="${X0}" x2="${X1}" y1="${yT(87)}" y2="${yT(87)}" stroke="#da1416" stroke-dasharray="4 3"/><rect x="${X0 + 4}" y="${yT(87) - 17}" width="132" height="14" fill="#fff" fill-opacity=".85"/><text x="${X0 + 8}" y="${yT(87) - 6}" font-size="12" fill="#da1416">Upper critical 87 \u00b0C</text>`;
    g += `<line x1="${X0}" x2="${X1}" y1="${yT(83)}" y2="${yT(83)}" stroke="#efc100" stroke-dasharray="2 3"/>`;
    g += `<path d="${path(aH, yT)}" fill="none" stroke="#999" stroke-width="1.2"/>`;
    g += `<path d="${path(gH, yT)}" fill="none" stroke="#2d60e5" stroke-width="1.5"/>`;
    for (const t of agent.crossEvents) if (t >= w.a && t < w.b) g += `<circle cx="${x(t)}" cy="${yT(87)}" r="3.5" fill="#da1416"/>`;
    for (const t of guard.crossEvents) if (t >= w.a && t < w.b) g += `<circle cx="${x(t)}" cy="${yT(87)}" r="4" fill="none" stroke="#da1416" stroke-width="1.5"/>`;
    // fan panel
    g += `<path d="${path(seriesFan(stock, w.a, w.b, w.step), yF)}" fill="none" stroke="#3f3f3f" stroke-dasharray="5 3" stroke-width="1.2"/>`;
    g += `<path d="${path(seriesFan(agent, w.a, w.b, w.step), yF)}" fill="none" stroke="#999" stroke-width="1.2"/>`;
    g += `<path d="${path(seriesFan(guard, w.a, w.b, w.step), yF)}" fill="none" stroke="#2d60e5" stroke-width="1.5"/>`;
    // drift marker
    const al = guard.driftAlarms[0];
    if (al && al.t >= w.a && al.t < w.b) g += `<line x1="${x(al.t)}" x2="${x(al.t)}" y1="${T0}" y2="${F1}" stroke="#efc100" stroke-width="1.5"/><text x="${x(al.t) + 6}" y="${T0 + 12}" font-size="12" fill="#3f3f3f">${al.sensor} drift alarm ${hm(al.t)}</text>`;
    g += `<line x1="${X0}" x2="${X0}" y1="${T0}" y2="${T1}" stroke="#ccc"/><line x1="${X0}" x2="${X0}" y1="${F0}" y2="${F1}" stroke="#ccc"/>`;
    $('chart').innerHTML = g;
    const nA = agent.crossEvents.filter(t => t >= w.a && t < w.b).length, nG = guard.crossEvents.filter(t => t >= w.a && t < w.b).length;
    $('legend').innerHTML = `<span><i style="background:#2d60e5"></i>With thermal guardrail</span><span><i style="background:#999"></i>DTM agent alone</span><span><i style="background:#3f3f3f;height:0;border-top:2px dashed #3f3f3f"></i>Stock fan table</span>` + (w.step <= 30 ? `<span><i style="background:#2d60e5;opacity:.35"></i>30 s upper bound</span>` : '') + `<span><i class="dot"></i>Agent crossings (${nA})</span><span><i class="dot" style="background:#fff;border:1.5px solid #da1416"></i>Guarded crossings (${nG})</span>`;
  }

  // ------------------------------------------------------------ tables
  function renderTables() {
    const { stock, agent, flat, guard, flatMargin, alpha } = res;
    const rows = [
      ['Stock fan table (PI to 75 \u00b0C)', stock, null],
      [`DTM agent alone (${prep.cfg.agentMargin} \u00b0C margin)`, agent, null],
      [`Agent + flat margin (${f1(flatMargin)} \u00b0C)`, flat, null],
      ['Agent + thermal guardrail', guard, 'sel']
    ];
    const head = `<thead><tr><th><span class="sort">&#9662;</span>Policy</th><th class="num">Fan energy</th><th class="num">vs stock</th><th>Crossings</th><th class="num">Slowdown (GPU-s)</th><th class="num">Peak GPU</th><th class="num">Risk per decision</th></tr></thead>`;
    $('cmp').innerHTML = head + '<tbody>' + rows.map(([name, x, cls]) => `<tr class="${cls || ''}"><td>${name}</td><td class="num">${x.fanKWh.toFixed(2)} kWh</td><td class="num">${x === stock ? '0.0%' : '\u2212' + f1(TG.savings(stock, x)) + '%'}</td><td>${statusIcon(x.crossingEvents === 0 ? 'success' : x.crossingEvents < 5 ? 'warning' : 'danger')}${x.crossingEvents}</td><td class="num">${x.throttleGpuS}</td><td class="num">${f1(x.peakGpuC)} \u00b0C</td><td class="num">${pct(x.decisionCrossRate || 0, 2)}</td></tr>`).join('') + '</tbody>';
    $('cmp-note').textContent = `Same 24 h workload, ambient and hidden faults for every policy (cooling derate ${hm(prep.cfg.hvac.start)} to ${hm(prep.cfg.hvac.end)}, inlet +${prep.cfg.hvac.delta} \u00b0C; ${prep.cfg.degrade.sensor} cooling path degrades from ${hm(prep.cfg.degrade.start)}; neither is in the training data). The flat margin is the ${pct(1 - alpha, 0)} quantile of all of yesterday's forecast errors, applied everywhere.`;
    $('cal').innerHTML = `<thead><tr><th>Predicted distance to limit</th><th class="num">Calibration scores</th><th class="num">Starting bound</th><th class="num">Decisions today</th><th class="num">Observed miss rate</th><th class="num">Target</th></tr></thead><tbody>` +
      guard.coverage.map(c => `<tr><td>${c.band}</td><td class="num">${c.calibN.toLocaleString()}</td><td class="num">+${f1(c.qhatStart)} \u00b0C</td><td class="num">${c.n.toLocaleString()}</td><td class="num">${statusIcon(c.missRate <= alpha * 1.25 ? 'success' : 'warning')}${pct(c.missRate, 2)}</td><td class="num">${pct(alpha, 0)}</td></tr>`).join('') + '</tbody>';
    const iv = guard.interventions.slice().sort((p, q) => (q.boundAgentC - q.limit) - (p.boundAgentC - p.limit) || p.t - q.t);
    $('iv-count').textContent = `${guard.interventions.length.toLocaleString()} items`;
    $('pg-last').textContent = String(Math.ceil(guard.interventions.length / 10));
    $('iv').innerHTML = `<thead><tr><th><span class="sort">&#9662;</span>ID</th><th><span class="sort">&#9662;</span>Severity</th><th><span class="sort">&#9662;</span>Date</th><th>Sensor</th><th>Description</th></tr></thead><tbody>` +
      iv.slice(0, 10).map((d) => {
        const id = guard.interventions.indexOf(d) + 1;
        const sev = d.boundAgentC > d.limit + 2 ? ['danger', 'Critical'] : d.boundAgentC > d.limit + 0.5 ? ['warning', 'Warning'] : ['info', 'OK'];
        return `<tr><td>${id}</td><td>${statusIcon(sev[0])}${sev[1]}</td><td class="mono">${DAY}<br>${hms(d.t)} UTC</td><td>${d.sensor}</td><td>Agent asked for ${Math.round(100 * d.qAgent)}% fan; guardrail held ${Math.round(100 * d.qGuard)}%. 30 s bound at ${Math.round(100 * d.qAgent)}%: ${f1(d.boundAgentC)} \u00b0C on ${d.sensor} (upper critical ${d.limit} \u00b0C).</td></tr>`;
      }).join('') + '</tbody>';
  }

  function render() { renderCards(); renderChart(); renderTables(); }
  function go() {
    $('loading').style.display = 'block';
    setTimeout(() => { run(parseFloat($('alpha').value)); render(); $('loading').style.display = 'none'; }, 20);
  }
  $('alpha').addEventListener('change', go);
  $('win').addEventListener('change', renderChart);
  $('hdr-refresh').addEventListener('click', go);
  $('iv-export').addEventListener('click', () => {
    const lines = ['id,time,sensor,agent_fan_pct,guard_fan_pct,bound_at_agent_C,limit_C'];
    res.guard.interventions.forEach((d, i) => lines.push([i + 1, DAY + 'T' + hms(d.t) + 'Z', d.sensor, Math.round(100 * d.qAgent), Math.round(100 * d.qGuard), d.boundAgentC.toFixed(2), d.limit].join(',')));
    const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' })); a.download = 'thermal_guardrail_interventions.csv'; a.click();
  });
  go();
})();
