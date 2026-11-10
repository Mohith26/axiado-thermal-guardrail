// Seeded pipeline run: writes the numbers quoted in the README. The page recomputes everything live.
'use strict';
const fs = require('fs'); const path = require('path');
const TG = require('../core.js');
const prep = TG.prepare();
const out = { seed: prep.cfg.seed, generated: 'node scripts/generate.js', fit: prep.fit.rmse, nTrain: prep.fit.nTrain, alphas: {} };
for (const a of [0.01, 0.02, 0.05]) {
  const r = TG.evaluate(prep, a);
  const row = {};
  for (const k of ['stock', 'agent', 'flat', 'guard']) {
    const x = r[k];
    row[k] = { fanKWh: +x.fanKWh.toFixed(3), savingsPct: +TG.savings(r.stock, x).toFixed(1), crossingEvents: x.crossingEvents, throttleGpuS: x.throttleGpuS, peakGpuC: +x.peakGpuC.toFixed(1), interventions: x.interventions.length };
  }
  row.flatMargin = +r.flatMargin.toFixed(2);
  row.guardMiss = +r.guard.overallMiss.toFixed(4);
  row.driftAlarms = r.guard.driftAlarms.map(d => ({ sensor: d.sensor, hour: +(d.t / 3600).toFixed(2) }));
  out.alphas[a] = row;
}
const rob = [];
for (const seed of [20261110, 101, 202, 303, 404]) {
  const p = TG.prepare({ seed }); const r = TG.evaluate(p, 0.02);
  rob.push({ seed, agent: { save: +TG.savings(r.stock, r.agent).toFixed(1), events: r.agent.crossingEvents }, flat: { save: +TG.savings(r.stock, r.flat).toFixed(1), events: r.flat.crossingEvents }, guard: { save: +TG.savings(r.stock, r.guard).toFixed(1), events: r.guard.crossingEvents, miss: +r.guard.overallMiss.toFixed(4) }, firstAlarm: r.guard.driftAlarms[0] ? r.guard.driftAlarms[0].sensor + '@' + (r.guard.driftAlarms[0].t / 3600).toFixed(2) + 'h' : null });
}
out.robustness = rob;
fs.writeFileSync(path.join(__dirname, '..', 'data', 'summary.json'), JSON.stringify(out, null, 2));
console.log(JSON.stringify(out, null, 1));
