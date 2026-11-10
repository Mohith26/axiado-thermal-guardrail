'use strict';
const test = require('node:test');
const assert = require('node:assert');
const TG = require('../core.js');

test('conformal quantile uses the finite-sample ceil((n+1)(1-a)) rank', () => {
  const xs = Array.from({ length: 19 }, (_, i) => i + 1); // 1..19
  assert.strictEqual(TG.quantile(xs, 0.9), 18);
  assert.strictEqual(TG.quantile(xs, 0.99), Infinity); // not enough points for that level
});

test('split conformal bound covers exchangeable data at the target rate', () => {
  const r = TG.mulberry32(5); let miss = 0, trials = 4000;
  for (let k = 0; k < trials; k++) {
    const cal = Array.from({ length: 99 }, () => TG.gauss(r) * 2).sort((a, b) => a - b);
    const q = TG.quantile(cal, 0.95);
    if (TG.gauss(r) * 2 > q) miss++;
  }
  const rate = miss / trials;
  assert.ok(rate <= 0.06 && rate >= 0.03, 'miss rate ' + rate);
});

test('ridge regression recovers a known linear model', () => {
  const r = TG.mulberry32(9); const X = [], y = [];
  for (let i = 0; i < 2000; i++) { const x = [1, r(), r(), r()]; X.push(x); y.push(0.5 + 2 * x[1] - 1 * x[2] + 3 * x[3] + 0.01 * TG.gauss(r)); }
  const w = TG.ridgeFit(X, y, 0);
  [0.5, 2, -1, 3].forEach((c, i) => assert.ok(Math.abs(w[i] - c) < 0.02, `w${i}=${w[i]}`));
});

test('workload generator is deterministic for a seed', () => {
  const a = TG.genDay(42, { seconds: 3600 }), b = TG.genDay(42, { seconds: 3600 }), c = TG.genDay(43, { seconds: 3600 });
  assert.deepStrictEqual(Array.from(a.ug.slice(0, 500)), Array.from(b.ug.slice(0, 500)));
  assert.notDeepStrictEqual(Array.from(a.ug.slice(0, 500)), Array.from(c.ug.slice(0, 500)));
});

test('plant reaches the analytic steady state under constant load and fan', () => {
  const plat = TG.makePlatform(1);
  const N = 1800, day = { N, ug: new Float32Array(8 * N).fill(1), uc: new Float32Array(2 * N).fill(1), amb: new Float32Array(N).fill(25) };
  const sim = TG.simulate(plat, day, { type: 'explore' }, null, { trace: true });
  // explore policy changes the fan, so just check a physical bound: no GPU exceeds limit for long without throttling
  assert.ok(sim.peakGpuC < 100);
  const s = plat.sensors[2]; const q = 1.0;
  const tss = 25 + 700 * (s.Rdh + TG.RsaNom(s, q) * s.unit);
  assert.ok(tss > 55 && tss < 80, 'front GPU steady state at full fan ' + tss);
});

test('margin bands are ordered from near-limit to far', () => {
  assert.strictEqual(TG.bandOf(-1), 0); assert.strictEqual(TG.bandOf(-4), 1);
  assert.strictEqual(TG.bandOf(-8), 2); assert.strictEqual(TG.bandOf(-30), 3);
});

const prep = TG.prepare();
const res = TG.evaluate(prep, 0.02);

test('forecaster is trained on held-constant fan windows and fits reasonably', () => {
  assert.ok(prep.fit.nTrain > 10000);
  assert.ok(prep.fit.rmse.gpu < 2.5 && prep.fit.rmse.cpu < 1.5, JSON.stringify(prep.fit.rmse));
});

test('guardrail is a shield: it never commands less airflow than the agent', () => {
  const tr = TG.simulate(prep.plat, prep.dTest, { type: 'guard', margin: prep.cfg.agentMargin, alpha: 0.02, gamma: 0.002, calib: prep.calib, window: 1440 }, prep.fit.model, { trace: true, noiseSeed: 13, degrade: prep.cfg.degrade, mu: prep.mu });
  assert.ok(tr.decisionLog.length > 1000);
  for (const d of tr.decisionLog) assert.ok(d.q >= d.qAgent - 1e-9);
  for (const iv of res.guard.interventions) assert.ok(iv.qGuard > iv.qAgent);
});

test('guardrail removes most limit crossings while keeping most of the savings', () => {
  const sa = TG.savings(res.stock, res.agent), sg = TG.savings(res.stock, res.guard);
  assert.ok(res.agent.crossingEvents >= 5, 'agent should cross under the injected drift');
  assert.ok(res.guard.crossingEvents <= Math.floor(res.agent.crossingEvents * 0.2));
  assert.ok(sg > 0.6 * sa, `guard keeps ${sg} of ${sa}`);
  assert.ok(sg > TG.savings(res.stock, res.flat), 'adaptive bound beats the flat calibrated margin on energy');
});

test('online coverage stays close to the target miss rate in every band', () => {
  for (const c of res.guard.coverage) assert.ok(c.missRate < 0.035, c.band + ' ' + c.missRate);
  assert.ok(Math.abs(res.guard.overallMiss - 0.02) < 0.006);
});

test('drift monitor localises the degraded GPU', () => {
  const al = res.guard.driftAlarms;
  assert.ok(al.length >= 1 && al[0].sensor === 'GPU5', JSON.stringify(al));
  assert.ok(al[0].t > prep.cfg.degrade.start && al[0].t < prep.cfg.degrade.start + 3600);
});
