/*
 * Thermal Guardrail core: a calibrated safety shield for a predictive fan-control agent.
 *
 *  1. A seeded synthetic platform: 8-GPU + 2-CPU air-cooled server, one fan zone,
 *     two-node (die/heatsink) thermal model, airflow-dependent heatsink resistance,
 *     back-row preheat, GPU slowdown (throttle) when a die crosses its limit.
 *  2. A forecaster: per-sensor ridge regression of the 30 s peak temperature rise
 *     as a function of the fan speed being considered (trained on an exploration log).
 *  3. A shield: Mondrian (per load regime) split-conformal bound on the forecast error of
 *     the worst sensor margin, kept honest online with adaptive conformal inference
 *     (Gibbs & Candes 2021). It vetoes any fan cut whose calibrated upper bound
 *     would cross a limit, and otherwise lets the agent's command through.
 *
 * Everything shown in the demo is computed by these functions; nothing is hard-coded.
 * Works in the browser (window.TG) and in Node (module.exports).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.TG = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ---------------------------------------------------------------- RNG
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    let u = 0; while (u === 0) u = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
  }
  const uni = (r, a, b) => a + (b - a) * r();

  // ---------------------------------------------------------------- platform
  const H = 30;          // forecast horizon (s)
  const DEC = 5;         // decision interval (s), matches a 5 s telemetry poll
  const QGRID = [];      // candidate fan duty (fraction), 20%..100% in 2% steps
  for (let p = 20; p <= 100; p += 2) QGRID.push(p / 100);

  function makePlatform(seed) {
    const r = mulberry32(seed);
    const sensors = [];
    for (let c = 0; c < 2; c++) sensors.push({
      id: 'CPU' + c, kind: 'cpu', tdp: 350, idle: 110, limit: 95, warn: 90, row: 0,
      Rdh: 0.030, R0: 0.030, a: 0.050, C: 900,
      unit: 1 + 0.06 * gauss(r) // hidden unit-to-unit heatsink variation
    });
    for (let g = 0; g < 8; g++) sensors.push({
      id: 'GPU' + g, kind: 'gpu', tdp: 700, idle: 85, limit: 87, warn: 83, row: g < 4 ? 0 : 1,
      Rdh: 0.018, R0: 0.015, a: 0.025, C: 1400,
      unit: 1 + 0.05 * gauss(r)
    });
    return {
      sensors,
      fans: { count: 10, wMax: 42, minPwm: 0.20 },
      Kpre: 2000,      // W/K of back-row preheat at 100% airflow
      throttleScale: 0.72,
      throttleHyst: 3
    };
  }
  // nominal (datasheet) heatsink resistance used by the model; the plant adds hidden unit variation
  const RsaNom = (s, q) => s.R0 + s.a / Math.pow(q, 0.8);
  const RsaTrue = (s, q, extra) => (s.R0 + s.a / Math.pow(q, 0.8)) * s.unit * (extra || 1);

  // ---------------------------------------------------------------- workload + ambient
  // Returns per-second GPU/CPU utilisation, ambient inlet temperature and a list of labelled segments.
  function genDay(seed, opts) {
    const r = mulberry32(seed);
    const N = opts.seconds;
    const startHour = opts.startHour || 0;
    const ug = new Float32Array(8 * N);
    const uc = new Float32Array(2 * N);
    const amb = new Float32Array(N);
    const phase = new Uint8Array(N); // 0 idle 1 train 2 infer 3 mixed
    const segs = [];
    let t = 0;
    const train = (i0, i1, gpus) => {
      let period = Math.round(uni(r, 6, 14));
      let nextCk = i0 + Math.round(uni(r, 900, 1500));
      const load = Math.min(Math.round(uni(r, 30, 70)), i1 - i0);
      for (let i = i0; i < i1; i++) {
        let u;
        if (i - i0 < load) u = 0.30 + 0.05 * gauss(r);            // data loading / warmup
        else if (i >= nextCk && i < nextCk + 55) u = 0.12;         // checkpoint write
        else u = ((i - i0) % period === 0) ? 0.55 : 0.92;          // step + all-reduce dip
        if (i === nextCk + 55) nextCk = i + Math.round(uni(r, 900, 1500));
        for (const g of gpus) ug[g * N + i] = Math.min(1, Math.max(0, u + 0.02 * gauss(r)));
      }
    };
    const infer = (i0, i1, gpus) => {
      let lam = 0.5;
      const burst = new Float32Array(8);
      const bleft = new Int32Array(8);
      for (let i = i0; i < i1; i++) {
        if ((i - i0) % 60 === 0) lam = Math.min(0.95, Math.max(0.1, 0.5 + 0.8 * (lam - 0.5) + 0.12 * gauss(r)));
        for (const g of gpus) {
          if (bleft[g] <= 0 && r() < 1 / 240) { bleft[g] = Math.round(uni(r, 15, 60)); burst[g] = uni(r, 0.85, 1.0); }
          let u = 0.22 + 0.6 * lam + 0.05 * gauss(r);
          if (bleft[g] > 0) { u = burst[g]; bleft[g]--; }
          ug[g * N + i] = Math.min(1, Math.max(0, u));
        }
      }
    };
    const all = [0, 1, 2, 3, 4, 5, 6, 7];
    while (t < N) {
      const hr = (startHour + t / 3600) % 24;
      const day = hr >= 8 && hr < 20;
      const x = r();
      let type, dur;
      if (day) type = x < 0.42 ? 2 : x < 0.78 ? 1 : x < 0.88 ? 3 : 0;
      else type = x < 0.55 ? 1 : x < 0.78 ? 0 : 2;
      if (type === 0) dur = uni(r, 5, 30); else if (type === 1) dur = uni(r, 40, 150);
      else if (type === 2) dur = uni(r, 20, 90); else dur = uni(r, 30, 90);
      const i0 = t, i1 = Math.min(N, t + Math.round(dur * 60));
      if (type === 0) { for (let i = i0; i < i1; i++) for (const g of all) ug[g * N + i] = Math.max(0, 0.03 + 0.01 * gauss(r)); }
      else if (type === 1) train(i0, i1, all);
      else if (type === 2) infer(i0, i1, all);
      else { train(i0, i1, [0, 1, 2, 3]); infer(i0, i1, [4, 5, 6, 7]); }
      for (let i = i0; i < i1; i++) phase[i] = type;
      segs.push({ type: ['idle', 'training', 'inference', 'mixed'][type], start: i0, end: i1 });
      t = i1;
    }
    let an = 0;
    for (let i = 0; i < N; i++) {
      const hr = (startHour + i / 3600) % 24;
      if (i % 30 === 0) an = 0.9 * an + 0.08 * gauss(r);
      let a = 23.5 + 1.8 * Math.sin(2 * Math.PI * (hr - 10) / 24) + an;
      if (opts.hvac) { // cooling-unit derate: inlet rises then recovers
        const { start, end, delta } = opts.hvac;
        if (i >= start && i < end) a += delta * (1 - Math.exp(-(i - start) / 180));
        else if (i >= end) a += delta * (1 - Math.exp(-(end - start) / 180)) * Math.exp(-(i - end) / 300);
      }
      amb[i] = a;
      let mg = 0; for (let g = 0; g < 8; g++) mg += ug[g * N + i]; mg /= 8;
      for (let c = 0; c < 2; c++) uc[c * N + i] = Math.min(1, Math.max(0.05, 0.22 + 0.45 * mg + 0.04 * gauss(r)));
    }
    return { N, startHour, ug, uc, amb, phase, segs };
  }

  // ---------------------------------------------------------------- features / forecaster
  const NF = 7;
  // Feature vector for sensor s at decision time, evaluated for candidate airflow q.
  // obs: {T, T5, P, P5, P15, tin, preW}  (preW = front-row GPU watts, for back-row preheat)
  function features(s, o, q, plat, out) {
    const tin = o.tin + (s.row === 1 ? o.preW / (plat.Kpre * q) : 0);
    const rsa = RsaNom(s, q);
    const tss = tin + o.P * (s.Rdh + rsa);
    const gap = tss - o.T;
    const approach = 1 - Math.exp(-H / (s.C * rsa));
    const phys = gap * approach; // first-order physics: expected approach to steady state within H
    out[0] = 1;
    out[1] = phys > 0 ? phys : 0;  // heating toward a hotter steady state
    out[2] = phys < 0 ? phys : 0;  // cooling: peak is near the current reading
    out[3] = (o.P - o.P5) / 100;
    out[4] = (o.P - o.P15) / 100;
    out[5] = o.T - o.T5;
    out[6] = o.P / s.tdp;
    return out;
  }
  function solve(A, b) { // Gaussian elimination with partial pivoting (small dense systems)
    const n = b.length; const M = A.map((row, i) => row.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c; for (let i = c + 1; i < n; i++) if (Math.abs(M[i][c]) > Math.abs(M[p][c])) p = i;
      [M[c], M[p]] = [M[p], M[c]];
      for (let i = c + 1; i < n; i++) { const f = M[i][c] / M[c][c]; for (let j = c; j <= n; j++) M[i][j] -= f * M[c][j]; }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; }
    return x;
  }
  function ridgeFit(X, y, lambda) {
    const n = y.length, d = X[0].length;
    const A = Array.from({ length: d }, () => new Array(d).fill(0));
    const b = new Array(d).fill(0);
    for (let k = 0; k < n; k++) { const x = X[k]; for (let i = 0; i < d; i++) { b[i] += x[i] * y[k]; for (let j = 0; j < d; j++) A[i][j] += x[i] * x[j]; } }
    for (let i = 1; i < d; i++) A[i][i] += lambda * n; // do not penalise intercept
    return solve(A, b);
  }
  // mean of a per-sensor series over seconds [t-a, t-b] (inclusive), clamped at 0
  function wmean(arr, base, t, a, b) { let s = 0, n = 0; for (let k = t - a; k <= t - b; k++) { s += arr[base + Math.max(0, k)]; n++; } return s / n; }
  function observe(read, pow, N, i, t, o) {
    o.T = wmean(read, i * N, t, 4, 0); o.T5 = wmean(read, i * N, t, 9, 5);
    o.P = wmean(pow, i * N, t, 4, 0); o.P5 = wmean(pow, i * N, t, 9, 5); o.P15 = wmean(pow, i * N, t, 19, 15);
    return o;
  }
  const dot = (w, x) => { let s = 0; for (let i = 0; i < w.length; i++) s += w[i] * x[i]; return s; };

  // ---------------------------------------------------------------- conformal helpers
  function quantile(sorted, level) { // finite-sample conformal quantile: ceil((n+1)level)-th smallest
    const n = sorted.length; if (n === 0) return Infinity;
    const k = Math.ceil((n + 1) * level);
    if (k > n) return Infinity; if (k < 1) return -Infinity;
    return sorted[k - 1];
  }
  function removeSorted(arr, v) { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < v) lo = m + 1; else hi = m; } if (lo < arr.length && arr[lo] === v) arr.splice(lo, 1); }
  function insertSorted(arr, v) { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < v) lo = m + 1; else hi = m; } arr.splice(lo, 0, v); }

  const REGIMES = ['idle', 'partial', 'sustained', 'ramp-up', 'ramp-down'];
  // Mondrian bins for the conformal score: how close the forecast says the hottest sensor will get to its limit.
  const BANDS = [
    { name: 'within 3 \u00b0C of a limit', lo: -3 },
    { name: '3 to 6 \u00b0C', lo: -6 },
    { name: '6 to 10 \u00b0C', lo: -10 },
    { name: 'more than 10 \u00b0C', lo: -Infinity }
  ];
  function bandOf(mhat) { for (let b = 0; b < BANDS.length; b++) if (mhat >= BANDS[b].lo) return b; return BANDS.length - 1; }
  function regimeOf(Ptot, Ptot15, tdpTot) {
    const d = Ptot - Ptot15;
    if (d > 0.06 * tdpTot) return 3;
    if (d < -0.06 * tdpTot) return 4;
    const lvl = Ptot / tdpTot;
    return lvl < 0.2 ? 0 : lvl < 0.6 ? 1 : 2;
  }

  // ---------------------------------------------------------------- closed-loop simulator
  /*
   * policy: { type: 'stock' | 'explore' | 'agent' | 'flat' | 'guard', margin, alpha, gamma, calib }
   * model: {gpu: w, cpu: w}  (required for agent/flat/guard and for residual logging)
   * Returns metrics, optional traces, the residual log (for calibration) and the intervention log.
   */
  function simulate(plat, day, policy, model, opts) {
    opts = opts || {};
    const S = plat.sensors, NS = S.length, N = day.N;
    const tdpG = S.filter(s => s.kind === 'gpu').reduce((a, s) => a + s.tdp, 0);
    const r = mulberry32(opts.noiseSeed || 7);
    const deg = opts.degrade || null; // {sensor:'GPU5', start, factor}
    const degIdx = deg ? S.findIndex(s => s.id === deg.sensor) : -1;
    // state
    const Ts = new Float64Array(NS), Td = new Float64Array(NS);
    const throttled = new Uint8Array(NS);
    let pwm = 0.6, cmd = 0.6;
    // init near steady state at 60%
    for (let i = 0; i < NS; i++) { const s = S[i]; const P = s.idle; Ts[i] = day.amb[0] + P * RsaTrue(s, pwm); Td[i] = Ts[i] + P * s.Rdh; }
    const read = new Float32Array(NS * N);   // sensor readings (integer C, like the BMC sees)
    const pow = new Float32Array(NS * N);    // actual power
    const fanTrace = new Float32Array(N);
    const cmdTrace = new Float32Array(N);
    let fanJ = 0, work = 0, workDemand = 0, throttleS = 0, cross = 0, peak = -1e9;
    const crossEvents = []; let inCross = false;
    // controller state
    let integ = 0, ePrev = 0, exHold = 0, exUntil = 0;
    const decisions = []; // {t, mhat(qApplied), qAgent, qGuard, q, regime, qhat}
    const resid = [];     // {regime, r}  realised worst margin - predicted
    // conformal state (guard)
    let bins = null, fifo = null, alphaB = null, missB = null, nB = null;
    const W = policy.window || 720;
    if (policy.type === 'guard') {
      // each regime keeps its most recent W scores, seeded with the calibration day
      fifo = BANDS.map((_, b) => (policy.calib[b] || []).slice(-W));
      bins = fifo.map(a => a.slice().sort((x, y) => x - y));
      alphaB = BANDS.map(() => policy.alpha); missB = BANDS.map(() => 0); nB = BANDS.map(() => 0);
    }
    // per-sensor drift monitor (one-sided CUSUM on each sensor's own forecast error)
    const cus = new Float64Array(NS), errMean = new Float64Array(NS), errN = new Float64Array(NS);
    const driftAlarms = [];
    const alarmed = new Uint8Array(NS);
    const sustSum = new Float64Array(NS), sustN = new Float64Array(NS);
    const sustWin = Array.from({ length: NS }, () => []), sustAll = Array.from({ length: NS }, () => []);
    const recentErr = Array.from({ length: NS }, () => []);
    let flatQ = policy.type === 'flat' ? policy.margin : 0;
    const fv = new Array(NF);
    const obs = { T: 0, T5: 0, P: 0, P5: 0, P15: 0, tin: 0, preW: 0 };
    const interventions = [];
    let uncontrollable = 0;
    const ptotAt = (t) => { let s = 0; for (let i = 2; i < NS; i++) s += wmean(pow, i * N, t, 4, 0); return s; };
    const lastPred = new Float64Array(NS);
    const mhatFor = (t, q) => { // predicted worst margin (C) if fans hold q for the next H seconds
      let m = -1e9;
      obs.tin = day.amb[t];
      let preW = 0; for (let i = 2; i < 6; i++) preW += wmean(pow, i * N, t, 4, 0);
      obs.preW = preW;
      for (let i = 0; i < NS; i++) {
        const s = S[i];
        observe(read, pow, N, i, t, obs);
        features(s, obs, q, plat, fv);
        const rise = dot(model[s.kind], fv);
        lastPred[i] = obs.T + rise;
        const v = lastPred[i] - s.limit;
        if (v > m) m = v;
      }
      return m;
    };
    const minFeasible = (t, margin) => {
      for (let k = 0; k < QGRID.length; k++) if (mhatFor(t, QGRID[k]) + margin <= 0) return QGRID[k];
      return 1.0;
    };
    const pending = [];
    for (let t = 0; t < N; t++) {
      // ---- plant step
      const q = Math.max(0.05, pwm);
      let preW = 0;
      for (let i = 0; i < NS; i++) {
        const s = S[i];
        const u = s.kind === 'gpu' ? day.ug[(i - 2) * N + t] : day.uc[i * N + t];
        let P = s.idle + (s.tdp - s.idle) * u;
        workDemand += s.kind === 'gpu' ? u : 0;
        if (throttled[i]) { P = s.idle + (P - s.idle) * plat.throttleScale; }
        if (s.kind === 'gpu') work += throttled[i] ? u * plat.throttleScale : u;
        pow[i * N + t] = P;
        if (s.kind === 'gpu' && s.row === 0) preW += P;
      }
      for (let i = 0; i < NS; i++) {
        const s = S[i];
        const P = pow[i * N + t];
        const tin = day.amb[t] + (s.row === 1 ? preW / (plat.Kpre * q) : 0);
        const extra = (i === degIdx && t >= deg.start) ? deg.factor : 1;
        const rsa = RsaTrue(s, q, extra);
        const tss = tin + P * rsa;
        Ts[i] = tss + (Ts[i] - tss) * Math.exp(-1 / (s.C * rsa));
        const rdh = s.Rdh * (i === degIdx && t >= deg.start ? deg.factor : 1);
        Td[i] = Ts[i] + P * rdh;
        read[i * N + t] = Math.round(Td[i] + 0.3 * gauss(r));
        if (Td[i] > peak && s.kind === 'gpu') peak = Td[i];
        if (Td[i] > s.limit) throttled[i] = 1; else if (Td[i] < s.limit - plat.throttleHyst) throttled[i] = 0;
        if (throttled[i] && s.kind === 'gpu') throttleS++;
      }
      let anyCross = false; for (let i = 0; i < NS; i++) if (Td[i] > S[i].limit) anyCross = true;
      if (anyCross) { cross++; if (!inCross) { crossEvents.push(t); inCross = true; } } else inCross = false;
      fanJ += plat.fans.count * plat.fans.wMax * Math.pow(pwm, 3);
      fanTrace[t] = pwm; cmdTrace[t] = cmd;

      // ---- resolve outcomes of decisions made H seconds ago (delayed feedback)
      while (pending.length && pending[0].t + H <= t) {
        const d = pending.shift();
        let m = -1e9;
        for (let i = 0; i < NS; i++) {
          let mx = -1e9; for (let k = 1; k <= H; k++) { const v = read[i * N + d.t + k]; if (v > mx) mx = v; }
          const v = mx - S[i].limit; if (v > m) m = v;
          // per-sensor forecast error -> drift monitor (steady-load decisions only, where the model is tight)
          const e = mx - d.pred[i];
          errMean[i] += e; errN[i]++;
          const mu = opts.mu ? opts.mu[i] : 0;
          if (d.regime === 2) {
            sustSum[i] += e; sustN[i]++; sustAll[i].push(e);
            const w = sustWin[i]; w.push(e); if (w.length > 120) w.shift();
            if (w.length === 120) {
              const med = w.slice().sort((x, y) => x - y)[60] - mu; cus[i] = med;
              if (!alarmed[i] && med > 1.0) { alarmed[i] = 1; driftAlarms.push({ sensor: S[i].id, t: t, medianShift: med }); }
            }
            if (alarmed[i]) recentErr[i].push(e - mu);
          }
        }
        const res = m - d.mhat;
        d.realised = m; d.res = res; d.crossed = m > 0;
        resid.push({ regime: d.regime, band: d.band, r: res });
        if (policy.type === 'guard') {
          const b = d.band; const miss = res > d.qhatApplied ? 1 : 0;
          nB[b]++; missB[b] += miss;
          alphaB[b] = Math.min(0.5, Math.max(0, alphaB[b] + policy.gamma * (policy.alpha - miss)));
          fifo[b].push(res); insertSorted(bins[b], res);
          if (fifo[b].length > W) removeSorted(bins[b], fifo[b].shift());
        }
      }

      // ---- control decision
      if (t % DEC === 0 && t >= 15) {
        const Ptot = ptotAt(t), Ptot15 = ptotAt(t - 15);
        const regime = regimeOf(Ptot, Ptot15, tdpG);
        let qNew = cmd, qAgent = null, qGuard = null, qhat = null, d_qh = null;
        if (policy.type === 'stock') {
          // incremental PI on the hottest margin to conservative setpoints (GPU 75 C, CPU 80 C)
          let e = -1e9; for (let i = 0; i < NS; i++) { const sp = S[i].kind === 'gpu' ? 75 : 80; const v = read[i * N + t] - sp; if (v > e) e = v; }
          qNew = cmd + 0.012 * (e - ePrev) + 0.0022 * e * DEC / 5;
          ePrev = e; qNew = Math.min(1, Math.max(0.30, qNew));
        } else if (policy.type === 'explore') {
          let hot = false; for (let i = 0; i < NS; i++) if (read[i * N + t] > S[i].limit - 3) hot = true;
          if (hot) { qNew = 1.0; exUntil = t + 60; }
          else if (t >= exUntil) { qNew = QGRID[Math.floor(r() * QGRID.length)]; qNew = Math.max(0.3, qNew); exUntil = t + Math.round(uni(r, 30, 120)); }
        } else {
          const margin = policy.type === 'flat' ? flatQ : policy.margin;
          qAgent = minFeasible(t, margin);
          qAgent = Math.max(qAgent, cmd - 0.04); // agent slews down gently, up immediately
          qNew = qAgent;
          if (policy.type === 'guard') {
            const qh = BANDS.map((_, b) => alphaB[b] <= 0 ? Infinity : quantile(bins[b], 1 - alphaB[b]));
            const U = (q) => { const m = mhatFor(t, q); return m + qh[bandOf(m)]; };
            qGuard = null;
            for (let k = 0; k < QGRID.length; k++) if (U(QGRID[k]) <= 0) { qGuard = QGRID[k]; break; }
            if (qGuard === null) {
              qGuard = 1.0; uncontrollable++; // no fan speed meets the bound: full airflow
            }
            const mA = mhatFor(t, qAgent);
            let hot = 0; for (let i = 1; i < NS; i++) if (lastPred[i] - S[i].limit > lastPred[hot] - S[hot].limit) hot = i;
            const hotPred = lastPred[hot];
            qhat = qh[bandOf(mA)];
            if (qGuard > qAgent + 1e-9) {
              qNew = qGuard;
              const mG = mhatFor(t, qGuard);
              interventions.push({ t, regime, qAgent, qGuard, sensor: S[hot].id, limit: S[hot].limit, predAgentC: hotPred, boundAgentC: hotPred + qhat, mhatAgent: mA, boundAgent: mA + qhat, mhatGuard: mG, boundGuard: mG + qh[bandOf(mG)], qhat });
            }
            d_qh = qh;
          }
        }
        cmd = qNew;
        if (model) {
          const mhat = mhatFor(t, cmd);
          const band = bandOf(mhat);
          const d = { t, regime, band, mhat, qAgent, qGuard, q: cmd, qhat, qhatApplied: d_qh ? d_qh[band] : null, bound: d_qh ? mhat + d_qh[band] : null, pred: Float32Array.from(lastPred) };
          decisions.push(d);
          if (t + H < N) pending.push(d);
        }
      }
      // fan slews toward command at 20%/s
      if (cmd > pwm) pwm = Math.min(cmd, pwm + 0.2); else pwm = Math.max(cmd, pwm - 0.2);
    }
    const resolved = decisions.filter(d => d.res !== undefined);
    const crossDec = resolved.filter(d => d.crossed).length;
    const out = {
      policy: policy.type,
      hours: N / 3600,
      fanKWh: fanJ / 3.6e6,
      meanPwm: fanTrace.reduce((a, b) => a + b, 0) / N,
      throttleGpuS: throttleS,
      crossingSeconds: cross,
      crossingEvents: crossEvents.length,
      crossEvents,
      lostWorkPct: 100 * (1 - work / Math.max(1e-9, workDemand)),
      peakGpuC: peak,
      decisions: resolved.length,
      decisionCrossRate: resolved.length ? crossDec / resolved.length : null,
      interventions,
      uncontrollable,
      resid,
      sensorBias: S.map((s, i) => ({ sensor: s.id, kind: s.kind, meanErr: errN[i] ? errMean[i] / errN[i] : 0, sustainedErr: sustN[i] ? sustAll[i].slice().sort((x, y) => x - y)[Math.floor(sustAll[i].length / 2)] : 0 })),
      driftAlarms: driftAlarms.map(a => { const i = S.findIndex(s => s.id === a.sensor); const arr = recentErr[i]; return Object.assign({}, a, { biasAfter: arr.length ? arr.reduce((x, y) => x + y, 0) / arr.length : null }); })
    };
    if (policy.type === 'guard') {
      out.coverage = BANDS.map((bd, b) => ({
        band: bd.name, n: nB[b], missRate: nB[b] ? missB[b] / nB[b] : null,
        alphaFinal: alphaB[b], calibN: (policy.calib[b] || []).length,
        qhatStart: quantile((policy.calib[b] || []).slice().sort((x, y) => x - y), 1 - policy.alpha)
      }));
      const tot = nB.reduce((a, b) => a + b, 0), mis = missB.reduce((a, b) => a + b, 0);
      out.overallMiss = tot ? mis / tot : null;
    }
    if (opts.trace) { out.read = read; out.fan = fanTrace; out.cmd = cmdTrace; out.decisionLog = resolved; out.pow = pow; }
    return out;
  }

  // ---------------------------------------------------------------- training from an exploration log
  function trainModel(plat, day, sim) {
    const S = plat.sensors, N = day.N, NS = S.length;
    const X = { gpu: [], cpu: [] }, Y = { gpu: [], cpu: [] };
    const read = sim.read, pow = sim.pow, cmd = sim.cmd;
    for (let t = 15; t + H < N; t += DEC) {
      // keep only windows where the fan command stayed constant over the horizon
      let same = true; for (let k = 1; k <= H; k++) if (Math.abs(cmd[t + k] - cmd[t]) > 1e-9) { same = false; break; }
      if (!same) continue;
      const q = cmd[t];
      let preW = 0; for (let i = 2; i < 6; i++) preW += wmean(pow, i * N, t, 4, 0);
      for (let i = 0; i < NS; i++) {
        const s = S[i];
        const o = observe(read, pow, N, i, t, { tin: day.amb[t], preW });
        let mx = -1e9; for (let k = 1; k <= H; k++) if (read[i * N + t + k] > mx) mx = read[i * N + t + k];
        X[s.kind].push(features(s, o, q, plat, new Array(NF)));
        Y[s.kind].push(mx - o.T);
      }
    }
    const model = { gpu: ridgeFit(X.gpu, Y.gpu, 1e-4), cpu: ridgeFit(X.cpu, Y.cpu, 1e-4) };
    // in-sample fit quality
    const rmse = {};
    for (const k of ['gpu', 'cpu']) { let se = 0; for (let j = 0; j < Y[k].length; j++) { const e = Y[k][j] - dot(model[k], X[k][j]); se += e * e; } rmse[k] = Math.sqrt(se / Y[k].length); }
    return { model, nTrain: Y.gpu.length + Y.cpu.length, rmse };
  }

  // ---------------------------------------------------------------- full pipeline
  const DEFAULTS = {
    seed: 20261110,
    agentMargin: 2.0,   // the agent's own heuristic safety margin (C)
    gammaFrac: 0.1,     // ACI step size as a fraction of the target miss rate
    window: 1440,       // recent scores kept per band (1440 decisions = 2 h at 5 s)
    trainHours: 12, calibHours: 12, testHours: 24,
    hvac: { start: Math.round(13.33 * 3600), end: Math.round(14.17 * 3600), delta: 5.5 },
    degrade: { sensor: 'GPU5', start: 15 * 3600, factor: 1.35 }
  };
  function prepare(cfg) {
    cfg = Object.assign({}, DEFAULTS, cfg || {});
    const plat = makePlatform(cfg.seed);
    const dTrain = genDay(cfg.seed + 1, { seconds: cfg.trainHours * 3600, startHour: 0 });
    const dCal = genDay(cfg.seed + 2, { seconds: cfg.calibHours * 3600, startHour: 8 });
    const dTest = genDay(cfg.seed + 3, { seconds: cfg.testHours * 3600, startHour: 0, hvac: cfg.hvac });
    const exp = simulate(plat, dTrain, { type: 'explore' }, null, { trace: true, noiseSeed: 11 });
    const fit = trainModel(plat, dTrain, exp);
    // calibration: run the unshielded agent and log its forecast errors per regime
    const cal = simulate(plat, dCal, { type: 'agent', margin: cfg.agentMargin }, fit.model, { noiseSeed: 12 });
    const calib = BANDS.map(() => []);
    for (const e of cal.resid) calib[e.band].push(e.r);
    const allCal = cal.resid.map(e => e.r).sort((a, b) => a - b);
    const mu = cal.sensorBias.map(b => b.sustainedErr); // each sensor's own median steady-load forecast error
    return { cfg, plat, dTrain, dCal, dTest, fit, cal, calib, allCal, mu };
  }
  function evaluate(prep, alpha, opts) {
    opts = opts || {};
    const { cfg, plat, dTest, fit, calib, allCal, mu } = prep;
    const sim = (pol, tr) => simulate(plat, dTest, pol, fit.model, { trace: tr, noiseSeed: 13, degrade: cfg.degrade, mu });
    const res = {};
    if (!opts.only || opts.only.includes('stock')) res.stock = opts.cache && opts.cache.stock || sim({ type: 'stock' }, opts.trace);
    if (!opts.only || opts.only.includes('agent')) res.agent = opts.cache && opts.cache.agent || sim({ type: 'agent', margin: cfg.agentMargin }, opts.trace);
    const flatMargin = quantile(allCal, 1 - alpha);
    if (!opts.only || opts.only.includes('flat')) res.flat = sim({ type: 'flat', margin: flatMargin }, opts.trace);
    if (!opts.only || opts.only.includes('guard')) res.guard = sim({ type: 'guard', margin: cfg.agentMargin, alpha, gamma: cfg.gammaFrac * alpha, calib, window: cfg.window }, opts.trace);
    res.flatMargin = flatMargin;
    return res;
  }
  function savings(base, x) { return 100 * (1 - x.fanKWh / base.fanKWh); }

  return { mulberry32, gauss, makePlatform, genDay, features, ridgeFit, solve, quantile, insertSorted, regimeOf,
    simulate, trainModel, prepare, evaluate, savings, REGIMES, BANDS, bandOf, QGRID, H, DEC, DEFAULTS, RsaNom };
});
