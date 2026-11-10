# Thermal Guardrail: a calibrated safety shield for an AI fan-control agent

**Live demo:** https://mohith26.github.io/axiado-thermal-guardrail/

Unofficial concept demo by Mohith Gajjela. Not affiliated with Axiado or the OpenBMC project. Synthetic data.

## What it is

Axiado's TCU runs a Dynamic Thermal Management (DTM) agent that forecasts temperatures and sets fan speed, claiming up to 50% lower air-cooling power. The open question for any operator letting an AI turn fans down is: *what stops it from cutting too far when its model is wrong?*

Thermal Guardrail is a shield that sits between the DTM agent and the fans. On every decision (every 5 s) it asks: "if the fans hold this speed for the next 30 s, how hot could the hottest sensor get, with a guaranteed error rate?" If the calibrated upper bound would cross a sensor's upper critical threshold, it vetoes the cut and holds the lowest fan speed that is safe. Otherwise the agent's command passes through untouched. A per-sensor drift monitor also tells you *which* part is running hotter than its model (for example a degrading thermal interface on one GPU).

The page is laid out as a new **Resource management > Thermal management** page in the OpenBMC web UI (webui-vue), which is the BMC web interface the TCU's OpenBMC distribution builds.

## Result on the seeded 24 h replay (target risk 2% per decision)

| Policy | Fan energy | vs stock fan table | Limit crossings | GPU-seconds in slowdown |
|---|---|---|---|---|
| Stock fan table (PI to 75 °C) | 2.901 kWh | 0% | 0 | 0 |
| DTM agent alone (2 °C margin) | 1.78 kWh | -38.6% | 26 | 45 |
| Agent + flat calibrated margin (5.32 °C) | 2.186 kWh | -24.6% | 0 | 0 |
| **Agent + Thermal Guardrail** | **1.962 kWh** | **-32.4%** | **0** | **0** |

The guardrail keeps 84% of the agent's savings and removes all 26 crossings; observed miss rate 2.17% against a 2% target. The drift monitor flags GPU5 at hour 15.18 (its cooling path was degraded at hour 15.0; nothing about that is in the training data).

Different risk targets (savings vs stock / crossings):

| target | agent alone | flat margin | guardrail | guardrail observed miss |
|---|---|---|---|---|
| 0.01 | 38.6% / 26 | -7.4% / 0 (margin 9.34 °C) | 24.4% / 6 | 1.16% |
| 0.02 | 38.6% / 26 | 24.6% / 0 (margin 5.32 °C) | 32.4% / 0 | 2.17% |
| 0.05 | 38.6% / 26 | 36.4% / 19 (margin 2.67 °C) | 35.4% / 3 | 5.26% |

### Robustness: five independent seeds (target 2%), savings / crossings

| seed | agent alone | flat margin | guardrail | observed miss | first drift alarm |
|---|---|---|---|---|---|
| 20261110 | 38.6% / 26 | 24.6% / 0 | 32.4% / 0 | 2.17% | GPU5@15.18h |
| 101 | 46.7% / 18 | 30.4% / 0 | 40.4% / 4 | 2.19% | GPU7@5.15h |
| 202 | 56.8% / 52 | 37.9% / 0 | 47.6% / 8 | 2.10% | GPU5@15.82h |
| 303 | 41.4% / 292 | 17.6% / 28 | 22.9% / 19 | 2.09% | GPU5@16.45h |
| 404 | 55.7% / 164 | 35.9% / 20 | 33.5% / 12 | 2.22% | GPU5@15.24h |

Across the five seeds the guardrail removes ~92% of the agent's crossings (552 to 43) and saves more fan energy than the flat margin in total (176.8 vs 146.4 percentage points summed), but it is not perfect: on seeds 303 and 404 the drift is severe enough that 12-19 crossings remain, and on seed 404 the flat margin saves slightly more. Coverage of the bound itself stays within 0.25 points of the 2% target on every seed. Crossings are rare events, so single-day counts are noisy (for example the 1% target shows 6 crossings on the default seed while 2% shows 0). Seed 101 has one false drift alarm (GPU7).

All numbers come from `node scripts/generate.js` (writes `data/summary.json`); the page recomputes everything live in the browser from the same seeded code.

## How it works

1. **Platform simulation** (`core.js`, seeded): an 8-GPU + 2-CPU air-cooled server with one fan zone. Each component has a die node and a heatsink node; heatsink-to-air resistance falls with airflow (R0 + a/q^0.8), back-row GPUs are preheated by the front row, fan power follows the cube law, and a GPU drops to 72% power (slowdown) above its 87 °C upper critical threshold. Workloads mix training jobs (step dips, checkpoint pauses, warm-up phases), bursty inference and idle gaps on a day/night schedule. The test day adds two surprises the model never saw: a cooling-unit derate (+5.5 °C inlet, 13:19 to 14:10) and a degraded cooling path on GPU5 from 15:00 (thermal resistance x1.35). Each unit also has hidden +-5% heatsink variation.
2. **Forecaster**: per sensor class, a ridge regression of the 30 s peak temperature rise on physics features (first-order approach to the steady state implied by the candidate fan speed, split into heating/cooling parts), recent power and temperature changes, and load level. Trained on 12 h of exploration telemetry (random held fan speeds, as Axiado describes training on "random and diverse intensity workloads").
3. **Stand-in DTM agent**: picks the lowest fan speed whose forecast keeps every sensor 2 °C below its limit, slewing down at most 4% per decision. This is my stand-in, not Axiado's DRL model; the guardrail only needs the agent's command and a forecast.
4. **Conformal bound**: the score is the error of the forecast worst margin, max_i(peak_i - limit_i), computed from what actually happened 30 s later. Scores are binned by how close the forecast says the hottest sensor will get (within 3 °C, 3-6, 6-10, more than 10), seeded from a 12 h calibration day of the unshielded agent, and each bin keeps its most recent 1,440 scores (2 h). The bound is the finite-sample conformal quantile, and adaptive conformal inference (Gibbs and Candes, 2021) adjusts each bin's miss rate online with step 0.1 x alpha, so the long-run miss rate tracks the target even under drift. Because crossing implies the realized margin exceeded the bound, the per-decision crossing probability is at most the miss rate.
5. **Shield**: final fan = the lowest speed whose bound is at or below 0 (never below the agent's own command); if no speed qualifies, full airflow.
6. **Drift monitor**: for each sensor, a rolling median (120 steady-load decisions) of its own forecast error minus its calibration-day median; alarm above +1.0 °C.

## Tests

```
node --test test/core.test.js
```

11 tests: finite-sample quantile rank, split-conformal coverage on exchangeable data, ridge recovery, deterministic seeds, physical steady state, band ordering, forecaster fit, the shield property (guard command never below agent), crossings and savings vs agent and flat margin, per-band online coverage, and the drift monitor localising GPU5.

## Honest limitations

- Synthetic platform and workloads. Parameters are plausible for an air-cooled 8-GPU server, not measured from an Axiado or NVIDIA system; the stock-vs-agent gap was not tuned to match Axiado's published 50% figure.
- The DTM agent here is a simple forecast-based controller, not Axiado's DRL/LLM agent.
- The bound assumes the fan holds the candidate speed for 30 s; later decisions can react, so the guarantee is conservative, and the "no speed qualifies, go to full airflow" fallback costs energy at idle (visible as fan spikes in the chart).
- Coverage is marginal within each band, not conditional on every situation. Calibration scores come from the unshielded agent, so the shielded loop's score distribution differs slightly; ACI corrects for that over time.
- Fan duty oscillates more than a production controller would allow; acoustic and fan-wear constraints are not modelled.
- The web UI is a static rebuild of upstream webui-vue's shell; it does not call Redfish.

## Novelty note

I checked Axiado's DTM announcement (Oct 2024), the OCP 2024 paper on DRL fan control, the AI-agent efficiency solution brief (Nov 2025), the Platform Intelligence Controller / SC26 page (Oct 2026), the Secure AI Copilot post, the Innovation Lab use cases and the news page through Oct 2026 for calibrated bounds, uncertainty, conformal prediction, guardrails, safety shields or drift detection on the DTM agent; the public material says the agent operates "while maintaining safe operating conditions" but describes no calibrated bound or drift-aware veto. Related work: Google DeepMind's 2018 "safety-first" data-center cooling (cited in Axiado's own OCP slides) filters actions by model uncertainty at the facility level; OpenBMC's phosphor-pid-control has a fixed failsafe fan percentage on sensor failure, which is not calibrated. A per-server, distribution-free bound with online adaptation and per-sensor drift localisation running next to the DTM agent appears to be new.

## Files

- `core.js`: simulation, forecaster, conformal shield, drift monitor (browser + Node)
- `app.js`, `index.html`, `icons.js`: the webui-vue shell (Carbon icons, Bootstrap 5 tokens from openbmc/webui-vue)
- `scripts/generate.js`: seeded run that writes `data/summary.json`
- `test/core.test.js`: unit and end-to-end tests
