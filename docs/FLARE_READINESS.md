# HybridDiffusion / FLARE readiness decision

Status: **SKIP / NO-GO**. Do not launch a FLARE workload from the current
repository state.

This is an exploratory same-checkpoint diagnostic readiness decision, not a new
research question and not an empirical result.

| Gate | Current evidence | Decision |
|---|---|---|
| Upstream model | `yuchen-zhu-zyc/HybridDiffusion-2B` | Identified |
| Immutable model revision | `configs/models/flare.yaml` has `revision: null` | **FAIL** |
| Upstream runtime | `yuchen-zhu-zyc/HybridDiffusion` at code commit `6ca547aebb72bfe897e80e0a683c776789e5f38c` | Repository pinned; full executable environment not qualified |
| License | HybridDiffusion-authored code, modifications, and checkpoints are reported under PolyForm Noncommercial 1.0.0; third-party portions retain their own terms | Research-only constraints require review before release |
| Native generation parity | No seed-matched native-versus-instrumented exact-token PASS artifact | **FAIL** |
| Telemetry parity | Native trace bridge exists; parity-safe full-logit telemetry is not qualified | **FAIL** |
| Runtime / GPU-hour estimate | No frozen revision and no parity-qualified canary | Not defensibly estimable |
| Scientific role | Same-checkpoint AR-versus-diffusion diagnostic | Potentially useful, but exploratory only |

## Scientific value and confounding

If one immutable HybridDiffusion checkpoint can be run through both its native
autoregressive and diffusion paths with exact seed-matched instrumentation
parity, the comparison would hold the checkpoint family more nearly fixed and
therefore reduce the paradigm/model-family confound. It would not eliminate
prompting, sampler, or mode-specific computation differences and would not
support a causal claim about observability level.

The current repository does not yet establish the immutable checkpoint,
executable environment, native output parity, or telemetry parity needed for
that interpretation. A GPU-hour estimate before those gates would be an
unsupported extrapolation.

## Reconsideration gate

Reconsider only through a new prospective exploratory freeze that binds:

1. the exact model/tokenizer revision and license snapshot;
2. the external runtime and dependency lock;
3. a calibration-only seed-matched native parity canary;
4. telemetry parity and finite-schema validation;
5. measured canary throughput and peak memory;
6. a fixed, small diagnostic workload and immutable analysis plan.

Until all six pass, the recommendation remains **SKIP**. Do not substitute a
different checkpoint, add a model, or create an RQ6.
