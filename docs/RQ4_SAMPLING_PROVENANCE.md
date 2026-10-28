# RQ4 sampled-executor provenance

Status: **QUALIFIED / GO** for the already-registered 19,420-generation
Llama-3.1 autoregressive control.

This record distinguishes the scientific registration from the later execution
support amendment:

- The five-sample RQ4 scientific settings were prospectively registered before
  sampled outcomes existed. The original freeze artifact has SHA-256
  `4f002417c736a7b3b7859e231fbf9cd0458f666d02e087cb7a364c56461093aa`.
- The original executor did not support the registered sampled settings: the
  Llama adapter rejected `do_sample=true`, and the generic generation path
  forced greedy decoding.
- Greedy RQ4 evidence existed before the sampled-executor amendment. It is not
  sampled evidence and is not used as sampled-throughput or parity evidence.
- The execution-only addendum has SHA-256
  `ed90a0ddb04d088b5004bb0437b1560e639ad97003a658326acdbeb2d831d9a5`.
  It added support for executing the registered sampling contract; it changed no seed, temperature, top-p, top-k, token limit, dataset, partition, model
  revision, telemetry feature, or statistical hypothesis.
- No sampled RQ4 outcome existed or was inspected before the addendum. The
  addendum is therefore not represented as part of the original executable
  freeze.

The outcome-blind calibration canary was executed from clean source commit
`5a3993c559128ed40bd9de1c504def687a8f861e`. Its immutable qualification report
has SHA-256
`db740a435a73b08f44657d4cc54fd3eb29104d48aac60fbd5fdce3c547861d7d`, and the
raw 12-row canary has SHA-256
`8dab2ee23120515ef94c41889684f3f2d93ee5eceb4785e061e9270238ee38b5`.
The gate passed 12/12 rows with exact native-versus-instrumented token-ID
parity, finite schema-valid telemetry, no outcome fields, no overwrite, and
259.01622335304575 examples/hour. At that measured rate, exactly 19,420
registered generations project to 74.97599860194873 A100 GPU-hours.

These qualification measurements authorize execution only. They are not RQ4
scientific outcomes and do not authorize an inferential claim. Final RQ4 claims
remain blocked until the complete campaign, scoring, registered analysis, and
claim gate pass.
