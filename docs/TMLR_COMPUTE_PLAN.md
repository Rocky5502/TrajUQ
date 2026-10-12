# TMLR compute and access plan

## Qualified worker specification

The accepted per-worker target is:

- NVIDIA A100-SXM4-80GB (80 GiB VRAM), exactly one visible GPU per worker;
- Ubuntu 24.04 with the provider CUDA 13.0 image used by the frozen campaign;
- 16 CPU cores and 48 GB host RAM;
- 200 GB system disk plus a 350 GB data disk mounted at `/data`;
- at least 250 Mbps effective network bandwidth during restoration/download;
- hourly billing, auto-renewal disabled;
- retain the disk until outputs have been collected and hash-validated locally.

The screenshot's top summary must show `A100-SXM4-80GB (80G)`, not `(0G)`,
before purchase. It must also show the intended bandwidth; a summary of 32 Mbps
means the 250 Mbps slider selection has not taken effect.

Prefer multiple independent one-GPU instances, each with 16 cores/48 GB, over a
multi-GPU instance that shares only 16 cores/48 GB across all GPUs. Independent
workers match the existing exclusive artifact-ownership controller and can be
released individually after their generation shards are collected.

## Existing outcome-blind projection

The frozen estimate contains 225.5 p90 planning GPU-hours for original primary,
RQ2, and RQ3 generation. It excludes downloads, loading, scoring, retries, and
RQ4. The controller's existing shard projections imply approximately:

| Qualified A100 workers | Original generation critical path | Operational recommendation including setup, RQ4 pilot/run, analysis, and final writing |
|---:|---:|---:|
| 1 | 9.4 days | 12--14 days |
| 2 | 4.7 days | 7--9 days |
| 3 | 3.2 days | 5--7 days |
| 4 | 2.4 days | 5--6 days |
| 6 | 1.7 days | 4--5 days |

The rightmost ranges are planning estimates, not guarantees. RQ4 must first run
an outcome-blind Llama throughput pilot; its measured trace will replace the
current conservative allowance. Four independent A100 workers are the preferred
cost/time compromise. Six are appropriate when minimum wall clock dominates
cost.

## Credential handoff

Do not paste passwords into chat or commit them. For the first new worker, put
the provider's SSH host, port, username, and password in the existing
operator-owned file `<controller-root>/gpu key.txt` using the provider labels
already parsed by the controller:

```text
SSH地址(移动): HOST
端口: PORT
SSH账户: USERNAME
密码: PASSWORD
```

For additional workers, create untracked files
`<controller-root>/trajuq_scaleout_2800205/credentials/a100-02.json` through
`a100-06.json`, contiguously, using the local
`worker_credential.template.json`. Never place credential files inside the Git
repository, evidence archive, or submission package.

## Launch gate

After credentials appear, the controller must verify GPU identity/VRAM, host
RAM, disk, exact commit, runtime and package versions, model/cache revisions,
and deterministic canaries. Generation begins only after qualification PASS and
immutable allocation authorization. Failed or partial workers cannot become
authoritative writers.
