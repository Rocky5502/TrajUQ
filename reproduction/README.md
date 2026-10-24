# Reviewer reproduction

The default workflow is intentionally source-only and CPU-safe. It validates the manuscript firewall, anonymity/template structure, and released authorized artifact hashes without downloading models or datasets.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-analysis.txt
python reproduction/reproduce_core.py
```

After every registered RQ is complete, the final workflow is:

```bash
python reproduction/reproduce_core.py --final
python reproduction/reproduce_tables.py
python reproduction/reproduce_figures.py
```

Final commands fail closed while required artifacts are missing or unauthorized. They do not convert partial progress into manuscript values. Raw benchmark rows, model weights, caches, and credentials are neither required for source-only validation nor redistributed.

