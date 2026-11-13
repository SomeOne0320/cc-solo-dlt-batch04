# Reproducible local environment

Source: dlt 1.32.0, upstream commit `90c0cea1f7b59795de91824be466c99efca0f574`.
Fixed image: `adminfather/benzhi-claude-code@sha256:159e0b167850ddc1993271972df001dcb9815b1bd25b4fa55b9c69aedae6b922`.
Platform: Linux ARM64 / Python 3.11.2.

The lock includes runtime dependencies plus local extraction/state testing tools,
DuckDB, pandas and Arrow. It does not require a cloud destination or credential.

```bash
export RUNTIME__DLTHUB_TELEMETRY=false
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/dlt-get-pip.py
python3 /tmp/dlt-get-pip.py --user --break-system-packages pip==24.3.1
python3 -m pip install --user --break-system-packages -r requirements-cc-solo.txt
python3 -m pip install --user --break-system-packages --no-build-isolation --no-deps -e .
python3 -m pytest -q tests/extract/test_state.py tests/extract/test_interval_context.py
```

Tests write temporary `_storage_gw0` data inside this checkout, isolated from host configuration. Use a fresh checkout per task. Select additional local tests for the changed components and preserve the exact command and output. Destination integration suites that require credentials or optional services are outside this local environment; do not report them as tested.

The repository normally uses uv for development. The commands above install the checked-in resolved lock directly in the fixed image, where pip is not initially available. No package installation into the host Python is needed.
