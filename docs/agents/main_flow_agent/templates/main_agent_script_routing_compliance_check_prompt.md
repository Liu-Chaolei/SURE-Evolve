# Script Routing Compliance v2

Verify that the route uses only validated model IDs, defaults to
`standard_system`, accepts explicit `strict_core`, reads published artifacts
only through the fixed NFS registries, writes only to repository staging, and
uses `scripts/run_evaluation_bridge.py` for evaluation. Any use of arbitrary
model/result paths or `scripts/evaluate_predictions.py` fails compliance.
