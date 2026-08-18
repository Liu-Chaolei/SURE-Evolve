# Main Flow Input v2

```yaml
schema: sure.eval.main_flow_input.v2
target:
  model_id: Org__Model
datasets:
  - name: source_dataset
    version: v1.0.0
    split: test
inference:
  protocol_id: standard_system
execution:
  mode: auto
```

Omit `inference` for the `standard_system` default. Set it to `strict_core`
only when deterministic constrained inference is explicitly required.
