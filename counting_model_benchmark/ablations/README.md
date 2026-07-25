# Counting ablation workspace

Keep one directory per ablation family, for example:

```text
ablations/
  a01_count_point/
    README.md
    configs/
  a02_boundary_hysteresis/
    README.md
    configs/
  a03_temporal_confirmation/
    README.md
    configs/
```

Each ablation README should state the single changed factor, unchanged controls,
validation split, parameter search range, selection rule, and expected output
directory. Generated caches and result CSV files belong under
`project_results/counting_model_benchmark/<run_id>/`, not in the source tree.

Do not recreate the withdrawn B0/E1/E2/E3 scheme automatically. New ablations
must be defined from the current research question and validated independently.
