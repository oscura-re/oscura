# Demonstration Data

This directory contains generated test data for demonstrations.

## Generation

All data is synthetic and reproducible. Generate it with:

```bash
# From project root
python demonstrations/generate_all_data.py
```

## Why Not in Git?

- All data is 100% synthetic
- Completely reproducible from generation scripts
- Reduces repository size significantly
- Faster clones for all users

## Structure

```
data/
├── outputs/           # Demonstration output files
│   └── <demo_name>/   # Per-demo output directory
├── signals/           # Generated signal files
├── protocols/         # Protocol capture files
└── formats/           # Various file format examples
```

**Total Size:** ~350 MB (not tracked in git)
