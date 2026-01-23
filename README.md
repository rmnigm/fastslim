# Faster SLIM
Rust implementation of [SLIM (**S**parse **Li**near **M**ethods)](https://ieeexplore.ieee.org/document/6137254) for item-based collaborative filtering with parallel computation. See [README](slim-rs/README.md) of library for details.

## Project Structure

```
slim-rs/
├── Cargo.toml
├── pyproject.toml
├── src/
│   └── lib.rs            # Rust solver
└── python/
    └── fastslim/
        ├── __init__.py   # Python API
        └── test_slim.py  # Tests
```