# KMeans++ with Earth Mover's Distance

This repository implements several variants of the K‑means clustering algorithm in Rust.
It was created as an experiment for clustering poker hand strength histograms using
advanced distance metrics and initialization strategies.

## Features

- **KMeans++ initialization** for better initial centroid placement.
- **Earth Mover's Distance (EMD)** as the primary distance metric, with an optional
  Euclidean distance variant.
- **Triangle inequality optimisation** to speed up distance calculations.
- Parallel computation using **rayon**.
- Data loading and saving via **Protocol Buffers** definitions found under
  [`src/proto`](src/proto).

## Project Layout

```
.
├── data_in/   # Expected input files (hand strength histograms, EMD matrix, ...)
├── data_out/  # Output folder for labels and centroids after clustering
├── src/       # Rust source code
│   ├── algorithm.rs        # Core K‑means implementations
│   ├── distance.rs         # Distance functions (EMD and Euclidean)
│   ├── initialization.rs   # KMeans++ initialisation logic
│   ├── load.rs             # Helpers for loading binary datasets
│   ├── inertia.rs          # Functions to compute clustering inertia
│   └── proto/              # .proto message definitions
└── Cargo.toml
```

## Building

This is a standard Cargo project. To build the project:

```bash
cargo build --release
```

Running `cargo run --release` will execute the demonstration present in
[`src/main.rs`](src/main.rs), which loads histograms from `data_in/` and
performs clustering.

## Input Data

The program expects binary files containing serialized protobuf messages. Example
message types include `HandStrengthHistograms`, `ClusteredDataLabels` and
`ClusteredDataCentroids`. Place these files in the `data_in` directory before running
the program. Results are written to `data_out`.