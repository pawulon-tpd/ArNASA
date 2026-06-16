# Chemical Space Explorer

Interactive UMAP map of chemical space, built with RDKit + UMAP + HDBSCAN + Dash.

Upload an `.sdf`, `.smi`, or `.csv` (with a `SMILES`/`SMI`/`canonical_smiles` column) file and
explore the resulting 2D embedding: hover for name/SMILES, click a point to see its 2D structure,
and color the plot by HDBSCAN cluster or by a computed property (MW, LogP, TPSA, HBD, HBA, RotB, QED)
or by any numeric column/SDF tag present in the input file.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open http://127.0.0.1:8050 in a browser.

## How it works

- `chem_io.py` — parses SDF/SMI/CSV into RDKit molecules + metadata (SDF tags, CSV columns).
- `features.py` — computes 2048-bit Morgan (ECFP4) fingerprints and RDKit descriptors.
- `embedding.py` — UMAP (jaccard metric on fingerprints) for the 2D layout, HDBSCAN for clustering.
- `app.py` — Dash UI: file upload, color-by dropdown, scatter plot, click-to-view structure panel.

A sample input file is provided at `sample_data/sample.smi`.
