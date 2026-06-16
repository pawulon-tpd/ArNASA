"""Parsing of SDF / SMI / CSV inputs into a list of RDKit molecules with metadata."""
import io

import pandas as pd
from rdkit import Chem

SMILES_COLUMN_CANDIDATES = ("smiles", "smi", "canonical_smiles")


def _mol_record(mol, name, extra_props):
    return {
        "mol": mol,
        "name": name,
        "smiles": Chem.MolToSmiles(mol),
        **extra_props,
    }


def parse_sdf(content_bytes):
    records = []
    suppl = Chem.ForwardSDMolSupplier(io.BytesIO(content_bytes), removeHs=False)
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        props = {k: mol.GetProp(k) for k in mol.GetPropNames()}
        numeric_props = {}
        for k, v in props.items():
            try:
                numeric_props[k] = float(v)
            except (TypeError, ValueError):
                pass
        name = mol.GetProp("_Name") if mol.HasProp("_Name") and mol.GetProp("_Name") else f"mol_{i}"
        records.append(_mol_record(mol, name, numeric_props))
    return records


def parse_smi(content_bytes):
    records = []
    text = content_bytes.decode("utf-8", errors="ignore")
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        smiles = parts[0]
        name = parts[1] if len(parts) > 1 else f"mol_{i}"
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        records.append(_mol_record(mol, name, {}))
    return records


def parse_csv(content_bytes):
    df = pd.read_csv(io.BytesIO(content_bytes))
    smiles_col = next(
        (c for c in df.columns if c.strip().lower() in SMILES_COLUMN_CANDIDATES), None
    )
    if smiles_col is None:
        raise ValueError(
            "No SMILES column found in CSV (expected one of: "
            f"{', '.join(SMILES_COLUMN_CANDIDATES)})"
        )
    name_col = next((c for c in df.columns if c.strip().lower() in ("name", "id", "title")), None)
    numeric_cols = [
        c for c in df.columns if c != smiles_col and pd.api.types.is_numeric_dtype(df[c])
    ]

    records = []
    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row[smiles_col]))
        if mol is None:
            continue
        name = str(row[name_col]) if name_col else f"mol_{i}"
        extra_props = {c: row[c] for c in numeric_cols if pd.notna(row[c])}
        records.append(_mol_record(mol, name, extra_props))
    return records


def parse_molecules(filename, content_bytes):
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "sdf":
        return parse_sdf(content_bytes)
    if ext in ("smi", "smiles", "txt"):
        return parse_smi(content_bytes)
    if ext == "csv":
        return parse_csv(content_bytes)
    raise ValueError(f"Unsupported file extension: .{ext}")
