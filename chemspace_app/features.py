"""Fingerprints and descriptors used to build the chemical space embedding."""
import numpy as np
from rdkit.Chem import Descriptors, QED
from rdkit.Chem import rdFingerprintGenerator

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

DESCRIPTOR_FUNCS = {
    "MW": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "HBD": Descriptors.NumHDonors,
    "HBA": Descriptors.NumHAcceptors,
    "RotB": Descriptors.NumRotatableBonds,
    "QED": QED.qed,
}


def compute_fingerprints(mols):
    return np.array([_FP_GEN.GetFingerprintAsNumPy(mol) for mol in mols], dtype=np.float32)


def compute_descriptors(mols):
    return {name: [func(mol) for mol in mols] for name, func in DESCRIPTOR_FUNCS.items()}
