"""Molecular datasets: SMILES → categorical DenseGraphState pipeline.

Compositional layout per
``internal notes (not included)``:

- :mod:`vocabulary` — frozen ``AtomBondVocabulary`` with QM9/MOSES/
  GuacaMol presets.
- :mod:`codec` — ``SMILESCodec``, the only place RDKit is imported.
- :mod:`dataset` — ``MolecularGraphDataset`` ABC with on-disk shard cache.
- :mod:`qm9`, :mod:`moses`, :mod:`guacamol` — concrete dataset subclasses.
"""

from __future__ import annotations

from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary

__all__ = ["AtomBondVocabulary"]
