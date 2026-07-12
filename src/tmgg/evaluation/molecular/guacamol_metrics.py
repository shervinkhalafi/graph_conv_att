"""GuacaMol Distribution-Learning metrics (Table 6 in DiGress).

Two metrics: :class:`KLDivPropertyMetric` (KL divergence on physchem
properties + Tanimoto similarity distribution) and
:class:`FCDChEMBLMetric` (Frechet ChemNet Distance against a caller-
supplied reference set).

The GuacaMol package is unmaintained: ``guacamol==0.5.2`` does
``from scipy import histogram`` (removed in scipy 1.0, 8 years ago) at
:mod:`guacamol.utils.chemistry` line 10, so the package no longer
imports under modern scipy. It also pulls in ``tensorflow==2.20.0`` as
a transitive dep, which conflicts with PyTorch's CUDA wheels. We
therefore vendor the four formulas we actually need
(:func:`_canonicalize_list`, :func:`_calculate_pc_descriptors`,
:func:`_continuous_kldiv`, :func:`_discrete_kldiv`) and drop the
``guacamol`` dependency.

Vendored from ``guacamol==0.5.2``:

* ``guacamol/distribution_learning_benchmark.py::KLDivBenchmark``
* ``guacamol/utils/chemistry.py``: ``calculate_pc_descriptors``,
  ``_calculate_pc_descriptors``, ``continuous_kldiv``,
  ``discrete_kldiv``, ``canonicalize``, ``canonicalize_list``,
  ``calculate_internal_pairwise_similarities``, ``get_mols``,
  ``get_fingerprints``.
* ``guacamol/utils/data.py::remove_duplicates``.

Parity verified by monkey-patching ``scipy.histogram = numpy.histogram``
to load the original ``KLDivBenchmark``, computing both scores on a
toy SMILES set, and asserting agreement to ~1e-3. The only behavioural
difference is that we accept pre-supplied canonicalised SMILES rather
than calling a ``DistributionMatchingGenerator.generate(N)`` adapter:
the upstream ``assess_model`` ends up treating ``model.generate(N)``'s
output the same way, so the two paths converge after the
``canonicalize_list`` step.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from tmgg.evaluation.molecular.metric import MolecularMetric

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol
    from rdkit.DataStructs.cDataStructs import ExplicitBitVect

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vendored GuacaMol helpers
# ---------------------------------------------------------------------------


def _remove_duplicates(items: Iterable[str]) -> list[str]:
    """Drop duplicates while preserving first-occurrence order.

    Vendored verbatim from ``guacamol.utils.data.remove_duplicates``.
    """

    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _canonicalize(smiles: str, include_stereocenters: bool = True) -> str | None:
    """Canonicalise a SMILES string with RDKit, returning ``None`` on failure.

    Vendored from ``guacamol.utils.chemistry.canonicalize``.
    """

    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)


def _canonicalize_list(
    smiles_list: Iterable[str],
    include_stereocenters: bool = True,
) -> list[str]:
    """Canonicalise, drop ``None``s, drop duplicates (order-preserving).

    Vendored from ``guacamol.utils.chemistry.canonicalize_list``.
    """

    canonicalised = [_canonicalize(s, include_stereocenters) for s in smiles_list]
    canonicalised = [s for s in canonicalised if s is not None]
    return _remove_duplicates(canonicalised)


def _get_mols(smiles_list: Iterable[str]) -> Iterator[Mol]:
    """Yield RDKit ``Mol`` objects, skipping invalid SMILES.

    Vendored from ``guacamol.utils.chemistry.get_mols``.
    """

    from rdkit import Chem

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            yield mol


def _get_fingerprints(
    mols: Iterable[Mol],
    radius: int = 2,
    length: int = 4096,
) -> list[ExplicitBitVect]:
    """Compute ECFP4 (radius=2, 4096 bits) bit-vector fingerprints.

    Vendored from ``guacamol.utils.chemistry.get_fingerprints`` but
    using the modern ``rdFingerprintGenerator`` API rather than the
    deprecated ``AllChem.GetMorganFingerprintAsBitVect``. The two
    routes produce numerically identical fingerprints.
    """

    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=length)
    return [gen.GetFingerprint(m) for m in mols]


def _calculate_internal_pairwise_similarities(
    smiles_list: Collection[str],
) -> NDArray[np.float64]:
    """Symmetric Tanimoto similarity matrix; diagonal zero.

    Vendored from
    ``guacamol.utils.chemistry.calculate_internal_pairwise_similarities``.
    """

    from rdkit import DataStructs

    mols = list(_get_mols(smiles_list))
    fps = _get_fingerprints(mols)
    nfps = len(fps)

    similarities = np.zeros((nfps, nfps), dtype=np.float64)
    for i in range(1, nfps):
        # rdkit-stubs misses ``BulkTanimotoSimilarity`` even though the
        # real ``rdkit.DataStructs`` re-exports it from the C++ extension.
        # Drop this marker once rdkit-stubs ships a fix.
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])  # type: ignore[reportAttributeAccessIssue]
        similarities[i, :i] = sims
        similarities[:i, i] = sims
    return similarities


def _calculate_pc_descriptors_single(
    smiles: str,
    pc_descriptors: Sequence[str],
) -> NDArray[np.float64] | None:
    """Compute a single SMILES' physchem-descriptor vector or ``None``.

    Vendored from ``guacamol.utils.chemistry._calculate_pc_descriptors``.
    """

    from rdkit import Chem
    from rdkit.ML.Descriptors import MoleculeDescriptors

    calc = MoleculeDescriptors.MolecularDescriptorCalculator(list(pc_descriptors))
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = np.asarray(calc.CalcDescriptors(mol), dtype=np.float64)
    mask = np.isfinite(fp)
    if (~mask).any():
        logger.warning("%s contains an NAN physchem descriptor", smiles)
        fp[~mask] = 0.0
    return fp


def _calculate_pc_descriptors(
    smiles: Iterable[str],
    pc_descriptors: Sequence[str],
) -> NDArray[np.float64]:
    """Compute ``(n_valid, n_descriptors)`` physchem-descriptor matrix.

    Vendored from ``guacamol.utils.chemistry.calculate_pc_descriptors``.
    """

    out: list[NDArray[np.float64]] = []
    for s in smiles:
        d = _calculate_pc_descriptors_single(s, pc_descriptors)
        if d is not None:
            out.append(d)
    return np.asarray(out, dtype=np.float64)


def _continuous_kldiv(
    x_baseline: NDArray[np.float64],
    x_sampled: NDArray[np.float64],
) -> float:
    """KL divergence on a 1000-point KDE-evaluated grid spanning both inputs.

    Vendored verbatim from ``guacamol.utils.chemistry.continuous_kldiv``.
    """

    from scipy.stats import entropy, gaussian_kde

    kde_p = gaussian_kde(x_baseline)
    kde_q = gaussian_kde(x_sampled)
    joined = np.hstack([x_baseline, x_sampled])
    x_eval = np.linspace(joined.min(), joined.max(), num=1000)
    p = kde_p(x_eval) + 1e-10
    q = kde_q(x_eval) + 1e-10
    return float(entropy(p, q))


def _discrete_kldiv(
    x_baseline: NDArray[np.float64],
    x_sampled: NDArray[np.float64],
) -> float:
    """KL divergence on density-normalised 10-bin histograms.

    Vendored from ``guacamol.utils.chemistry.discrete_kldiv``. Uses
    :func:`numpy.histogram` directly (the original code imports
    ``scipy.histogram``, removed in scipy 1.0).
    """

    from scipy.stats import entropy

    p, bins = np.histogram(x_baseline, bins=10, density=True)
    p = p + 1e-10
    q, _ = np.histogram(x_sampled, bins=bins, density=True)
    q = q + 1e-10
    return float(entropy(p, q))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Vendored from ``KLDivBenchmark.__init__`` in guacamol 0.5.2: the first
# four are continuous, the last five integer-valued.
_PC_DESCRIPTOR_SUBSET: tuple[str, ...] = (
    "BertzCT",
    "MolLogP",
    "MolWt",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "NumAliphaticRings",
    "NumAromaticRings",
)
_N_CONTINUOUS_DESCRIPTORS = 4


class KLDivPropertyMetric(MolecularMetric):
    """KL divergence on the GuacaMol physchem-property distribution.

    Reimplements ``guacamol.distribution_learning_benchmark.KLDivBenchmark``
    against the caller-supplied generated and reference SMILES lists.
    Returns ``mean(exp(-KL_d))`` over the 9 physchem descriptors plus a
    Tanimoto-similarity-based "internal_similarity" KL term, i.e. a
    score in ``[0, 1]`` (higher = closer to ref).
    """

    name = "kl_div_property"

    def __init__(self, n_samples: int = 10_000) -> None:
        self.n_samples = n_samples

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        if reference is None:
            raise ValueError("KLDivPropertyMetric requires reference SMILES.")

        # Match the upstream pipeline: canonicalise both sets, drop
        # invalid/duplicates. ``include_stereocenters=False`` matches
        # the upstream call inside ``assess_model``.
        gen_unique = _canonicalize_list(generated, include_stereocenters=False)
        ref_unique = _canonicalize_list(reference, include_stereocenters=False)
        if not gen_unique or not ref_unique:
            return 0.0

        d_sampled = _calculate_pc_descriptors(gen_unique, _PC_DESCRIPTOR_SUBSET)
        d_chembl = _calculate_pc_descriptors(ref_unique, _PC_DESCRIPTOR_SUBSET)

        kldivs: list[float] = []
        for i in range(_N_CONTINUOUS_DESCRIPTORS):
            kldivs.append(
                _continuous_kldiv(
                    x_baseline=d_chembl[:, i],
                    x_sampled=d_sampled[:, i],
                )
            )
        for i in range(_N_CONTINUOUS_DESCRIPTORS, len(_PC_DESCRIPTOR_SUBSET)):
            kldivs.append(
                _discrete_kldiv(
                    x_baseline=d_chembl[:, i],
                    x_sampled=d_sampled[:, i],
                )
            )

        # Per-row max Tanimoto similarity, then continuous KL between
        # the two row-max distributions. Matches upstream `KLDivBenchmark`.
        chembl_sim = _calculate_internal_pairwise_similarities(ref_unique).max(axis=1)
        sampled_sim = _calculate_internal_pairwise_similarities(gen_unique).max(axis=1)
        kldivs.append(_continuous_kldiv(x_baseline=chembl_sim, x_sampled=sampled_sim))

        partial_scores = [float(np.exp(-k)) for k in kldivs]
        return float(np.mean(partial_scores))


class FCDChEMBLMetric(MolecularMetric):
    """Frechet ChemNet Distance against a caller-supplied reference set.

    The historical name "ChEMBL" reflects DiGress' use of GuacaMol's
    ChEMBL training split as the reference; the metric itself is
    distribution-agnostic. Backed by the maintained :pkg:`fcd` package
    (``bioinf-jku/FCD``, PyPI ``fcd>=1.2.2``); we never depended on
    ``guacamol`` for this metric. The previous backend was the now-
    frozen ``fcd_torch==1.0.7`` (2019); the new ``fcd>=1.2.2`` ships
    the same ChemNet weights and Frechet formula, and parity holds to
    ~3e-3 absolute on toy SMILES sets.
    """

    name = "fcd_chembl"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        # Cached ChemNet ``torch.nn.Module``. The ``fcd`` package ships
        # no type stubs, so we keep the slot as ``Any`` rather than
        # ``object`` to satisfy ``get_fcd``'s ``model: Module | None``
        # parameter without an extra cast.
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from fcd import load_ref_model

            self._model = load_ref_model()
        return self._model

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        if reference is None:
            raise ValueError("FCDChEMBLMetric requires reference SMILES.")
        from fcd import canonical_smiles, get_fcd

        # ``_canonicalize_list`` already drops invalid SMILES and
        # duplicates via RDKit; we still funnel through ``fcd``'s own
        # ``canonical_smiles`` for byte-exact parity with the upstream
        # GuacaMol/FCD pipeline.
        gen_valid = _canonicalize_list(generated, include_stereocenters=False)
        if not gen_valid:
            return float("inf")
        gen_canonical = [s for s in canonical_smiles(list(gen_valid)) if s]
        ref_canonical = [s for s in canonical_smiles(list(reference)) if s]
        if not gen_canonical or not ref_canonical:
            return float("inf")
        model = self._ensure_model()
        return float(
            get_fcd(gen_canonical, ref_canonical, model=model, device=self.device)
        )
