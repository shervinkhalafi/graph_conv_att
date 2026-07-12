"""Composer: a list of :class:`MolecularMetric` + a :class:`SMILESCodec`.

Mirrors :class:`tmgg.evaluation.graph_evaluator.GraphEvaluator`'s
surface so :class:`DiffusionModule` accepts either via duck typing.
The ``results.to_dict()`` method returns a flat ``{name: float}``
mapping that the existing ``on_validation_epoch_end`` loop logs as
``gen-val/<name>``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary

if TYPE_CHECKING:
    from tmgg.data.datasets.graph_types import DenseGraphState
    from tmgg.evaluation.molecular.metric import MolecularMetric


@dataclass
class MolecularEvaluationResults:
    """Flat results container, mirrors :class:`EvaluationResults`."""

    values: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        return dict(self.values)


class MolecularEvaluator:
    """Compose a list of metrics + a codec for SMILES decoding."""

    def __init__(
        self,
        metrics: Sequence[MolecularMetric],
        codec: SMILESCodec,
        eval_num_samples: int = 500,
    ) -> None:
        self.metrics = list(metrics)
        self.codec = codec
        # Mirror :class:`GraphEvaluator.eval_num_samples` so the
        # trainer's on_validation_epoch_end can read this off either
        # evaluator flavour without a type check (universal-transport
        # refactor 2026-05-01).
        self.eval_num_samples = eval_num_samples

    # ------------------------------------------------------------------
    # public surface (duck-types GraphEvaluator)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        refs: Sequence[DenseGraphState],
        generated: Sequence[DenseGraphState],
    ) -> MolecularEvaluationResults:
        """Decode both sets to SMILES; run each metric.

        Both ``refs`` and ``generated`` are sequences of per-graph
        :class:`tmgg.data.datasets.graph_types.DenseGraphState` carriers
        with a leading batch dim of 1 — the universal-transport format
        produced by
        :meth:`tmgg.data.data_modules.base_data_module.BaseGraphDataModule.get_reference_graphs`
        and by sampler-side per-graph chunking. The codec
        (:meth:`tmgg.data.datasets.molecular.codec.SMILESCodec.decode`)
        consumes that batched-of-one shape directly, so the evaluator
        does not flatten or re-batch.
        """
        ref_smiles = self._decode_all(refs)
        gen_smiles = self._decode_all(generated)
        results = MolecularEvaluationResults()
        for metric in self.metrics:
            value = metric.compute(gen_smiles, ref_smiles or None)
            if isinstance(value, dict):
                # dict[str, float] → flat keys
                for sub, sv in value.items():
                    results.values[f"{metric.name}/{sub}"] = sv
            else:
                results.values[metric.name] = float(value)
        return results

    def _decode_all(self, batch: Sequence[DenseGraphState]) -> list[str]:
        out: list[str] = []
        for data in batch:
            decoded = self.codec.decode(data)
            if decoded is not None:
                out.append(decoded)
        return out

    # ------------------------------------------------------------------
    # classmethod presets
    # ------------------------------------------------------------------

    @classmethod
    def for_qm9(
        cls, *, eval_num_samples: int = 500, **_kwargs: object
    ) -> MolecularEvaluator:
        # ``**_kwargs`` swallows fields leaked from the
        # ``discrete_sbm_official`` model preset's ``evaluator:`` block
        # via Hydra deep-merge (notably ``p_intra``, ``p_inter``,
        # ``clustering_sigma``). ``eval_num_samples`` is the sole
        # field we now thread through to the constructor — pre-2026-05-04
        # it was silently dropped here, which left every molecular eval
        # at the constructor default 500 regardless of the yaml.
        from tmgg.evaluation.molecular.rdkit_metrics import (
            NoveltyMetric,
            UniquenessMetric,
            ValidityMetric,
        )

        codec = SMILESCodec(
            vocab=AtomBondVocabulary.qm9(remove_h=True),
            max_atoms=9,
        )
        return cls(
            metrics=[ValidityMetric(), UniquenessMetric(), NoveltyMetric()],
            codec=codec,
            eval_num_samples=eval_num_samples,
        )

    @classmethod
    def for_moses(
        cls, *, eval_num_samples: int = 500, **_kwargs: object
    ) -> MolecularEvaluator:
        # See ``for_qm9`` for the ``**_kwargs`` rationale and the
        # ``eval_num_samples`` thread-through history.
        from tmgg.evaluation.molecular.moses_metrics import (
            FCDMetric,
            FiltersMetric,
            IntDivMetric,
            ScaffoldSplitMetric,
            SNNMetric,
        )
        from tmgg.evaluation.molecular.rdkit_metrics import (
            NoveltyMetric,
            UniquenessMetric,
            ValidityMetric,
        )

        codec = SMILESCodec(
            vocab=AtomBondVocabulary.moses(),
            max_atoms=30,
        )
        return cls(
            metrics=[
                ValidityMetric(),
                UniquenessMetric(),
                NoveltyMetric(),
                FCDMetric(),
                SNNMetric(),
                IntDivMetric(),
                FiltersMetric(),
                ScaffoldSplitMetric(),
            ],
            codec=codec,
            eval_num_samples=eval_num_samples,
        )

    @classmethod
    def for_guacamol(
        cls, *, eval_num_samples: int = 500, **_kwargs: object
    ) -> MolecularEvaluator:
        # See ``for_qm9`` for the ``**_kwargs`` rationale and the
        # ``eval_num_samples`` thread-through history.
        from tmgg.evaluation.molecular.guacamol_metrics import (
            FCDChEMBLMetric,
            KLDivPropertyMetric,
        )
        from tmgg.evaluation.molecular.rdkit_metrics import (
            NoveltyMetric,
            UniquenessMetric,
            ValidityMetric,
        )

        codec = SMILESCodec(
            vocab=AtomBondVocabulary.guacamol(),
            max_atoms=88,
        )
        return cls(
            metrics=[
                ValidityMetric(),
                UniquenessMetric(),
                NoveltyMetric(),
                KLDivPropertyMetric(),
                FCDChEMBLMetric(),
            ],
            codec=codec,
            eval_num_samples=eval_num_samples,
        )
