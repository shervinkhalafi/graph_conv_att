"""Loss helpers for the 4-type GraphData grid.

Each helper takes a :class:`_DistributionGraph` prediction and a
:class:`_StateGraph` target. Mixing two distributions or two states is a
runtime ``TypeError`` (and a static type error to the extent the carriers
are statically known); the helpers refuse to silently densify or sparsify
across the carrier axis.

Carrier-matched dispatch
------------------------
Both helpers accept dense+dense or sparse+sparse pairs. Cross-carrier
calls MUST be converted by the caller using the ``GraphData`` conversion
APIs (``to_dense`` / ``to_sparse``) before invoking the loss.
"""

from tmgg.training.losses.masked_ce import masked_ce_loss
from tmgg.training.losses.masked_mse import masked_mse_loss

__all__ = ["masked_ce_loss", "masked_mse_loss"]
