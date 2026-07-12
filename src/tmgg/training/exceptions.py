"""Custom exceptions for experiment utilities."""


class CheckpointMismatchError(Exception):
    """Raised when checkpoint doesn't match expected model signature.

    This occurs when loading a checkpoint that contains hyperparameters
    not accepted by the model's ``__init__`` method.
    """
