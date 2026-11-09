"""Domain exceptions shared across the pipeline."""


class JobInputError(Exception):
    """The inputs can't be used. The job stops with this code; nothing is retried.

    Attributes:
        code: Short machine-readable identifier (reported as ``error_code`` in the job result).
    """

    def __init__(self, code: str, message: str):
        """
        Args:
            code: Machine-readable error code.
            message: Human-readable explanation, used as the exception message.
        """
        super().__init__(message)
        self.code = code


class LLMOutputError(Exception):
    """The model's answer still failed validation after the repair attempt.

    Attributes:
        errors: Every validation problem found in the final (rejected) answer.
    """

    def __init__(self, errors: list[str]):
        """
        Args:
            errors: Validation problems; only the first five are included in the exception message.
        """
        super().__init__("; ".join(errors[:5]))
        self.errors = errors
