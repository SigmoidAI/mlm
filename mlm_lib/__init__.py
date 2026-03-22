"""
mlm_lib
=======

Minimal import — this is all you need for most use cases:

    from mlm_lib import CascadeRunner

    result = CascadeRunner("simple").run("What is the capital of France?")
    print(result.answer)

For full config control:

    from mlm_lib.settings import SimpleConfig, ComplexConfig, JudgeConfig, MLflowConfig, CascadeRunnerConfig
"""

from .runner import CascadeRunner
from .results import RunResult, RunType

__all__ = ["CascadeRunner", "RunResult", "RunType"]