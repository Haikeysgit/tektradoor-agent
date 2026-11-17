"""tektradoor-agent: trades practice money with the judge's locked rules.

Importing this package first puts the vendored judge (vendor/tektradoor-judge,
a git submodule) on sys.path, so ``import judge`` loads exactly that code.
"""
from agent import judge_link  # noqa: F401  (sets up the judge import)
