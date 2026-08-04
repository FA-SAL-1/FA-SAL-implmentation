

from __future__ import annotations

import os
from typing import Dict


BLAS_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_blas_threads(default_threads: int = 1) -> Dict[str, str]:


    requested = os.environ.get("FA_SAL_BLAS_THREADS", str(int(default_threads)))
    configured: Dict[str, str] = {}
    for name in BLAS_THREAD_ENV_VARS:
        os.environ.setdefault(name, requested)
        configured[name] = os.environ[name]
    return configured
