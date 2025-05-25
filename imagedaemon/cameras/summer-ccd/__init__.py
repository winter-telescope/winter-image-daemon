# imagedaemon/cameras/summer-ccd/__init__.py
from .meta import SummerCCDMeta
from .pipelines import SummerCCDPipelines

# ① Create the meta *instance* the adapter will use
meta = SummerCCDMeta()

# ② Expose the Pipelines class under the expected name
Pipelines = SummerCCDPipelines

# ③ Document the public symbols
__all__ = ["meta", "Pipelines"]
