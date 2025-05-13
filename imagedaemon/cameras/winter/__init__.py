# imagedaemon/cameras/winter/__init__.py
from .meta import WinterMeta
from .pipelines import WinterPipelines

# ① Create the meta *instance* the adapter will use
meta = WinterMeta()

# ② Expose the Pipelines class under the expected name
Pipelines = WinterPipelines

# ③ Document the public symbols
__all__ = ["meta", "Pipelines"]
