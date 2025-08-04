# imagedaemon/cameras/pirt/__init__.py
from .meta import PirtMeta
from .pipelines import QcmosPipelines

# ① Create the meta *instance* the adapter will use
meta = PirtMeta()

# ② Expose the Pipelines class under the expected name
Pipelines = PirtPipelines

# ③ Document the public symbols
__all__ = ["meta", "Pipelines"]
