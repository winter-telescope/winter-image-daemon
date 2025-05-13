# imagedaemon/cameras/winter/__init__.py
from .meta import QcmosMeta
from .pipelines import QcmosPipelines

# ① Create the meta *instance* the adapter will use
meta = QcmosMeta()

# ② Expose the Pipelines class under the expected name
Pipelines = QcmosPipelines

# ③ Document the public symbols
__all__ = ["meta", "Pipelines"]
