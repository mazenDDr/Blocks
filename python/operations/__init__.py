"""Built-in operations. Importing this package registers them."""
from . import core, layers  # noqa: F401
from . import tabular_ops, sklearn_ops, stats_ops  # noqa: F401  (graph kind "tabular")
from . import connector_ops  # noqa: F401  (connector sources + cross-source join)
from . import tensor_ops  # noqa: F401  (tensor primitives, Milestone 3)
from . import diag_ops  # noqa: F401  (diagnostic blocks + structural block catalog)
from . import code_ops  # noqa: F401  (code blocks, Milestone 3)
