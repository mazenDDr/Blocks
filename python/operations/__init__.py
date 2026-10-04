"""Built-in operations. Importing this package registers them."""
from . import core, layers  # noqa: F401
from . import backend_ops  # noqa: F401  (backend-specific nodes, Milestone 6)
from . import tabular_ops, sklearn_ops, stats_ops  # noqa: F401  (graph kind "tabular")
from . import unsup_ops  # noqa: F401  (clustering / representation, Milestone 5)
from . import connector_ops  # noqa: F401  (connector sources + cross-source join)
from . import tensor_ops  # noqa: F401  (tensor primitives, Milestone 3)
from . import diag_ops  # noqa: F401  (diagnostic blocks + structural block catalog)
from . import code_ops  # noqa: F401  (code blocks, Milestone 3)
from agent import blocks as agent_blocks  # noqa: F401  (graph kind "agent", Milestone 4; heavy libraries are imported lazily at run time)
from rl import ops as rl_ops  # noqa: F401  (graph kind "rl", Milestone 5)
