from baikal._version import __version__

# Make the most relevant classes importable from root
from baikal._core.model import Model
from baikal.steps import Step, Input, make_step
from baikal._core.config import get_config, set_config
