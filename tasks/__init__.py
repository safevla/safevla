from importlib import import_module
from inspect import isclass
from pathlib import Path
from pkgutil import iter_modules
from typing import Dict, Type

from tasks.abstract_task import AbstractSPOCTask
from tasks.abstract_task_sampler import AbstractSPOCTaskSampler
from utils.type_utils import REGISTERED_TASK_PARAMS

REGISTERED_TASKS: Dict[str, Type[AbstractSPOCTask]] = {}


def register_task(cls):
    if cls.task_type_str not in REGISTERED_TASK_PARAMS:
        return cls

    REGISTERED_TASKS[cls.task_type_str] = cls
    return cls


# iterate through the modules in the current package
package_dir = str(Path(__file__).resolve().parent)
for _, module_name, _ in iter_modules([package_dir]):
    # import the module and iterate through its attributes
    module = import_module(f"{__name__}.{module_name}")
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if isclass(attribute):
            # Add the class to this package's variables
            if issubclass(attribute, AbstractSPOCTask) and attribute != AbstractSPOCTask:
                globals()[attribute_name] = attribute
                register_task(attribute)
