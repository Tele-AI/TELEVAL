"""
from https://github.com/OpenBMB/UltraEval-Audio/blob/main/src/prompt/base.py
"""
import json
from functools import singledispatch
from typing import Any, Dict, List
from jinja2 import StrictUndefined, Template
from jinja2.exceptions import UndefinedError
from src.config import TemplateStruct

@singledispatch
def _load(t: Any, **kwargs: Any) -> Any:
    return t


@_load.register
def _(t: str, **kwargs: Any) -> str:
    def getvar(name: str, default=None):  # for multiturn
        return kwargs.get(name, default)
    
    template = Template(t, undefined=StrictUndefined)
    try:
        rendered = template.render(**kwargs, getvar=getvar)
        # add for multiturn template
        try:
            return json.loads(rendered)
        except json.JSONDecodeError:
            return rendered
    except UndefinedError as e:
        raise ValueError("{}: template is {}\ndoc is {}".format(e, t, kwargs))


@_load.register
def _(t: list, **kwargs: Any) -> List[Any]:
    return [_load(item, **kwargs) for item in t]


@_load.register
def _(t: dict, **kwargs: Any) -> Dict[Any, Any]:
    return {k: _load(v, **kwargs) for k, v in t.items()}


class DataTemplate:
    def __init__(self, template: TemplateStruct):
        self.template = template

    def load(self, **kwargs):
        return _load(self.template, **kwargs)
