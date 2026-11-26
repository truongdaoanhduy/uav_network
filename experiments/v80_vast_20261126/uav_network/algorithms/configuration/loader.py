"""Single strict YAML+--set interface for every MATD3/MASAC experiment.

Unknown fields, implicit boolean coercion and numeric/str type confusion are
errors. The user can override *declared* configuration, not code constants.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from .schema import LearnerConfig


def _type_checked(value: Any, template: Any, path: str):
    if type(template) is bool:
        if type(value) is not bool:
            raise ValueError(f'{path} requires a boolean')
    elif type(template) is int:
        if type(value) is not int:
            raise ValueError(f'{path} requires an integer')
    elif type(template) is float:
        if type(value) not in (float, int):
            raise ValueError(f'{path} requires a number')
        value = float(value)
    elif type(template) is str:
        if type(value) is not str:
            raise ValueError(f'{path} requires a string')
    elif type(template) is list:
        if not isinstance(value, list) or any(type(x) is not int or x <= 0 for x in value):
            raise ValueError(f'{path} requires a list of positive integer sizes')
    else:
        raise ValueError(f'{path}: unsupported setting type')
    return value


def load_learner_config(
    path: str | Path | None = None,
    *,
    overrides: Iterable[str] = (),
) -> LearnerConfig:
    """Load partial learner YAML and typed overrides atop an audited default."""
    base = LearnerConfig().to_dict()
    payload = {}
    if path is not None:
        payload = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            raise ValueError('Learner config must be a mapping')
    if set(payload) - set(base):
        raise ValueError(f'Unexpected learner sections: {set(payload)-set(base)}')
    for section, section_values in payload.items():
        if isinstance(base[section], dict):
            if not isinstance(section_values, dict):
                raise ValueError(f'{section} must be a mapping')
            for key, val in section_values.items():
                if key not in base[section]:
                    raise ValueError(f'Unknown learner field {section}.{key}')
                base[section][key] = _type_checked(val, base[section][key], f'{section}.{key}')
        else:
            base[section] = _type_checked(section_values, base[section], section)
    for item in overrides:
        key, sep, raw = item.partition('=')
        if not sep or not raw or key.count('.') != 1:
            raise ValueError(f'Expected --set SECTION.FIELD=VALUE, got {item!r}')
        section, name = key.split('.')
        if section not in base or not isinstance(base[section], dict) or name not in base[section]:
            raise ValueError(f'Unknown learner setting {key}')
        expected = base[section][name]
        if type(expected) is str and not (raw.startswith('"') or raw.startswith("'")):
            value = raw
        else:
            value = yaml.safe_load(raw)
        base[section][name] = _type_checked(value, expected, key)
    return LearnerConfig.from_dict(base)
