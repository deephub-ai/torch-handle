from typing import (
    Any,
    Dict,
)


class ObjectDict(Dict[str, Any]):

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]

