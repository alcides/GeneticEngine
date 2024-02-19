from __future__ import annotations


def abstract(clazz):
    get_gengy(clazz)["abstract"] = True
    return clazz


__builtin_module_name__ = dict().__class__.__module__


def is_builtin(t):
    return t.__module__ == __builtin_module_name__


def get_gengy(t: type) -> dict:
    dic = t.__dict__
    if "__gengy__" not in dic and not is_builtin(t):
        setattr(t, "__gengy__", {})
    return dic.get("__gengy__", {})


def weight(w):
    def weight_w(clazz):
        get_gengy(clazz)["weight"] = w
        return clazz

    return weight_w
