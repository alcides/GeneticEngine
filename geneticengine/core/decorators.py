

def abstract(clazz):
    get_gengy(clazz)["abstract"] = True
    return clazz


__builtin_module_name__ = dict().__class__.__module__


def is_builtin(t):
    return t.__module__ == __builtin_module_name__


def get_gengy(t: type) -> dict:
    dic = t.__dict__
    if "__gengy__" not in dic and not is_builtin(t):
        t.__gengy__ = {}
    return dic.get("__gengy__", {})
