def build_graph(ts):
    return ts


def build_graph2(op, tensor, y):
    if tensor.ops:
        return
    if not tensor.ops:
        pass


def get_ops_result(op, tensor, y):
    print(op.__name__, tensor, y)
    if not tensor.ops:
        return op(tensor.values.values(), y)
    else:
        return op(get_ops_result(*tensor.ops), y)

# get_ops_result(*t.ops)
