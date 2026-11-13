"""Shared trailer-type metadata, without ROS or PyTorch dependencies."""

DEFAULT_TYPE = 'default'


def normalize_type(value):
    return str(value).strip() if value is not None and str(value).strip() else DEFAULT_TYPE


def row_type(row):
    return normalize_type(row.get('trailer_type'))


def check_type_names(names):
    names = list(names)
    if not names or len(names) != len(set(names)) or any(normalize_type(name) != name for name in names):
        raise ValueError('车型列表不能为空、重复或包含空白名称')
    return names
