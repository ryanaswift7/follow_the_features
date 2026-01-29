def config_hash(config_dict):
    import hashlib
    import json as _json

    return hashlib.sha256(
        _json.dumps(config_dict, sort_keys=True).encode()
    ).hexdigest()[:10]


def string_hash(input_string):
    import hashlib

    return hashlib.sha256(input_string.encode("utf-8")).hexdigest()[:10]
