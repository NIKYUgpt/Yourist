import hashlib

from app.modules.deid.identity import PersonResolution


def resolution(text, *people):
    bindings, offset = [], 0
    for value, participant in people:
        start = text.index(value, offset)
        bindings.append({"start": start, "end": start + len(value), "participant": participant})
        offset = start + len(value)
    return PersonResolution(
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), bindings=bindings
    )
