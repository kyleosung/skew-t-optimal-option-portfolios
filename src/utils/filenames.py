def dot_to_p(dot: str) -> str:
    """
    Convert a dot-separated string to a path with forward slashes.
    """
    return dot.replace(".", "p")
