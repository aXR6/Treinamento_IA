def detect_doc_profile(path: str, metadata: dict) -> str:
    pages = metadata.get("numpages", 0)
    is_scanned = metadata.get("is_scanned", False)
    if is_scanned or path.lower().endswith(".tiff"):
        return "scanned"
    if pages > 100:
        return "long_text"
    return "default"