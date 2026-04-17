from src.utils import (
    normalized_levenshtein,
    jaro_winkler,
    keyboard_proximity,
    check_homoglyph,
    longest_common_substring_ratio,
    qgram_similarity,
    jaccard_similarity,
    has_homoglyph_substitution,
    contains_brand_variant,
    levenshtein_distance,
    normalize_homoglyphs,
)


def predict(orig: str, susp: str) -> tuple:
    """
    Algorithmic phishing detection using heuristic rules.

    Returns:
        tuple: (label, score) where label is "Phishing", "Temiz", or "Şüpheli"
    """
    orig = orig.lower().strip()
    susp = susp.lower().strip()

    orig_clean = orig.split(".")[0]
    susp_clean = susp.split(".")[0]

    if orig == susp:
        return ("Temiz", 0.01)

    if orig_clean == susp_clean:
        susp_after = susp[len(susp_clean) :]
        if "-" in susp_after or len(susp_after) > 7:
            return ("Phishing", 0.90)
        return ("Temiz", 0.05)

    if susp_clean.startswith("xn--"):
        return ("Phishing", 0.90)

    if contains_brand_variant(susp_clean, orig_clean) and orig_clean != susp_clean:
        return ("Phishing", 0.93)

    if len(orig_clean) <= 4 and len(susp_clean) <= 4:
        if has_homoglyph_substitution(orig_clean, susp_clean):
            return ("Phishing", 0.90)
        if len(orig_clean) <= 3 and len(susp_clean) <= 3:
            return ("Temiz", 0.10)

    if has_homoglyph_substitution(orig_clean, susp_clean):
        return ("Phishing", 0.92)

    char_jaccard = jaccard_similarity(orig_clean, susp_clean)
    if char_jaccard < 0.2:
        return ("Temiz", 0.05)

    norm_orig = normalize_homoglyphs(orig_clean)
    norm_susp = normalize_homoglyphs(susp_clean)
    if orig_clean != susp_clean and levenshtein_distance(norm_orig, norm_susp) <= 1:
        orig_tld = orig.split(".", 1)[1] if "." in orig else ""
        susp_tld = susp.split(".", 1)[1] if "." in susp else ""
        if orig_tld == susp_tld:
            return ("Phishing", 0.88)

    if set(orig_clean) == set(susp_clean) and orig_clean != susp_clean:
        ratio = min(len(orig_clean), len(susp_clean)) / max(
            len(orig_clean), len(susp_clean)
        )
        if ratio > 0.7:
            return ("Phishing", 0.85)

    norm_lev = normalized_levenshtein(orig_clean, susp_clean)
    jaro_wink = jaro_winkler(orig_clean, susp_clean)
    keyboard = keyboard_proximity(orig_clean, susp_clean)
    homoglyph = check_homoglyph(orig, susp)
    lcs = longest_common_substring_ratio(orig_clean, susp_clean)
    qgram = qgram_similarity(orig_clean, susp_clean)

    weights = {
        "norm_lev": 0.20,
        "jaro_wink": 0.20,
        "keyboard": 0.10,
        "homoglyph": 0.25,
        "lcs": 0.15,
        "qgram": 0.10,
    }

    phishing_score = (
        (1 - norm_lev) * weights["norm_lev"]
        + jaro_wink * weights["jaro_wink"]
        + keyboard * weights["keyboard"]
        + homoglyph * weights["homoglyph"]
        + lcs * weights["lcs"]
        + qgram * weights["qgram"]
    )

    orig_tld = orig.split(".", 1)[1] if "." in orig else ""
    susp_tld = susp.split(".", 1)[1] if "." in susp else ""
    if orig_tld != susp_tld:
        phishing_score *= 0.5

    if phishing_score > 0.60:
        return ("Phishing", phishing_score)
    elif phishing_score > 0.40:
        return ("Şüpheli", phishing_score)
    else:
        return ("Temiz", phishing_score)
