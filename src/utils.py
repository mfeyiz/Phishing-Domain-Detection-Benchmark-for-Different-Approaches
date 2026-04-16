import math
import random
import string
import numpy as np
from typing import List, Tuple, Dict

KEYBOARD_ADJACENCY: dict = {
    "q": set("wa"),
    "w": set("qeas"),
    "e": set("wsdr"),
    "r": set("edft"),
    "t": set("rfgy"),
    "y": set("tghu"),
    "u": set("yhji"),
    "i": set("ujko"),
    "o": set("iklp"),
    "p": set("ol"),
    "a": set("qwsxz"),
    "s": set("awedxz"),
    "d": set("ercfv"),
    "f": set("rtgvb"),
    "g": set("tyhbn"),
    "h": set("yujnm"),
    "j": set("uikmn"),
    "k": set("iojl"),
    "l": set("op"),
    "z": set("asx"),
    "x": set("zsdc"),
    "c": set("xdfv"),
    "v": set("cfgb"),
    "b": set("vghn"),
    "n": set("bhjm"),
    "m": set("njk"),
    "0": set("9oO"),
    "1": set("2qQ"),
    "2": set("1qwW"),
    "3": set("2weE"),
    "4": set("3erR"),
    "5": set("4rtT"),
    "6": set("5tyY"),
    "7": set("6yuU"),
    "8": set("7uiI"),
    "9": set("8uoO"),
}

HOMOGLYPHS: dict = {
    "0": set("oO"),
    "1": set("lLiI|"),
    "o": set("0O"),
    "O": set("0o"),
    "l": set("1iI|"),
    "i": set("1lL|"),
    "I": set("1l|"),
    "a": set("@4"),
    "@": set("a"),
    "4": set("a"),
    "e": set("3"),
    "3": set("e"),
    "s": set("5$"),
    "5": set("s"),
    "$": set("s"),
    "g": set("9q"),
    "9": set("g"),
    "q": set("9g"),
    "b": set("8"),
    "8": set("b"),
}

UNICODE_HOMOGLYPHS: dict = {
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "х": "x",
    "у": "y",
    "ɑ": "a",
    "ο": "o",
    "і": "i",
    "ј": "j",
    "ѕ": "s",
    "ԁ": "d",
    "ɡ": "g",
}


def calculate_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    entropy = 0.0
    length = len(text)
    for count in freq.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    return entropy


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[len(s1)][len(s2)]


def normalized_levenshtein(s1: str, s2: str) -> float:
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    return levenshtein_distance(s1, s2) / max_len


def jaro_winkler(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    max_dist = max(0, (max(len1, len2) // 2) - 1)
    match = 0
    hash1, hash2 = [0] * len1, [0] * len2
    for i in range(len1):
        for j in range(max(0, i - max_dist), min(len2, i + max_dist + 1)):
            if s1[i] == s2[j] and hash2[j] == 0:
                hash1[i] = 1
                hash2[j] = 1
                match += 1
                break
    if match == 0:
        return 0.0
    t = 0
    point = 0
    for i in range(len1):
        if hash1[i]:
            while hash2[point] == 0:
                point += 1
            if s1[i] != s2[point]:
                t += 1
            point += 1
    t /= 2
    jaro = (match / len1 + match / len2 + (match - t) / match) / 3.0
    prefix = 0
    for i in range(min(len1, len2, 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * 0.1 * (1 - jaro)


def keyboard_proximity(s1: str, s2: str) -> float:
    dist = levenshtein_distance(s1, s2)
    if dist == 0:
        return 1.0
    if dist > 2:
        return 0.0
    if len(s1) == len(s2) and dist == 1:
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                if s2[i] in KEYBOARD_ADJACENCY.get(s1[i], set()):
                    return 0.8
    return 0.2


def normalize_homoglyphs(text: str) -> str:
    return "".join([UNICODE_HOMOGLYPHS.get(c, c) for c in text])


def check_homoglyph(s1: str, s2: str) -> float:
    if s1 == s2:
        return 0.0
    if normalize_homoglyphs(s1) == normalize_homoglyphs(s2):
        return 1.0
    score = 0
    common_len = min(len(s1), len(s2))
    for i in range(common_len):
        if s1[i] != s2[i]:
            if s2[i] in HOMOGLYPHS.get(s1[i], set()) or s1[i] in HOMOGLYPHS.get(
                s2[i], set()
            ):
                score += 1
    return score / max(len(s1), len(s2))


def longest_common_substring_ratio(s1: str, s2: str) -> float:
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return 0.0
    lengths = [[0] * (m + 1) for _ in range(n + 1)]
    longest = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
                longest = max(longest, lengths[i][j])
    return longest / max(n, m)


def qgram_similarity(s1: str, s2: str, q: int = 2) -> float:
    def get_qgrams(s, q):
        return [s[i : i + q] for i in range(len(s) - q + 1)]

    q1, q2 = get_qgrams(s1, q), get_qgrams(s2, q)
    if not q1 or not q2:
        return 0.0
    set1, set2 = set(q1), set(q2)
    return len(set1.intersection(set2)) / len(set1.union(set2))


def jaccard_similarity(s1: str, s2: str) -> float:
    set1, set2 = set(s1), set(s2)
    if not set1 and not set2:
        return 1.0
    return len(set1.intersection(set2)) / len(set1.union(set2))


def has_homoglyph_substitution(s1: str, s2: str) -> bool:
    if s1 == s2:
        return False
    if normalize_homoglyphs(s1) == normalize_homoglyphs(s2):
        return True
    if len(s1) != len(s2):
        return False
    for i in range(len(s1)):
        if s1[i] != s2[i] and s2[i] not in HOMOGLYPHS.get(s1[i], set()):
            return False
    return True


def contains_brand_variant(susp: str, brand: str) -> bool:
    if brand in susp:
        return True
    return normalize_homoglyphs(brand) in normalize_homoglyphs(susp)


def extract_features(orig: str, susp: str) -> Dict[str, float]:
    orig_clean = orig.split(".")[0].lower()
    susp_clean = susp.split(".")[0].lower()
    return {
        "entropy_diff": abs(
            calculate_entropy(orig_clean) - calculate_entropy(susp_clean)
        ),
        "levenshtein_dist": float(levenshtein_distance(orig_clean, susp_clean)),
        "len_ratio": len(susp) / (len(orig) + 1e-6),
        "digit_count": sum(c.isdigit() for c in susp) / (len(susp) + 1e-6),
        "special_char": sum(1 for c in susp if c in "-_@") / (len(susp) + 1e-6),
    }


class PhishGenerator:
    def __init__(self):
        self.brands = [
            "google",
            "facebook",
            "amazon",
            "apple",
            "microsoft",
            "netflix",
            "paypal",
            "twitter",
            "instagram",
        ]
        self.tlds = [".com", ".net", ".org", ".co"]
        self.keywords = ["login", "secure", "verify", "update", "account"]

    def _typosquat(self, brand: str) -> str:
        s = list(brand)
        mode = random.choice(["missing", "extra", "swapped", "adjacent"])
        if mode == "missing" and len(s) > 1:
            s.pop(random.randint(0, len(s) - 1))
        elif mode == "extra":
            s.insert(random.randint(0, len(s)), random.choice(string.ascii_lowercase))
        elif mode == "swapped" and len(s) > 1:
            idx = random.randint(0, len(s) - 2)
            s[idx], s[idx + 1] = s[idx + 1], s[idx]
        elif mode == "adjacent":
            idx = random.randint(0, len(s) - 1)
            char = s[idx]
            if char in KEYBOARD_ADJACENCY:
                s[idx] = random.choice(list(KEYBOARD_ADJACENCY[char]))
        return "".join(s)

    def generate_dataset(self, n_samples: int = 1000) -> List[Dict]:
        data = []
        for _ in range(n_samples // 2):
            brand = random.choice(self.brands)
            orig = brand + ".com"
            # Phishing
            method = random.choice(["typo", "homo", "combo", "sub"])
            if method == "typo":
                susp = self._typosquat(brand) + ".com"
            elif method == "homo":
                susp = brand.replace("o", "0").replace("l", "1") + ".com"
            elif method == "combo":
                susp = f"{brand}-{random.choice(self.keywords)}.com"
            else:
                susp = f"account.{brand}.com"
            if susp == orig:
                susp = brand + "-update.com"
            data.append({"orig": orig, "susp": susp, "label": 1})
            # Benign
            data.append({"orig": orig, "susp": orig, "label": 0})
        return data


def generate_dataset(n_samples: int = 3000) -> List[Dict]:
    return PhishGenerator().generate_dataset(n_samples)
