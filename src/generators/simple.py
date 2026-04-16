import random
import string
from typing import List, Dict

from src.utils import (
    KEYBOARD_ADJACENCY,
    HOMOGLYPHS,
    calculate_entropy,
    levenshtein_distance,
    extract_features,
)


class SimpleGenerator:
    """Simple data generator for phishing detection - easy examples."""

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
            data.append({"orig": orig, "susp": orig, "label": 0})

        return data


def generate_dataset(n_samples: int = 3000) -> List[Dict]:
    return SimpleGenerator().generate_dataset(n_samples)
