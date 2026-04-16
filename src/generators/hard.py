import random
import string
from typing import List, Dict, Tuple

from src.utils import (
    KEYBOARD_ADJACENCY,
    HOMOGLYPHS,
    UNICODE_HOMOGLYPHS,
)


class HardGenerator:
    """Hard/realistic phishing examples for testing model robustness."""

    def __init__(self, seed: int = 42):
        random.seed(seed)

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
            "linkedin",
            "dropbox",
            "adobe",
            "github",
            "gitlab",
            "whatsapp",
            "telegram",
            "discord",
            "slack",
            "zoom",
            "spotify",
            "steam",
            "twitch",
            "yahoo",
            "outlook",
            "chase",
            "wellsfargo",
            "bankofamerica",
            "citi",
            "capitalone",
            "amex",
            "ebay",
            "etsy",
            "shopify",
            "walmart",
            "target",
            "bestbuy",
        ]

        self.tlds = [".com", ".net", ".org", ".io", ".co", ".info", ".biz", ".xyz"]

        self.keywords_login = [
            "login",
            "signin",
            "auth",
            "verify",
            "account",
            "secure",
            "update",
            "password",
            "recovery",
            "reset",
            "support",
            "help",
        ]

        self.keywords_financial = [
            "payment",
            "wallet",
            "billing",
            "invoice",
            "transaction",
            "banking",
            "transfer",
            "confirm",
            "validate",
            "kyc",
        ]

        self.dummy_words = [
            "help",
            "service",
            "customer",
            "portal",
            "dashboard",
            "webmail",
            "cloud",
            "drive",
            "docs",
            "files",
            "storage",
            "backup",
            "sync",
            "admin",
            "manager",
            "center",
            "hub",
            "pro",
            "plus",
            "premium",
            "2024",
            "2025",
            "secure",
            "verify",
            "valid",
            "official",
        ]

        self.unrelated_brands = [
            "news",
            "shop",
            "tech",
            "web",
            "blog",
            "news",
            "media",
            "auto",
            "health",
            "food",
            "travel",
            "sports",
            "music",
            "video",
            "game",
            "photo",
            "art",
            "book",
            "learn",
            "work",
            "home",
        ]

    def _multi_step_typo(self, brand: str, steps: int = 2) -> str:
        """Multiple typo operations in sequence."""
        s = list(brand)
        operations = ["missing", "extra", "swap", "adjacent", "double"]

        for _ in range(steps):
            if not s:
                break
            mode = random.choice(operations)

            if mode == "missing" and len(s) > 2:
                idx = random.randint(0, len(s) - 1)
                s.pop(idx)
            elif mode == "extra":
                idx = random.randint(0, len(s))
                s.insert(idx, random.choice(string.ascii_lowercase))
            elif mode == "swap" and len(s) > 2:
                idx = random.randint(0, len(s) - 2)
                s[idx], s[idx + 1] = s[idx + 1], s[idx]
            elif mode == "adjacent" and len(s) > 0:
                idx = random.randint(0, len(s) - 1)
                char = s[idx]
                if char in KEYBOARD_ADJACENCY:
                    s[idx] = random.choice(list(KEYBOARD_ADJACENCY[char]))
            elif mode == "double" and len(s) > 1:
                idx = random.randint(0, len(s) - 1)
                s.insert(idx, s[idx])

        return "".join(s)

    def _double_char(self, brand: str) -> str:
        """Add double character somewhere."""
        s = list(brand)
        idx = random.randint(0, len(s) - 1)
        s.insert(idx, s[idx])
        return "".join(s)

    def _hyphen_abuse(self, brand: str) -> str:
        """Insert hyphen in the middle."""
        if len(brand) < 3:
            return brand
        idx = random.randint(1, len(brand) - 2)
        return brand[:idx] + "-" + brand[idx:]

    def _tld_variation(self, brand: str) -> str:
        """Same brand, different TLD (could be legitimate or phishing)."""
        return brand + random.choice(self.tlds)

    def _stacked_keywords(self, brand: str) -> str:
        """Multiple keywords stacked: paypal-login-verify.com"""
        keywords = random.sample(
            self.keywords_login + self.keywords_financial, random.randint(2, 4)
        )
        return f"{brand}-{'-'.join(keywords)}"

    def _multi_subdomain(self, brand: str) -> str:
        """Deep nesting: login.paypal.secure.net"""
        subdomains = random.sample(self.dummy_words, random.randint(2, 4))
        return f"{'.'.join(subdomains)}.{brand}"

    def _mixed_attack(self, brand: str) -> str:
        """Combine multiple attack types."""
        attacks = [
            lambda b: b.replace("o", "0").replace("l", "1"),
            lambda b: self._hyphen_abuse(b),
            lambda b: b + "-" + random.choice(self.keywords_login),
        ]

        result = brand
        for _ in range(random.randint(1, 2)):
            result = random.choice(attacks)(result)

        return result

    def _unicode_homoglyph(self, brand: str) -> str:
        """Unicode homoglyph substitution."""
        result = []
        for char in brand:
            if char in UNICODE_HOMOGLYPHS:
                if random.random() < 0.6:
                    result.append(random.choice([UNICODE_HOMOGLYPHS[char], char]))
                else:
                    result.append(char)
            else:
                result.append(char)
        return "".join(result)

    def _random_suffix(self, brand: str) -> str:
        """Add random suffix: google-help-2024-verify.com"""
        suffix = random.choice(self.dummy_words)
        year = random.choice(["2024", "2025", "2026", ""])
        return f"{brand}-{suffix}{year}"

    def _random_chars(self, brand: str) -> str:
        """Add random chars: googlexyz123.com"""
        chars = "".join(
            random.choices(
                string.ascii_lowercase + string.digits, k=random.randint(3, 8)
            )
        )
        return f"{brand}{chars}"

    def _brand_lookalike(self, brand: str) -> str:
        """Similar but different brand: googles.com (real company)"""
        suffixes = ["s", "ly", "er", "hub", "app", "io", "co"]
        return f"{brand}{random.choice(suffixes)}"

    def _phishing_attack(self) -> Tuple[str, str]:
        """Generate a sophisticated phishing domain."""
        brand = random.choice(self.brands)
        orig = f"{brand}{random.choice(self.tlds)}"

        attack_type = random.choices(
            [
                "multi_step_typo",
                "double_char",
                "hyphen_abuse",
                "stacked_keywords",
                "multi_subdomain",
                "mixed",
                "unicode",
                "random_suffix",
                "random_chars",
            ],
            weights=[15, 10, 10, 20, 10, 15, 10, 5, 5],
        )[0]

        if attack_type == "multi_step_typo":
            susp = self._multi_step_typo(brand)
        elif attack_type == "double_char":
            susp = self._double_char(brand)
        elif attack_type == "hyphen_abuse":
            susp = self._hyphen_abuse(brand)
        elif attack_type == "stacked_keywords":
            susp = self._stacked_keywords(brand)
        elif attack_type == "multi_subdomain":
            susp = self._multi_subdomain(brand)
        elif attack_type == "mixed":
            susp = self._mixed_attack(brand)
        elif attack_type == "unicode":
            susp = self._unicode_homoglyph(brand)
        elif attack_type == "random_suffix":
            susp = self._random_suffix(brand)
        else:
            susp = self._random_chars(brand)

        tld = random.choice(self.tlds)
        if not susp.endswith(tld):
            susp = f"{susp}{tld}"

        if susp == orig:
            susp = f"{brand}-secure{random.randint(100, 999)}{tld}"

        return orig, susp

    def _benign_attack(self) -> Tuple[str, str]:
        """Generate benign (non-phishing) domain - hard cases."""
        brand = random.choice(self.brands)
        orig = f"{brand}{random.choice(self.tlds)}"

        benign_type = random.choices(
            ["tld_variation", "brand_lookalike", "unrelated", "same_domain"],
            weights=[20, 15, 35, 30],
        )[0]

        if benign_type == "tld_variation":
            susp = self._tld_variation(brand)
        elif benign_type == "brand_lookalike":
            susp = self._brand_lookalike(brand)
        elif benign_type == "unrelated":
            other = random.choice(self.unrelated_brands)
            random_suffix = random.randint(100, 9999)
            tld = random.choice(self.tlds)
            susp = f"{other}{random_suffix}{tld}"
        else:
            susp = orig

        if susp == orig:
            other_brand = random.choice([b for b in self.brands if b != brand])
            tld = random.choice(self.tlds)
            susp = f"{other_brand}{tld}"

        return orig, susp

    def generate_sample(self) -> Dict:
        """Generate a single sample."""
        is_phishing = random.random() < 0.5

        if is_phishing:
            orig, susp = self._phishing_attack()
            label = 1
        else:
            orig, susp = self._benign_attack()
            label = 0

        return {"orig": orig, "susp": susp, "label": label}

    def generate_dataset(self, n_samples: int = 1000) -> List[Dict]:
        """Generate a dataset of hard examples."""
        return [self.generate_sample() for _ in range(n_samples)]

    def generate_train_test_split(
        self, n_train: int = 10000, n_test: int = 1000
    ) -> Tuple[List[Dict], List[Dict]]:
        """Generate train/test split with hard test examples."""
        train_data = self.generate_dataset(n_train)
        test_data = self.generate_dataset(n_test)
        return train_data, test_data
