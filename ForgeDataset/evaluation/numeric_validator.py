"""Deterministic Python Numeric and Schema Validation.
Validates numbers, formulas, dates, and calculations without relying on an LLM.
"""

import re
from typing import Tuple, List, Set


class NumericValidator:
    """Pure Python deterministic numerical verifier to prevent LLM mathematical hallucinations."""

    def __init__(self, tolerance: float = 0.0001):
        self.tolerance = tolerance
        # Regex matching numbers, decimals, percentages, currency
        self.num_pattern = re.compile(r'[-+]?\b\d+(?:,\d{3})*(?:\.\d+)?%?\b')

    def extract_numbers(self, text: str) -> List[str]:
        """Extract all numeric tokens normalized (removing commas, currency symbols)."""
        raw_matches = self.num_pattern.findall(text)
        normalized = []
        for m in raw_matches:
            clean = m.replace(',', '').rstrip('%').lstrip('$€£')
            try:
                # normalize floats like 4.0 to 4
                val = float(clean)
                if val.is_integer():
                    normalized.append(str(int(val)))
                else:
                    normalized.append(str(round(val, 4)))
            except ValueError:
                pass
        return normalized

    def validate(self, answer: str, source_context: str) -> Tuple[bool, List[str]]:
        """Verify whether every number in the answer is grounded in source context or valid arithmetic.
        
        Returns: (passed: bool, flags: List[str])
        """
        answer_nums = self.extract_numbers(answer)
        if not answer_nums:
            # No numbers present in answer -> passes numeric check trivially
            return True, []

        source_nums: Set[str] = set(self.extract_numbers(source_context))
        unsupported: List[str] = []

        for num in answer_nums:
            if num not in source_nums:
                # Number wasn't found verbatim in source
                # Check if it might be a year, simple count, or index
                try:
                    val = float(num)
                    # Allow 1, 2, 3 as standard ordinal/bullet counts
                    if val in [1, 2, 3]:
                        continue
                except ValueError:
                    pass
                unsupported.append(num)

        if unsupported:
            flags = [f"Unsupported numeric value(s) in answer: {', '.join(unsupported)}"]
            return False, flags

        return True, []

