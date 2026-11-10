"""Metadata filters for search: category (any of) and publication year range.

What it does and why (plain language)
-------------------------------------
"Only papers in cs.IR or cs.LG from 2020 on". The filter is applied INSIDE the
search (candidates that fail are never scored or returned), not after taking the
top k, so a filter never leaves you with fewer results than exist. Each segment
stores the year and categories of its papers (attrs.json) and a zone map (min /
max year, set of categories) so a whole segment that cannot match is skipped.

Year = year of `created` (first version), or of `datestamp` when created is empty.
"""
from __future__ import annotations

from dataclasses import dataclass


def paper_year(created: str, datestamp: str) -> int:
    return int((created or datestamp)[:4])


@dataclass(frozen=True)
class Filter:
    categories: tuple = ()          # match if the paper has ANY of these (empty = no category filter)
    year_from: int = 0              # 0 = no lower bound
    year_to: int = 0                # 0 = no upper bound

    @property
    def empty(self) -> bool:
        return not self.categories and not self.year_from and not self.year_to

    def matches(self, year: int, cats) -> bool:
        if self.year_from and year < self.year_from:
            return False
        if self.year_to and year > self.year_to:
            return False
        if self.categories and not any(c in self.categories for c in cats):
            return False
        return True

    def may_match_zone(self, zone: dict) -> bool:
        """Cheap whole-segment test from the zone map; True means 'cannot rule it out'."""
        if self.year_from and zone["year_max"] < self.year_from:
            return False
        if self.year_to and zone["year_min"] > self.year_to:
            return False
        if self.categories and not set(self.categories) & set(zone["categories"]):
            return False
        return True
