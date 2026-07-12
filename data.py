#Deterministic analogy data with template-disjoint train/evaluation splits
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class AnalogyTemplate:
    a: str
    b: str
    c: str
    targets: tuple[str, ...]

    @property
    def prompt(self) -> str:
        return f"{self.a} is to {self.b} as {self.c} is to"


TEMPLATES = (
    AnalogyTemplate("sun", "bright", "moon", ("dim", "pale", "glowing", "luminous", "reflective")),
    AnalogyTemplate("king", "man", "queen", ("woman", "lady", "female")),
    AnalogyTemplate("cat", "kitten", "dog", ("puppy",)),
    AnalogyTemplate("teacher", "school", "doctor", ("hospital", "clinic")),
    AnalogyTemplate("rain", "wet", "snow", ("cold", "white", "icy")),
    AnalogyTemplate("fire", "hot", "ice", ("cold", "freezing", "chilly")),
    AnalogyTemplate("bird", "fly", "fish", ("swim", "swimming")),
    AnalogyTemplate("word", "sentence", "note", ("melody", "tune")),
    AnalogyTemplate("ear", "hear", "eye", ("see", "look", "watch")),
    AnalogyTemplate("lion", "courage", "fox", ("cunning", "clever", "sly")),
    AnalogyTemplate("knife", "cut", "pen", ("write", "scribble")),
    AnalogyTemplate("car", "road", "boat", ("water", "sea", "river")),
    AnalogyTemplate("winter", "cold", "summer", ("hot", "warm")),
    AnalogyTemplate("seed", "plant", "egg", ("bird", "chick")),
    AnalogyTemplate("up", "down", "left", ("right",)),
    AnalogyTemplate("strong", "strength", "wise", ("wisdom", "insight")),
    AnalogyTemplate("mother", "parent", "son", ("child", "kid")),
    AnalogyTemplate("glass", "transparent", "brick", ("opaque", "solid")),
    AnalogyTemplate("bee", "honey", "cow", ("milk",)),
    AnalogyTemplate("author", "book", "composer", ("music", "symphony", "song")),
)


def split_templates(seed: int, held_out_templates: int = 4) -> tuple[list[AnalogyTemplate], list[AnalogyTemplate]]:
    # Return a template-disjoint split; no relation appears in both partitions.
    if not 0 < held_out_templates < len(TEMPLATES):
        raise ValueError("held_out_templates must be between 1 and len(TEMPLATES)-1")
    order = list(TEMPLATES)
    random.Random(seed).shuffle(order)
    return order[held_out_templates:], order[:held_out_templates]


def repeat_templates(templates: Iterable[AnalogyTemplate], repeats: int, seed: int) -> list[AnalogyTemplate]:
    examples = list(templates) * repeats
    random.Random(seed).shuffle(examples)
    return examples
