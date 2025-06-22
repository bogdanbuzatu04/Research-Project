import re
import pycld2 as cld2
import fasttext
from functools import lru_cache

_CAMEL_RE  = re.compile(r"^[a-z][a-z0-9]*(?:[A-Z][a-z0-9]+)*$")
_PASCAL_RE = re.compile(r"^[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)*$")
_SNAKE_RE  = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_KEBAB_RE  = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_CAP_SNAKE_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$")
_SPLIT_CAMEL = re.compile(r"[A-Z]?[a-z0-9]+")      # safe: no acronyms present

ft = fasttext.load_model('lid.176.bin')

@lru_cache(maxsize=1024)
def split_variable_name(s: str) -> list[str]:

    if _CAMEL_RE.fullmatch(s):  # camelCase
        parts = _SPLIT_CAMEL.findall(s)

    elif _PASCAL_RE.fullmatch(s):  # PascalCase
        parts = _SPLIT_CAMEL.findall(s)

    elif _SNAKE_RE.fullmatch(s):  # snake_case
        parts = s.split('_')

    elif _KEBAB_RE.fullmatch(s):  # kebab-case
        parts = s.split('-')

    elif _CAP_SNAKE_RE.fullmatch(s):  # SCREAMING_SNAKE_CASE
        parts = s.split('_')

    else:  # not a recognised style → reject
        return []

    return [p.lower() for p in parts if p]

def determine_language_in_variable_name(starting_position_in_code: int, variable_name: str) -> list:

    words = split_variable_name(variable_name)
    if not words:
        return [
            (starting_position_in_code + index, "-1")
            for index in range(len(variable_name))
        ]

    text = ' '.join(words)

    if not text:
        return [(starting_position_in_code + i, "-1") for i in range(len(variable_name))]



    is_reliable, _, details = cld2.detect(
            text,
            isPlainText=True,
            bestEffort=True,
            debugScoreAsQuads=True
        )

    labels, probs = ft.predict(text, k=3)
    labels = [label.replace("__label__", "") for label in labels]

    if len(text) < 5:
        return [(starting_position_in_code + i, "-1") for i in range(len(variable_name))]

    if is_reliable and details:


        if details[0][1] != 'un':
            if labels[0] == details[0][1] and probs[0] > 0.4:
                return [
                    (starting_position_in_code + index, labels[0])
                    for index in range(len(variable_name))
                ]
            elif probs[0] > 0.7:
                return [
                    (starting_position_in_code + index, labels[0])
                    for index in range(len(variable_name))
                ]
            elif labels[0] == 'en' and probs[0] > 0.3:
                return [
                    (starting_position_in_code + index, labels[0])
                    for index in range(len(variable_name))
                ]

    else:
        if probs[0] > 0.7:
            return [
                (starting_position_in_code + index, labels[0])
                for index in range(len(variable_name))
            ]

    return [
        (starting_position_in_code + index, "-1")
        for index in range(len(variable_name))
    ]

__all__ = [
    'split_variable_name',
    'determine_language_in_variable_name',
]

if __name__ == '__main__':
    import sys
    for var in sys.argv[1:]:
        lang = determine_language_in_variable_name(var)
        print(f"{var!r} -> {lang}")