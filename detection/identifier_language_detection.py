import re
import pycld2 as cld2
from functools import lru_cache

@lru_cache(maxsize=1024)
def split_variable_name(s: str) -> list[str]:
    """
    Split a variable name into its component words.
    Supports camelCase, snake_case, kebab-case, and SCREAMING_SNAKE_CASE.
    Uses regular expressions to identify the format of the variable name.
    The function returns a list of words in lower-case.
    """

    camel_regular_expression = re.compile(r"^[a-z]+(?:[A-Z](?:[a-z0-9]+|[A-Z]*(?=[A-Z]|$)))*$")
    snake_regular_expression = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
    kebab_regular_expression = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
    cap_snake_regular_expression = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$")

    if not s:
        return []

    # camelCase or PascalCase (allows embedded acronyms)
    if camel_regular_expression.fullmatch(s):
        parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=$|[A-Z][a-z0-9])", s)

    # snake_case
    elif snake_regular_expression.fullmatch(s):
        parts = s.split('_')

    # kebab-case
    elif kebab_regular_expression.fullmatch(s):
        parts = s.split('-')

    # SCREAMING_SNAKE_CASE
    elif cap_snake_regular_expression.fullmatch(s):
        parts = s.split('_')

    # nothing matched – return the string as-is
    else:
        parts = [s]

    # --- normalise --------------------------------------------------------
    # 1. drop accidental empty chunks (e.g. double underscores, trailing dash)
    # 2. force every chunk to lower-case
    return [p.lower() for p in parts if p]

def determine_language_in_variable_name(starting_position_in_code: int, variable_name: str) -> list:
    """
    Determine the language of a variable name based on its components.
    Returns the most likely language code or 'unknown' if no language is detected.
    """
    words = split_variable_name(variable_name)
    if not words:
        return [
            (starting_position_in_code + index, "-1")
            for index in range(len(variable_name))
        ]

    # Join the words to form a single text for language detection
    text = ' '.join(words)

    # Detect the language using pycld2
    is_reliable, _, details = cld2.detect(
            text,
            isPlainText=True,
            bestEffort=True,
            debugScoreAsQuads=True
        )

    if is_reliable and details:
        # Return the most likely language code
        return [
            (starting_position_in_code + index, details[0][1])
            for index in range(len(variable_name))
        ]  # details[0] is (language_code, language_name, percent, ...)

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