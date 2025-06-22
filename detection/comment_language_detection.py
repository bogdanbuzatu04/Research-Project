import fasttext
import pycld2 as cld2
import re
import string
from collections import Counter, defaultdict

TAG_PATTERN = re.compile(r'''
    /\*\*            |   # /**  
    \*/              |   # */

    //               |   # C++/Java single-line
    \#               |   # Python single-line

    ^\s*\*           |   # “ * this line”

    \b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b  |  # PascalCase (≥2 segments)
    \b[a-z]+(?:[A-Z][a-z0-9]+)+\b          |  # camelCase
    \b[a-z]+(?:_[a-z0-9]+)+\b              |  # snake_case
    \b[a-z]+(?:-[a-z0-9]+)+\b              |  # kebab-case

    (?xi:
        @(?:param|return|throws)    # @param, @return, @throws
        \s+                         #   plus one or more spaces
        [A-Za-z_]\w*                #   the parameter or exception name
    )                              

    | (?xi:
        :(?:param|return|raises):  # :param:, :return:, :raises:
        \s*
        [A-Za-z_]\w*                #   the name after the colon-tag
    )

    (?i:<\/?\w+[^>]*>)  |  # <p>, <code>, …
    [`\[\]\(\)]        |  # backticks, brackets
    [\*\#\-\=]{2,}       # **bold**, ## headers, == …

    (?i:\b(?:Args|Returns?|Raises?|Throws?|Type|Example|Usage|See\s+Also)\b)

''', re.VERBOSE | re.MULTILINE)

PUNCTUATION_TABLE = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

ft = fasttext.load_model('lid.176.bin')

def smooth(final_langs, index_map, half_window=3):
    smoothed = []
    array_size = len(final_langs)

    for i, (pos, lang) in enumerate(final_langs):
        # always skip masked chars
        if index_map[i] is None:
            smoothed.append((pos, "-1"))
            continue
        else:
            start = max(0, min(i - half_window, array_size - half_window * 2))
            end = min(array_size, start + half_window * 2)

            window = final_langs[start:end]

            window_langs = [l for _, l in window]

            counts = Counter(window_langs)
            top_lang, top_count = counts.most_common(1)[0]

            if top_count >= 2:
                if top_lang == "-1":
                    smoothed.append((pos, lang))
                else:
                    smoothed.append((pos, top_lang))
            else:
                smoothed.append((pos, "-1"))
    return smoothed

def clean_and_map(input_text: str) -> tuple:
    """
    Fast-clean input text by masking tags/symbols and tracking original indices.
    Returns:
        cleaned_text: string ready for FastText
        index_map: list mapping cleaned indices to original indices
    """
    cleaned = []
    index_map = []

    i = 0
    length = len(input_text)
    while i < length:
        # Try matching tag pattern
        match = TAG_PATTERN.match(input_text, i)
        if match:
            span_len = match.end() - match.start()
            cleaned.extend([' '] * span_len)
            index_map.extend([None] * span_len)
            i += span_len
            continue

        # Replace punctuation with space (except underscore and alphanum)
        c = input_text[i]
        if c.isalpha():
            # letter → keep it and map back
            cleaned.append(c)
            index_map.append(i)
        else:
            # anything else → space and no mapping
            cleaned.append(' ')
            index_map.append(None)
        i += 1

    return ''.join(cleaned), index_map

def fallback_lang(input_text: str, index:int, window_size:int):
    start = max(0, min(index - window_size // 2, len(input_text) - window_size))
    end = min(len(input_text), start + window_size)
    snippet = input_text[start:end]
    try:
        reliable, _, details = cld2.detect(snippet,
            isPlainText=True,
            bestEffort=True,
            debugScoreAsQuads=True)

        if reliable:
            return details[0][1].lower()
    except Exception:
        pass
    return "-1"

def pick_window_params(char_count: int) -> tuple:
    """Return (window, step) based on length."""
    if char_count <= 60:
        return char_count, char_count          # one shot
    elif char_count <= 180:
        return 45, 20
    elif char_count <= 360:
        return 60, 30
    else:
        return 90, 45

def sliding_windows_comments(input_text: str, top_k=3):
    """
    Slide a window over `text`, yielding (start, end, info) tuples where
    `info` is { 'labels': [...], 'probs': [...] } of the top_k FastText predictions.
    """

    length = len(input_text)

    window_size, step_size = pick_window_params(length)

    for start in range(0, length, step_size):
        end = min(start + window_size, length)
        snippet = input_text[start:end]

        # ask for the top_k predictions!
        labels, probs = ft.predict(snippet, k=top_k)

        labels = [label.replace("__label__", "") for label in labels]

        yield start, end, {
            # labels should be like ('en', 'fr', 'ro', ...)
            'labels': labels,  # strip "__label__"
            'probs':  probs    # e.g. (0.98, 0.01, ...)
        }

        if end == length:
            break


def detect_language_in_comment(starting_position_in_code: int, input_text: str) -> list:

    if len(input_text) < 5:
        return [(starting_position_in_code + i, "-1") for i in range(len(input_text))]

    cleaned_text, index_map = clean_and_map(input_text)
    cleaned_text = cleaned_text.replace('\n', ' ').replace('\r', ' ')

    labels, probs = ft.predict(cleaned_text, k=1)
    full_lang, full_conf = labels[0].replace('__label__', ''), probs[0]

    if full_conf >= 0.95:
        return [
            (starting_position_in_code + i, full_lang if index_map[i] is not None else "-1")
            for i in range(len(index_map))
        ]

    input_length = len(cleaned_text) # for convenience

    langs_per_char = [defaultdict(float) for _ in range(input_length)]
    total_weight = [0.0 for _ in range(input_length)]

    window_length, step_size = pick_window_params(input_length)

    resulting_windows = []

    for window_start, window_end, window_info in sliding_windows_comments(
        cleaned_text,
        top_k=3  # get top 3 language predictions
    ):
        resulting_windows.append((window_start, window_end, window_info['labels'], window_info['probs']))

    for (window_start, window_end, labels, probs) in resulting_windows:
        center_of_window = (window_start + window_end) / 2.0

        for index in range(window_start, window_end):
            distance_center_char = abs(center_of_window  - index)
            weight = 1 - (distance_center_char / window_length)

            # Now iterate through each (label, probability) pair for this window:
            for label, probability in zip(labels, probs):
                if window_start == 0 and index < step_size:
                    # This character is present only in the first window
                    weight = 1
                    total_weight[index] =  1
                elif window_end == len(cleaned_text) and index >= len(cleaned_text) - step_size:
                    # This character is present only in the last window
                    weight = 1
                    total_weight[index] = 1
                else:
                    total_weight[index] += weight * probability

                langs_per_char[index][label] += weight * probability

    for index in range(input_length):
        if total_weight[index] > 0:
            for lang in langs_per_char[index]:
                langs_per_char[index][lang] /= total_weight[index]


    # Now we have a list of dictionaries, each with language probabilities per character
    final_langs = []
    for i, lang_scores in enumerate(langs_per_char):
        if index_map[i] is None:
            # This character was masked out, skip it
            final_langs.append((starting_position_in_code + i, "-1"))
            continue
        if not lang_scores:
            final_langs.append((starting_position_in_code + i, "-1"))
            continue
        lang, score = max(lang_scores.items(), key=lambda x: x[1])
        if score < 0.70:
            lang = fallback_lang(cleaned_text, i, window_length)
        final_langs.append((starting_position_in_code + i,lang))

    # Smooth the final languages
    final_langs = smooth(final_langs, index_map, half_window=3)

    return final_langs


__all__ = [
    'detect_language_in_comment',
]