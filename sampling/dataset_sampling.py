from functools import partial
from datasets import Dataset

def _row_pred(example, *, non_english: bool) -> bool:
    """
    Return True iff the *file* (row) is single-language
    and, depending on the flag, either English or non-English.

    We ignore "" and "-1" tags and reject rows that contain any
    multi-language token like "de/en".
    """
    codes = set()

    # merge the three language-list columns into one set
    for col in ("lang_identifiers", "lang_comments", "lang_strings"):
        for code in example[col]:
            if code in ("", "-1"):          # drop unknown tags
                continue
            if "/" in code:                # mixed-language token -> reject row
                return False
            codes.add(code)

    if len(codes) != 1:                    # not exactly one language
        return False

    (lang,) = codes                        # unpack the single item
    return (lang != "en") if non_english else (lang == "en")


def keep_single_language_dataset(
    ds: Dataset,
    non_english: bool = True,
) -> Dataset:
    """
    Keep only rows whose entire file is ONE language.
    If `non_english=True`  keep rows that are *not* English.
    If `non_english=False` keep rows that are     English.
    """
    predicate = partial(_row_pred, non_english=non_english)

    return ds.filter(
        predicate,            # row-level predicate
        batched=False,        # ← run once per example
        num_proc=16,
    )

def _row_pred_language_present(example, *, language: str, in_comments: bool = True, in_identifiers: bool = True, in_strings: bool = True) -> bool:
    """
    Return True iff the *file* (row) contains the given `language`
    """
    comment_codes = set()
    identifier_codes = set()
    string_codes = set()

    if not (in_comments or in_identifiers or in_strings):
        return False

    for code in example["lang_comments"]:
        if code in ("", "-1"):          # drop unknown tags
            continue
        for lang in code.split("/"):
            comment_codes.add(lang)

    for code in example["lang_identifiers"]:
        if code in ("", "-1"):          # drop unknown tags
            continue
        for lang in code.split("/"):
            identifier_codes.add(lang)

    for code in example["lang_strings"]:
        if code in ("", "-1"):          # drop unknown tags
            continue
        for lang in code.split("/"):
            string_codes.add(lang)

    if (in_comments and language in comment_codes) or (in_identifiers and language in identifier_codes) or (in_strings and language in string_codes):
        return True
    return False


def is_language_present(
    ds: Dataset,
    language: str,
    in_comments: bool = False,
    in_identifiers: bool = False,
    in_strings: bool = False
) -> Dataset:
    """ Keep only rows that contain the given `language`"""
    predicate = partial(_row_pred_language_present, language=language, in_comments=in_comments, in_identifiers=in_identifiers, in_strings=in_strings)

    return ds.filter(
        predicate,            # row-level predicate
        batched=False,        # ← run once per example
        num_proc=16,
    )


def keep_files_with_comments(ds):
    # Arrow lets us pull a whole column cheaply
    has_comment_idx = [
        i for i, comments in enumerate(ds["lang_comments"])
        if comments          # non-empty list → keep row
    ]
    return ds.select(has_comment_idx)


__all__ = ["keep_single_language_dataset", "keep_files_with_comments", "is_language_present"]