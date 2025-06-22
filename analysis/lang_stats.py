from collections import Counter, defaultdict
import pandas as pd
import numpy as np

def top_k(counter, k=10):
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

def save_stats(stats, single_lang=False):
    if not stats:
        return

    comments_stats = stats['comments']
    identifiers_stats = stats['identifiers']
    strings_stats = stats['strings']

    comments_path = "comments_stats.csv"
    identifiers_path = "identifiers_stats.csv"
    strings_path = "strings_stats.csv"

    if single_lang:
        comments_path = "single_lang_comments_stats.csv"
        identifiers_path = "single_lang_identifiers_stats.csv"
        strings_path = "single_lang_strings_stats.csv"

    save_stats_to_csv(comments_stats, comments_path)
    save_stats_to_csv(identifiers_stats, identifiers_path)
    save_stats_to_csv(strings_stats, strings_path)

def save_stats_to_csv(stats, path):
    if not stats:
        return

    languages = stats['languages'].tolist()
    frequencies = stats['freq'].tolist()
    file_counts = stats['file_count'].tolist()

    df = pd.DataFrame({
        'language': languages,
        'frequency': frequencies,
        'file_count': file_counts
    })

    df.to_csv(path)


def get_stats(partials):
    if not partials:
        return {}

    buckets = ["identifiers", "comments", "strings"]
    freq_tot = {b: Counter() for b in buckets}
    max_size = {b: defaultdict(int) for b in buckets}
    file_cnt = {b: Counter() for b in buckets}

    for shard in partials:
        for b in buckets:
            langs = shard[f"{b}_langs"]  # already list[str]
            freqs = shard[f"{b}_freq"]
            mxs = shard[f"{b}_max_size"]
            fcs = shard[f"{b}_file_count"]

            for lang, f, mx, fc in zip(langs, freqs, mxs, fcs):
                freq_tot[b][lang] += f
                max_size[b][lang] = max(max_size[b][lang], mx)
                file_cnt[b][lang] += fc
    stats = {}
    for b in buckets:
        ordered = top_k(freq_tot[b], 10)
        langs = [l for l, _ in ordered]
        stats[b] = {
            "languages": langs,
            "freq": [freq_tot[b][l] for l in langs],
            "max_size": [max_size[b][l] for l in langs],
            "file_count": [file_cnt[b][l] for l in langs],
        }

    return stats

def get_file_stats(partials):
    files_only_lang = Counter()

    for shard in partials:
        langs = shard["only_langs"]  # list[str]
        counts = shard["only_counts"]  # list[int]  (same length)
        for lang, n in zip(langs, counts):
            files_only_lang[lang] += n

    return files_only_lang

def compute_language_stats(batch):
    buckets   = ["comments", "identifiers", "strings"]
    freq_tot  = {b: Counter()        for b in buckets}
    max_size  = {b: defaultdict(int) for b in buckets}
    file_seen = {b: Counter()        for b in buckets}

    for keys_i, mx_i, fr_i, keys_c, mx_c, fr_c, keys_s, mx_s, fr_s in zip(
            batch["lang_identifiers"], batch["lang_max_identifiers"], batch["lang_freq_identifiers"],
            batch["lang_comments"],    batch["lang_max_comments"],    batch["lang_freq_comments"],
            batch["lang_strings"],     batch["lang_max_strings"],     batch["lang_freq_strings"]):

        triplets = [
            ("identifiers", keys_i, mx_i, fr_i),
            ("comments",    keys_c, mx_c, fr_c),
            ("strings",     keys_s, mx_s, fr_s),
        ]
        for bucket, keys, mx, fr in triplets:
            langs_in_file = set()
            for k, sz, f in zip(keys, mx, fr):
                for lang in k.split("/"):
                    freq_tot [bucket][lang] += f
                    max_size[bucket][lang]  = max(max_size[bucket][lang], sz)
                    langs_in_file.add(lang)
            for lang in langs_in_file:
                file_seen[bucket][lang] += 1

    row = {}
    for b in buckets:
        langs = list(freq_tot[b].keys())  # order doesn't matter here
        row[f"{b}_langs"]      = [langs]
        row[f"{b}_freq"]       = [[freq_tot [b][l]  for l in langs]]
        row[f"{b}_max_size"]   = [[max_size[b][l]   for l in langs]]
        row[f"{b}_file_count"] = [[file_seen[b][l]  for l in langs]]

    return row

def compute_single_language_stats(batch):
    buckets   = ["identifiers", "comments", "strings"]
    freq_tot  = {b: Counter()        for b in buckets}
    max_size  = {b: defaultdict(int) for b in buckets}
    file_seen = {b: Counter()        for b in buckets}

    for keys_i, mx_i, fr_i, keys_c, mx_c, fr_c, keys_s, mx_s, fr_s in zip(
        batch["lang_identifiers"], batch["lang_max_identifiers"], batch["lang_freq_identifiers"],
        batch["lang_comments"],    batch["lang_max_comments"],    batch["lang_freq_comments"],
        batch["lang_strings"],     batch["lang_max_strings"],     batch["lang_freq_strings"]
    ):
        triplets = [
            ("identifiers", keys_i, mx_i, fr_i),
            ("comments",    keys_c, mx_c, fr_c),
            ("strings",     keys_s, mx_s, fr_s),
        ]
        for bucket, keys, mx, fr in triplets:
            langs_in_file = set()
            for lang, sz, f in zip(keys, mx, fr):
                # `lang` is already a single code
                if "/" in lang:
                    continue
                else:
                    freq_tot [bucket][lang] += f
                    max_size[bucket][lang]  = max(max_size[bucket][lang], sz)
                    langs_in_file.add(lang)
            for lang in langs_in_file:
                file_seen[bucket][lang] += 1

    row = {}
    for b in buckets:
        langs = list(freq_tot[b].keys())           # order arbitrary
        row[f"{b}_langs"]      = [langs]                               # [[...]]
        row[f"{b}_freq"]       = [[freq_tot [b][l]  for l in langs]]
        row[f"{b}_max_size"]   = [[max_size[b][l]   for l in langs]]
        row[f"{b}_file_count"] = [[file_seen[b][l]  for l in langs]]
    return row

def compute_multi_language_stats(batch):
    buckets   = ["identifiers", "comments", "strings"]
    freq_tot  = {b: Counter()        for b in buckets}
    max_size  = {b: defaultdict(int) for b in buckets}
    file_seen = {b: Counter()        for b in buckets}

    for keys_i, mx_i, fr_i, keys_c, mx_c, fr_c, keys_s, mx_s, fr_s in zip(
        batch["lang_identifiers"], batch["lang_max_identifiers"], batch["lang_freq_identifiers"],
        batch["lang_comments"],    batch["lang_max_comments"],    batch["lang_freq_comments"],
        batch["lang_strings"],     batch["lang_max_strings"],     batch["lang_freq_strings"]
    ):
        triplets = [
            ("identifiers", keys_i, mx_i, fr_i),
            ("comments",    keys_c, mx_c, fr_c),
            ("strings",     keys_s, mx_s, fr_s),
        ]
        for bucket, keys, mx, fr in triplets:
            langs_in_file = set()
            for lang, sz, f in zip(keys, mx, fr):
                if "/" in lang:
                    for language in lang.split("/"):
                        freq_tot[bucket][language] += f
                        max_size[bucket][language] = max(max_size[bucket][language], sz)
                        langs_in_file.add(language)

            for lang in langs_in_file:
                file_seen[bucket][lang] += 1

    row = {}
    for b in buckets:
        langs = list(freq_tot[b].keys())           # order arbitrary
        row[f"{b}_langs"]      = [langs]                               # [[...]]
        row[f"{b}_freq"]       = [[freq_tot [b][l]  for l in langs]]
        row[f"{b}_max_size"]   = [[max_size[b][l]   for l in langs]]
        row[f"{b}_file_count"] = [[file_seen[b][l]  for l in langs]]
    return row

def compute_single_language_in_files(batch):

    file_counter = Counter()

    for keys_i, keys_c, keys_s in zip(
        batch["lang_identifiers"],
        batch["lang_comments"],
        batch["lang_strings"]
    ):
        langs = set()

        def _add(lcodes):
            for code in lcodes:
                if "/" in code:          # mixed token, whole file discounted
                    langs.add(None)
                else:
                    langs.add(code)

        _add(keys_i)
        _add(keys_c)
        _add(keys_s)

        if None in langs or not langs:  # mixed-language or no lang seen
            continue
        if len(langs) == 1:             # pure single-language file
            lang = next(iter(langs))
            file_counter[lang] += 1

    # ---- serialise for Arrow (double-wrap) ----
    langs = list(file_counter.keys())
    return {
        "only_langs"  : [langs],
        "only_counts" : [[file_counter[l] for l in langs]],
    }


__all__ = [
    "save_stats", "get_stats", "get_file_stats", "compute_language_stats", "compute_single_language_stats", "compute_multi_language_stats", "compute_single_language_in_files"
]