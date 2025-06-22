import tree_sitter_java
from tree_sitter import Language, Parser, Query
from collections import Counter
from detection.comment_language_detection import detect_language_in_comment
from detection.identifier_language_detection import determine_language_in_variable_name

JAVA_LANGUAGE = Language(tree_sitter_java.language())
parser = Parser(JAVA_LANGUAGE)
JAVA_QUERY = JAVA_LANGUAGE.query(
    """
    (method_declaration
       name: (identifier)        @method.name)
    (class_declaration
         name: (identifier)        @class.name)
    (variable_declarator
        name: (identifier)        @variable.name)
    (string_literal
       (string_fragment)         @string.content)
    (block_comment)       @comment.block
    (line_comment)        @comment.line
    """
)

def parse_code(input_code):
    encoded_code = bytes(input_code, 'utf8')
    tree = parser.parse(encoded_code)
    root_node = tree.root_node


    captures: dict[str, list] = JAVA_QUERY.captures(root_node)
    # ← this is a bytes object
    results = []

    for cap_name, nodes in captures.items():
        for node in nodes:
            start = node.start_byte
            end = node.end_byte

            c_start = len(encoded_code[:start].decode('utf-8', errors='replace'))
            c_end = len(encoded_code[:end].decode('utf-8', errors='replace'))
            snippet = input_code[c_start:c_end]

            results.append(
                (cap_name, c_start, c_end, snippet)
            )

    return results


def annotate_code(input_code):

    """
    Annotate the input code with language information for comments and variable names.
    Returns a list of tuples containing the starting position in code and the detected language.
    """

    annotations = [
            (index, "-1")  # Default annotation for each character
            for index in range(len(input_code))
    ]

    # Parse the code to get variable names
    parsed_results = parse_code(input_code)

    for cap_name, start, end, match_str in parsed_results:
        if cap_name == 'variable.name' or cap_name == 'method.name' or cap_name == 'class.name':
            variable_annotations = determine_language_in_variable_name(start, match_str)
            annotations[start:end] = variable_annotations
        if cap_name == 'comment.block' or cap_name == 'comment.line' or cap_name == 'string.content':
            # For block and line comments, we can also detect language
            comment_annotations = detect_language_in_comment(start, match_str)
            annotations[start:end] = comment_annotations


    resulting_array = [x[1] for x in annotations]

    return resulting_array

def get_numerical_data_from_code(input_code, annotations):

    # 1) First pass: collect raw (lang_key, length) tuples per bucket
    parsed = parse_code(input_code)
    raw = { 'identifiers': [], 'comments': [], 'strings': [] }

    for cap_name, start, end, match_str in parsed:
        if cap_name in {'variable.name', 'method.name', 'class.name'}:
            bucket = 'identifiers'
        elif cap_name in {'comment.block', 'comment.line'}:
            bucket = 'comments'
        elif cap_name == 'string.content':
            bucket = 'strings'
        else:
            continue

        # collect all non-"-1" langs for this token
        langs = [lang for lang in annotations[start:end] if lang != "-1"]
        if not langs:
            continue

        # pick up to 3 most frequent, sort & join only if the frequency is > 1
        top_three = [l for l,_ in Counter(langs).most_common(3) if langs.count(l) > 1]
        lang_key  = "/".join(sorted(top_three))

        raw[bucket].append((lang_key, len(match_str)))

    # 2) Second pass: aggregate per lang_key in each bucket
    data = {}
    for bucket, tuples in raw.items():
        stats = {}  # lang_key -> {'max':…, 'count':…}
        for key, length in tuples:
            if key not in stats:
                stats[key] = {'max': length, 'count': 1}
            else:
                stats[key]['max']   = max(stats[key]['max'], length)
                stats[key]['count'] += 1

        # sort lang_keys by descending freq, then code
        ordered = sorted(stats.keys(),
                         key=lambda k: (-stats[k]['count'], k))

        data[f'lang_{bucket}']      = ordered
        data[f'lang_max_{bucket}']  = [stats[k]['max']   for k in ordered]
        data[f'lang_freq_{bucket}'] = [stats[k]['count'] for k in ordered]

    return data

__all__ = [
    'parse_code', 'annotate_code', 'get_numerical_data_from_code'
]