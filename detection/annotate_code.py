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

def point_to_char_index(input_code, point):
    """
    Given a Tree-sitter point (row, col), return the
    absolute Python string index into input_code.
    """
    row, col = point
    # split *once* per file, keeping line breaks
    lines = input_code.splitlines(keepends=True)
    # sum the lengths of all full lines before `row`
    # then add the column offset
    return sum(len(lines[i]) for i in range(row)) + col

def parse_code(input_code):
    encoded_code = bytes(input_code, 'utf8')
    tree = parser.parse(encoded_code)
    root_node = tree.root_node


    captures: dict[str, list] = JAVA_QUERY.captures(root_node)
    # ← this is a bytes object
    results = []

    for cap_name, nodes in captures.items():
        for node in nodes:
            c_start = point_to_char_index(input_code, node.start_point)
            c_end = point_to_char_index(input_code, node.end_point)
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
        if cap_name == 'comment.block':
            # For block and line comments, we can also detect language
            comment_annotations = detect_language_in_comment(start, match_str)
            annotations[start:end] = comment_annotations


    # resulting_array = [x[1] for x in annotations]

    return annotations



__all__ = [
    'parse_code', 'annotate_code'
]