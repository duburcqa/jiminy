import os
import re
import sys
import pathlib
from queue import deque
from textwrap import indent
from typing import Sequence


def redent(txt: str, n: int = 0) -> str:
    """Re-indent a multiline text using a given number of space. Note that it
    completely gets rid of the original indentation contrary to `dedent`.
    """
    return indent(''.join([
        line.strip() + '\n' for line in txt.splitlines(True)]), ' ' * n)


def find_include_files(input_files_fullpath: Sequence[str],
                       include_dir: str) -> Sequence[str]:
    """Scan recursively the list of header files included in given header files
    and contain in a specific include directory.
    """
    include_dir = pathlib.Path(include_dir)
    lookup_file_list = deque(input_files_fullpath)
    include_path_list = []
    while len(lookup_file_list):
        input_file_fullpath = lookup_file_list.pop()
        try:
            with open(input_file_fullpath, "r") as input_file:
                for line in input_file:
                    if "#include" in line:
                        relative_path = re.search(r'(?<=")[^ ]+(?="$)', line)
                        if relative_path:
                            relative_path = relative_path.group()
                            try:
                                full_path = str(next(
                                    include_dir.rglob(relative_path)))
                            except StopIteration:
                                break
                            # Prevent potential infinite recursion
                            if not full_path in include_path_list:
                                include_path_list.append(full_path)
                                lookup_file_list.append(full_path)
        except (FileNotFoundError, PermissionError):
            continue  # It may happen for temporary files and some dependencies
    return include_path_list


def extract_doxygen_doc(include_filenames: str,
                        doc_pattern: str) -> str:
    """Get C++ docstring associated with a given class and method.
    """
    method_line = -1
    class_name, method_name = doc_pattern.split("::", 1)
    for include_filename in include_filenames:
        if class_name in include_filename:
            with open(include_filename, "r") as include_file:
                for i, line in enumerate(include_file):
                    if re.findall(fr"(?<= ){method_name}(?= *\()", line):
                        method_filename = include_filename
                        method_line = i
                        break
    if (method_line > 0):
        with open(method_filename, "r") as include_file:
            lines = include_file.readlines()
        for doc_start_line in list(range(method_line))[::-1]:
            if not "///" in lines[doc_start_line]:
                break
        doc = lines[(doc_start_line+1):method_line]
        return re.sub(r"^[ \n]+?(?= +\\)|[ \n]+$", "",
                      "".join(doc).replace("/",""))
    else:
        return ""


if __name__ == "__main__":
    _, include_dir, input_file_fullpath, output_file_name =  sys.argv

    # Extract the fullpath of included files available in include_dir
    include_files = find_include_files([input_file_fullpath], include_dir)
    include_files.append(input_file_fullpath)

    # Create output file
    pathlib.Path(os.path.dirname(output_file_name)).mkdir(
        parents=True, exist_ok=True)

    # Read input file
    with open(input_file_fullpath, "r") as input_file:
        input_str = input_file.read()

    # Extract boost python method definitions
    pat_head = r'(\.def\("(?!__)[a-zA-Z0-9_\r\n", <>\*\(\):&]*?&)'
    pat_class_method = r'([^ ,\(\)*]+)::([^ ,\(\)*]+)'
    pat_tail = r'([a-zA-Z0-9_\r\n", <>\(\):=\-@]*?)\)'
    pat_delim = r'(?=(?: *\n? *?(?:;|\.|\n\n)| *//))'
    bp_def_list = re.findall(
        fr'{pat_head}{pat_class_method}{pat_tail}{pat_delim}', input_str)

    # Parse @copydoc macro, and use best-guess otherwise if order to add C++
    # docstring to Python.
    for (head, class_name, method_name, tail) in bp_def_list:
        # Check if the def already has docstring argument
        docstring = re.search(r'"(.*)"$', tail)
        if docstring is not None:
            docstring = docstring.group(1)
            has_copydoc = "@copydoc" in docstring
        else:
            has_copydoc = False

        # Get declaration from which to extract docstring, if anu
        if has_copydoc:
            doc_pattern = re.search(
                r"@copydoc +([a-zA-Z0-9:_]+)", docstring).group(1)
        else:
            if docstring:
                continue
            class_name_clean = re.sub("^Py|Visitor$", "", class_name)
            doc_pattern = f"{class_name_clean}::{method_name}"

        # Extract raw docstring, if any. Continue otherwise.
        doc_str = extract_doxygen_doc(include_files, doc_pattern)
        if not doc_str:
            continue

        # Remove global indentation
        doc_str = redent(doc_str)

        # Convert to reStructuredText format manually
        make_flag = lambda txt: f":{txt}: "
        make_directive = lambda txt: f".. {txt}::\n"
        tag_map = {
            'remark': make_directive('note'),
            'return': make_flag('return')
        }

        pat_block = r'((?:.|\n)+?\n(?=\\|\.\.|$))'

        for (_head, tag, txt) in re.findall(
                fr"( *\\)(details|brief){pat_block}", doc_str):
            doc_str = doc_str.replace(
                f"{_head}{tag}{txt}",
                f"{redent(txt)}")

        for (_head, tag, txt) in re.findall(
                fr"( *\\)(remark|note|return){pat_block}", doc_str):
            doc_str = doc_str.replace(
                f"{_head}{tag}{txt}",
                f"\n{tag_map.get(tag, make_directive(tag))}{redent(txt, 4)}")

        for (_head, name, descr) in re.findall(
                fr"( *\\param\[(?:in|out)\] +)([^ ]+){pat_block}", doc_str):
            doc_str = doc_str.replace(
                f"{_head}{name}{descr}",
                f":param {name}:\n{redent(descr, 4)}")

        # Replace newline by string '\n' because it is not properly supported
        doc_str = doc_str.strip().replace("\n", r'\\n')

        # Add docstring
        def_match = f"{head}{class_name}::{method_name}{tail}"
        def_orig = f"{def_match})"
        if has_copydoc:
            doc_str = doc_str.replace(r'\\n', r'\\\\n')
            def_new = re.sub(r'(?<=")@copydoc.+(?="\)$)', doc_str, def_orig)
        else:
            def_new =  f'{def_match}, "{doc_str}")'
        def_orig = re.sub(r"(\(|\)|\\)", r"\\\g<1>", def_orig)
        input_str = re.sub(def_orig, def_new, input_str, count=1)

    # Save post-processed file
    with open(output_file_name, "w") as output_file:
        output_file.write(input_str)
