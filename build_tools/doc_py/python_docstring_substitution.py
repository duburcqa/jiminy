#!/usr/bin/env python

import os
import sys
import re

def mkdir_rec(path):
    try:
        os.makedirs(path)
    except OSError as e:
        pass

def find_file(relative_path, base_dir):
    relative_dir = os.path.dirname(relative_path)
    filename = os.path.basename(relative_path)
    for root, dirs, files in os.walk(base_dir):
        if (not relative_dir or root[-len(relative_dir):] == relative_dir) and filename in files:
            return os.path.join(root, filename)

def find_include_files(input_file_name, include_dir):
    include_path_list = []
    with open(input_file_name, "r") as input_file:
        for line in input_file:
            if "#include" in line:
                relative_path = re.findall(r'(?<=")[^ ]+(?="$)', line)
                if relative_path:
                    full_path = find_file(relative_path[0], include_dir)
                    if full_path is not None:
                        include_path_list.append(full_path)
    return include_path_list

def extract_doxygen_doc(include_filenames, doc_pattern):
    method_line = -1
    class_name, method_name = doc_pattern.split("::", 1)
    for include_filename in include_filenames:
        if class_name in include_filename:
            with open(include_filename, "r") as include_file:
                for i, line in enumerate(include_file):
                    if len(re.findall(r'(?<= )' + method_name + r'(?= *\()', line)) > 0:
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
        return re.sub(r'^[ \n]+?(?= +\\)|[ \n]+$', "", "".join(doc).replace("/",""))
    else:
        return ""

if __name__ == "__main__":
    include_dir =  sys.argv[1]
    input_file_name =  sys.argv[2]
    output_file_name =  sys.argv[3]

    # Extract the fullpath of included files available in include_dir
    include_files = find_include_files(input_file_name, include_dir)
    include_files.append(input_file_name)

    mkdir_rec(os.path.dirname(output_file_name))
    with open(input_file_name, "r") as input_file:
        with open(output_file_name, "w") as output_file:
            for line in input_file:
                # docstring substitution
                if "@copydoc" in line:
                    doc_pattern = re.findall(r'(?<=@copydoc )[a-zA-Z0-9:_]+', line)[0]
                    line = line.replace("@copydoc " + doc_pattern,
                                        extract_doxygen_doc(include_files, doc_pattern).replace("\\",r'@').replace("\n",r'\n'))
                output_file.write(line)
