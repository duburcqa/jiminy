#!/usr/bin/env python

import os
import doxypy

if __name__ == "__main__":
    file_name = doxypy.optParse()
    file_input = doxypy.loadFile(file_name)

    # doxypy filter
    fsm = doxypy.Doxypy()
    output = fsm.parse(file_input)

    print output
