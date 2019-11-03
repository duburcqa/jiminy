#!/usr/bin/env python
# Doxypy main script
# mypy page: https://pypi.org/project/doxypy/
# source from: https://gitlab.cern.ch:8443/lhcb-core
# (30/08/2019): https://gitlab.cern.ch:8443/lhcb-core/LbScripts/blob/f02cc5370f8e7911e2549e9238c0d665dbbff3cc/LbUtils/scripts/doxypy

###############################################################################
# (c) Copyright 2000-2018 CERN for the benefit of the LHCb Collaboration      #
#                                                                             #
# This software is distributed under the terms of the GNU General Public      #
# Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################

__applicationName__ = "doxypy"
__blurb__ = """
doxypy is an input filter for Doxygen. It preprocesses python
files so that docstrings of classes and functions are reformatted
into Doxygen-conform documentation blocks.
"""

__doc__ = __blurb__ + \
    """
In order to make Doxygen preprocess files through doxypy, simply
add the following lines to your Doxyfile:
	FILTER_SOURCE_FILES = YES
	INPUT_FILTER = "python /path/to/doxypy.py"
"""

__version__ = "0.3rc1"
__date__ = "1st December 2007"
__website__ = "http://code.foosel.org/doxypy"

__author__ = (
    "Philippe 'demod' Neumann (doxypy at demod dot org)",
    "Gina 'foosel' Haeussge (gina at foosel dot net)"
)

__licenseName__ = "GPL v2"
__license__ = """This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import re

from optparse import OptionParser, OptionGroup


class FSM(object):
    """	FSM implements a finite state machine. Transitions are given as
            4-tuples, consisting of an origin state, a target state, a condition
            for the transition (given as a reference to a function which gets called
            with a given piece of input) and a pointer to a function to be called
            upon the execution of the given transition. 
    """

    def __init__(self, start_state=None, transitions=[]):
        self.transitions = transitions
        self.current_state = start_state
        self.current_input = None
        self.current_transition = None

    def setStartState(self, state):
        self.current_state = state

    def addTransition(self, from_state, to_state, condition, callback):
        self.transitions.append([from_state, to_state, condition, callback])

    def makeTransition(self, input):
        """ Makes a transition based on the given input.
                @param	input	input to parse by the FSM
        """
        for transition in self.transitions:
            [from_state, to_state, condition, callback] = transition
            if from_state == self.current_state:
                match = condition(input)
                if match:
                    self.current_state = to_state
                    self.current_input = input
                    self.current_transition = transition
                    callback(match)
                    return


class Doxypy(object):
    def __init__(self):
        self.start_single_comment_re = re.compile("^\s*(''')")
        self.end_single_comment_re = re.compile("(''')\s*$")

        self.start_double_comment_re = re.compile("^\s*(\"\"\")")
        self.end_double_comment_re = re.compile("(\"\"\")\s*$")

        self.single_comment_re = re.compile("^\s*(''').*(''')\s*$")
        self.double_comment_re = re.compile("^\s*(\"\"\").*(\"\"\")\s*$")

        self.defclass_re = re.compile("^(\s*)(def .+:|class .+:)")
        self.empty_re = re.compile("^\s*$")
        self.hashline_re = re.compile("^\s*#.*$")
        self.importline_re = re.compile("^\s*(import |from .+ import)")

        # Transition list format
        #  ["FROM", "TO", condition, action]
        transitions = [
            # FILEHEAD

            # single line comments
            ["FILEHEAD", "FILEHEAD", self.single_comment_re.search,
             self.appendCommentLine],
            ["FILEHEAD", "FILEHEAD", self.double_comment_re.search,
             self.appendCommentLine],

            # multiline comments
            ["FILEHEAD", "FILEHEAD_COMMENT_SINGLE",
             self.start_single_comment_re.search, self.appendCommentLine],
            ["FILEHEAD_COMMENT_SINGLE", "FILEHEAD",
             self.end_single_comment_re.search, self.appendCommentLine],
            ["FILEHEAD_COMMENT_SINGLE", "FILEHEAD_COMMENT_SINGLE",
             self.catchall, self.appendCommentLine],
            ["FILEHEAD", "FILEHEAD_COMMENT_DOUBLE",
             self.start_double_comment_re.search, self.appendCommentLine],
            ["FILEHEAD_COMMENT_DOUBLE", "FILEHEAD",
             self.end_double_comment_re.search, self.appendCommentLine],
            ["FILEHEAD_COMMENT_DOUBLE", "FILEHEAD_COMMENT_DOUBLE",
             self.catchall, self.appendCommentLine],

            # other lines
            ["FILEHEAD", "FILEHEAD", self.empty_re.search,
             self.appendFileheadLine],
            ["FILEHEAD", "FILEHEAD", self.hashline_re.search,
             self.appendFileheadLine],
            ["FILEHEAD", "FILEHEAD", self.importline_re.search,
             self.appendFileheadLine],
            ["FILEHEAD", "DEFCLASS", self.defclass_re.search,
             self.resetCommentSearch],
            ["FILEHEAD", "DEFCLASS_BODY", self.catchall, self.appendFileheadLine],

            # DEFCLASS

            # single line comments
            ["DEFCLASS", "DEFCLASS_BODY",
             self.single_comment_re.search, self.appendCommentLine],
            ["DEFCLASS", "DEFCLASS_BODY",
             self.double_comment_re.search, self.appendCommentLine],

            # multiline comments
            ["DEFCLASS", "COMMENT_SINGLE",
             self.start_single_comment_re.search, self.appendCommentLine],
            ["COMMENT_SINGLE", "DEFCLASS_BODY",
             self.end_single_comment_re.search, self.appendCommentLine],
            ["COMMENT_SINGLE", "COMMENT_SINGLE",
             self.catchall, self.appendCommentLine],
            ["DEFCLASS", "COMMENT_DOUBLE",
             self.start_double_comment_re.search, self.appendCommentLine],
            ["COMMENT_DOUBLE", "DEFCLASS_BODY",
             self.end_double_comment_re.search, self.appendCommentLine],
            ["COMMENT_DOUBLE", "COMMENT_DOUBLE",
             self.catchall, self.appendCommentLine],

            # other lines
            ["DEFCLASS", "DEFCLASS", self.empty_re.search,
             self.appendDefclassLine],
            ["DEFCLASS", "DEFCLASS", self.defclass_re.search,
             self.resetCommentSearch],
            ["DEFCLASS", "DEFCLASS_BODY", self.catchall, self.stopCommentSearch],

            # DEFCLASS_BODY

            ["DEFCLASS_BODY", "DEFCLASS",
             self.defclass_re.search, self.startCommentSearch],
            ["DEFCLASS_BODY", "DEFCLASS_BODY",
             self.catchall, self.appendNormalLine],
        ]

        self.fsm = FSM("FILEHEAD", transitions)

        self.output = []

        self.comment = []
        self.filehead = []
        self.defclass = []
        self.indent = ""

    def catchall(self, input):
        """	The catchall-condition, always returns true. """
        return True

    def resetCommentSearch(self, match):
        """ Restarts a new comment search for a different triggering line.
                Closes the current commentblock and starts a new comment search.
        """
        self.__closeComment()
        self.startCommentSearch(match)

    def startCommentSearch(self, match):
        """ Starts a new comment search.
                Saves the triggering line, resets the current comment and saves
                the current indentation.
        """
        self.defclass = [self.fsm.current_input]
        self.comment = []
        self.indent = match.group(1)

    def stopCommentSearch(self, match):
        """ Stops a comment search.
                Closes the current commentblock, resets	the triggering line and
                appends the current line to the output.
        """
        self.__closeComment()

        self.defclass = []
        self.output.append(self.fsm.current_input)

    def appendFileheadLine(self, match):
        """	Appends a line in the FILEHEAD state.
                Closes the open comment	block, resets it and appends the current line.
        """
        self.__closeComment()
        self.comment = []
        self.output.append(self.fsm.current_input)

    def appendCommentLine(self, match):
        """	Appends a comment line.
                The comment delimiter is removed from multiline start and ends as
                well as singleline comments.
        """
        (from_state, to_state, condition, callback) = self.fsm.current_transition

        # single line comment
        if (from_state == "DEFCLASS" and to_state == "DEFCLASS_BODY") \
                or (from_state == "FILEHEAD" and to_state == "FILEHEAD"):
            # remove comment delimiter from begin and end of the line
            activeCommentDelim = match.group(1)
            line = self.fsm.current_input
            self.comment.append(line[line.find(
                activeCommentDelim)+len(activeCommentDelim):line.rfind(activeCommentDelim)])
            if (to_state == "DEFCLASS_BODY"):
                self.__closeComment()
                self.defclass = []
        # multiline start
        elif from_state == "DEFCLASS" or from_state == "FILEHEAD":
            # remove comment delimiter from begin of the line
            activeCommentDelim = match.group(1)
            line = self.fsm.current_input
            self.comment.append(
                line[line.find(activeCommentDelim)+len(activeCommentDelim):])
        # multiline end
        elif to_state == "DEFCLASS_BODY" or to_state == "FILEHEAD":
            # remove comment delimiter from end of the line
            activeCommentDelim = match.group(1)
            line = self.fsm.current_input
            self.comment.append(line[0:line.rfind(activeCommentDelim)])
            if (to_state == "DEFCLASS_BODY"):
                self.__closeComment()
                self.defclass = []
        # in multiline comment
        else:
            # just append the comment line
            self.comment.append(self.fsm.current_input)

    def appendNormalLine(self, match):
        """ Appends a line to the output. """
        self.output.append(self.fsm.current_input)

    def appendDefclassLine(self, match):
        """ Appends a line to the triggering block. """
        self.defclass.append(self.fsm.current_input)

    def __closeComment(self):
        """ Appends any open comment block and triggering block to the output. """
        if self.comment:
            block = self.makeCommentBlock()
            self.output.extend(block)
        if self.defclass:
            self.output.extend(self.defclass)

    def makeCommentBlock(self):
        """ Indents the current comment block with respect to the current
                indentation level.
                @returns a list of indented comment lines
        """
        doxyStart = "##"
        commentLines = self.comment

        if options.strip:
            commentLines = map(str.strip, commentLines)

        commentLines = map(lambda x: "%s# %s" % (self.indent, x), commentLines)
        l = [self.indent + doxyStart]
        l.extend(commentLines)

        return l

    def parse(self, input):
        """ Parses a python file given as input string and returns the doxygen-
                compatible representation.
                @param	input	the python code to parse
                @returns the modified python code
        """
        lines = input.split("\n")

        for line in lines:
            self.fsm.makeTransition(line)

        if self.fsm.current_state == "DEFCLASS":
            self.__closeComment()

        return "\n".join(self.output)


def loadFile(filename):
    """ Loads file "filename" and returns the content.
            @param   filename	The name of the file to load
            @returns the content of the file.
    """
    f = open(filename, 'r')

    try:
        content = f.read()
        return content
    finally:
        f.close()


def optParse():
    """ Parses commandline options. """
    parser = OptionParser(prog=__applicationName__,
                          version="%prog " + __version__)

    parser.set_usage("%prog [options] filename")
    parser.add_option("--trim", "--strip",
                      action="store_true", dest="strip",
                      help="enables trimming of docstrings, might be useful if you get oddly spaced output"
                      )

    # parse options
    global options
    (options, filename) = parser.parse_args()

    if not filename:
        print >>sys.stderr, "No filename given."
        sys.exit(-1)

    return filename[0]


def main():
    """ Opens the file given as first commandline argument and processes it,
            then prints out the processed file.
    """
    filename = optParse()

    try:
        file_input = loadFile(filename)
    except IOError, (errno, msg):
        print >>sys.stderr, msg
        sys.exit(-1)

    fsm = Doxypy()
    output = fsm.parse(file_input)
    print output


if __name__ == "__main__":
    main()
