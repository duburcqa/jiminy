# Jiminy python documentation

@note This documentation is currently in beta.

## Usage
 This documentation groups together all the native Python code and bindings, that is part of the jiminy project.
 The source code is split in several python packages, each containing both source code for the package
 and command line scripts, both of which are covered here.

  - For a list of command-line tools (standalone scripts that can be called from the command line), refer to the
   [Command Line Scripts](./group__scripts.html) page.
  - Documentation for the package sources is available by three different tabs:
    - [Packages](./namespaces.html), where file are sorted by python package.
    - [Classes](./annotated.html), where only python classes are listed.
    - [Files](./files.html), where one may browse the file directory tree directly.
      This is the most complete reference (i.e. some file might be inaccessible from the other two interfaces).
  - Python bindings of the C++ source code are documented as well: typically look for <ClassName>Visitor class.

## Contributing to this documentation

 Here are some guidelines on how this documentation is generated:

 - doxygen is used to parse the documentation, thus everything works like in C code:
  - use double comment (##, like /// in C) to start a doxygen line.
  - comments must be placed before the declaration of the function they comment, or the class creation.
  - all [special commands](https://www.stack.nl/~dimitri/doxygen/manual/commands.html) are available. For declaring
    a command, \\ or \@ character can be used: for backward compatibility with previously written code, please use
    \@ symbol only in python. Here is an example ({at} stands for \@ character):

        ## {at}brief Gives the answer.
        ## {at}details This function gives the answer life, the universe and everything.
        ## {at}note Not everybody might be psychologically ready for the answer.
        ## {at}param question The question input.
        ## {at}return 42
        def answer(the_question):
          return 42

 - File parsed: the following file are parsed:
  - all python files (files with extension .py) in jiminy, **except** __init__.py  and setup.py files and
    files in a build (sub)directory or in a unit (sub)directory.
  - all files in a scripts directory, regardless of extension (command line scripts have no extensions). Files
    without extensions are considered to be python scripts.
  - .h files inside a directory named python.

 - To add a file to the [Command Line Scripts](./group__scripts.html) page, simply add `\@ingroup scripts` at the top of
   the file.
