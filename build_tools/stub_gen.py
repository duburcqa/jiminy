# Monkey-patch pybind11-stubgen to handle Boost Python
#
# Based on https://github.com/jcarpent/pybind11-stubgen/blob/master/pybind11_stubgen/__init__.py
# Copyright (c) 2014-2020, CNRS
# Copyright (c) 2018-2020, INRIA
import re
import ast
import logging
from functools import partialmethod

import pybind11_stubgen
from pybind11_stubgen import logger, main, replace_default_pybind11_repr, FunctionSignature


issubclass_orig = issubclass


def _issubclass(cls, class_or_tuple, /):
    try:
        return issubclass_orig(cls, class_or_tuple)
    except TypeError as e:
        if not isinstance(class_or_tuple, tuple):
            class_or_tuple = (class_or_tuple,)
        if all(issubclass(cls_, dict) for cls_ in class_or_tuple):
            return issubclass_orig(cls, dict)
        raise e


def extract_boost_python_signature(args: str) -> str:
    find_optional_args = re.search('\[(.*?)\]$', args)
    if find_optional_args:
        optional_args = find_optional_args.group(1)
        nominal_args = args.replace("[" + optional_args + "]", "")
    else:
        optional_args, nominal_args = None, args

    num_nominal_args = 0
    if nominal_args:
        nominal_args = nominal_args.split(",")
        num_nominal_args = len(nominal_args)

    num_optional_args = 0
    if optional_args:
        optional_args = optional_args.split("[,")
        num_optional_args = len(optional_args)
        if num_optional_args > 1:
            optional_args[-1] = re.sub(
                (num_optional_args - 1)  * ']' + '$', '', optional_args[-1])
    new_args = ""

    if nominal_args:
        for k, arg in enumerate(nominal_args):
            type_name = re.search('\((.*?)\)', arg).group(1)
            _, arg_name = map(str.strip, arg.split(")", 1))
            arg_name = arg_name.replace(' ','_')

            new_args += arg_name + ": " + type_name
            if k < num_nominal_args - 1:
                new_args += ", "

    if num_optional_args > 0 and num_nominal_args > 0:
        new_args += ", "

    if optional_args and True:
        for k, arg in enumerate(optional_args):
            main_arg, *optional_args = map(str.strip, arg.split('=', 1))
            type_name = re.search('\((.*?)\)', main_arg).group(1)

            _, arg_name = map(str.strip, main_arg.split(")", 1))
            arg_name = arg_name.replace(' ','_')
            new_args += arg_name + ": " + type_name
            optional_value = None
            if len(optional_args) > 1:
                optional_value, *_ = optional_args
                new_args += " = " + optional_value

            if k < num_optional_args - 1:
                new_args += ", "

    return new_args.replace(" ,", ",")


def init_function_signature(self, name, args='*args, **kwargs', rtype='None', validate=True):
    self.name = name
    self.args = args
    self.rtype = rtype

    if validate:
        invalid_defaults, self.args = replace_default_pybind11_repr(self.args)
        if invalid_defaults:
            FunctionSignature.n_invalid_default_values += 1
            lvl = logging.WARNING if FunctionSignature.ignore_invalid_defaultarg else logging.ERROR
            logger.log(lvl, "Default argument value(s) replaced with ellipses (...):")
            for invalid_default in invalid_defaults:
                logger.log(lvl, "    {}".format(invalid_default))

        try:
            if args:
                self.args = extract_boost_python_signature(args)
            self.rtype = rtype.split(" :")[0]
        except IndexError:
            lvl = logging.WARNING if FunctionSignature.ignore_invalid_signature else logging.ERROR
            logger.log(lvl, "[%s] Bad signature formatting: '%s'", name, args)

        function_def_str = "def {sig.name}({sig.args}) -> {sig.rtype}: ...".format(sig=self)
        try:
            ast.parse(function_def_str)
        except SyntaxError as e:
            FunctionSignature.n_invalid_signatures += 1
            if FunctionSignature.signature_downgrade:
                self.name = name
                self.args = "*args, **kwargs"
                self.rtype = "typing.Any"
                lvl = logging.WARNING if FunctionSignature.ignore_invalid_signature else logging.ERROR
                logger.log(lvl, "Generated stubs signature is degraded to `(*args, **kwargs) -> typing.Any` for")
            else:
                lvl = logging.WARNING
                logger.warning("Ignoring invalid signature:")
            logger.log(lvl, function_def_str)
            logger.log(lvl, " " * (e.offset - 1) + "^-- Invalid syntax")


def to_lines_classmember_replace_self(self):
    result = []
    docstring = self.sanitize_docstring(self.member.__doc__)
    if not docstring and not (
        self.name.startswith("__") and self.name.endswith("__")
    ):
        logger.debug(
            "Docstring is empty for '%s'" % self.fully_qualified_name(self.member)
        )
    for sig in self.signatures:
        args = sig.args
        sargs = args.strip()
        # =============================================================
        # Boost::Python does not name "self" as such automatically.
        # Thus, the condition should be based on type rather than name.
        # =============================================================
        args_splitted = sig.split_arguments()
        if args_splitted[0].split(':', 1)[-1].strip() not in (self.member.__module__, "object"):
            if sargs.startswith("cls"):
                result.append("@classmethod")
                # remove type of cls
                args = ",".join(["cls"] + args_splitted[1:])
            else:
                result.append("@staticmethod")
        else:
            # remove type of self
            args = ",".join(["self"] + args_splitted[1:])
        if len(self.signatures) > 1:
            result.append("@typing.overload")

        result.append(
            "def {name}({args}) -> {rtype}: {ellipsis}".format(
                name=sig.name,
                args=args,
                rtype=sig.rtype,
                ellipsis="" if docstring else "...",
            )
        )
        if docstring:
            result.append(self.format_docstring(docstring))
            docstring = None  # don't print docstring for other overloads
    return result


pybind11_stubgen.__builtins__['issubclass'] = _issubclass
pybind11_stubgen.ClassMemberStubsGenerator.to_lines = to_lines_classmember_replace_self
pybind11_stubgen.FunctionSignature.__init__ = init_function_signature
pybind11_stubgen.ClassStubsGenerator.__init__ = partialmethod(
    pybind11_stubgen.ClassStubsGenerator.__init__,
    base_class_blacklist=("object", "instance"))

if __name__ == "__main__":
    main()
