"""This module contains the _StepPrettyPrinter class used in
Step.__repr__ for pretty-printing steps.
It is derived from the one in scikit-learn:
https://github.com/scikit-learn/scikit-learn/blob/5a4340834d23c4bdcd813ccda24a690ae174c168/sklearn/utils/_pprint.py

TODO: Although we trust this borrowed code, it'd be nice to have test coverage.
"""

# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
# 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018 Python Software Foundation;
# All Rights Reserved

# Authors: Fred L. Drake, Jr. <fdrake@acm.org> (built-in CPython pprint module)
#          Nicolas Hug (scikit-learn specific changes)
#          Alejandro González Tineo (baikal specific changes)

# License: PSF License version 2 (see below)

# PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
# --------------------------------------------

# 1. This LICENSE AGREEMENT is between the Python Software Foundation ("PSF"),
# and the Individual or Organization ("Licensee") accessing and otherwise
# using this software ("Python") in source or binary form and its associated
# documentation.

# 2. Subject to the terms and conditions of this License Agreement, PSF hereby
# grants Licensee a nonexclusive, royalty-free, world-wide license to
# reproduce, analyze, test, perform and/or display publicly, prepare
# derivative works, distribute, and otherwise use Python alone or in any
# derivative version, provided, however, that PSF's License Agreement and
# PSF's notice of copyright, i.e., "Copyright (c) 2001, 2002, 2003, 2004,
# 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
# 2017, 2018 Python Software Foundation; All Rights Reserved" are retained in
# Python alone or in any derivative version prepared by Licensee.

# 3. In the event Licensee prepares a derivative work that is based on or
# incorporates Python or any part thereof, and wants to make the derivative
# work available to others as provided herein, then Licensee hereby agrees to
# include in any such work a brief summary of the changes made to Python.

# 4. PSF is making Python available to Licensee on an "AS IS" basis. PSF MAKES
# NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED. BY WAY OF EXAMPLE, BUT
# NOT LIMITATION, PSF MAKES NO AND DISCLAIMS ANY REPRESENTATION OR WARRANTY OF
# MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF
# PYTHON WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

# 5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON FOR ANY
# INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS A RESULT OF
# MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON, OR ANY DERIVATIVE
# THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

# 6. This License Agreement will automatically terminate upon a material
# breach of its terms and conditions.

# 7. Nothing in this License Agreement shall be deemed to create any
# relationship of agency, partnership, or joint venture between PSF and
# Licensee. This License Agreement does not grant permission to use PSF
# trademarks or trade name in a trademark sense to endorse or promote products
# or services of Licensee, or any third party.

# 8. By copying, installing or otherwise using Python, Licensee agrees to be
# bound by the terms and conditions of this License Agreement.


# Brief summary of changes to original code:
# - "compact" parameter is supported for dicts, not just lists or tuples
# - estimators have a custom handler, they're not just treated as objects
# - long sequences (lists, tuples, dict items) with more than N elements are
#   shortened using ellipsis (', ...') at the end.

# Brief summary of changes to the scikit-learn code:
# - Prints changed parameters only, by default, unless sklearn is importable and
#   print_changed_only is set to False in the global config.
# - It has a local definition of is_scalar_nan.
# - Includes the step-specific parameters in the repr.
# - Revise logic to extract default parameters from the step signature.
# - Migrate brute-force ellipsis post-processing logic from BaseEstimator.__repr__
# - Rename all instances of "estimator[s]" to "steps" in class/function/variable names.

from inspect import signature
import numbers
import pprint
import re
from collections import OrderedDict

import numpy as np

from baikal import Step, get_config


def is_scalar_nan(x):
    """Tests if x is NaN

    This function is meant to overcome the issue that np.isnan does not allow
    non-numerical types as input, and that np.nan is not np.float('nan').

    Parameters
    ----------
    x : any type

    Returns
    -------
    boolean

    Examples
    --------
    >>> is_scalar_nan(np.nan)
    True
    >>> is_scalar_nan(float("nan"))
    True
    >>> is_scalar_nan(None)
    False
    >>> is_scalar_nan("")
    False
    >>> is_scalar_nan([np.nan])
    False
    """
    # convert from numpy.bool_ to python bool to ensure that testing
    # is_scalar_nan(x) is True does not fail.
    return bool(isinstance(x, numbers.Real) and np.isnan(x))


class KeyValTuple(tuple):
    """Dummy class for correctly rendering key-value tuples from dicts."""
    def __repr__(self):
        # needed for _dispatch[tuple.__repr__] not to be overridden
        return super().__repr__()


class KeyValTupleParam(KeyValTuple):
    """Dummy class for correctly rendering key-value tuples from parameters."""
    pass


def _get_params(step, changed_only):
    params = step.get_params(deep=False) if hasattr(step, "get_params") else {}
    if changed_only and params:
        init_func = step._get_super_class_with_init().__init__
        init_params = {
            name: param.default
            for name, param in signature(init_func).parameters.items()
        }
        params = _changed_params(params, init_params)

    params = OrderedDict(
        (name, val) for (name, val) in sorted(params.items(), key=pprint._safe_tuple)
    )
    # step-specific parameters
    params.update({"name": step.name,})
    params.update({"n_outputs": step.n_outputs})
    return params


def _changed_params(params, init_params):
    """Return dict (param_name: value) of parameters that were given to
    the step with non-default values."""
    filtered_params = {}
    for k, v in params.items():
        if (repr(v) != repr(init_params[k]) and
                not (is_scalar_nan(init_params[k]) and is_scalar_nan(v))):
            filtered_params[k] = v
    return filtered_params


class _StepPrettyPrinter(pprint.PrettyPrinter):
    """Pretty Printer class for step objects.

    This extends the pprint.PrettyPrinter class, because:
    - we need steps to be printed with their parameters, e.g.
      Step(param1=value1, ...) which is not supported by default.
    - the 'compact' parameter of PrettyPrinter is ignored for dicts, which
      may lead to very long representations that we want to avoid.

    Quick overview of pprint.PrettyPrinter (see also
    https://stackoverflow.com/questions/49565047/pprint-with-hex-numbers):

    - the entry point is the _format() method which calls format() (overridden
      here)
    - format() directly calls _safe_repr() for a first try at rendering the
      object
    - _safe_repr formats the whole object reccursively, only calling itself,
      not caring about line length or anything
    - back to _format(), if the output string is too long, _format() then calls
      the appropriate _pprint_TYPE() method (e.g. _pprint_list()) depending on
      the type of the object. This where the line length and the compact
      parameters are taken into account.
    - those _pprint_TYPE() methods will internally use the format() method for
      rendering the nested objects of an object (e.g. the elements of a list)

    In the end, everything has to be implemented twice: in _safe_repr and in
    the custom _pprint_TYPE methods. Unfortunately PrettyPrinter is really not
    straightforward to extend (especially when we want a compact output), so
    the code is a bit convoluted.

    This class overrides:
    - format() to support the changed_only parameter
    - _safe_repr to support printing of steps (for when they fit on a
      single line)
    - _format_dict_items so that dict are correctly 'compacted'
    - _format_items so that ellipsis is used on long lists and tuples

    When steps cannot be printed on a single line, the builtin _format()
    will call _pprint_step() because it was registered to do so (see
    _dispatch[Step.__repr__] = _pprint_step).

    both _format_dict_items() and _pprint_step() use the
    _format_params_or_dict_items() method that will format parameters and
    key-value pairs respecting the compact parameter. This method needs another
    subroutine _pprint_key_val_tuple() used when a parameter or a key-value
    pair is too long to fit on a single line. This subroutine is called in
    _format() and is registered as well in the _dispatch dict (just like
    _pprint_step). We had to create the two classes KeyValTuple and
    KeyValTupleParam for this.
    """

    def __init__(self, indent=1, width=80, depth=None, stream=None, *,
                 compact=False, indent_at_name=True,
                 n_max_elements_to_show=None):
        super().__init__(indent, width, depth, stream, compact=compact)
        self._indent_at_name = indent_at_name
        if self._indent_at_name:
            self._indent_per_level = 1  # ignore indent param

        self._changed_only = get_config()['print_changed_only']

        # Max number of elements in a list, dict, tuple until we start using
        # ellipsis. This also affects the number of arguments of a step
        # (they are treated as dicts)
        self.n_max_elements_to_show = n_max_elements_to_show

    def format(self, object, context, maxlevels, level):
        return _safe_repr(object, context, maxlevels, level,
                          changed_only=self._changed_only)

    def _pprint_step(self, object, stream, indent, allowance, context, level):
        stream.write(object.__class__.__name__ + '(')
        if self._indent_at_name:
            indent += len(object.__class__.__name__)

        params = _get_params(object, self._changed_only)

        self._format_params(params.items(), stream, indent, allowance + 1,
                            context, level)
        stream.write(')')

    def _format_dict_items(self, items, stream, indent, allowance, context,
                           level):
        return self._format_params_or_dict_items(
            items, stream, indent, allowance, context, level, is_dict=True)

    def _format_params(self, items, stream, indent, allowance, context, level):
        return self._format_params_or_dict_items(
            items, stream, indent, allowance, context, level, is_dict=False)

    def _format_params_or_dict_items(self, object, stream, indent, allowance,
                                     context, level, is_dict):
        """Format dict items or parameters respecting the compact=True
        parameter. For some reason, the builtin rendering of dict items doesn't
        respect compact=True and will use one line per key-value if all cannot
        fit in a single line.
        Dict items will be rendered as <'key': value> while params will be
        rendered as <key=value>. The implementation is mostly copy/pasting from
        the builtin _format_items().
        This also adds ellipsis if the number of items is greater than
        self.n_max_elements_to_show.
        """
        write = stream.write
        indent += self._indent_per_level
        delimnl = ',\n' + ' ' * indent
        delim = ''
        width = max_width = self._width - indent + 1
        it = iter(object)
        try:
            next_ent = next(it)
        except StopIteration:
            return
        last = False
        n_items = 0
        while not last:
            if n_items == self.n_max_elements_to_show:
                write(', ...')
                break
            n_items += 1
            ent = next_ent
            try:
                next_ent = next(it)
            except StopIteration:
                last = True
                max_width -= allowance
                width -= allowance
            if self._compact:
                k, v = ent
                krepr = self._repr(k, context, level)
                vrepr = self._repr(v, context, level)
                if not is_dict:
                    krepr = krepr.strip("'")
                middle = ': ' if is_dict else '='
                rep = krepr + middle + vrepr
                w = len(rep) + 2
                if width < w:
                    width = max_width
                    if delim:
                        delim = delimnl
                if width >= w:
                    width -= w
                    write(delim)
                    delim = ', '
                    write(rep)
                    continue
            write(delim)
            delim = delimnl
            class_ = KeyValTuple if is_dict else KeyValTupleParam
            self._format(class_(ent), stream, indent,
                         allowance if last else 1, context, level)

    def _format_items(self, items, stream, indent, allowance, context, level):
        """Format the items of an iterable (list, tuple...). Same as the
        built-in _format_items, with support for ellipsis if the number of
        elements is greater than self.n_max_elements_to_show.
        """
        write = stream.write
        indent += self._indent_per_level
        if self._indent_per_level > 1:
            write((self._indent_per_level - 1) * ' ')
        delimnl = ',\n' + ' ' * indent
        delim = ''
        width = max_width = self._width - indent + 1
        it = iter(items)
        try:
            next_ent = next(it)
        except StopIteration:
            return
        last = False
        n_items = 0
        while not last:
            if n_items == self.n_max_elements_to_show:
                write(', ...')
                break
            n_items += 1
            ent = next_ent
            try:
                next_ent = next(it)
            except StopIteration:
                last = True
                max_width -= allowance
                width -= allowance
            if self._compact:
                rep = self._repr(ent, context, level)
                w = len(rep) + 2
                if width < w:
                    width = max_width
                    if delim:
                        delim = delimnl
                if width >= w:
                    width -= w
                    write(delim)
                    delim = ', '
                    write(rep)
                    continue
            write(delim)
            delim = delimnl
            self._format(ent, stream, indent,
                         allowance if last else 1, context, level)

    def _pprint_key_val_tuple(self, object, stream, indent, allowance, context,
                              level):
        """Pretty printing for key-value tuples from dict or parameters."""
        k, v = object
        rep = self._repr(k, context, level)
        if isinstance(object, KeyValTupleParam):
            rep = rep.strip("'")
            middle = '='
        else:
            middle = ': '
        stream.write(rep)
        stream.write(middle)
        self._format(v, stream, indent + len(rep) + len(middle), allowance,
                     context, level)

    # mypy error: "Type[PrettyPrinter]" has no attribute "_dispatch"
    _dispatch = pprint.PrettyPrinter._dispatch.copy()  # type: ignore
    _dispatch[Step.__repr__] = _pprint_step
    _dispatch[KeyValTuple.__repr__] = _pprint_key_val_tuple


def _safe_repr(object, context, maxlevels, level, changed_only=False):
    """Same as the builtin _safe_repr, with added support for Step
    objects."""
    typ = type(object)

    if typ in pprint._builtin_scalars:
        return repr(object), True, False

    r = getattr(typ, "__repr__", None)
    if issubclass(typ, dict) and r is dict.__repr__:
        if not object:
            return "{}", True, False
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return "{...}", False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        saferepr = _safe_repr
        items = sorted(object.items(), key=pprint._safe_tuple)
        for k, v in items:
            krepr, kreadable, krecur = saferepr(
                k, context, maxlevels, level, changed_only=changed_only)
            vrepr, vreadable, vrecur = saferepr(
                v, context, maxlevels, level, changed_only=changed_only)
            append("%s: %s" % (krepr, vrepr))
            readable = readable and kreadable and vreadable
            if krecur or vrecur:
                recursive = True
        del context[objid]
        return "{%s}" % ", ".join(components), readable, recursive

    if (issubclass(typ, list) and r is list.__repr__) or \
       (issubclass(typ, tuple) and r is tuple.__repr__):
        if issubclass(typ, list):
            if not object:
                return "[]", True, False
            format = "[%s]"
        elif len(object) == 1:
            format = "(%s,)"
        else:
            if not object:
                return "()", True, False
            format = "(%s)"
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return format % "...", False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        for o in object:
            orepr, oreadable, orecur = _safe_repr(
                o, context, maxlevels, level, changed_only=changed_only)
            append(orepr)
            if not oreadable:
                readable = False
            if orecur:
                recursive = True
        del context[objid]
        return format % ", ".join(components), readable, recursive

    if issubclass(typ, Step):
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return "{...}", False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True
        context[objid] = 1
        readable = True
        recursive = False
        params = _get_params(object, changed_only)
        components = []
        append = components.append
        level += 1
        saferepr = _safe_repr
        for k, v in params.items():
            krepr, kreadable, krecur = saferepr(
                k, context, maxlevels, level, changed_only=changed_only)
            vrepr, vreadable, vrecur = saferepr(
                v, context, maxlevels, level, changed_only=changed_only)
            append("%s=%s" % (krepr.strip("'"), vrepr))
            readable = readable and kreadable and vreadable
            if krecur or vrecur:
                recursive = True
        del context[objid]
        return ("%s(%s)" % (typ.__name__, ", ".join(components)), readable,
                recursive)

    rep = repr(object)
    return rep, (rep and not rep.startswith('<')), False


def post_process_repr(repr_, n_char_max):
    """Post-process repr produced by _StepPrettyPrinter.
    Adapted from the original in scikit-learn BaseEstimator.__repr__
    """
    # Use bruteforce ellipsis when there are a lot of non-blank characters
    n_nonblank = len("".join(repr_.split()))
    if n_nonblank > n_char_max:
        lim = n_char_max // 2  # apprx number of chars to keep on both ends
        regex = r"^(\s*\S){%d}" % lim
        # The regex '^(\s*\S){%d}' % n
        # matches from the start of the string until the nth non-blank
        # character:
        # - ^ matches the start of string
        # - (pattern){n} matches n repetitions of pattern
        # - \s*\S matches a non-blank char following zero or more blanks
        left_lim = re.match(regex, repr_).end()
        right_lim = re.match(regex, repr_[::-1]).end()

        if "\n" in repr_[left_lim:-right_lim]:
            # The left side and right side aren't on the same line.
            # To avoid weird cuts, e.g.:
            # categoric...ore',
            # we need to start the right side with an appropriate newline
            # character so that it renders properly as:
            # categoric...
            # handle_unknown='ignore',
            # so we add [^\n]*\n which matches until the next \n
            regex += r"[^\n]*\n"
            right_lim = re.match(regex, repr_[::-1]).end()

        ellipsis = "..."
        if left_lim + len(ellipsis) < len(repr_) - right_lim:
            # Only add ellipsis if it results in a shorter repr
            repr_ = repr_[:left_lim] + "..." + repr_[-right_lim:]
    return repr_
