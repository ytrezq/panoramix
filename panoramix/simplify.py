import collections
import time
import logging
import sys
from copy import copy

import panoramix.core.arithmetic as arithmetic
from panoramix.core.algebra import (
    _max_op,
    add_ge_zero,
    add_op,
    apply_mask,
    apply_mask_to_storage,
    bits,
    calc_max,
    div_op,
    divisible_bytes,
    flatten_adds,
    ge_zero,
    get_sign,
    le_op,
    lt_op,
    mask_op,
    max_op,
    max_to_add,
    min_op,
    minus_op,
    mul_op,
    neg_mask_op,
    or_op,
    safe_ge_zero,
    safe_le_op,
    safe_lt_op,
    safe_max_op,
    safe_min_op,
    shr_op,
    signextend_op,
    simplify,
    simplify_max,
    sub_op,
    to_bytes,
    try_add,
)
from panoramix.core.arithmetic import changed_reads, is_zero, to_real_int
from panoramix.core.masks import get_bit, to_mask, to_neg_mask
from panoramix.core.memloc import (
    apply_mask_to_range,
    fill_mem,
    memloc_overwrite,
    range_overlaps,
    sizeof,
    split_setmem,
    split_store,
    splits_mem,
)
from panoramix.matcher import Any, match
from panoramix.prettify import (
    explain,
    pformat_trace,
    pprint_repr,
    pprint_trace,
    pretty_repr,
)
from panoramix.utils.helpers import (
    C,
    MAX_EXP_SIZE,
    cached,
    contains,
    exp_size,
    find_f_list,
    find_f_set,
    find_op_list,
    opcode,
    replace,
    replace_f,
    replace_f_stop,
    rewrite_trace,
    rewrite_trace_full,
    rewrite_trace_ifs,
    to_exp2,
    walk_trace,
)

from panoramix.postprocess import cleanup_mul_1
from panoramix.rewriter import postprocess_exp, postprocess_trace, rewrite_string_stores

logger = logging.getLogger(__name__)


"""

    Simplifier engine.

    It takes the trace, and performs rewrites in the following loop:
        - simplify expressions
        - inline and cleanup variables (when possible)
        - inline and cleanup memory (when possible)
        - split setmems
        - split storage writes into separate lines
        - replace msize with actual values if possible
        - if there are any obvious always-fulfilled conditions, remove them
        - look for loops that are really setmems and clean those too
        - for loops that access storage, rewrite them so they are easier to parse by storage parser

    Finally, after all the processing is done:
    - replace `max(2+x, 3+y)` with `2 + max(x, 1+y)` - more human-readable, but would cause problems in the loop
    - do some other stuff that is human readable but would make the processing in the loop difficult


    If you want to see the engine in all it's glory, try:
        panoramix.py kitties tokenMetadata

    If you want to figure out exactly what happens, pprint_trace(trace) and pprint_repr(trace)
    are your friend. Put them in various places in simplify_trace and and see how the code changes.

"""

"""

    Pro tips:

        - rules in simplifier are quite independent. you can disable and move around a lot of the stuff
          and still maintain the mathematical correctedness of the output

        - the worst that will happen is that some edge cases in some contracts won't get simplified, but they
          will still be correct

        - bulk_compare.py is your friend for testing your changes. in Eveem I use something like it for initial
          tests and then do a huge integration comparison before each release, checking thousands of contracts for
          changes

        - if you like this approach, be sure to check out the work of people from Kestrel Institute, and their use
          of ACL2. They built a decompiler with a similar structure, but with simplification rules formally proven(!).

"""


def simplify_trace(trace, timeout=0):
    time_start = time.monotonic()

    def should_quit():
        q = timeout and (time.monotonic() - time_start > timeout)
        if q:
            logger.warning("simplify_trace timed out.")
        return q

    old_trace = None
    count = 0
    while trace != old_trace and count < 40 and not should_quit():
        count += 1

        old_trace = trace

        # fun fact:
        # you can do prettify.pprint_trace(trace) or prettify.pprint_repr(trace)
        # between every stage here to see changes that happen to the code.

        trace = replace_f(trace, simplify_exp)
        explain("simplify expressions", trace)

        trace = cleanup_vars(trace)
        explain("cleanup variables", trace)

        trace = cleanup_mems(trace)
        explain("cleanup mems", trace)

        trace = rewrite_trace(trace, split_setmem)
        trace = rewrite_trace_full(trace, split_store)
        explain("split setmems & storages", trace)

        trace = cleanup_vars(trace)
        explain("cleanup vars", trace)

        trace = replace_f(trace, simplify_exp)
        explain("simplify expressions", trace)

        trace = cleanup_mul_1(trace)
        explain("simplify expressions", trace)

        trace = cleanup_msize(trace)
        explain("calculate msize", trace)

        trace = replace_bytes_or_string_length(trace)
        explain("replace storage with length", trace)

        trace = cleanup_conds(trace)
        explain("cleanup unused ifs", trace)

        trace = rewrite_trace(trace, loop_to_setmem)
        explain("convert loops to setmems", trace)

        trace = propagate_storage_in_loops(trace)
        explain("move loop indexes outside of loops", trace)

        # there is a logic to this ordering, but it would take a long
        # time to explain. if you play with it, just run through bulk_compare.py
        # and see how it affects the code.

    # final lightweight postprocessing
    # introduces new variables, simplifies code for human readability
    # and does other stuff that would break the above loop

    trace = replace_f(trace, max_to_add)

    trace = replace_f(trace, postprocess_exp)
    trace = replace_f(trace, postprocess_exp)
    trace = rewrite_trace_ifs(trace, postprocess_trace)

    trace = rewrite_string_stores(trace)
    explain("using heuristics to clean up some things", trace)

    if not should_quit():
        # the most expensive pass, skipped if we're already out of time
        for _ in range(3):
            trace = cleanup_mems(trace)

    trace = cleanup_conds(trace)
    explain("final setmem/condition cleanup", trace)

    def fix_storages(exp):
        # a field moved left (see algebra.apply_mask_to_storage): the field,
        # shifted
        if (m := match(exp, ("storage", ":size", ":int:off", ":loc"))) and m.off < 0:
            return ("mask_shl", m.size, 0, -m.off, ("storage", m.size, 0, m.loc))
        return exp

    trace = replace_f(trace, fix_storages)
    trace = cleanup_conds(trace)
    explain("cleaning up storages slightly", trace)

    trace = readability(trace)
    explain("adding nicer variable names", trace)

    trace = cleanup_mul_1(trace)

    return trace


# The operands that are bytes (see memloc, "bytes"): the rest are numbers.
BYTES_OPERANDS = {
    "data": None,  # all of them
    "arr": range(2, 2**32),  # after the length
    "sha3": None,
    "return": (1,),
    "revert": (1,),
    "log": (1,),
    "call": (4, 5),
    "staticcall": (4, 5),
    "callcode": (4, 5),
    "delegatecall": (3, 4),
    "create": (2,),
    "create2": (2,),
    "precompiled": (3,),
}


def unwrap_bytes(exp):
    """exp, with the operands that are numbers not wrapped in "bytes"."""
    op = opcode(exp)
    if op == "bytes":
        return exp

    keep = BYTES_OPERANDS.get(op, ())
    if keep is None:
        return exp

    if not any(opcode(e) == "bytes" for e in exp[1:]):
        return exp

    return (op,) + tuple(
        e[2] if opcode(e) == "bytes" and idx + 1 not in keep else e
        for idx, e in enumerate(exp[1:])
    )


def simplify_bytes(exp):
    _, size, val = exp
    val = simplify_exp(val)

    if (m := match(val, ("bytes", ":inner_size", ":inner"))) and safe_le_op(
        m.inner_size, size
    ) is True:
        val = m.inner

    if val != 0 and sub_op(sizeof(val), bits(size)) == 0:
        return val

    return ("bytes", size, val)


def data_elements(res):
    """
    The elements of a data without "bytes" when they're as wide as their
    value, zeroes followed by a value merged into it, and zeroes into one.
    """
    res = [
        e[2] if opcode(e) == "bytes" and sub_op(sizeof(e[2]), bits(e[1])) == 0 else e
        for e in res
    ]

    def zeroes(e):
        # the width of e if it's zeroes, None otherwise
        if e == 0:
            return 256
        if (m := match(e, ("bytes", ":size", 0))) and type(m.size) is int:
            return bits(m.size)
        return None

    merged = []
    for e in res:
        if merged and (z := zeroes(merged[-1])) is not None:
            w = sizeof(e)
            if zeroes(e) is not None:
                merged[-1] = ("bytes", (z + zeroes(e)) // 8, 0)
                continue
            if type(w) is int and w % 8 == 0 and opcode(merged[-1]) == "bytes":
                inner = e[2] if opcode(e) == "bytes" else e
                merged[-1] = ("bytes", (z + w) // 8, inner)
                continue
        merged.append(e)

    return [
        e[2] if opcode(e) == "bytes" and sub_op(sizeof(e[2]), bits(e[1])) == 0 else e
        for e in merged
    ]


@cached
def simplify_exp(exp):
    if type(exp) == list:
        return exp

    if type(exp) is tuple:
        exp = unwrap_bytes(exp)

    if opcode(exp) == "bytes":
        return simplify_bytes(exp)

    if m := match(exp, ("shr", ":int:off", ":val")):
        # a shift by an amount the vm didn't know, known now
        exp = shr_op(m.val, m.off)

    if m := match(exp, ("signextend", ":b", ":val")):
        exp = signextend_op(m.b, m.val)

    if match(exp, ("mask_shl", Any, Any, Any, 0)):
        # whatever its size and offset
        return 0

    if opcode(exp) == "and":
        _, *terms = exp
        real = 2**256 - 1
        symbols = []
        for t in terms:
            if type(t) == int and t >= 0:
                real = real & t
            elif opcode(t) == "and":
                symbols += t[1:]
            else:
                symbols.append(t)

        if real != 2**256 - 1:
            res = (real,)
        else:
            res = tuple()

        res += tuple(symbols)
        exp = ("and",) + res

    if m := match(exp, ("iszero", ("iszero", ":e"))):
        exp = ("bool", m.e)

    if m := match(exp, ("bool", ("bool", ":e"))):
        exp = ("bool", m.e)

    if (m := match(exp, ("eq", ":sth", 0))) or (m := match(exp, ("eq", 0, ":sth"))):
        exp = ("iszero", m.sth)

    if (
        (m := match(exp, ("mask_shl", ":int:size", 5, 0, ("add", ":int:num", ...))))
        and m.size > 240
        and m.num % 32 == 31
        and m.num > 32
    ):
        # floor32(num + x) = (num - 31) + floor32(31 + x), num - 31 being
        # a multiple of 32
        add_terms = exp[-1][2:]  # the "..."
        exp = (
            "add",
            m.num - 31,
            (
                "mask_shl",
                256,
                5,
                0,
                (
                    "add",
                    31,
                )
                + add_terms,
            ),
        )

    if m := match(exp, ("iszero", ("mask_shl", ":size", ":off", ":shl", ":val"))):
        exp = ("iszero", ("mask_shl", m.size, m.off, 0, m.val))

    if m := match(exp, ("max", ":single")):
        exp = m.single

    if m := match(exp, ("mem", ("range", Any, 0))):
        return None  # sic. this happens usually in params to logs etc, we probably want None here

    if (
        (m := match(exp, ("mod", ":exp2", ":int:num")))
        and (size := to_exp2(m.num))
        and size > 1
    ):
        return mask_op(m.exp2, size=size)

    if m := match(exp, ("mod", 0, Any)):
        exp = 0

    # same thing is added in both expressions ?
    if (m := match(exp, (":op", ("add", ...), ("add", ...)))) and m.op in (
        "lt",
        "le",
        "gt",
        "ge",
    ):
        e1 = exp[1][1:]  # first ...
        e2 = exp[2][1:]  # second ...
        t1 = tuple(t for t in e1 if t not in e2)
        t2 = tuple(t for t in e2 if t not in e1)
        exp = (m.op, add_op(*t1), add_op(*t2))

    if m := match(exp, ("add", ":e")):
        logger.debug("single add")
        return simplify_exp(m.e)

    if m := match(exp, ("mul", 1, ":e")):
        return simplify_exp(m.e)

    if m := match(exp, ("div", ":e", 1)):
        return simplify_exp(m.e)

    if m := match(exp, ("mask_shl", 256, 0, 0, ":val")):
        return simplify_exp(m.val)

    if m := match(exp, ("mask_shl", ":int:size", ":int:offset", ":int:shl", ":e")):
        exp = mask_op(simplify_exp(m.e), m.size, m.offset, m.shl)

    if m := match(
        exp, ("mask_shl", ":size", 0, 0, ("div", ":expr", ("exp", 256, ":shr")))
    ):
        shift = bits(m.shr)
        exp = mask_op(simplify_exp(m.expr), m.size, shift, shr=shift)

    if (
        m := match(exp, ("mask_shl", Any, Any, ":shl", ("storage", ":size", Any, Any)))
    ) and safe_le_op(m.size, minus_op(m.shl)):
        return 0

    if m := match(exp, ("or", ":sth", 0)):
        return m.sth

    if opcode(exp) == "add":
        terms = exp[1:]
        res = 0
        for el in terms:
            el = simplify_exp(el)
            if opcode(el) == "add":
                for e in el[1:]:
                    res = add_op(res, e)
            else:
                res = add_op(res, el)
        exp = res

    if opcode(exp) == "mask_shl":
        exp = cleanup_mask_data(exp)

    if m := match(
        exp, ("mask_shl", ":size", 0, 0, ("mem", ("range", ":mem_loc", ":mem_size")))
    ):
        if divisible_bytes(m.size) and safe_le_op(to_bytes(m.size)[0], m.mem_size):
            return (
                "mem",
                apply_mask_to_range(("range", m.mem_loc, m.mem_size), m.size, 0),
            )

    if (
        m := match(
            exp,
            (
                "mask_shl",
                ":size",
                ":off",
                ":shl",
                ("mem", ("range", ":mem_loc", ":mem_size")),
            ),
        )
    ) and m.shl == minus_op(m.off):
        if (
            divisible_bytes(m.size)
            and safe_le_op(to_bytes(m.size)[0], m.mem_size)
            and divisible_bytes(m.off)
        ):
            return (
                "mem",
                apply_mask_to_range(("range", m.mem_loc, m.mem_size), m.size, m.off),
            )

    elif opcode(exp) == "data":
        params = exp[1:]
        res = []

        # simplify inner expressions, and remove nested 'data's
        for e in params:
            e = simplify_exp(
                e
            )  # removes further nested datas, and does other simplifications
            if opcode(e) == "data":
                res.extend(e[1:])
            else:
                res.append(e)

        # sometimes we may have on expression split into two masks next to each other
        # e.g.  (mask_shl 192 64 -64 (cd 36)))) (mask_shl 64 0 0 (cd 36))
        #       (in solidstamp withdrawRequest)
        # merge those into one.

        res2 = res
        res = None
        while (
            res2 != res
        ):  # every swipe merges up to two elements next to each other. repeat until there are no new merges
            # there's a more efficient way of doing this for sure.
            res, res2 = res2, []
            idx = 0
            while idx < len(res):
                el = res[idx]
                if idx == len(res) - 1:
                    res2.append(el)
                    break

                next_el = res[idx + 1]
                idx += 1

                if (
                    (m := match(el, ("mask_shl", ":size", ":offset", ":shl", ":val")))
                    and next_el == ("mask_shl", m.offset, 0, 0, m.val)
                    and add_op(m.offset, m.shl) == 0
                ):
                    res2.append(("mask_shl", add_op(m.size, m.offset), 0, 0, m.val))
                    idx += 1

                else:
                    res2.append(el)

        res = res2

        # could do the same for mem slices, but no case of that happening yet

        res = data_elements(res)

        if len(res) == 1:
            return res[0]
        else:
            return ("data",) + tuple(res)

    if m := match(
        exp, ("mul", -1, ("mask_shl", ":size", ":offset", ":shl", ("mul", -1, ":val")))
    ):
        return (
            "mask_shl",
            simplify_exp(m.size),
            simplify_exp(m.offset),
            simplify_exp(m.shl),
            simplify_exp(m.val),
        )

    if type(exp) == int and to_real_int(exp) > -(
        8**22
    ):  # if it's larger than 30 bytes, it's probably
        # an address, not a negative number
        return to_real_int(exp)

    if m := match(exp, ("and", ":num", ":num2")):
        num = arithmetic.eval(m.num)
        num2 = arithmetic.eval(m.num2)
        if type(num) == int or type(num2) == int:
            return simplify_mask(exp)

    if type(exp) == list:
        r = simplify_exp(tuple(exp))
        return list(r)

    if type(exp) != tuple:
        return exp

    if m := match(
        exp, ("mask_shl", ":int:size", ":int:offset", ":int:shl", ":int:val")
    ):
        return apply_mask(m.val, m.size, m.offset, m.shl)

    if (m := match(exp, ("mask_shl", ":size", 5, ":shl", ("add", 31, ":val")))) and (
        match(m.val, ("mask_shl", 251, 0, 5, Any))
    ):
        # the val is a multiple of 32: adding 31 doesn't change its bits above
        # the 5 lowest (ceil32 of a multiple of 32 is the number)
        return simplify_exp(("mask_shl", m.size, 5, m.shl, m.val))

    if opcode(exp) == "mul":
        terms = exp[1:]
        res = 1
        for e in terms:
            res = mul_op(res, simplify_exp(e))

        if m := match(res, ("mul", 1, ":res")):
            res = m.res

        return res

    if opcode(exp) == "max":
        terms = exp[1:]
        els = [simplify_exp(e) for e in terms]
        res = 0
        for e in els:
            res = _max_op(res, e)
        return res

    res = tuple()
    for e in exp:
        res += (simplify_exp(e),)

    return res


def simplify_mask(exp):
    op = opcode(exp)

    if op in arithmetic.OPCODES:
        exp = arithmetic.eval(exp)

    if m := match(exp, ("and", ":left", ":right")):
        left, right = m.left, m.right

        if mask := to_mask(left):
            exp = mask_op(right, *mask)

        elif mask := to_mask(right):
            exp = mask_op(left, *mask)

        elif bounds := to_neg_mask(left):
            exp = neg_mask_op(right, *bounds)

        elif bounds := to_neg_mask(right):
            exp = neg_mask_op(left, *bounds)

    elif (m := match(exp, ("div", ":left", ":int:right"))) and (
        shift := to_exp2(m.right)
    ):
        exp = mask_op(m.left, size=256 - shift, offset=shift, shr=shift)

    elif (m := match(exp, ("mul", ":int:left", ":right"))) and (
        shift := to_exp2(m.left)
    ):
        exp = mask_op(m.right, size=256 - shift, shl=shift)

    elif (m := match(exp, ("mul", ":left", ":int:right"))) and (
        shift := to_exp2(m.right)
    ):
        exp = mask_op(m.left, size=256 - shift, shl=shift)

    return exp


def cleanup_mask_data(exp):
    # if there is a mask over some data,
    # it removes pieces of data that for sure won't
    # fit into the mask (doesn't truncate if something)
    # partially fits
    # e.g. mask(200, 8, 8 data(123, mask(201, 0, 0, sth), mask(8,0, 0, sth_else)))
    #      -> mask(200, 0, 0, mask(201, 0, 0, sth))

    def _cleanup_right(exp):
        # removes elements that are cut off by offset
        m = match(exp, ("mask_shl", ":size", ":offset", ":shl", ":val"))
        assert m
        size, offset, shl, val = m.size, m.offset, m.shl, m.val

        if opcode(val) != "data":
            return exp

        last = val[-1]
        if sizeof(last) is not None and safe_le_op(sizeof(last), offset):
            offset = sub_op(offset, sizeof(last))
            shl = add_op(shl, sizeof(last))
            if len(val) == 3:
                val = val[1]
            else:
                val = val[:-1]

            return mask_op(val, size, offset, shl)

        return exp

    def _cleanup_left(exp):
        # removes elements that are cut off by size+offset
        m = match(exp, ("mask_shl", ":size", ":offset", ":shl", ":val"))
        assert m
        size, offset, shl, val = m.size, m.offset, m.shl, m.val

        if opcode(val) != "data":
            return exp

        total_size = add_op(size, offset)  # simplify_exp

        sum_sizes = 0
        val = list(val[1:])
        res = []
        while len(val) > 0:
            last = val.pop()
            if sizeof(last) is None:
                return exp
            sum_sizes = simplify_exp(add_op(sum_sizes, sizeof(last)))
            res.insert(0, last)
            if safe_le_op(total_size, sum_sizes):
                return exp[:4] + (("data",) + tuple(res),)

        return exp

    assert opcode(exp) == "mask_shl"

    prev_exp = None
    while prev_exp != exp:
        prev_exp = exp
        exp = _cleanup_right(exp)

    prev_exp = None
    while prev_exp != exp:
        prev_exp = exp
        exp = _cleanup_left(exp)

    if m := match(exp, ("mask_shl", ":size", 0, 0, ("data", ...))):
        size = m.size
        terms = exp[-1][1:]  # the "..."
        # if size of data is size of mask, we can remove the mask altogether
        sum_sizes = 0
        for e in terms:
            s = sizeof(e)
            if s is None:
                return exp
            sum_sizes = add_op(sum_sizes, s)

        if sub_op(sum_sizes, size) == 0:
            return ("data",) + terms

    return exp


def replace_while_var(rest, counter_idx, new_idx):
    while contains(rest, ("var", new_idx)):
        new_idx += 1

    def r(exp):
        if exp == ("var", counter_idx):
            return ("var", new_idx)

        elif m := match(exp, ("setvar", counter_idx, ":val")):
            return ("setvar", new_idx, m.val)

        else:
            return exp

    return simplify_exp(replace_f(rest, r)), new_idx


def canonise_max(exp):
    if opcode(exp) == "max":
        args = []
        for e in exp[1:]:
            if m := match(e, ("mul", 1, ":num")):
                args.append(m.num)
            else:
                args.append(e)

        args.sort(key=lambda x: str(x) if type(x) != int else " " + str(x))
        return ("max",) + tuple(args)
    else:
        return exp


assert canonise_max(("max", ("mul", 1, ("x", "y")), 4)) == ("max", 4, ("x", "y"))


def readability(trace):
    """
    - replaces variable names with nicer ones,
    - fixes empty memory in calls
    - replaces 'max..' in setmems with msize variable
        (max can only appear because of this)
    """

    trace = replace_f(trace, canonise_max)

    res = []
    for idx, line in enumerate(trace):
        if m := match(line, ("setmem", ("range", ("add", ...), Any), ":mem_val")):
            add_params = line[1][1][1:]  # the "..."
            for m in add_params:
                if opcode(m) == "max":
                    res.append(("setvar", "_msize", m))

                    def x(line):
                        return [replace(line, m, ("var", "_msize"))]

                    rest = rewrite_trace(trace[idx:], x)
                    res.extend(readability(rest))
                    return res

        elif m := match(line, ("if", ":cond", ":if_true", ":if_false")):
            cond, if_true, if_false = m.cond, m.if_true, m.if_false

            # if if_false ~ [('revert', ...)]: # no lists in Tilde... yet :,)
            if len(if_false) == 1 and opcode(if_false[0]) == "revert":
                res.append(
                    ("if", is_zero(cond), readability(if_false), readability(if_true))
                )
            else:
                res.append(("if", cond, readability(if_true), readability(if_false)))

            continue

        elif opcode(line) == "while":
            # for whiles, normalize variable names

            a = parse_counters(line)

            rest = trace[idx:]

            if "counter" in a:
                counter_idx = a["counter"]
                rest, _ = replace_while_var(rest, counter_idx, 0)

            else:
                counter_idx = -1

            new_idx = 1

            cond, path, jds, vars = line[1:]

            for _, v_idx, _ in vars:
                if v_idx != counter_idx:
                    rest, new_idx = replace_while_var(rest, v_idx, new_idx)

            line, rest = rest[0], rest[1:]
            cond, path, jds, vars = line[1:]

            path = readability(path)
            res.append(("while", cond, path, jds, vars))

            res.extend(readability(rest))
            return res

        res.append(line)

    return res


def replace_bytes_or_string_length(trace):
    # see unicorn contract, version/name

    def replace(expr):
        m = match(
            expr,
            (
                "mask_shl",
                ":size",
                ":offset",
                -1,
                (
                    "and",
                    ("storage", Any, 0, ":key"),
                    (
                        "add",
                        -1,
                        (
                            "mask_shl",
                            Any,
                            Any,
                            Any,
                            ("iszero", ("storage", Any, 0, ":key")),
                        ),
                    ),
                ),
            ),
        ) or match(
            expr,
            (
                "mask_shl",
                ":size",
                ":offset",
                -1,
                (
                    "and",
                    (
                        "add",
                        -1,
                        (
                            "mask_shl",
                            Any,
                            Any,
                            Any,
                            ("iszero", ("storage", Any, 0, ":key")),
                        ),
                    ),
                    ("storage", Any, 0, ":key"),
                ),
            ),
        )
        if not m:
            return
        key, size, offset = m.key, m.size, m.offset

        if type(key) == int:
            key = ("loc", key)

        if size == 255 and offset == 1:
            return ("storage", 256, 0, ("length", key))
        assert offset >= 1
        return ("mask_shl", size, offset - 1, 0, ("storage", 256, 0, ("length", key)))

    return replace_f_stop(trace, replace)


def loop_to_setmem(line):
    if opcode(line) == "while":
        r = _loop_to_setmem(line)

        if r is not None:
            return r

        r = loop_to_setmem_from_storage(line)

        if r is not None:
            return r

    return [line]


def vars_in_expr(expr):
    if m := match(expr, ("var", ":var_id")):
        return frozenset([m.var_id])

    s = frozenset()

    if type(expr) not in (tuple, list):
        return s

    for e in expr:
        s = s | vars_in_expr(e)
    return s


def only_add_in_expr(op):
    if m := match(op, ("setvar", ":idx", ":val")):
        return only_add_in_expr(m.val)
    if opcode(op) == "add":
        terms = op[1:]
        return all(only_add_in_expr(o) for o in terms)
    if match(op, ("var", Any)):
        return True
    if m := match(op, ("sha3", ":term")):  # sha3(constant) is allowed.
        return opcode(m.term) is None
    if opcode(op) is not None:
        return False
    return True


assert only_add_in_expr(("setvar", 100, ("mul", ("var", 100), 1))) is False
assert only_add_in_expr(("setvar", 100, ("add", ("var", 100), 1))) is True


def propagate_storage_in_loop(line):
    op, cond, path, jds, setvars = line
    assert op == "while"

    def storage_sha3(value):
        if opcode(value) == "add":
            terms = value[1:]
            for op in terms:
                if storage_sha3(op) is not None:
                    return storage_sha3(op)

        if m := match(value, ("sha3", ":val")):
            if type(m.val) != int or m.val < 1000:  # used to be int:val here, why?
                return value

    def path_only_add_in_continue(path):
        for op in path:
            if opcode(op) == "continue":
                _, _, instrs = op
                if any(not only_add_in_expr(instr) for instr in instrs):
                    return False
        return True

    new_setvars = []

    for setvar in setvars:
        op, var_id, value = setvar
        assert op == "setvar"

        sha3 = storage_sha3(value)
        if not sha3:
            new_setvars.append(setvar)
            continue

        # If the "continue" instructions don't only add stuff to the index variable,
        # it's not safe to proceed. If we would do "i = i * 2", then it doesn't
        # make sense to substract a constant to "i".
        if not path_only_add_in_continue(path):
            new_setvars.append(setvar)
            continue

        new_setvars.append(("setvar", var_id, sub_op(value, sha3)))

        def add_sha3(t):
            # We replace occurrences of var by "var + sha3"
            if t == ("var", var_id):
                return add_op(t, sha3)
            # Important: for "continue" we don't want to touch the variable.
            # TODO: This is only valid if the "continue" contains only
            # operators like "+" or "-". We should check that.
            if opcode(t) == "continue":
                return t

        path = replace_f_stop(path, add_sha3)
        cond = replace_f_stop(cond, add_sha3)

    return [("while", cond, path, jds, new_setvars)]


def propagate_storage_in_loops(trace):
    def touch(line):
        if opcode(line) == "while":
            r = propagate_storage_in_loop(line)
            if r is not None:
                return r

        return [line]

    return rewrite_trace(trace, touch)


def _loop_to_setmem(line):
    def memidx_to_memrange(mem_idx, setvars, stepvars, endvars):
        mem_idx_next = mem_idx
        for v in stepvars:
            op, v_idx, v_val = v
            assert op == "setvar"
            mem_idx_next = replace(mem_idx_next, ("var", v_idx), v_val)

        diff = sub_op(mem_idx_next, mem_idx)

        if diff not in (32, -32):
            return None, None

        mem_idx_last = mem_idx
        for v_idx, v_val in endvars.items():
            mem_idx_last = replace(mem_idx_last, ("var", v_idx), v_val)

        mem_idx_first = mem_idx
        for v in setvars:
            op, v_idx, v_val = v
            assert op == "setvar"

            mem_idx_first = replace(mem_idx_first, ("var", v_idx), v_val)

        if diff == 32:
            mem_len = sub_op(mem_idx_last, mem_idx_first)

            return ("range", mem_idx_first, mem_len), diff

        else:
            assert diff == -32

            mem_idx_last = add_op(32, mem_idx_last)
            mem_idx_first = add_op(32, mem_idx_first)

            mem_len = sub_op(mem_idx_first, mem_idx_last)

            return ("range", mem_idx_last, mem_len), diff

    op, cond, path, jds, setvars = line
    assert op == "while"

    if len(path) != 2:
        return None

    if opcode(path[1]) != "continue":
        return None

    if opcode(path[0]) != "setmem":
        return None

    setmem = path[0]
    cont = path[1]

    mem_idx, mem_val = setmem[1], setmem[2]

    op, i, l = mem_idx
    assert op == "range"
    if l != 32:
        return None
    mem_idx = i

    stepvars = cont[2]

    a = parse_counters(line)
    if "endvars" not in a:
        return None

    setvars, endvars = a["setvars"], a["endvars"]

    rng, diff = memidx_to_memrange(mem_idx, setvars, stepvars, endvars)

    if rng is None:
        return None

    if mem_val == 0:
        res = [("setmem", rng, 0)]

    elif opcode(mem_val) == "mem":
        mem_val_idx = mem_val[1]
        op, i, l = mem_val_idx
        assert op == "range"
        mem_val_idx = i
        if l != 32:
            return None

        val_rng, val_diff = memidx_to_memrange(mem_val_idx, setvars, stepvars, endvars)

        if val_rng is None:
            return None

        if val_diff != diff:
            return None  # possible but unsupported

        # we should check for overwrites here, but skipping for now
        # if the part of memcopy loop overwrites source before it's copied,
        # we can end up with unexpected behaviour, could at least show some warning,
        # or set that mem to 'complicated' or sth

        res = [("setmem", rng, ("mem", val_rng))]

    else:
        return None

    for v_idx, v_val in endvars.items():
        res.append(("setvar", v_idx, v_val))

    return res


def loop_to_setmem_from_storage(line):
    assert opcode(line) == "while"
    cond, path, jds, setvars = line[1:]

    if len(path) != 2 or opcode(path[0]) != "setmem" or opcode(path[1]) != "continue":
        return None

    setmem = path[0]
    cont = path[1]

    logger.debug("loop_to setmem_from_storage: %s\n%s\n%s", setmem, cont, cond)

    # (setmem, mem_idx, mem_val)
    mem_idx, mem_val = setmem[1], setmem[2]

    # Extract the interesting variable from mem_idx
    vars_in_idx = vars_in_expr(mem_idx)
    if len(vars_in_idx) != 1:
        return
    if not only_add_in_expr(mem_idx):
        return
    memory_index_var = next(iter(vars_in_idx))

    # Same from mem_val
    vars_in_val = vars_in_expr(mem_val)
    if len(vars_in_val) != 1:
        return
    storage_key_var = next(iter(vars_in_val))
    if opcode(mem_val) != "storage":
        return
    if mem_val[1] != 256 or mem_val[2] != 0:
        return
    storage_key = mem_val[3]
    if not only_add_in_expr(storage_key):
        return

    logger.debug("now look at the continue")
    update_memory_index = (
        "setvar",
        memory_index_var,
        ("add", 32, ("var", memory_index_var)),
    )
    update_storage_key = (
        "setvar",
        storage_key_var,
        ("add", 1, ("var", storage_key_var)),
    )
    if set(cont[2]) != {update_memory_index, update_storage_key}:
        return

    logger.debug("setvars")
    memory_index_start = None
    storage_key_start = None
    memory_index_init = None
    for setvar in setvars:
        if setvar[1] == memory_index_var:
            memory_index_start = replace(mem_idx, ("var", memory_index_var), setvar[2])
            memory_index_init = setvar[2]
        elif setvar[1] == storage_key_var:
            storage_key_start = replace(
                storage_key, ("var", storage_key_var), setvar[2]
            )
        else:
            return

    logger.debug("while condition")
    if memory_index_var not in vars_in_expr(cond):
        return
    if opcode(cond) != "gt":
        return

    mem_count = ("add", cond[1], ("mul", -1, cond[2]))
    mem_count = replace(mem_count, ("var", memory_index_var), memory_index_init)

    mem_rng = ("range", memory_index_start, mem_count)
    storage_rng = ("range", storage_key_start, ("div", mem_count, 32))

    logger.debug("mem_rng: %s, storage_rng: %s", mem_rng, storage_rng)
    return [("setmem", mem_rng, ("storage", 256, 0, storage_rng))]


"""

    simplifier

"""


def apply_constraint(exp, constr):
    # for constraints like "isZero XX % 32", applies them to expression

    return exp

    if match(constr, ("mask_shl", 5, 0, 0, Any)):

        def f(x):
            if m := match(
                x, ("mask_shl", ":int:size", 5, ":int:shl", ("add", 31, ":val"))
            ):
                return ("add", 32 * (2**m.shl), ("mask_shl", m.size, 5, m.shl, m.val))
            if m := match(x, ("mask_shl", ":int:size", 5, 0, ":val")):
                return ("mask_shl", m.size, 5, 0, m.val)

            return x

        return replace_f(exp, f)

    if match(constr, ("iszero", ("mask_shl", 5, 0, 0, Any))):

        def f(x):
            if m := match(x, ("mask_shl", ":int:size", 5, 0, ("add", 31, ":val"))):
                return ("mask_shl", m.size + 5, 0, 0, m.val)
            if m := match(x, ("mask_shl", ":int:size", 5, 0, ":val")):
                return ("mask_shl", m.size + 5, 0, 0, m.val)
            if match(x, ("mask_shl", 5, 0, 0, ":val")):
                return 0

            return x

        return replace_f(exp, f)

    return exp


def cleanup_conds(trace):
    """

    removes ifs/whiles with conditions that are obviously true
    and replace variables that need to be equal to a constant by that constant

    """

    res = []

    for line in trace:
        if opcode(line) == "while":
            _, cond, path, jds, setvars = line
            # we're not evaluating symbolically, otherwise stuff like
            # stor0 <= stor0 + 1 gets evaluated to `True` - this happens
            # because we're truncating mask256. it should really be
            # mask(256, stor0) <= mask(256, stor0 + 1)
            # which is not always true
            # see 0x014B50466590340D41307Cc54DCee990c8D58aa8.transferFrom
            path = cleanup_conds(path)
            ev = arithmetic.eval_bool(cond, symbolic=False)
            if ev is True:
                res.append(("while", ("bool", 1), path, jds, setvars))
            elif ev is False:
                pass  # removing loop altogether
            else:
                res.append(("while", cond, path, jds, setvars))

        elif opcode(line) == "if":
            _, cond, if_true, if_false = line
            if_true = cleanup_conds(if_true)
            if_false = cleanup_conds(if_false)

            # If the condition is always true/false, remove the if.
            ev = arithmetic.eval_bool(cond, symbolic=False)
            if ev is True:
                res.extend(if_true)
            elif ev is False:
                res.extend(if_false)
            elif not if_true and not if_false:
                # can happen once the branches of a merged if got simplified
                pass
            elif not if_true:
                res.append(("if", is_zero(cond), if_false, []))
            else:
                res.append(("if", cond, if_true, if_false))

        else:
            res.append(line)

    return res


@cached
def find_mems(exp):
    def f(exp):
        if opcode(exp) == "mem":
            return set([exp])
        else:
            return set()

    return find_f_set(exp, f)


test_e = ("x", "sth", ("mem", 4), ("t", ("mem", 4), ("mem", 8), ("mem", ("mem", 64))))
assert find_mems(test_e) == {
    ("mem", 64),
    ("mem", ("mem", 64)),
    ("mem", 4),
    ("mem", 8),
}, find_mems(test_e)


def _eval_msize(cond):
    if opcode(cond) not in ("lt", "le", "gt", "ge"):
        return None

    left, right = cond[1], cond[2]

    if opcode(left) != "max" and opcode(right) != "max":
        return None

    if opcode(left) == "max" and opcode(right) == "max":
        return None

    if opcode(right) == "max":
        cond = swap_cond(cond)
        left, right = cond[1], cond[2]

    assert opcode(left) == "max"

    if opcode(cond) in ("lt", "le"):
        if opcode(cond) == "le":
            cond = ("lt", left, add_op(1, right))
            left, right = cond[1], cond[2]

        # max(2,3) <= 3
        # max(2,3) < 4

        # cond == (lt, max(....), right)
        # any .... > right -> True
        # any .... ? right -> ?
        # else -> all ... < right -> False

        if all([safe_lt_op(l, right) is True for l in left[1:]]):
            return False

        if any([safe_lt_op(right, l) is False for l in left[1:]]):
            return False

        if all([safe_le_op(right, l) is True for l in left[1:]]):
            return True

    if opcode(cond) in ("gt", "ge"):
        assert False, cond  # unsupported yet

    return None


def cleanup_msize(trace, current_msize=0):
    res, _ = _cleanup_msize(trace, current_msize)
    return res


def _cleanup_msize(trace, current_msize=0):
    """Returns the cleaned up trace, and the msize at the end of it."""
    res = []

    for line in trace:
        if opcode(line) == "setmem":
            line = replace(line, "msize", current_msize)

            mem_right = memloc_right(line)

            current_msize = _max_op(current_msize, mem_right)

            res.append(line)

        elif opcode(line) == "while":
            new_one = while_max_memidx(line)
            current_msize = _max_op(current_msize, new_one)
            res.append(line)

        elif opcode(line) == "if":
            cond, if_true, if_false = line[1:]
            if "msize" in str(cond) and opcode(current_msize) == "max":
                tmp_cond = replace(cond, "msize", current_msize)

                tmp_evald = _eval_msize(tmp_cond)

                if tmp_evald is not None:
                    cond = 1 if tmp_evald is True else 0

            else:
                new_msize = max_to_add(current_msize)
                cond = replace(cond, "msize", new_msize)

            if_true, msize_true = _cleanup_msize(if_true, current_msize)
            if_false, msize_false = _cleanup_msize(if_false, current_msize)
            res.append(("if", cond, if_true, if_false))

            # for what follows the if, when its branches merge again
            if msize_true == msize_false:
                current_msize = msize_true
            else:
                # we could take the max of both, but that gets expensive
                # quickly, and msize isn't used by any recent compiler.
                current_msize = "msize"

        else:
            line = replace(line, "msize", current_msize)
            res.append(line)

    #    print('done')
    return res, current_msize


def overwrites_mem(line, mem_idx):
    """
    for a given line, returns True if it potentially
    overwrites *any part* of memory index, False if it *for sure* doesn't

    """
    if m := match(line, ("setmem", ":set_idx", Any)):
        return range_overlaps(m.set_idx, mem_idx) is not False

    if opcode(line) == "while":
        return while_touches_mem(line, mem_idx)

    if opcode(line) == "if":
        # matters for what comes after the if, when its branches merge again
        return any(overwrites_mem(l, mem_idx) for l in line[2] + line[3])

    return False


def changes_reads(line, exp):
    """
    True if the line may change something that exp reads from the state (see
    arithmetic.changed_by): the storage it writes, what a call can change...
    """
    op = opcode(line)
    if op == "store":
        return bool(changed_reads(exp, "store", line[3]))
    if op == "tstore":
        return bool(changed_reads(exp, "tstore", line[1]))
    if op in (
        "call",
        "staticcall",
        "delegatecall",
        "callcode",
        "codecall",
        "create",
        "create2",
    ):
        return bool(changed_reads(exp, op))
    if op == "if":
        return any(changes_reads(l, exp) for l in line[2] + line[3])
    if op == "while":
        return any(changes_reads(l, exp) for l in line[2])
    return False


def affects(line, exp):
    if changes_reads(line, exp):
        return True

    if type(exp) != tuple and exp != "msize":
        return False

    s = str(exp)

    if "msize" in s:
        if overwrites_mem(line, ("range", 0, "undefined")):
            return True

    if "mem" not in s:
        return False

    mems = find_mems(exp)

    for m in mems:
        m_idx = m[1]
        if overwrites_mem(line, m_idx):
            return True

    return False


line_test = ("setmem", ("range", 65, 32), "x")
exp_test = ("mul", 8, ("mem", ("range", 64, 32)))
assert affects(line_test, exp_test) == True
exp_test = ("mul", 8, ("mem", ("range", 100, 32)))
assert affects(line_test, exp_test) == False

line_test = ("setmem", ("range", 65, 32), "x")
exp_test = ("mul", 8, ("mem", ("range", 64, 32)))
assert affects(line_test, exp_test) == True
exp_test = ("mul", 8, ("mem", ("range", 100, 32)))
assert affects(line_test, exp_test) == False

line_test = ("setmem", ("range", 65, "sth"), "x")
exp_test = ("mul", 8, ("mem", ("range", 64, 32)))
assert affects(line_test, exp_test) == True
exp_test = ("mul", 8, ("mem", ("range", 100, 32)))
assert affects(line_test, exp_test) == True

line_test = ("setmem", ("range", 65, 32), "x")
exp_test = ("mul", 8, ("mem", ("range", 64, 1)))
assert affects(line_test, exp_test) == False
exp_test = ("mul", 8, ("mem", ("range", 64, "sth")))
assert affects(line_test, exp_test) == True


"""

    Memory cleanup

"""


def trace_uses_mem(trace, mem_idx):
    """

    checks if memory is used anywhere in the trace

    """
    return _mem_use(trace, mem_idx) == USED


USED, OVERWRITTEN, NEITHER = "used", "overwritten", "neither"


def _mem_use(trace, mem_idx):
    """
    USED if the trace reads the memory, OVERWRITTEN if every path through it
    overwrites the memory (or ends the execution) before reading it, NEITHER
    if it does neither and what comes after the trace may still read it.
    """

    for idx, line in enumerate(trace):
        if m := match(line, ("setmem", ":memloc", ":memval")):
            memloc, memval = m.memloc, m.memval
            memval = simplify_exp(memval)

            if exp_uses_mem(memval, mem_idx):
                return USED

            if any(overwrites_mem(line, inner[1]) for inner in find_mems(mem_idx)):
                # where mem_idx is depends on the memory the line writes:
                # it isn't the same memory afterwards
                return USED

            split = memloc_overwrite(
                mem_idx, memloc
            )  # returns range that we're confident wasn't overwritten by memloc
            res2 = trace[idx + 1 :]
            res = OVERWRITTEN
            for s_idx in split:
                r = _mem_use(res2, s_idx)
                if r == USED:
                    return USED
                if r == NEITHER:
                    res = NEITHER

            return res

        elif opcode(line) == "while":
            if while_uses_mem(line, mem_idx):
                return USED

        elif m := match(line, ("if", ":cond", ":if_true", ":if_false")):
            if exp_uses_mem(m.cond, mem_idx):
                return USED

            res = []
            for branch in (m.if_true, m.if_false):
                r = _mem_use(branch, mem_idx)
                if r == USED:
                    return USED
                if r == NEITHER and trace_ends_execution(branch):
                    r = OVERWRITTEN
                res.append(r)

            if res == [OVERWRITTEN, OVERWRITTEN]:
                return OVERWRITTEN

        elif opcode(line) == "continue":
            return USED

        else:
            if exp_uses_mem(line, mem_idx):
                return USED

            if opcode(line) in ENDS_EXECUTION:
                return OVERWRITTEN

    return NEITHER


ENDS_EXECUTION = ("revert", "return", "stop", "invalid", "assert_fail", "selfdestruct")


def trace_ends_execution(trace):
    """True if every path through the trace ends the execution."""
    if len(trace) == 0:
        return False

    last = trace[-1]

    if opcode(last) == "if":
        return trace_ends_execution(last[2]) and trace_ends_execution(last[3])

    return opcode(last) in ENDS_EXECUTION


def cleanup_mems(trace, used_after=None):
    """
    for every setmem, replace future occurences of it with it's value,
    if possible

    `used_after` is what gets executed after `trace`, when `trace` is a
    branch of an if whose branches merge again.

    """

    used_after = used_after or []
    res = []

    for idx, line in enumerate(trace):
        if match(line, ("setmem", ":rng", ("mem", ":rng"))):
            continue

        if opcode(line) in ["call", "staticcall", "delegatecall", "codecall"]:
            fname, fdata = line[-2:]

            if match(fdata, ("mem", ("range", Any, -4))):
                line = line[:-2] + (None, None)

            res.append(line)

        elif m := match(line, ("setmem", ":mem_idx", ":mem_val")):
            mem_idx, mem_val = m.mem_idx, m.mem_val

            # find all the future occurences of var and replace if possible
            if affects(line, mem_val):
                remaining_trace = trace[idx + 1 :]
            else:
                # The following line is the cause for O(N^2) complexity,
                # as here we iterate over the trace, and below we call `replace_mem`
                # over everything that's left in the trace.
                remaining_trace = replace_mem(trace[idx + 1 :], mem_idx, mem_val)

            if trace_uses_mem(remaining_trace + used_after, mem_idx):
                res.append(line)

            res.extend(cleanup_mems(remaining_trace, used_after))

            break

        elif opcode(line) == "while":
            _, cond, path, *rest = line
            path = cleanup_mems(path)
            res.append(
                (
                    "while",
                    cond,
                    path,
                )
                + tuple(rest)
            )

        elif opcode(line) == "if":
            _, cond, if_true, if_false = line
            after = trace[idx + 1 :] + used_after
            if_true = cleanup_mems(if_true, after)
            if_false = cleanup_mems(if_false, after)
            res.append(("if", cond, if_true, if_false))

        else:
            res.append(line)

    return res


@cached
def replace_mem_exp(exp, mem_idx, mem_val):
    if type(exp) != tuple:
        return exp

    res = tuple(
        replace_mem_exp(e, mem_idx, mem_val) if type(e) == tuple else e for e in exp
    )

    if opcode(mem_val) not in ("mem", "var", "data"):
        if m := match(
            res, ("delegatecall", ":gas", ":addr", ("mem", ":func"), ("mem", ":args"))
        ):
            op, f_begin, f_len = m.func
            assert op == "range"
            op, a_begin, a_len = m.args
            assert op == "range"
            if f_len == 4 and sub_op(add_op(f_begin, f_len), a_begin) == 0:
                # we have a situation when inside memory is sth like: (range 96 4) (100 ...)
                # let's merge those two memories, and try to replace with mem exp
                res_range = simplify_exp(("range", f_begin, add_op(f_len, a_len)))
                if res_range == mem_idx:
                    res = ("delegatecall", m.gas, m.addr, None, mem_val)

        if m := match(
            res, ("call", ":gas", ":addr", ":value", ("mem", ":func"), ("mem", ":args"))
        ):
            op, f_begin, f_len = m.func
            assert op == "range"
            op, a_begin, a_len = m.args
            assert op == "range"
            if f_len == 4 and sub_op(add_op(f_begin, f_len), a_begin) == 0:
                # we have a situation when inside memory is sth like: (range 96 4) (100 ...)
                # let's merge those two memories, and try to replace with mem exp
                res_range = simplify_exp(("range", f_begin, add_op(f_len, a_len)))
                if res_range == mem_idx:
                    res = ("call", m.gas, m.addr, m.value, None, mem_val)

    if res != exp:
        res = simplify_exp(res)

    if opcode(res) == "mem":
        assert len(res) == 2
        res = fill_mem(res, mem_idx, mem_val)

    return res


def replace_mem(trace, mem_idx, mem_val):
    """

    replaces any reference to mem_idx in the trace
    with a value of mem_val, up until a point of that mem being
    overwritten

    mem[64] = 'X'
    log mem[64]
    mem[65] = 'Y'
    log mem[64 len 1]
    log mem[65]
    mem[63] = 'Z'
    ...

    into

    mem[64] = 'X'
    log 'X'
    mem[65] = 'Y'
    log mask(1, 'X')
    log mem[65]
    ... (the rest unchanged)

    """
    mem_idx = simplify_exp(mem_idx)
    mem_val = simplify_exp(mem_val)
    mem_id = ("mem", mem_idx)

    if type(mem_val) is tuple and opcode(mem_val) != "mem":
        mem_val = arithmetic.eval(mem_val)

    res = []

    for idx, line in enumerate(trace):
        if m := match(line, ("setmem", ":memloc", Any)):
            # replace in val, and in where it's written - that is known
            # before the line changes the memory
            line = replace_mem_exp(line, mem_idx, mem_val)
            res.append(line)
            memloc = simplify_exp(line[1])
            # (None: it may overlap)
            if range_overlaps(memloc, mem_idx) is not False:
                split = splits_mem(mem_idx, memloc, mem_val)
                res2 = trace[idx + 1 :]
                for s in split:
                    res2 = replace_mem(res2, s[0], s[1])

                res.extend(res2)
                return res
            if affects(line, mem_val) or affects(line, mem_id):
                # (mem_id: where mem_idx is may depend on the memory the
                # line writes)
                res.extend(copy(trace[idx + 1 :]))
                return res

        elif opcode(line) == "if":
            _, cond, if_true, if_false = line
            cond = replace_mem_exp(cond, mem_idx, mem_val)
            mem_idx_true = apply_constraint(mem_idx, cond)
            mem_val_true = apply_constraint(mem_val, cond)
            mem_idx_false = apply_constraint(mem_idx, is_zero(cond))
            mem_val_false = apply_constraint(mem_val, is_zero(cond))

            if_true = replace_mem(if_true, mem_idx_true, mem_val_true)
            if_false = replace_mem(if_false, mem_idx_false, mem_val_false)

            res.append(("if", cond, if_true, if_false))

            if affects(line, mem_val) or affects(line, mem_id):
                # one of the branches may have changed the memory, so
                # what comes after the if (if anything) is left alone
                res.extend(copy(trace[idx + 1 :]))
                return res

        elif affects(line, mem_val) or affects(line, mem_id):
            res.extend(copy(trace[idx:]))
            return res

        elif opcode(line) == "while":
            _, cond, path, jds, vars = line
            # shouldn't this go above the affects if above? and also update vars even if
            # the loops affects the memidx?

            vars = [replace_mem_exp(v, mem_idx, mem_val) for v in vars]

            if not affects(line, ("mem", mem_idx)) and not affects(line, (mem_val)):
                cond = replace_mem_exp(cond, mem_idx, mem_val)
                path = replace_mem(path, mem_idx, mem_val)

            res.append(("while", cond, path, jds, vars))

        else:
            # speed
            test = "mem" in str(line)
            if (
                test
                and (m := match(mem_idx, ("add", Any, ("var", ":num"))))
                and str(("var", m.num)) not in str(line)
            ):
                test = False
            # / speed

            if test:
                l = replace_mem_exp(line, mem_idx, mem_val)
            else:
                l = line

            res.append(l)

    return res


"""

    Variables cleanup

"""


def cleanup_vars(trace, required_after=None):
    required_after = required_after or []
    """
        for every var = mem declaration, replace future
        occurences of it, if possible

        var1 = mem[64]
        log var1
        mem[65] = 'Y'
        log var1

        into

        var1 = mem[64]
        log mem[64]
        mem[65] = 'Y'
        log var1

        for var declarations that are no longer in use, remove them

    """

    res = []

    for idx, line in enumerate(trace):
        if opcode(line) == "setvar":
            _, var_idx, var_val = line
            # find all the future occurences of var and replace if possible

            if exp_size(var_val) > MAX_EXP_SIZE:
                # the vm gave a name to this expression because it's huge,
                # inlining it would make everything else huge as well
                res.append(line)
                res.extend(
                    cleanup_vars(trace[idx + 1 :], required_after=required_after)
                )
                return res

            remaining_trace = replace_var(trace[idx + 1 :], var_idx, var_val)
            if (
                contains(remaining_trace, ("var", var_idx))
                or ("var", var_idx) in required_after
            ):
                res.append(line)

            res.extend(cleanup_vars(remaining_trace, required_after=required_after))
            return res

        elif opcode(line) == "while":
            _, cond, path, *rest = line
            path = cleanup_vars(
                path,
                required_after=required_after + find_op_list(trace[idx + 1 :], "var"),
            )
            res.append(
                (
                    "while",
                    cond,
                    path,
                )
                + tuple(rest)
            )

            a = parse_counters(line)

            if "endvars" in a:
                remaining_trace = trace[idx + 1 :]
                for var_idx, var_val in a["endvars"].items():
                    remaining_trace = replace_var(remaining_trace, var_idx, var_val)

                res.extend(
                    cleanup_vars(
                        remaining_trace,
                        required_after=required_after
                        + find_op_list(trace[idx + 1 :], "var"),
                    )
                )
                return res

        elif opcode(line) == "if":
            _, cond, if_true, if_false = line
            # a variable set in a branch may be used after the if, when the
            # branches merge again
            required = required_after + find_op_list(trace[idx + 1 :], "var")
            if_true = cleanup_vars(
                if_true,
                required_after=(
                    required_after if trace_ends_execution(if_true) else required
                ),
            )
            if_false = cleanup_vars(
                if_false,
                required_after=(
                    required_after if trace_ends_execution(if_false) else required
                ),
            )
            res.append(("if", cond, if_true, if_false))
        else:
            res.append(line)

    return res


def replace_var(trace, var_idx, var_val):
    """
    replace occurences of var, if possible

    """

    var_id = ("var", var_idx)
    res = []

    for idx, line in enumerate(trace):
        if m := match(line, ("setmem", ":mem_idx", Any)):
            # this all seems incorrect, (plus 'affects' checks below, needs to be revisited)
            # memloc = ("mem", m.mem_idx)
            # replace in val
            res.append(replace(line, var_id, var_val))

            if affects(line, var_val):
                res.extend(copy(trace[idx + 1 :]))
                return res
            else:
                continue

        if opcode(line) == "while":
            _, cond, path, jd, setvars = line
            setvars = replace(setvars, var_id, var_val)

            if not affects(
                line, var_val
            ):  # and not find_f(path, lambda e: e ~ ('setvar', var_idx, _)):
                cond = replace(cond, var_id, var_val)
                path = replace_var(path, var_idx, var_val)

            line = ("while", cond, path, jd, setvars)

        if opcode(line) == "if":
            _, cond, if_true, if_false = line
            cond = replace(cond, var_id, var_val)
            if_true = replace_var(if_true, var_idx, var_val)
            if_false = replace_var(if_false, var_idx, var_val)
            res.append(("if", cond, if_true, if_false))

            if affects(line, var_val):
                # one of the branches may have changed the memory the value
                # depends on, so what comes after the if is left alone
                res.extend(copy(trace[idx + 1 :]))
                return res

        elif affects(line, var_val):
            # what the line itself reads is read before it changes anything
            if opcode(line) not in ("while", "if"):
                line = replace(line, var_id, var_val)
            res.append(line)
            res.extend(copy(trace[idx + 1 :]))
            return res

        elif opcode(line) == "while":
            assert not affects(line, var_val)
            res.append(line)  # could replace vars inside of while, skipping for now

        else:
            res.append(replace(line, var_id, var_val))

    return res


"""

    loop parsing

"""


def find_conts(trace):
    def check(line):
        if opcode(line) == "continue":
            return [line]
        else:
            return []

    return find_f_list(trace, check)


def swap_cond(cond):
    replacement = {
        "lt": "gt",
        "le": "ge",
        "gt": "lt",
        "ge": "le",
    }

    return (replacement[cond[0]], cond[2], cond[1])


def move_right(left, right, exp):
    assert type(right) != list
    assert type(left) != list
    if left == exp:
        return right

    if opcode(left) == "add":
        terms = left[1:]
        if exp not in terms:
            return None  # deep embedding unsupported
        for e in terms:
            if e != exp:
                if type(e) == int:
                    e = to_real_int(e)
                right = sub_op(right, e)

        return right

    if opcode(left) == "mul":
        terms = left[1:]
        if exp not in terms:
            return None  # deep embedding unsupported
        for e in terms:
            if e != exp:
                assert type(e) != list
                assert type(right) != list
                right = div_op(right, e)

        return right


def normalize(cond):
    if type(cond) == tuple:
        cond = tuple(cleanup_mul_1(cond))

    if opcode(cond) not in ("lt", "le", "gt", "ge"):
        cond = ("lt", 0, cond)
        return normalize(cond)

    left, right = cond[1], cond[2]
    vars_left = find_op_list(left, "var")
    vars_right = find_op_list(right, "var")

    left_vars = tuple(
        [e for e in vars_left if match(e, ("var", int))]
    )  # int = loop vars
    right_vars = tuple([e for e in vars_right if match(e, ("var", int))])

    if len(left_vars) + len(right_vars) != 1:
        return None

    if len(right_vars) == 1:
        return normalize(swap_cond(cond))

    assert len(left_vars) == 1 and len(right_vars) == 0, cond

    var = left_vars[0]

    if opcode(left) != "var":
        assert type(right) != list
        assert type(left) != list
        right = move_right(left, right, var)
        if right is None:
            return None
        left = var
        cond = (cond[0], left, right)

    if m := match(cond, ("lt", ":left", ":right")):
        cond = ("le", m.left, sub_op(m.right, 1))

    if m := match(cond, ("gt", ":left", ":right")):
        cond = ("ge", m.left, add_op(m.right, 1))

    return cond  # we end up with (gt/lt (var int) sth)


def find_setmems(trace):
    def check(line):
        if m := match(line, ("while", Any, ":path", ...)):
            sm = find_setmems(m.path)
            if len(sm) == 0:
                return []

            # for s in sm:
            #     s_idx = s[1]
            #                if 'var' in str(s_idx):
            #                    print(s_idx)
            #                    assert False

            return sm

        elif opcode(line) == "setmem":
            return [line]

        else:
            return []

    return walk_trace(trace, check)


def memloc_left(setmem):
    assert opcode(setmem) in ("setmem", "mem")
    memloc = setmem[1]
    op, loc, _ = memloc
    assert op == "range"
    return loc


def memloc_right(setmem):
    assert opcode(setmem) in ("setmem", "mem")
    memloc = setmem[1]

    op, loc, rlen = memloc
    assert op == "range"
    return add_op(loc, rlen)


def make_range(left, right):
    r_len = sub_op(right, left)

    if safe_ge_zero(r_len) is False:
        return ("range", left, 0)
    else:
        return ("range", left, r_len)


def while_max_memidx(line):
    # returns the rightmost memory index for a setmem

    a = parse_counters(line)
    op, cond, path, jds, setvars = line
    assert op == "while"

    try:
        setmems = find_setmems(path)
    except Exception:
        logger.exception("Error in find_setmems")
        return "unknown"

    if len(setmems) == 0:
        return 0

    collected = 0

    if "lastvars" not in a:
        for s in setmems:
            collected = _max_op(collected, memloc_right(s))

        return collected

    setmems_begin = setmems_end = setmems

    for v in a["setvars"]:
        v_idx, v_start = v[1], v[2]
        v_end = a["lastvars"][v_idx]

        setmems_begin = replace_var(setmems_begin, v_idx, v_start)
        setmems_end = replace_var(setmems_end, v_idx, v_end)

    for idx, _ in enumerate(setmems):
        collected = _max_op(collected, memloc_right(setmems_begin[idx]))
        collected = _max_op(collected, memloc_right(setmems_end[idx]))

    return collected


def extract_paths(while_exp):
    op, _, trace, jd, setvars = while_exp
    assert op == "while"

    def f(trace, jd, so_far):
        # extract all the paths leading up to jd
        if len(trace) == 0:
            return []

        line = trace[0]

        # assert opcode(line) != 'while'

        if opcode(line) == "if":
            _, cond, if_true, if_false = line
            rest = trace[1:]  # non-empty when the branches merge again
            res_true = f(if_true + rest, jd, so_far + [("require", cond)])
            res_false = f(if_false + rest, jd, so_far + [("require", is_zero(cond))])
            return res_true + res_false

        if len(trace) == 1:
            if opcode(line) == "continue":
                return [so_far]
            else:
                return []

        return f(trace[1:], jd, so_far + [line])

    return f(trace, jd, [])


def extract_setmems(while_exp):
    """
    The setmems of the loop body that are on a path to a `continue` (the
    ones on the paths leaving the loop don't matter to what comes after it).

    Same as collecting them from extract_paths, without enumerating the paths,
    which are exponential in the number of ifs in the body.
    """
    op, _, trace, jd, setvars = while_exp
    assert op == "while"

    def f(trace, after):
        # `after`: whether a continue is reachable from the end of `trace`.
        # Returns the setmems on the paths through `trace` that reach a
        # continue, and whether there are such paths.
        res = []
        reach = after

        for line in reversed(trace):
            if opcode(line) == "continue":
                reach = True

            elif opcode(line) in ENDS_EXECUTION:
                reach = False

            elif opcode(line) == "if":
                res_true, reach_true = f(line[2], reach)
                res_false, reach_false = f(line[3], reach)
                res = res_true + res_false + res
                reach = reach_true or reach_false

            elif reach and opcode(line) in ("setmem", "while"):
                res = find_setmems([line]) + res

        return res, reach

    res, _ = f(trace, False)

    return list(dict.fromkeys(res))


def extract_mems(while_exp):
    paths = extract_paths(while_exp)
    res = []
    for p in paths:
        res += find_mems(p)
    return res


#        mems = extract_mems(path)


def while_touches_mem(line, mem_idx):
    a = parse_counters(line)
    op, cond, path, jds, setvars = line
    assert op == "while"

    #    try:
    setmems = extract_setmems(line)
    #    setmems = find_setmems(path)
    #    except Exception:
    #        return True

    if len(setmems) == 0:
        return False

    setmems_begin = setmems_end = setmems

    if "lastvars" not in a:
        for (
            s
        ) in (
            setmems
        ):  # if no endvars, comparing just with a 'var' assumes 'var' is any natural number
            if range_overlaps(mem_idx, s[1]) is not False:
                return True

        return False

    for v in a["setvars"]:
        v_idx, v_start = v[1], v[2]
        v_end = a["lastvars"][v_idx]

        setmems_begin = replace_var(setmems_begin, v_idx, v_start)
        setmems_end = replace_var(setmems_end, v_idx, v_end)

    for idx, _ in enumerate(setmems):
        r_begin = memloc_left(setmems_begin[idx])
        r_end = memloc_right(setmems_end[idx])

        r = make_range(r_begin, r_end)
        if range_overlaps(mem_idx, r) is not False:
            return True

        r_begin = memloc_left(setmems_end[idx])
        r_end = memloc_right(setmems_begin[idx])

        r = make_range(r_begin, r_end)
        if range_overlaps(mem_idx, r) is not False:
            return True

    return False


def while_uses_mem(line, mem_idx):
    op, cond, path, jds, setvars = line
    assert op == "while"
    a = parse_counters(line)

    mems = find_mems(line)

    #    mems = extract_mems(line)

    if len(mems) == 0:
        return False

    mems_begin = mems_end = mems

    if "lastvars" not in a:
        for s in mems:
            if range_overlaps(mem_idx, s[1]) is not False:
                return True

        return False

    for v in a["setvars"]:
        v_idx, v_start = v[1], v[2]
        v_end = a["lastvars"][v_idx]

        mems_begin = replace_var(mems_begin, v_idx, v_start)
        mems_end = replace_var(mems_end, v_idx, v_end)

    for idx, _ in enumerate(mems):
        r_begin = memloc_left(mems_begin[idx])
        r_end = memloc_right(mems_end[idx])

        r = make_range(r_begin, r_end)
        if range_overlaps(mem_idx, r) is not False:
            return True

        r_begin = memloc_left(mems_end[idx])
        r_end = memloc_right(mems_begin[idx])

        r = make_range(r_begin, r_end)
        if range_overlaps(mem_idx, r) is not False:
            return True

    return False


def exp_uses_mem(exp, mem_idx):
    mems = find_mems([exp])

    for m in mems:
        op, m_idx = m
        assert op == "mem"
        if range_overlaps(m_idx, mem_idx) is not False:
            return True

    return False


# what can have a different value in every iteration of a loop: its
# variables, and what reads the state
LOOP_VARIANT = (
    "var",
    "storage",
    "tload",
    "mem",
    "msize",
    "gas",
    "balance",
    "extcodesize",
    "extcodehash",
    "returndatasize",
    "ext_call",
    "return_data",
    "return_code",
    "new_address",
    ".result",
)


def is_loop_invariant(exp):
    """True if exp is sure to have the same value in every iteration of a loop."""
    if type(exp) is int:
        return True

    if type(exp) is str:
        return not any(name in exp for name in LOOP_VARIANT)

    if type(exp) is not tuple:
        return False

    return all(is_loop_invariant(e) for e in exp)


def parse_counters(line):
    a = {}
    op, cond, path, jds, setvars = line
    assert op == "while"

    a["setvars"] = setvars
    a["jds"] = jds

    # the continues of this loop, not of the ones nested in it
    conts = [c for c in find_conts(path) if c[1] == jds]

    startvars = {}
    for v in setvars:
        assert (m := match(v, ("setvar", ":vidx", ":vval")))
        startvars[m.vidx] = m.vval

    cond = normalize(cond)
    if cond is None or not conts:
        return {}

    cont = conts[0]

    stepvars = {}
    for v in cont[2]:
        var_idx, var_val = v[1], v[2]
        stepvars[var_idx] = var_val

    a["stepvars"] = stepvars

    counter = cond[1][1]
    counter_stop = cond[2]
    if counter not in startvars:
        logger.warning("counter %s not in %s", counter, startvars)
        return {}
    counter_start = startvars[counter]
    a["counter"] = counter
    a["start"] = counter_start
    a["stop"] = counter_stop
    if counter not in stepvars:
        logger.warning("counter not in stepvars")
        counter_diff = 0
    else:
        counter_diff = stepvars[counter]

    if len(conts) > 1:
        return a

    if opcode(counter_diff) != "add":
        return a

    assert type(counter_diff[1]) == int

    counter_diff = (counter_diff[0], to_real_int(counter_diff[1]), counter_diff[2])

    # counter_diff[2] ~ ('mul', 1, X) -> counter_diff[2] = X
    if opcode(counter_diff[2]) == "mul" and counter_diff[2][1] == 1:
        counter_diff = (counter_diff[0], counter_diff[1], counter_diff[2][2])

    assert counter_diff[2] == ("var", counter), counter_diff

    counter_step = to_real_int(counter_diff[1])
    a["step"] = counter_step

    num_loops = div_op(
        add_op(sub_op(counter_stop, counter_start), counter_step), counter_step
    )

    if opcode(num_loops) == "div":  # so, no obvious divider
        a["counter_stop"] = counter_stop
        a["counter_start"] = counter_start
        a["counter_step"] = counter_step
        return a

    a["num_loops"] = num_loops

    lastvars = {}
    for v in setvars:
        var_idx, var_val = v[1], to_real_int(v[2])
        step = stepvars.get(var_idx)
        if not (
            (
                match(step, ("add", ":diff", ("var", var_idx)))
                or match(step, ("add", ":diff", ("mul", 1, ("var", var_idx))))
            )
            and is_loop_invariant(step[1])
        ):
            # not a var that gets incremented by the same amount on every
            # iteration (e.g. one that is set in a branch, or `s += i * i`),
            # we can't tell its value after the loop
            return a
        var_diff = to_real_int(step[1])
        assert type(num_loops) != list
        var_stop = add_op(var_val, mul_op(var_diff, num_loops))
        lastvars[var_idx] = var_stop

    # the values after the last iteration if there is one, which is enough
    # to tell what memory the loop touches
    a["lastvars"] = lastvars

    # num_loops is the number of iterations only if the counter doesn't start
    # beyond where it stops - otherwise there are none, and it's negative
    if cond[0] == "le" and counter_step > 0:
        starts_before_stop = safe_le_op(
            counter_start, add_op(counter_stop, counter_step)
        )
    elif cond[0] == "ge" and counter_step < 0:
        starts_before_stop = safe_le_op(
            add_op(counter_stop, counter_step), counter_start
        )
    else:
        starts_before_stop = False

    if starts_before_stop is True:
        # the values after the loop
        a["endvars"] = lastvars

    return a
