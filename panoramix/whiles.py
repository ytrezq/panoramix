import collections
import logging

from panoramix.core import arithmetic
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
    simplify,
    simplify_max,
    sub_op,
    to_bytes,
    try_add,
)
from panoramix.core.arithmetic import is_zero, to_real_int
from panoramix.core.masks import get_bit, to_mask, to_neg_mask
from panoramix.core.memloc import (
    apply_mask_to_range,
    fill_mem,
    memloc_overwrite,
    range_overlaps,
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
from panoramix.simplify import ENDS_EXECUTION, simplify_trace
from panoramix.utils.helpers import (
    C,
    contains,
    find_f_list,
    find_f_set,
    find_op_list,
    opcode,
    replace_f,
    replace_vars,
    replace_f_stop,
    rewrite_trace,
    rewrite_trace_full,
    rewrite_trace_ifs,
    rewrite_trace_multiline,
    to_exp2,
    walk_trace,
)

from panoramix.postprocess import cleanup_mul_1

logger = logging.getLogger(__name__)

"""

    Rube Goldberg would be proud.

"""


def make_whiles(trace, timeout=0):
    trace = make(trace)
    explain("Loops -> whiles", trace)

    # clean up jumpdests
    trace = rewrite_trace(
        trace, lambda line: [] if opcode(line) == "jumpdest" else [line]
    )
    trace = simplify_trace(trace, timeout=timeout)

    return trace


"""

    make whiles

"""


def make(trace):
    res = []

    for idx, line in enumerate(trace):
        if m := match(line, ("if", ":cond", ":if_true", ":if_false")):
            res.append(("if", m.cond, make(m.if_true), make(m.if_false)))

        elif m := match(line, ("label", ":jd", ":vars", ...)):
            jd, vars = m.jd, m.vars
            try:
                before, inside, remaining, cond = to_while(trace[idx + 1 :], jd)
            except Exception:
                logger.exception("couldn't make loop for line %s, omitting it.", line)
                continue

            inside = make(inside)
            remaining = make(remaining)

            before = replace_vars(before, {v_idx: v_val for _, v_idx, v_val in vars})
            before = make(before)

            res.extend(before)
            res.append(("while", cond, inside, repr(jd), vars))
            res.extend(remaining)

            return res

        elif m := match(line, ("goto", ":jd", ":setvars")):
            res.append(("continue", repr(m.jd), m.setvars))

        else:
            res.append(line)

    return res


def get_jds(line):
    if m := match(line, ("goto", ":jd", ...)):
        return [m.jd]
    return []


def is_revert(trace):
    if len(trace) != 1:
        return False

    return opcode(trace[0]) in ("revert", "invalid")


def falls_through(trace):
    """
    Whether a path of trace gets to its end: goes on with what follows it,
    rather than to a label (goto) or out of the call.
    """
    if not trace:
        return True
    last = trace[-1]
    if opcode(last) == "if":
        return falls_through(last[2]) or falls_through(last[3])
    return opcode(last) not in ("goto", "undefined") + ENDS_EXECUTION


def to_while(trace, jd, path=None):
    """
    `trace` is what follows a loop label, `jd` the label. Returns
    (before, inside, remaining, cond) so that the loop can be written as:

        before
        while cond:
            inside
        remaining

    """
    path = path or []

    def add_path(line):
        # the lines preceding the exit condition are executed again after
        # the body, before the next iteration
        if m := match(line, ("goto", Any, ":svs")):
            # (the values of the next iteration, all at once)
            path2 = replace_vars(path, {v_idx: v_val for _, v_idx, v_val in m.svs})

            return path2 + [line]
        else:
            return [line]

    while trace:
        line, *trace = trace

        if m := match(line, ("if", ":cond", ":if_true", ":if_false")):
            cond, if_true, if_false = m.cond, m.if_true, m.if_false

            # `trace` is what comes after the if - if its branches merge
            # again (see vm.merge_branches), that's what follows on the merged
            # path. Nothing otherwise.

            # a branch that reverts is a check on the way to the exit
            # condition, kept as it is: a require would drop the data of the
            # revert (a Panic code, a message), and tell an invalid from a
            # revert no more
            if is_revert(if_true):
                path.append(("if", cond, if_true, []))
                trace = if_false + trace
                continue
            if is_revert(if_false):
                path.append(("if", is_zero(cond), if_false, []))
                trace = if_true + trace
                continue

            jds_true = find_f_list(if_true, get_jds)
            jds_false = find_f_list(if_false, get_jds)

            if (
                trace
                and (jd in jds_true or jd in jds_false)
                and jd not in find_f_list(trace, get_jds)
            ):
                # The branches merge again, but the loop doesn't go on after
                # that: what follows the if is where the loop ends up when it
                # is left - by the exit condition, a break, a return from
                # inside it... The if is the body of the loop, which it leaves
                # when it doesn't continue, and that follows it.
                return [], path + [line], trace, ("bool", 1)

            if trace or (jd not in jds_true and jd not in jds_false):
                # The branches merge again and the loop goes on after that:
                # a statement of the loop body (a goto inside it is a
                # `continue`), not the exit condition, which is the last
                # thing on its path.
                path.append(line)
                continue

            if jd in jds_true and jd in jds_false:
                # the loop goes on whichever way the if goes - if it can be
                # left at all, it's from inside the branches (a return, or
                # an if with an exit of its own)
                return [], path + [line], trace, ("bool", 1)

            body, out = (if_true, if_false) if jd in jds_true else (if_false, if_true)
            if out and falls_through(body):
                # A path of the body gets to its end: it leaves the loop -
                # a break, an early return (solc's return code, shared) - and
                # goes on with what follows the if, not with the exit
                # branch. After a `while cond:`, it would.
                return [], path + [line], trace, ("bool", 1)

            if jd in jds_true:
                if_true = rewrite_trace(if_true, add_path)
                return path, if_true, if_false + trace, cond
            else:
                if_false = rewrite_trace(if_false, add_path)
                return path, if_false, if_true + trace, is_zero(cond)

        elif match(line, ("goto", jd, ...)):
            # the path loops back unconditionally: the exits, if any, are
            # the reverts and returns along the way
            return [], rewrite_trace([line], add_path), trace, ("bool", 1)

        elif opcode(line) == "label":
            # a loop in this one, before its exit condition (Vyper tests it
            # at the end): the rest of the trace is that loop, and what
            # follows it, where this one goes on or ends
            return [], path + [line] + trace, [], ("bool", 1)

        else:
            path.append(line)

    assert False, f"no if after label?{jd}"
