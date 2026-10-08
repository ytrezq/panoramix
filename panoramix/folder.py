import logging

from panoramix.core.arithmetic import comp_bool, is_zero, simplify_bool
from panoramix.matcher import match
from panoramix.prettify import prettify
from panoramix.utils.helpers import (
    COLOR_BLUE,
    COLOR_BOLD,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_HEADER,
    COLOR_OKGREEN,
    COLOR_UNDERLINE,
    COLOR_WARNING,
    ENDC,
    FAIL,
    car,
    contains,
    find_f_list,
    get_op,
    opcode,
    replace_f,
)

logger = logging.getLogger(__name__)


"""

    Folder takes takes all the execution paths, and merges them into
    one, as concise as possible.

    In other words - it turns intermediate representation available
    through API/.json, that is easy to parse by machine, and turns
    it into a representation that is easy to read by humans, and
    a much shorter one.


    This (and the vm.py/loops) algorithm is one of the two most difficult
    ones within the decompiler.

    While the code can be cleaned up and refactored, I'm not sure the algorithm
    could be much simpler.

"""


def fold(trace):
    """
    as_paths unfolds the trace into a list of branchless paths that the contract can
    possibly take. 'if's are converted into condition assertions, e.g.

    [x,
     y,
     z,
     (if, cond,
            [a,
             b,
             c,
             ...,
             q,
             w,
             e]
           ,
             [d,
              e,
              f,
              ...,
              q,
              w,
              e])]

    gets turned into

    [
        [x,
         y,
         z,
         cond,
         a,
         b,
         c,
         ...,
         q,
         w,
         e],

         [x,
          y,
          z,
          is_zero(cond),
          d,
          e,
          f,
          ...,
          q,
          w,
          e]
    ]
    """

    try:
        paths = as_paths(trace)

        """

            fold_paths turns those paths again into folded ones, but also with simple
            ifs (that is, not if-else), and not necessarily ifs ending a trace, so - in the
            example above
            [x,y,z,(if, cond, [a,b,c..], [d,e,f..]),q,w,e]

            it is a more succint form of the program display (q,w,e doesn't get repeated
            any more, etc), but it's more difficult to parse automatically (you're now
            dealing with an acyclic graph instead of a tree), so Panoramix uses this form
            only for a display that's readable to humans. Api/json output has the original
            form.

        """

        log = meta_fold_paths(paths)

        """
            some additional folding below, e.g. when there is an if-else like below:
            (if, cond, [revert], [a,b,c]) # if cond revert else a,b,c
            it can be turned into
            (if, cond, [revert]), [a,b,c] # if cond revert; a,b,c
            - no need to use 'else', since the 'if-true' trace always returns, and 'else'
            is unnecessary

        """
        log = fold_aux(log)

        assert not find_f_list(log, lambda e: [e] if opcode(e) == MERGED_IF else [])

        return log

    except Exception:
        # make folder fail gracefuly
        logger.exception(f"folder failed in a function.")
        return trace


# An 'if' whose branches join again (see vm.merge_branches) is folded by
# as_paths on its own, branch by branch, and carried through the folding of
# the surrounding paths as a single opaque line under this opcode. fold_aux
# turns it back into a regular 'if' - it's the only place that knows it
# shouldn't fold the branches again, folding isn't idempotent.
MERGED_IF = "merged_if"


def make_fands(exp):
    # see `ferlan.getOrderDataClaim` for why it's necessary
    if opcode(exp) == "or":
        return ("for",) + exp[1:]
    elif opcode(exp) == "and":
        return ("fand",) + exp[1:]
    else:
        return exp


def unmake_fands(exp):
    if opcode(exp) == "for":
        return ("or",) + exp[1:]
    elif opcode(exp) == "fand":
        return ("and",) + exp[1:]
    else:
        return exp


def as_paths(trace, path=None):
    assert type(trace) == list

    path = path or tuple()

    #    self.find_offsets()

    trace = replace_f(trace, make_fands)

    for idx, line in enumerate(trace):
        if opcode(line) == "if":
            cond, if_true, if_false = line[1], line[2], line[3]

            if idx + 1 < len(trace):
                # The branches merge again and the trace goes on after the if
                # (see vm.merge_branches). Fold each branch on its own and
                # keep the if as a single line - unfolding it would double
                # the number of paths, which is what merging avoided.
                path += ((MERGED_IF, cond, fold(if_true), fold(if_false)),)
                continue

            return as_paths(if_true, path + (cond,)) + as_paths(
                if_false, path + (is_zero(cond),)
            )

        if opcode(line) == "LOOP":
            path += (("LOOP", line[2]),)
            return as_paths(line[1], path)

        path += (line,)

    #    pprint_logic()

    return (list(path),)


"""

    Folder_aux is a set of additional rules for a last-minute folding,
    after everything else is done.

    Done especially with
    0xd883209C4DCd497f24633C627a4E451013424841 , sendFoods
    in mind.


"""

TERMINATING = (
    "return",
    "stop",
    "selfdestruct",
    "invalid",
    "assert_fail",
    "revert",
    "continue",
    "undefined",
)


def fold_aux(trace):
    if type(trace) != list:
        trace = trace.trace
    assert type(trace) == list, type(trace)

    out = []

    for idx, line in enumerate(trace):
        if opcode(line) == "while":
            cond, body, jds, setvars = line[1:]

            body = fold(body)
            line = ("while", cond, body, jds, setvars)

            out.append(line)

        elif opcode(line) == MERGED_IF:
            # the branches are folded already
            _, cond, if_true, if_false = line
            lines, merged = try_merge_ifs(cond, if_true, if_false)
            out.extend(lines)
            out.extend(join_if(*merged[1:]))

        elif opcode(line) == "if":
            if m := match(line, ("if", ":cond", ":if_true")):
                cond, if_true = m.cond, m.if_true
                if_false = trace[idx + 1 :]
                last_true = if_true[-1] if if_true else None
                if (
                    if_false == [("return", 0)]
                    or if_false == [("revert", None)]
                    or opcode(car(if_false)) == "invalid"
                ):  # and\
                    if (
                        opcode(last_true) not in TERMINATING
                    ):  # or (last_true ~ ('if', _, _))):
                        if_true.append(if_false[0])

                    if_true = fold_aux(if_true)
                    line = ("if", cond, if_true, if_false)

                    out.append(line)
                    return out

                else:
                    if_true = fold_aux(if_true)

                    line = ("if", cond, if_true)
                    out.append(line)

            elif m := match(line, ("if", ":cond", ":if_true", ":if_false")):
                cond, if_true, if_false = m.cond, m.if_true, m.if_false
                if_true = fold_aux(if_true)
                if_false = fold_aux(if_false)

                out.extend(join_if(cond, if_true, if_false))
            else:
                assert False, line

        else:
            out.append(line)

    return out


def join_if(cond, if_true, if_false):
    # `if_true` and `if_false` went through fold_aux already
    if not if_true and not if_false:
        return []

    if not if_true:
        cond, if_true, if_false = is_zero(cond), if_false, []

    if not if_false:
        return [("if", cond, if_true)]

    if opcode(if_true[-1]) in TERMINATING and opcode(car(if_false)) not in (
        "invalid",
        "revert",
    ):
        # ^ should be some more generic check that all if_true
        #   paths end with exit

        return [("if", cond, if_true)] + if_false

    return [("if", cond, if_true, if_false)]


"""

    Helper functions, similar to prettify and algebra, but different
    enough to have a separate implementation here.

"""


def pprint(exp):
    for e in exp:
        print(" ", prettify(e, add_color=True))


def pprint_logic(exp, indent=2):
    if type(exp) == list:
        for idx, line in enumerate(exp):
            pprint_logic(line, indent)

    elif opcode(exp) == "or" and len(exp) > 1:
        print(" " * indent + "if")
        pprint_logic(exp[1], indent + 2)
        for line in exp[2:]:
            print(" " * indent + "or")
            pprint_logic(line, indent + 2)
    else:
        print(" " * indent + prettify(exp, add_color=True))


def sorted_tuple(line):
    row = list(line[1:])
    row.sort(key=lambda x: len(x) if type(x) == list else 0)

    return ("or",) + tuple(row)


def or_op(*args):
    ret = tuple()
    for r in args:
        if type(r) == list:
            r = and_op(*r)

        if opcode(r) == "or":
            ret += r[1:]
        else:
            ret += (r,)

    return sorted_tuple(("or",) + ret)


def and_op(*args):
    # returns a list that doesn't have elements with 'or' inside
    idx = 0

    ret = []
    for idx, exp in enumerate(args):
        if type(exp) == list:
            ret += exp
        else:
            ret.append(exp)

        if opcode(exp) == "or":
            ret = []
            for variant in exp[1:]:
                ret.append(and_op(*(args[:idx] + (variant,) + args[idx + 1 :])))

            return or_op(*ret)

    return ret


assert and_op(1, 2, 3, 4) == [1, 2, 3, 4]
assert or_op(1, 2) == ("or", 1, 2)

c = and_op(1, 2, 3)
d = or_op(c, c)
assert d == ("or", [1, 2, 3], [1, 2, 3])

b = or_op(1, 2)
a = and_op("a", b, "c")

assert a == ("or", ["a", 1, "c"], ["a", 2, "c"])


def starting_with(or_tuple, starting):
    ret = []

    for exp in or_tuple[1:]:
        if exp[: len(starting)] == starting:
            ret.append(exp[len(starting) :])

    return ret


def ending_with(or_tuple, ending):
    assert len(ending) > 0

    ret = []

    for exp in or_tuple[1:]:
        if exp[-len(ending) :] == ending:
            ret.append(exp[: -len(ending)])

    return ret


"""

    Folder proper.

    It merges the beginnings and endings of paths recursively in
    fold_paths, and then does some cleanup operations.

    If you have trouble visualising how the algorithm
    works, welcome to the club. I wrote it, but I don't
    fully get it myself.

    It works similarly to the algorithms that do diffs
    of two files, but with way more than two objects
    to compare at the same time.

"""


def meta_fold_paths(paths):
    for_merge = []
    for r in paths:
        assert type(r) in (tuple, list)
        if len(r) > 0:
            for_merge.append(r)

    output = fold_paths(for_merge)

    assert type(output) == list

    output = flatten(
        output
    )  # converts if-else statements into if statements, if possible
    output = cleanup_ors(output)
    output = make_ifs(output)
    output = merge_ifs(output)

    output = replace_f(output, unmake_fands)

    return output


def flatten(path):
    def ends_exec(path):  # check if all the subpaths end execution
        # only checking the last line, previous ones may end execution as well
        # but at least one leading up to the last line didn't - otherwise
        # we wouldn't see it

        line = path[-1]

        if opcode(line) in (
            "return",
            "stop",
            "selfdestruct",
            "invalid",
            "assert_fail",
            "revert",
            "continue",
            "undefined",
        ):
            return True
        elif opcode(line) == "or":
            assert len(line) == 3
            return ends_exec(line[1]) and ends_exec(line[2])
        elif opcode(line) == "while":
            # well, 'while True' ends execution, but all the other
            # ones most likely don't. if we miss some cases nothing
            # bad will happen - just slightly less readable code
            return False
        else:
            return False

    res = []

    for idx, line in enumerate(path):
        if opcode(line) != "or":
            res.append(line)
            continue

        assert (
            len(line) == 3
        ), line  # ('or', [exec1], [exec2]) - we're dealing only with if-else at this stage

        if len(line[1]) == 1 and len(line[2]) == 1:
            # sometimes, after folding, both paths are identical,
            # so we can skip 'ifs', e.g. fallback func in 0xBe46324018124B2d604c2f3eaE91D3De9b388b09
            continue

        elif ends_exec(line[1]):
            res.extend(try_merge(flatten(line[1]), flatten(line[2])))

        #        elif idx == len(path) - 1:
        # the last or = we're flattening
        #            res.extend(try_merge(flatten(line[1]), flatten(line[2])))
        else:
            res.append(("or", flatten(line[1]), flatten(line[2])))

    return res


def try_merge(one, two):
    assert (type(one), type(two)) == (list, list)

    if len(one) > len(two):
        shorter = two
    else:
        shorter = one

    idx = 1
    while one[-idx] == two[-idx] and idx < len(shorter):
        idx += 1

    idx -= 1

    # (the line after a one-sided or is the guard of what follows it, that
    # cleanup_ors skips: it stays with its or)
    while idx > 0 and any(
        opcode(x[-idx - 1]) == "or" and len(x[-idx - 1]) == 2 for x in (one, two)
    ):
        idx -= 1

    if idx > 0:
        return [("or", one[:-idx], two[:-idx])] + one[-idx:]
    else:
        return [("or", one)] + two


def cleanup_ors(path):
    assert type(path) == list
    ret = []

    idx = 0
    while idx < len(path):
        if type(path[idx]) == list:
            path = path[:idx] + path[idx] + path[idx + 1 :]

        line = path[idx]

        if opcode(line) != "or":
            ret.append(line)

        elif len(line) == 2:  # one-sided or
            # clean up the inside, skip the next line in the main path
            # condition = line[1][0]
            line = ("or", cleanup_ors(line[1]))
            ret.append(line)
            idx += 1

        elif len(line) == 3:  # two-sided or
            if len(line[1]) == 1:
                assert comp_bool(simplify_bool(line[1][0]), is_zero(line[2][0]))
                line = ("or", cleanup_ors(line[2]))
                ret.append(line)
            else:
                assert comp_bool(is_zero(line[1][0]), simplify_bool(line[2][0]))
                line = ("or", cleanup_ors(line[1]), cleanup_ors(line[2][1:]))
                ret.append(line)

        else:  # three-sided ors? madness!
            assert False

        idx += 1

    return ret


def make_ifs(path):
    assert type(path) == list

    ret = []
    for line in path:
        if opcode(line) != "or":
            ret.append(line)

        elif len(line) == 2:
            ret.append(("if", line[1][0], make_ifs(line[1][1:])))

        elif len(line) == 3:
            ret.append(("if", line[1][0], make_ifs(line[1][1:]), make_ifs(line[2])))

        else:
            assert False

    return ret


# what the lines moved out of an if may change, that its condition reads
STATE_READS = (
    "st",
    "sall",
    "storage",
    "stor",
    "mem",
    "var",
    "tload",
    "balance",
    "extcodesize",
    "extcodehash",
)
STATE_SYMBOLS = (
    "ext_call",
    "return_data",
    "returndatasize",
    "return_code",
    "new_address",
    ".result",
    "memcopy",
    "gas",
    "msize",
)


def reads_state(exp, reads=STATE_READS):
    """whether exp reads what a line may change: storage, memory, a variable..."""
    if type(exp) == str:
        return any(k in exp for k in STATE_SYMBOLS)
    if type(exp) in (tuple, list):
        return opcode(exp) in reads or any(reads_state(e, reads) for e in exp)
    return False


def changes_read(line, cond):
    """whether the line may change what cond reads"""
    if contains(cond, "gas") or contains(cond, "msize"):
        # (the gas left, the size of the memory: any line may change them)
        return True
    op = opcode(line)
    if op == "setvar":
        return contains(cond, ("var", line[1]))
    if op == "setmem":
        return get_op(cond, "mem") is not None
    if op == "store":
        return any(get_op(cond, k) is not None for k in ("st", "sall", "storage", "stor"))
    if op == "if":
        # (one-sided, or with an else)
        return any(changes_read(l, cond) for branch in line[2:] for l in branch)
    if type(line) == str or op in ("log", "jd") or op in TERMINATING:
        return False
    if op in ("call", "staticcall", "delegatecall", "callcode", "create", "create2"):
        # (the state, the memory, what it returns - not a variable)
        return reads_state(cond, tuple(r for r in STATE_READS if r != "var"))
    return reads_state(cond)


def try_merge_ifs(cond, if_true, if_false):
    # the lines both branches begin with go before the if - before its
    # condition is evaluated: up to one that may change what it reads
    idx = 0
    while (
        idx < min(len(if_true), len(if_false))
        and if_true[idx] == if_false[idx]
        and not changes_read(if_true[idx], cond)
    ):
        idx += 1

    assert if_true[:idx] == if_false[:idx]

    if idx > 0:
        lines = if_true[:idx]
        merged = ("if", cond, if_true[idx:], if_false[idx:])
    else:
        lines = []
        merged = ("if", cond, if_true, if_false)

    return lines, merged


def merge_ifs(path):
    # detects if-else sections that have the same beginnings, and moves
    # if upstream, merging some of the code

    assert type(path) == list
    ret = []

    for idx, line in enumerate(path):
        assert type(line) != list
        if opcode(line) != "if":
            ret.append(line)
            continue
        elif len(line) == 3:
            # one-sided if
            cond, if_true, if_false = (
                line[1],
                merge_ifs(line[2]),
                merge_ifs(path[idx + 1 :]),
            )
            if not if_true or opcode(if_true[-1]) not in TERMINATING:
                # what follows the if runs after its body too: it's not an
                # else, nothing of the two is the same code
                ret.append(("if", cond, if_true))
                ret.extend(if_false)
                break
            lines, merged = try_merge_ifs(cond, if_true, if_false)
            ret.extend(lines)
            ret.append(merged[:3])
            ret.extend(merged[3])
            break
        else:
            assert len(line) == 4
            cond, if_true, if_false = line[1], merge_ifs(line[2]), merge_ifs(line[3])
            lines, merged = try_merge_ifs(cond, if_true, if_false)
            ret.extend(lines)
            ret.append(merged)
            # don't break

    return ret


def fold_paths(for_merge):
    if len(for_merge) == 0:
        return []

    if len(for_merge) == 1:
        return for_merge[0]

    for_merge.sort(key=lambda r: -len(r))

    or_paths = ("or",) + tuple(for_merge)

    # merge beginnings

    idx = 0
    while len(starting_with(or_paths, for_merge[0][:idx])) == len(for_merge):
        idx += 1

    begin_offset = idx - 1

    # merge endings

    idx = 1
    while len(ending_with(or_paths, for_merge[0][-idx:])) == len(for_merge):
        idx += 1

    end_offset = idx - 1  # for_merge[0][:idx]

    s_with = starting_with(or_paths, for_merge[0][:begin_offset])
    if end_offset > 0:
        e_with = ending_with(("or",) + tuple(s_with), for_merge[0][-end_offset:])
        merged = (
            for_merge[0][:begin_offset] + [or_op(*e_with)] + for_merge[0][-end_offset:]
        )
    else:
        merged = for_merge[0][:begin_offset] + [or_op(*s_with)]

    output = []
    for line in merged:
        if opcode(line) != "or":
            output.append(line)

        else:

            def fold_or(line):
                longest = line[-1]

                i = 1
                while longest[0] == line[i][0]:
                    i += 1

                shortest = line[i]

                assert longest[0] != shortest[0], (longest, shortest)

                for l in line[1:]:
                    if l[0] in (longest[0], shortest[0]):
                        pass
                    else:
                        pprint_logic(shortest)
                        print()
                        pprint_logic(longest)
                        print()
                        pprint_logic(l[0])
                        raise
                #                        , (longest[0], shortest[0], l[0])

                # find two longest stretches that split the line or into exactly two parts

                best = None

                for idx1 in range(1, len(shortest)):
                    s1 = starting_with(line, shortest[:idx1])
                    for idx2 in range(1, len(longest)):
                        s2 = starting_with(line, longest[:idx2])
                        if (
                            best is None
                            and s1 == s2
                            and len(s1) + len(s2) + 1 == len(line)
                        ):
                            best = (idx1, idx2)
                            best_s = s1

                if best is not None:
                    return or_op(shortest[: best[0]], longest[: best[1]]), best_s

                # cut the first line, merge the remaining paths if possible
                s1 = starting_with(line, [shortest[0]])
                s2 = starting_with(line, [longest[0]])

                s1 = fold_paths(s1)
                s2 = fold_paths(s2)

                shorter_path = and_op(shortest[0], s1)
                longer_path = and_op(longest[0], s2)

                return ("or",) + (shorter_path,) + (longer_path,), []

            ors, paths = fold_or(line)

            output.append(ors)
            output += fold_paths(paths)

    return output
