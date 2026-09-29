import logging
import time
import sys
from copy import copy

from panoramix.core import arithmetic
import panoramix.utils.opcode_dict as opcode_dict
from panoramix.core.algebra import (
    add_op,
    bits,
    lt_op,
    mask_op,
    mul_op,
    or_op,
    shr_op,
    sub_op,
    to_bytes,
    CannotCompare,
)
from panoramix.core.arithmetic import (
    VOLATILE,
    changed_reads,
    is_zero,
    mentions,
    simplify_bool,
)
from panoramix.matcher import match
from panoramix.prettify import pprint_trace
from panoramix.utils.helpers import (
    C,
    MAX_EXP_SIZE,
    EasyCopy,
    all_concrete,
    exp_size,
    opcode,
    precompiled,
    precompiled_var_names,
    replace,
)

from .loader import apply_entry, entry_known, selector_test
from .stack import Stack, fold_stacks, stack_vars

logger = logging.getLogger(__name__)


"""

    A symbolic EVM.

    It executes the contract, and returns the resulting `trace` of execution - which is the decompiled form.


    The most difficult part of this module is the loop detection and simplification algorithm.
    In over 10 iterations I didn't find a simpler way that is just as effective.

    Unfortunately, because of the complexity, I don't fully understand how it works.
    Ergo, I cannot explain it to you :) Good luck!

    On the upside, some stuff, like apply_stack is quite straightforward.

"""


def mem_load(pos, size=32):
    return ("mem", ("range", pos, size))


def find_nodes(node, f):
    """Find the nodes below `node` (itself included) where f(node) returns true."""
    assert type(node) == Node

    res = []
    to_visit = [node]

    while to_visit:
        n = to_visit.pop()
        if f(n):
            res.append(n)
        to_visit.extend(reversed(n.next))

    return res


MAX_NODE_COUNT = 5_000
node_count = 0


"""

    Path conditions.

    Every node carries `known`: the tuple of conditions that are known to hold
    on the execution path leading to it (one per `jumpi` taken along the way).
    When the VM reaches a `jumpi` whose condition is implied by one of them,
    only the feasible branch is followed. Without this, every conditional
    that is (partially) decided by an earlier one doubles the number of paths
    to explore, and contracts that repeat the same checks - e.g. the
    `success`/`returndatasize`/`extcodesize` checks in the SafeERC20 and
    Address libraries, or `token == address(0)` special cases - blow past
    MAX_NODE_COUNT and only get a truncated, exponentially unrolled
    decompilation.

    Facts about values that can change while the contract runs (storage,
    balances, the result of the last external call...) are forgotten as soon
    as the relevant state may have changed, see `forget`. Everything else
    (calldata, msg.sender, chainid...) is constant for the whole execution.

"""

STATE_CHANGING_OPS = (
    "call",
    "staticcall",
    "delegatecall",
    "callcode",
    "create",
    "create2",
    "selfdestruct",
)

# A codecopy of more bytes than that is left as code.data: the number would be
# unreadable anyway, and one of more than 4300 digits (1786 bytes) can't even be
# turned into a string since python 3.11, which aborts the whole function.
MAX_CODECOPY_SIZE = 1024


def forget(known, names):
    return tuple(fact for fact in known if not mentions(fact, names))


def is_known(exp, known):
    """Evaluate `exp` to True/False if it is decided by the known conditions, None otherwise."""
    for fact in reversed(known):
        res = arithmetic.eval_bool(exp, fact, symbolic=False)
        if res is not None:
            return res

    return None


class Node:
    def __str__(self):
        return f"Node({self.jd})"

    def __repr__(self):
        # technically not a proper _repr_, but some trace printout functions use this
        # instead of a proper str
        return self.__str__()

    def __init__(self, vm, start, safe, stack, condition=True, trace=None, known=()):
        global node_count

        node_count += 1

        self.vm = vm
        self.prev = []
        self.next = []
        self.trace = trace
        self.start = start
        self.safe = safe
        self.stack = stack
        self.history = {}
        self.depth = 0
        self.label_history = {}
        self.label = None

        # condition: the one under which this node is reached from its parent.
        # known: all the conditions that hold on the path leading here.
        self.condition = condition
        self.known = known

        # set when the execution from this node on got merged with other
        # paths, see merge_branches
        self.merged = False

        stack_obj = Stack(stack)
        self.jd = (start, len(stack), tuple(stack_obj.jump_dests(vm.loader.jump_dests)))

    def make_trace(self):
        res = []
        node = self

        # the nodes form long chains (a jump at the end of each), walked
        # here in a loop rather than recursively - the recursion is only as
        # deep as the ifs are nested
        while node is not None:
            res.extend(node._begin_trace())

            if node.trace is None:
                break

            next_node = None

            for line in node.trace:
                if opcode(line) == "jump" and isinstance(line[1], Node):
                    # always the last line
                    next_node = line[1]

                elif opcode(line) == "if" and isinstance(line[2], Node):
                    # the last line, unless the paths merge again after the if -
                    # see merge_branches - in which case a jump to the merged
                    # node follows.
                    _, cond, if_true, if_false = line
                    res.append(
                        ("if", cond, if_true.make_trace(), if_false.make_trace())
                    )

                else:
                    res.append(line)

            node = next_node

        return res

    def _begin_trace(self):
        """What goes before the node's own lines in the decompiled trace."""
        if self.trace is None:
            return [("undefined", "decompilation didn't finish")]

        if self.vm.just_fdests and (
            self.safe and self.vm.lines.get(self.start, (None, None))[1] == "jumpdest"
        ):
            # the loader looks for these to find the default function
            begin = [("jd", str(self.jd[0]))]
        elif self.vm.just_fdests and self.trace != [("revert", None)]:
            t = self.trace[0]
            if match(t, ("jump", ":target_node", ...)):
                begin = [("jd", str(self.jd[0]))]  # , str(self.trace))]
            else:
                begin = ["?"]
        else:
            begin = []

        if self.is_label():
            begin_vars = []
            for _, var_idx, var_val, _ in self.label.begin_vars:
                begin_vars.append(("setvar", var_idx, var_val))

            if find_nodes(
                self,
                lambda n: n.trace
                and opcode(n.trace[-1]) == "goto"
                and n.trace[-1][1] in (self, self.label),
            ):
                begin.append(("label", self, tuple(begin_vars)))
            else:
                # Nothing loops back here any more: a merge (see
                # merge_branches) took the paths that did to the
                # continuation of an if above, where the loop is found
                # again. What's left here is the first iteration.
                begin.extend(begin_vars)

        return begin

    def set_label(self, loop_dest, vars, stack):
        self.label = loop_dest
        loop_dest.begin_vars = vars

        assert len(self.stack) == len(stack)
        self.stack = stack
        loop_dest.prev_trace = loop_dest.trace
        loop_dest.trace = [("jump", self)]
        loop_dest.next = []
        self.set_prev(loop_dest)

        # This node is now the body of a loop, executed for every iteration,
        # and what we learned about the state during the first one doesn't
        # necessarily hold for the next ones.
        self.known = forget(self.known, VOLATILE)

    def set_prev(self, prev):
        self.prev = prev
        self.depth = prev.depth + 1

        self.history = copy(prev.history)
        self.history[prev.jd] = prev

        self.label_history = copy(prev.label_history)
        if prev.label:
            self.label_history[prev.jd] = prev.label

        prev.next.append(self)

    def is_label(self):
        return self.label is not None

    def run(self):
        logger.debug("Node.run(%s)", self)
        self.prev_trace = self.trace
        self.trace = self.vm._run(
            self.start, self.safe, self.stack, self.condition, self.known
        )

        last = self.trace[-1]

        if opcode(last) == "jump":
            n = last[1]

            n.set_prev(self)

        if opcode(last) == "if":
            if_true, if_false = last[2], last[3]

            if_true.set_prev(self)
            if_false.set_prev(self)


class VM(EasyCopy):
    def __init__(self, loader, just_fdests=False):
        self.loader = loader

        # (line_no, op, param)
        self.lines = loader.lines  # a shortcut

        self.just_fdests = just_fdests

        self.counter = 0
        self.known = ()
        self.should_quit = lambda: False
        global node_count
        node_count = 0

    def run(
        self,
        start,
        history={},
        condition=None,
        re_run=False,
        stack=(),
        timeout=0,
        known=(),
        entry=None,
    ):
        """
        `known` is a tuple of conditions known to hold when `start` is reached,
        e.g. for the default function: that no function selector matched.

        `entry` is what runs before `start` (see loader.entry_paths), and
        goes at the beginning of the trace. If it isn't known, the free memory
        pointer is assumed to be 0x60, as the old compilers set it.
        """
        time_start = time.monotonic()

        def should_quit():
            return node_count > MAX_NODE_COUNT or (
                timeout and (time.monotonic() - time_start > timeout)
            )

        self.should_quit = should_quit

        if entry is None:
            before = [("setmem", ("range", 0x40, 32), 0x60)]
        else:
            before = []
            known = tuple(known) + entry_known(entry)

        func_node = Node(
            vm=self, start=start, safe=True, stack=list(stack), known=tuple(known)
        )
        trace = before + [
            ("jump", func_node, "safe", tuple()),
        ]

        root = Node(
            vm=self,
            trace=trace,
            start=start,
            safe=True,
            stack=list(stack),
            known=tuple(known),
        )
        func_node.set_prev(root)

        """

            BFS symbolic execution, ends up with a decompiled
            code, with labels and gotos.

            Depth-first would be way easier to implement, but it tends
            to work way slower because of the loops.

        """

        for j in range(20):
            for i in range(200):
                """

                Find all the jumps, and expand them until
                the next jump.

                """

                self.expand_trace(root)

                """
                    find all the jumps that lead to an already
                    reached jumpdest (with similar stack, otherwise
                    we'd catch function calls as all).

                    replace them with 'loop' identifier
                """

                self.replace_loops(root)

                """
                    turn them into loops right away: the later this is
                    done, the bigger the subtree that gets thrown away and
                    explored again when a loop is set up.
                """

                self.continue_loops(root)

                """
                    find the ifs whose branches all end up at the same
                    jumpdest, and continue from there only once.
                """

                self.merge_branches(root)

                """
                    repeat until there are no more jumps
                    to explore (so, until the trace didn't change)

                """

                nodes = find_nodes(root, lambda n: n.trace is None)

                if len(nodes) == 0 or should_quit():
                    break

            trace = self.continue_loops(root)

            # tr = root.make_trace()
            nodes = find_nodes(root, lambda n: n.trace is None)

            if len(nodes) == 0 or should_quit():
                break

        if should_quit():
            logger.warning(
                "VM stopped prematurely. Node count %i, after %.2f seconds.",
                node_count,
                time.monotonic() - time_start,
            )

        tr = root.make_trace()
        if entry:
            tr = apply_entry(entry, tr)
        return tr

    def expand_trace(self, root):
        nodes = find_nodes(root, lambda n: n.trace is None)

        for node in nodes:
            if self.should_quit():
                # symbolic execution of a single node can take very long when
                # the expressions get big, so the timeout is checked here too
                break

            node.run()

    def replace_loops(self, root):
        nodes = find_nodes(root, lambda n: n.trace is None)

        for node in nodes:
            if (
                node.jd in node.history
                and node.jd[1] > 0
                and len(node.history[node.jd].stack) == len(node.stack)
            ):  # jd[1] == stack_len
                folded, vars = fold_stacks(
                    node.history[node.jd].stack, node.stack, node.depth
                )
                loop_line = (
                    "loop",
                    node.history[node.jd],
                    node.stack,
                    folded,
                    tuple(vars),
                )
                node.trace = [loop_line]

    def continue_loops(self, root):
        loop_list = find_nodes(
            root,
            lambda n: n.trace is not None
            and len(n.trace) == 1
            and opcode(n.trace[0]) == "loop",
        )

        for node in loop_list:
            (line,) = node.trace
            op, loop_dest, stack, new_stack, vars = line
            assert op == "loop"

            if loop_dest.is_label():
                old_stack = loop_dest.stack
                beginvars = loop_dest.label.begin_vars
                set_vars = []

                for _, var_idx, val, stack_pos in beginvars:
                    sv = ("setvar", var_idx, stack[stack_pos])
                    set_vars.append(sv)

                if not set_vars:
                    folded, var_list = fold_stacks(
                        old_stack, stack, loop_dest.label.depth
                    )
                    node.trace = None
                    node.set_label(loop_dest, tuple(var_list), folded)
                    continue

                var_positions = set(stack_pos for *_, stack_pos in beginvars)
                changed = set(
                    idx
                    for idx, (before, after) in enumerate(zip(old_stack, stack))
                    if idx not in var_positions and before != after
                )

                if changed:
                    # The loop variables were found by comparing the stack
                    # before and after the first iteration, and this one
                    # changes something else: e.g. `s += i * i` leaves s at 0
                    # the first time, `if (x[i] > m) m = x[i]` may not change m.
                    # The loop gets explored again, with a variable there too.
                    first = loop_dest.label
                    folded, var_list = stack_vars(
                        first.stack, var_positions | changed, loop_dest.depth
                    )
                    loop_dest.trace = None
                    loop_dest.next = []
                    loop_dest.set_label(first, tuple(var_list), folded)
                    # the other loops found in this pass may be in the part
                    # of the tree that was just thrown away
                    return

                node.trace = [("goto", loop_dest, tuple(set_vars))]

            else:
                node.trace = None
                node.set_label(loop_dest, tuple(vars), new_stack)

    def merge_branches(self, root):
        """

        When every path going out of an `if` either ends the execution or
        reaches the same jumpdest with the same stack layout, decompile what
        follows that jumpdest once, as the continuation of the `if`, with
        variables standing for the stack values that differ between paths:

            if cond:                        if cond:
                ...                             ...
                jump X (with a on stack)        _1 = a
            else:                           else:
                ...                             ...
                jump X (with b on stack)        _1 = b
                                            X (with _1 on stack)

        Without this, everything after X gets decompiled once for every path
        leading to it, so the number of paths doubles at each such `if`. This
        is what the compiler emits around every external call (was the call
        successful? is there return data? is it shorter than 32 bytes?) and
        a function doing a few of them can't be decompiled at all.

        """
        if self.just_fdests:
            return

        unexpanded = find_nodes(root, lambda n: n.trace is None)

        if not unexpanded:
            return

        by_jd = {}
        for n in find_nodes(root, lambda n: True):
            by_jd.setdefault(n.jd, []).append(n)

        tried = set()

        for node in unexpanded:
            if node.trace is not None:
                # merged into another node during this pass
                continue

            if len(by_jd[node.jd]) < 2 or self.ends_execution(node.jd[0]):
                # nothing to merge with, or no point (e.g. a shared revert block)
                continue

            for other in by_jd[node.jd]:
                if other is node:
                    continue

                # the paths to node and other diverged at their closest common
                # ancestor. if it's an `if` that's where they could be merged.
                p = self.common_ancestor(node, other)

                if (
                    p == []
                    or not (p.trace and opcode(p.trace[-1]) == "if")
                    or (id(p), node.jd) in tried
                ):
                    continue

                tried.add((id(p), node.jd))

                if self._merge_at(p, node.jd):
                    break

    @staticmethod
    def common_ancestor(a, b):
        while a != [] and a.depth > b.depth:
            a = a.prev
        while b != [] and b.depth > a.depth:
            b = b.prev
        while a != [] and a is not b:
            a, b = a.prev, b.prev
        return a

    def ends_execution(self, line):
        """True if the basic block starting at `line` can only end the execution."""
        while line in self.lines:
            op = self.lines[line][1]
            if op in (
                "revert",
                "return",
                "stop",
                "invalid",
                "assert_fail",
                "selfdestruct",
            ):
                return True
            if op in ("jump", "jumpi"):
                return False
            line = self.loader.next_line(line)
        return False

    def _merge_at(self, p, jd):
        """
        Merge the paths going out of the `if` node `p` at jumpdest `jd`.
        Returns True if it was done. (If it couldn't be done because some
        paths were not explored yet, it will be tried again when another
        node reaches `jd`.)
        """

        TERMINAL = (
            "revert",
            "return",
            "stop",
            "invalid",
            "assert_fail",
            "selfdestruct",
            "undefined",
        )

        if jd == p.jd or not (p.trace and opcode(p.trace[-1]) == "if"):
            # a loop rather than a merge, or `p` isn't an if any more
            return False

        hits = []

        def visit(start):
            """
            Returns True if every path below `start` either ends, reaches
            `jd`, or loops back (a `continue`: it doesn't get to what follows
            the if either), False if not, None if it's too early to tell.
            """
            result = True
            to_visit = [start]

            while to_visit:
                n = to_visit.pop()

                if n.merged:
                    # goes on in the continuation of an `if` below `p`, which
                    # will be visited as well
                    continue

                if n.is_label():
                    if n.jd == jd:
                        # the head of a loop, let's keep it that way
                        return False

                    # Otherwise the paths inside the loop body are fair game.
                    # If one of them gets merged before it loops back, the loop
                    # gets peeled: its first iteration stays in the branch, and
                    # what follows is decompiled again from the merge point,
                    # where the loop will be found again. See make_trace for
                    # the label left behind.

                elif n.jd == jd:
                    hits.append(n)
                    continue

                if n.trace is None:
                    result = None
                    continue

                if n.next:
                    to_visit.extend(reversed(n.next))
                    continue

                op = opcode(n.trace[-1]) if n.trace else None

                if op in TERMINAL or op == "goto":
                    continue

                # e.g. 'loop', not yet processed by continue_loops
                result = None

            return result

        _, _, if_true, if_false = p.trace[-1]

        if visit(if_true) is not True or visit(if_false) is not True:
            return False

        if len(hits) < 2:
            return False

        stacks = [list(h.stack) for h in hits]
        merged = list(stacks[0])
        setvars = [[] for _ in hits]

        for idx in range(len(merged)):
            vals = [s[idx] for s in stacks]

            if all(v == vals[0] for v in vals):
                continue

            if any(type(v) == int and v in self.loader.jump_dests for v in vals):
                # let's not turn a jump destination into a variable
                return False

            self.counter += 1
            name = f"_{self.counter}"
            merged[idx] = ("var", name)

            for k in range(len(hits)):
                setvars[k].append(("setvar", name, vals[k]))

        logger.debug("merging %i paths at %s", len(hits), hits[0])

        for h, sv in zip(hits, setvars):
            h.trace = sv
            h.next = []
            h.merged = True

        # what's known at the merge point is what's known on every path
        known = tuple(
            fact for fact in hits[0].known if all(fact in h.known for h in hits[1:])
        )

        node = Node(
            self,
            start=jd[0],
            safe=hits[0].safe,
            stack=tuple(merged),
            condition=True,
            known=known,
        )
        node.set_prev(p)
        p.trace.append(("jump", node))

        return True

    def _run(self, start, safe, stack, condition, known=()):
        logger.debug("VM._run stack=%s", stack)
        self.stack = Stack(stack)
        self.known = known
        trace = []

        i = start
        lines = self.lines

        if i not in lines:
            if type(i) != int:
                return [("undefined", "jump to a parameter computed at runtime", i)]
            else:
                return [("invalid", "jumdest", i)]

        if not safe and lines[i][1] != "jumpdest":
            return [("invalid", "jump")]

        if lines[i][1] == "jumpdest":
            # This node stands for this jumpdest already: don't create another
            # one for it below, it would look like a one-node loop.
            # (e.g. when this is the fallthrough branch of a jumpi, and the
            # next instruction happens to be a jumpdest)
            i = self.loader.next_line(i)
            if i not in lines:
                return [("invalid", "eof?")]

        while True:
            try:
                line = lines[i]
            except KeyError:
                trace.append(("invalid", "jumpdest"))
                return trace

            res = self.handle_jumps(trace, line, condition)
            if res is not None:
                return res

            if line[1] == "jumpdest":
                n = Node(
                    self,
                    start=i,
                    safe=False,
                    stack=tuple(self.stack.stack),
                    condition=condition,
                    known=self.known,
                )
                logger.debug("jumpdest %s", n)
                trace.append(("jump", n))
                return trace

            else:
                self.apply_stack(trace, line)

            i = self.loader.next_line(i)

        assert False

    def handle_jumps(self, trace, line, condition):
        i, op = line[0], line[1]
        stack = self.stack

        if "--explain" in sys.argv and op in (
            "jump",
            "jumpi",
            "selfdestruct",
            "stop",
            "return",
            "invalid",
            "assert_fail",
            "revert",
        ):
            trace.append(C.asm(f"       {stack}"))
            trace.append("")
            trace.append(f"[{line[0]}] {C.asm(op)}")

        if op in (
            "jump",
            "jumpi",
            "selfdestruct",
            "stop",
            "return",
            "invalid",
            "assert_fail",
            "revert",
        ):
            logger.debug("[%s] %s", i, op)

        if op == "jump":
            target = stack.pop()

            n = Node(
                self,
                start=target,
                safe=False,
                stack=tuple(self.stack.stack),
                condition=condition,
                known=self.known,
            )

            trace.append(("jump", n))
            return trace

        elif op == "jumpi":
            target = stack.pop()
            if_condition = simplify_bool(stack.pop())

            tuple_stack = tuple(self.stack.stack)
            n_true = Node(
                self,
                start=target,
                safe=False,
                stack=tuple_stack,
                condition=if_condition,
                known=self.known + (if_condition,),
            )
            n_false = Node(
                self,
                start=self.loader.next_line(i),
                safe=True,
                stack=tuple_stack,
                condition=is_zero(if_condition),
                known=self.known + (is_zero(if_condition),),
            )

            if self.just_fdests and (test := selector_test(if_condition)):
                fx_hash, taken = test
                if taken:
                    n_true.trace = [("funccall", fx_hash, target, tuple_stack)]
                else:
                    n_false.trace = [("funccall", fx_hash, n_false.start, tuple_stack)]

            bool_condition = arithmetic.eval_bool(if_condition, symbolic=False)

            if bool_condition is None:
                bool_condition = is_known(if_condition, self.known)

            if bool_condition is not None:
                if bool_condition:
                    trace.append(("jump", n_true))
                    return trace  # res, False

                else:
                    trace.append(("jump", n_false))
                    return trace

            trace.append(
                (
                    "if",
                    if_condition,
                    n_true,
                    n_false,
                )
            )
            logger.debug("jumpi -> if %s", trace[-1])
            return trace

        elif op in ["return", "revert"]:
            p = stack.pop()
            n = stack.pop()

            if n == 0 and op == "return":
                # returning nothing is the same as stopping, while
                # ("return", 0) returns a word, 0
                trace.append(("stop",))
            elif n == 0:
                # likewise, ("revert", 0) reverts with a word, 0
                trace.append((op, None))
            else:
                return_data = mem_load(p, n)
                trace.append(
                    (
                        op,
                        return_data,
                    )
                )

            return trace

        elif op == "selfdestruct":
            trace.append(
                (
                    "selfdestruct",
                    stack.pop(),
                )
            )
            return trace

        elif op in ["stop", "assert_fail", "invalid"]:
            trace.append((op,))
            return trace

        elif op == "UNKNOWN":
            trace.append(("invalid",))
            return trace

        return None

    def snapshot(self, trace, op, target=None):
        """
        Before an instruction that changes the state, what was read from it
        and is still on the stack gets a variable: afterwards, the expression
        of the read (e.g. ("storage", 256, 0, 4)) stands for what's there
        then, not for what was read.

        e.g. arr.push(x) reads the length, stores it plus one, and stores x
        at the old length.
        """
        reads = []
        for item in self.stack.stack:
            for r in changed_reads(item, op, target):
                if r not in reads:
                    reads.append(r)

        for r in reads:
            self.counter += 1
            vname = f"_{self.counter}"
            trace(("setvar", vname, r))
            self.stack.stack = [
                replace(item, r, ("var", vname)) for item in self.stack.stack
            ]

    def apply_stack(self, ret, line):
        def trace(exp, *format_args):
            try:
                logger.debug("Trace: %s", str(exp).format(*format_args))
            except Exception:
                pass

            if type(exp) == str:
                ret.append(exp.format(*format_args))
            else:
                ret.append(exp)

        stack = self.stack

        op = line[1]

        previous_len = stack.len()

        if op in ("sstore", "tstore"):
            self.snapshot(trace, op, stack.stack[-1])
        elif op in STATE_CHANGING_OPS and op != "selfdestruct":
            self.snapshot(trace, op)

        if op == "sstore":
            self.known = forget(self.known, ("storage",))
        elif op == "tstore":
            self.known = forget(self.known, ("tload",))
        elif op in STATE_CHANGING_OPS:
            # Anything can happen in the callee, including reentering this
            # contract and changing its storage.
            self.known = forget(self.known, VOLATILE)

        if "--verbose" in sys.argv or "--explain" in sys.argv:
            trace(C.asm("       " + str(stack)))
            trace("")

            if "push" not in op and "dup" not in op and "swap" not in op:
                trace("[{}] {}", line[0], C.asm(op))
            else:
                if type(line[2]) == str:
                    trace("[{}] {} {}", line[0], C.asm(op), C.asm(" ”" + line[2] + "”"))
                elif line[2] > 0x1000000000:
                    trace("[{}] {} {}", line[0], C.asm(op), C.asm(hex(line[2])))
                else:
                    trace("[{}] {} {}", line[0], C.asm(op), C.asm(str(line[2])))

        param = 0
        if len(line) > 2:
            param = line[2]

        if op in [
            "exp",
            "and",
            "eq",
            "div",
            "lt",
            "gt",
            "slt",
            "sgt",
            "mod",
            "xor",
            "signextend",
            "smod",
            "sdiv",
        ]:
            stack.append(
                arithmetic.eval(
                    (
                        op,
                        stack.pop(),
                        stack.pop(),
                    )
                )
            )

        elif op[:4] == "push":
            stack.append(param)

        elif op == "pop":
            stack.pop()

        elif op == "dup":
            stack.dup(param)

        elif op == "mul":
            stack.append(mul_op(stack.pop(), stack.pop()))

        elif op == "or":
            stack.append(or_op(stack.pop(), stack.pop()))

        elif op == "add":
            stack.append(add_op(stack.pop(), stack.pop()))

        elif op == "sub":
            left = stack.pop()
            right = stack.pop()

            if type(left) == int and type(right) == int:
                stack.append(arithmetic.sub(left, right))
            else:
                stack.append(sub_op(left, right))

        elif op in ["mulmod", "addmod"]:
            stack.append((op, stack.pop(), stack.pop(), stack.pop()))

        elif op == "shl":
            off = stack.pop()
            exp = stack.pop()
            if all_concrete(off, exp):
                stack.append(exp << off)
            else:
                stack.append(mask_op(exp, shl=off))

        elif op == "shr":
            off = stack.pop()
            exp = stack.pop()
            if all_concrete(off, exp):
                stack.append(exp >> off)
            else:
                stack.append(shr_op(exp, off))

        elif op == "sar":
            off = stack.pop()
            exp = stack.pop()
            if all_concrete(off, exp):
                sign = exp & (1 << 255)
                if off >= 256:
                    if sign:
                        stack.append(2**256 - 1)
                    else:
                        stack.append(0)
                else:
                    shifted = exp >> off
                    if sign:
                        shifted |= (2**256 - 1) << (256 - off)
                    stack.append(shifted)
            else:
                # left as it is: no mask copies the sign bit into the bits
                # shifted in
                stack.append(("sar", off, exp))

        elif op in ["not", "iszero"]:
            stack.append((op, stack.pop()))

        elif op == "sha3":
            p = stack.pop()
            n = stack.pop()
            res = mem_load(p, n)

            self.counter += 1
            vname = f"_{self.counter}"
            vval = (
                "sha3",
                res,
            )

            trace(("setvar", vname, vval))
            stack.append(("var", vname))

        elif op == "calldataload":
            stack.append(
                (
                    "cd",
                    stack.pop(),
                )
            )

        elif op == "byte":
            # the idx-th byte of val, the most significant one first
            idx = stack.pop()
            val = stack.pop()
            off = sub_op(248, mul_op(8, idx))
            stack.append(mask_op(val, 8, off, shr=off))

        elif op == "selfbalance":
            stack.append(
                (
                    "balance",
                    "address",
                )
            )

        elif op == "balance":
            addr = stack.pop()
            if addr[:4] == ("mask_shl", 160, 0, 0):
                stack.append(
                    (
                        "balance",
                        addr[4],
                    )
                )
            else:
                stack.append(
                    (
                        "balance",
                        addr,
                    )
                )

        elif op == "swap":
            stack.swap(param)

        elif op[:3] == "log":
            p = stack.pop()
            s = stack.pop()
            topics = []
            param = int(op[3])
            for i in range(param):
                el = stack.pop()
                topics.append(el)

            trace(
                (
                    "log",
                    mem_load(p, s),
                )
                + tuple(topics)
            )

        elif op == "sload":
            sloc = stack.pop()
            stack.append(("storage", 256, 0, sloc))

        elif op == "sstore":
            sloc = stack.pop()
            val = stack.pop()
            trace(("store", 256, 0, sloc, val))

        elif op == "tload":
            stack.append(("tload", stack.pop()))

        elif op == "tstore":
            key = stack.pop()
            val = stack.pop()
            trace(("tstore", key, val))

        elif op == "mload":
            memloc = stack.pop()

            self.counter += 1
            vname = f"_{self.counter}"
            trace(("setvar", vname, ("mem", ("range", memloc, 32))))
            stack.append(("var", vname))

        elif op == "mstore":
            memloc = stack.pop()
            val = stack.pop()
            trace(
                (
                    "setmem",
                    ("range", memloc, 32),
                    val,
                )
            )

        elif op == "mstore8":
            memloc = stack.pop()
            val = stack.pop()

            # the lowest byte of val, into one byte of memory
            trace(
                (
                    "setmem",
                    ("range", memloc, 1),
                    mask_op(val, 8),
                )
            )

        elif op == "mcopy":
            dst = stack.pop()
            src = stack.pop()
            size = stack.pop()

            if size != 0:
                trace(("setmem", ("range", dst, size), ("mem", ("range", src, size))))

        elif op == "extcodecopy":
            addr = stack.pop()
            mem_pos = stack.pop()
            code_pos = stack.pop()
            data_len = stack.pop()

            trace(
                (
                    "setmem",
                    ("range", mem_pos, data_len),
                    ("extcodecopy", addr, ("range", code_pos, data_len)),
                )
            )

        elif op == "codecopy":
            mem_pos = stack.pop()
            call_pos = stack.pop()
            data_len = stack.pop()

            if (
                (type(call_pos), type(data_len)) == (int, int)
                and call_pos + data_len <= len(self.loader.binary)
                and data_len <= MAX_CODECOPY_SIZE
            ):
                res = 0
                for i in range(call_pos, call_pos + data_len):
                    res = res << 8
                    res += self.loader.binary[i]
                trace(
                    ("setmem", ("range", mem_pos, data_len), res)
                )  # ('bytes', data_len, res)))

            else:
                trace(
                    (
                        "setmem",
                        ("range", mem_pos, data_len),
                        (
                            "code.data",
                            call_pos,
                            data_len,
                        ),
                    )
                )

        elif op == "codesize":
            stack.append(len(self.loader.binary))

        elif op == "calldatacopy":
            mem_pos = stack.pop()
            call_pos = stack.pop()
            data_len = stack.pop()

            if data_len != 0:
                call_data = ("call.data", call_pos, data_len)
                #                call_data = mask_op(('call.data', bits(add_op(data_len, call_pos))), size=bits(data_len), shl=bits(call_pos))
                trace(("setmem", ("range", mem_pos, data_len), call_data))

        elif op == "returndatacopy":
            mem_pos = stack.pop()
            ret_pos = stack.pop()
            data_len = stack.pop()

            if data_len != 0:
                return_data = ("ext_call.return_data", ret_pos, data_len)
                #                return_data = mask_op(('ext_call.return_data', bits(add_op(data_len, ret_pos))), size=bits(data_len), shl=bits(ret_pos))
                trace(("setmem", ("range", mem_pos, data_len), return_data))

        elif op == "call":
            self.handle_call(op, trace)

        elif op == "staticcall":
            self.handle_call(op, trace)

        elif op == "delegatecall":
            gas = stack.pop()
            addr = stack.pop()

            arg_start = stack.pop()
            arg_len = stack.pop()
            ret_start = stack.pop()
            ret_len = stack.pop()

            call_trace = (
                "delegatecall",
                gas,
                addr,
            )  # arg_start, arg_len, ret_start, ret_len)

            if arg_len == 0:
                fname = None
                fparams = None

            elif arg_len == 4:
                fname = mem_load(arg_start, 4)
                fparams = None

            else:
                fname = mem_load(arg_start, 4)
                fparams = mem_load(add_op(arg_start, 4), sub_op(arg_len, 4))

            call_trace += (fname, fparams)

            trace(call_trace)

            self.call_len = ret_len
            stack.append("delegate.return_code")

            if 0 != ret_len:
                return_data = ("delegate.return_data", 0, ret_len)

                trace(("setmem", ("range", ret_start, ret_len), return_data))

        elif op == "callcode":
            gas = stack.pop()
            addr = stack.pop()
            value = stack.pop()

            arg_start = stack.pop()
            arg_len = stack.pop()
            ret_start = stack.pop()
            ret_len = stack.pop()

            call_trace = (
                "callcode",
                gas,
                addr,
                value,
            )

            if arg_len == 0:
                fname = None
                fparams = None

            elif arg_len == 4:
                fname = mem_load(arg_start, 4)
                fparams = None

            else:
                fname = mem_load(arg_start, 4)
                fparams = mem_load(add_op(arg_start, 4), sub_op(arg_len, 4))

            call_trace += (fname, fparams)

            trace(call_trace)

            self.call_len = ret_len
            stack.append("callcode.return_code")

            if 0 != ret_len:
                return_data = ("callcode.return_data", 0, ret_len)

                trace(("setmem", ("range", ret_start, ret_len), return_data))

        elif op == "create":
            wei, mem_start, mem_len = stack.pop(), stack.pop(), stack.pop()

            call_trace = ("create", wei)

            code = mem_load(mem_start, mem_len)
            call_trace += (code,)

            trace(call_trace)

            stack.append("create.new_address")

        elif op == "create2":
            wei, mem_start, mem_len, salt = (
                stack.pop(),
                stack.pop(),
                stack.pop(),
                stack.pop(),
            )

            call_trace = ("create2", wei, ("mem", ("range", mem_start, mem_len)), salt)

            trace(call_trace)

            stack.append("create2.new_address")

        elif op == "pc":
            stack.append(line[0])

        elif op == "msize":
            self.counter += 1
            vname = f"_{self.counter}"
            trace(("setvar", vname, "msize"))
            stack.append(("var", vname))

        elif op in ("extcodesize", "extcodehash", "blockhash", "blobhash"):
            stack.append(
                (
                    op,
                    stack.pop(),
                )
            )

        elif op in [
            "callvalue",
            "caller",
            "address",
            "number",
            "gas",
            "origin",
            "timestamp",
            "chainid",
            "difficulty",
            "gasprice",
            "coinbase",
            "gaslimit",
            "calldatasize",
            "returndatasize",
            "basefee",
            "blobbasefee",
        ]:
            stack.append(op)

        else:
            # TODO: Maybe raise an error directly?
            assert op not in [
                "jump",
                "jumpi",
                "revert",
                "return",
                "stop",
                "jumpdest",
                "UNKNOWN",
            ]

        if stack.len() - previous_len != opcode_dict.stack_diffs[op]:
            logger.error("line: %s", line)
            logger.error("stack: %s", stack)
            logger.error(
                "expected %s, got %s stack diff",
                opcode_dict.stack_diffs[op],
                stack.len() - previous_len,
            )
            assert False, f"opcode {op} not processed correctly"

        if (
            op not in ("dup", "swap")
            and not op.startswith("push")
            and type(stack.peek()) == tuple
            and exp_size(stack.peek()) > MAX_EXP_SIZE
        ):
            # Some arithmetic (e.g. the Newton iterations in mulDiv) makes
            # expressions grow exponentially, and the algebra with them.
            # Give the big ones a name, as is done for memory reads.
            self.counter += 1
            vname = f"_{self.counter}"
            trace(("setvar", vname, stack.pop()))
            stack.append(("var", vname))

        stack.cleanup()

    def handle_call(self, op, trace):
        stack = self.stack

        gas = stack.pop()
        addr = stack.pop()
        if op == "call":
            wei = stack.pop()
        else:
            assert op == "staticcall"
            wei = 0

        arg_start = stack.pop()
        arg_len = stack.pop()
        ret_start = stack.pop()
        ret_len = stack.pop()

        if addr == 4:  # Identity
            m = mem_load(arg_start, arg_len)
            trace(("setmem", ("range", ret_start, arg_len), m))

            stack.append("memcopy.success")

        elif type(addr) == int and addr in precompiled:
            m = mem_load(arg_start, arg_len)
            args = mem_load(arg_start, arg_len)
            var_name = precompiled_var_names[addr]

            trace(("precompiled", var_name, precompiled[addr], args))
            trace(("setmem", ("range", ret_start, ret_len), ("var", var_name)))

            stack.append("{}.result".format(precompiled[addr]))

        else:
            assert op in ("call", "staticcall")
            call_trace = (
                op,
                gas,
                addr,
                wei,
            )

            if arg_len == 0:
                call_trace += None, None

            elif arg_len == 4:
                call_trace += mem_load(arg_start, 4), None

            else:
                fname = mem_load(arg_start, 4)
                fparams = mem_load(add_op(arg_start, 4), sub_op(arg_len, 4))
                call_trace += fname, fparams

            trace(call_trace)
            #           trace(('comment', mem_load(arg_start, arg_len)))

            self.call_len = ret_len

            stack.append("ext_call.success")

            try:
                if lt_op(0, ret_len):
                    return_data = ("ext_call.return_data", 0, ret_len)
                    trace(("setmem", ("range", ret_start, ret_len), return_data))
            except CannotCompare:
                return_data = ("ext_call.return_data", 0, ret_len)
                trace(("setmem", ("range", ret_start, ret_len), return_data))
