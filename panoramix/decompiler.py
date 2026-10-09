import dataclasses
import io
import json
import logging
import os
import sys
from contextlib import contextmanager, redirect_stdout

import timeout_decorator

import panoramix.folder as folder
from panoramix.contract import Contract
from panoramix.function import Function
from panoramix.loader import Loader
from panoramix.prettify import explain, pprint_repr, pprint_trace
from panoramix.safemath import pretty_defs
from panoramix.storage import pretty_def
from panoramix.vm import VM, entry_memory
from panoramix.whiles import make_whiles
from panoramix.utils.helpers import C, rewrite_trace

logger = logging.getLogger(__name__)

# The time given to each step of a function (the execution, the
# simplification) and to a function as a whole, in seconds. PANORAMIX_TIMEOUT
# scales them (e.g. 10 for a slow machine, or to compare outputs).
_scale = float(os.environ.get("PANORAMIX_TIMEOUT", "1"))
STEP_TIMEOUT = 60 * _scale
FUNCTION_TIMEOUT = 60 * 3 * _scale

# The ifs of a trace nest as deep as the jumps of the code take them, and
# the walks of a trace (the simplifier's, the folder's, the printer's)
# recurse into them: python's 1000 frames failed a function past some 300
# levels (a RecursionError, the function lost).
RECURSION_LIMIT = 100_000
FRAME_SIZE = 2048  # bytes of the stack a frame takes, at most (pypy's: ~750)


@contextmanager
def deep_recursion():
    """
    Python's recursion as deep as RECURSION_LIMIT frames, the main thread's
    stack let grow for them - or as deep as the stack holds, where the
    system doesn't let it grow.
    """
    limit = sys.getrecursionlimit()
    stack = None
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        size = RECURSION_LIMIT * FRAME_SIZE
        if soft != resource.RLIM_INFINITY and soft < size:
            if hard != resource.RLIM_INFINITY:
                size = min(size, hard)
            if size > soft:
                resource.setrlimit(resource.RLIMIT_STACK, (size, hard))
                stack = (soft, hard)
        sys.setrecursionlimit(max(limit, size // FRAME_SIZE))
    except (ImportError, ValueError, OSError):
        pass

    try:
        yield
    finally:
        sys.setrecursionlimit(limit)
        if stack is not None:
            resource.setrlimit(resource.RLIMIT_STACK, stack)


@dataclasses.dataclass
class Decompilation:
    text: str = ""
    asm: list = dataclasses.field(default_factory=list)
    json: dict = dataclasses.field(default_factory=dict)


# Derives from BaseException so it bypasses all the "except Exception" that are
# all around Panoramix code.
class TimeoutInterrupt(BaseException):
    """Thrown when a timeout occurs in the `timeout` context manager."""

    def __init__(self, value="Timed Out"):
        self.value = value

    def __str__(self):
        return repr(self.value)


def decompile_bytecode(code: str, only_func_name=None) -> Decompilation:
    loader = Loader()
    loader.load_binary(code)  # Code is actually hex.
    with deep_recursion():
        return _decompile_with_loader(loader, only_func_name)


def decompile_address(address: str, only_func_name=None) -> Decompilation:
    loader = Loader()
    loader.load_addr(address)
    with deep_recursion():
        return _decompile_with_loader(loader, only_func_name)


def _decompile_with_loader(loader, only_func_name=None) -> Decompilation:
    """

    But the main decompilation process looks like this:

        loader = Loader()
        loader.load(this_addr)

    loader.lines contains disassembled lines now

        loader.run(VM(loader, just_fdests=True))

    After this, loader.func_list contains a list of functions and their locations in the contract.
    Passing VM here is a pretty ugly hack, sorry about it.

        trace = VM(loader).run(target)

    Trace now contains the decompiled code, starting from target location.
    you can do pprint_repr or pprint_logic to see how it looks

        trace = make_whiles(trace)

    This turns gotos into whiles
    then it simplifies the code.
    (should be two functions really)

        functions[hash] = Function(hash, trace)

    Turns trace into a Function class.
    Function class constructor figures out it's kind (e.g. read-only, getter, etc),
    and some other things.

        contract = Contract(addr=this_addr,
                            ver=VER,
                            problems=problems,
                            functions=functions)

    Contract is a class containing all the contract decompiled functions and some other data.

        contract.postprocess()

    Figures out storage structure (you have to do it for the whole contract at once, not function by function)
    And folds the trace (that is, changes series of ifs into simpler forms)

    Finally...

        loader.disasm() -- contains disassembled version
        contract.json() -- contains json version of the contract

    Decompiled, human-readable version of the contract is done within this .py file,
    starting from `with redirect_stdout...`


    To anyone going into this code:
        - yes, it is chaotic
        - yes, there are way too many interdependencies between some modules
        - this is the first decompiler I've written in my life :)

    """

    """
        Fetch code from Web3, and disassemble it.

        Loader holds the disassembled line by line code,
        and the list of functions within the contract.
    """

    logger.info("Running light execution to find functions.")

    loader.run(VM(loader, just_fdests=True))

    if len(loader.lines) == 0:
        # No code.
        return Decompilation(text=C.gray + "# No code found for this contract." + C.end)

    """

        Main decompilation loop

    """

    problems = {}
    functions = {}

    for hash, fname, target, stack in loader.func_list:
        """
        hash contains function hash
        fname contains function name
        target contains line# for the given function
        """

        if only_func_name is not None and not fname.startswith(only_func_name):
            # if user provided a function_name in command line,
            # skip all the functions that are not it
            continue

        logger.info("Decompiling %s...", fname)
        logger.debug("stack %s", stack)

        try:
            if loader.lines.get(target, (None, None))[1] == "jumpdest" and target > 1:
                target += 1

            @timeout_decorator.timeout(
                FUNCTION_TIMEOUT, timeout_exception=TimeoutInterrupt
            )
            def dec():
                logger.info(" -> Interpreting EVM on function...")
                # (the default function where it's all of the dispatcher, not
                # where it starts: what's known there is what its entries,
                # each of the paths to it, have in common - see loader.entry)
                if hash == "_fallback" and target == 0:
                    known = loader.fallback_known
                else:
                    known = ()
                trace = VM(loader).run(
                    target,
                    stack=stack,
                    timeout=STEP_TIMEOUT,
                    known=known,
                    entry=loader.entry(hash),
                    memory=entry_memory(loader.entries.get(hash)),
                )
                explain("Initial decompiled trace", trace)

                if "--explain" in sys.argv:
                    trace = rewrite_trace(
                        trace, lambda line: [] if type(line) == str else [line]
                    )
                    explain("Without assembly", trace)

                logger.info(" -> Cleaning up AST, identifying loops...")
                trace = make_whiles(trace, timeout=STEP_TIMEOUT)
                explain("final", trace)

                if "--explain" in sys.argv:
                    explain("folded", folder.fold(trace))

                return trace

            trace = dec()
            functions[hash] = Function(hash, trace)
        except (Exception, TimeoutInterrupt):
            problems[hash] = fname
            logger.exception("Problem with %s%s", fname, C.end)

    logger.info("Functions decompilation finished, now doing post-processing.")

    """

        Store decompiled contract into .json

    """

    contract = Contract(
        problems=problems,
        functions=functions,
        code=bytes(loader.binary or []),
    )

    contract.postprocess()

    decompilation = Decompilation()

    for l in loader.disasm():
        decompilation.asm.append(l)

    try:
        decompilation.json = contract.json()
        # This would raise a TypeError if it's not serializable, which is an
        # important assumption people can make.
        json.dump(decompilation.json, open(os.devnull, "w"))
    except Exception:
        logger.exception("Failed json serialization.")
        decompilation.json = {}

    text_output = io.StringIO()
    with redirect_stdout(text_output):
        """
        Print out decompilation header
        """

        print(C.gray + "# Palkeoramix decompiler. " + C.end)

        if len(problems) > 0:
            print(C.gray + "#")
            print("#  I failed with these: ")
            for p in problems.values():
                print(f"{C.end}{C.gray}#  - {C.end}{C.fail}{p}{C.end}{C.gray}")
            print("#  All the rest is below.")
            print("#" + C.end)

        print()

        """
            Print out constants & storage
        """

        shown_already = set()

        for func in contract.consts:
            shown_already.add(func.hash)
            print(func.print())

        if shown_already:
            print()

        if len(contract.stor_defs) > 0:
            lang = " (vyper)" if contract.lang == "vyper" else ""
            print(f"{C.green}def {C.end}storage{lang}:")

            for s in contract.stor_defs:
                print(pretty_def(s))

            print()

        if contract.safemath:
            print(f"{C.green}def {C.end}SafeMath:")
            for line in pretty_defs(contract.safemath):
                print(line)
            print()

        """
            Print out getters
        """

        for hash, func in functions.items():
            if func.getter is not None:
                shown_already.add(hash)
                print(func.print())

                if "--repr" in sys.argv:
                    print()
                    pprint_repr(func.trace)

                print()

        """
            Print out regular functions
        """

        func_list = list(contract.functions)
        func_list.sort(
            key=lambda f: f.priority()
        )  # sort func list by length, with some caveats

        if shown_already and any(1 for f in func_list if f.hash not in shown_already):
            # otherwise no irregular functions, so this is not needed :)
            print(C.gray + "#\n#  Regular functions\n#" + C.end + "\n")

        for func in func_list:
            hash = func.hash

            if hash not in shown_already:
                shown_already.add(hash)

                print(func.print())

                if "--returns" in sys.argv:
                    for r in func.returns:
                        print(r)

                if "--repr" in sys.argv:
                    pprint_repr(func.orig_trace)

                print()

    """
        Wrap up
    """

    decompilation.text = text_output.getvalue()
    text_output.close()

    return decompilation
