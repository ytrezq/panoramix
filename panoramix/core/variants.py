from panoramix.utils.helpers import is_array, opcode

"""

    for a given expression, returns variants of it with each value being 0 or 2^256-1

    e.g. for ('ADD', ('mem', 64), ('mem', 256))
         returns
            ('ADD', 0, 0),
            ('ADD', 0, MAX_number),
            ('ADD', MAX_number, 0),
            ('ADD', MAX_number, MAX_number)

"""


MAX_number = 2**230 - 1
MAX_number2 = 2**230 - 1


def variants(exp):
    var = extract_variables(exp)
    for p in possibilities(list(var)):
        yield replace_dict(exp, p)


"""

    implementation

"""


def extract_variables(exp):
    if type(exp) == int:
        return set()

    if opcode(exp) in (
        "var",
        "mem",
        "cd",
        "storage",
        "call.data",
        "sha3",
        "calldatasize",
    ) or is_array(opcode(exp)):
        return set([exp])

    if type(exp) == str and exp in (
        "x",
        "y",
        "z",
        "sth",
        "unknown",
        "undefined",
        "callvalue",
        "number",
        "timestamp",
        "address",
    ):
        return set([exp])

    if type(exp) == str and exp != "data" and "data" in exp:
        return set([exp])

    if type(exp) != tuple:
        return set([exp])

    res = set()
    for e in exp[1:]:
        res = res.union(extract_variables(e))

    return res


def possibilities(var):
    if len(var) > 0:
        current = var[0]
        if len(var) == 1:
            yield {current: MAX_number}
            yield {current: MAX_number2}

            if current == ("mem", ("range", 64, 32)):
                yield {current: 96}
            elif current == "calldatasize":
                yield {
                    current: 6
                }  # nasty hack for `sweeper` contract, try to remove and see what happens
            else:  # can theoretically cause bugs in other ones
                yield {current: 0}

        else:
            for p in possibilities(var[1:]):
                p[current] = MAX_number
                yield p
                p[current] = MAX_number2
                yield p

                if current == ("mem", ("range", 64, 32)):
                    # (used to yield {current: 96} alone, without the other
                    # variables: a variant that isn't concrete, so the
                    # result depended on the order of the set of variables,
                    # i.e. on the hash seed)
                    p[current] = 96
                elif current == "calldatasize":
                    p[current] = 6
                else:
                    p[current] = 0
                yield p


def replace(exp, idx, val):
    if exp == idx:
        return val

    if type(exp) != tuple:
        return exp

    return tuple(replace(e, idx, val) for e in exp)


def replace_dict(exp, dic):
    """
    replaces every variable of `dic` in `exp` with its value, all at once:
    a variable that contains another one (e.g. mem[_5] and _5) must be
    replaced before it, otherwise the result depends on the order of the
    dict, which depends on the hash seed
    """
    try:
        if exp in dic:
            return dic[exp]
    except TypeError:  # unhashable
        pass

    if type(exp) != tuple:
        return exp

    return tuple(replace_dict(e, dic) for e in exp)
