Reading the output
==================

The decompiled code is meant to be read one way only: this is that way. It
is python-like, with the conventions below; where the output can't say
something for sure, it says less (a raw `stor[...]`, `mem[...]`, a
`Bytes(n, ...)`) rather than something else.

## Values

- Every value is a 256-bit word, and arithmetic is modulo 2**256: `x - 1`
  with `x` 0 is 2**256 - 1. Division and modulo by 0 give 0.
- Operations are unsigned; a `′` marks the signed ones: `<′ >′ <=′ >=′`,
  `/′ %′ >>′` (arithmetic shift), `*′`. Numbers compared unsigned are printed
  unsigned (`x > 0xffff...ff00`, not `x > -256`); a big number added is
  printed as the word it is (`+ 0xf652...`), not as a negative one.
- Casts keep the low bits: `uint8(x)` is `x % 2**8`, `address(x)` the low 160
  bits; `int8(x)` is the low 8 bits sign-extended; `bool(x)` is 1 when `x`
  isn't 0. `Mask(size, off, x)` is the bits `off` to `off + size` of `x`, in
  place (the others 0). `ceil32(x)` / `floor32(x)` round to a multiple of 32.

## Operators

Their precedence is python's, from the loosest: `or`, `and`, `not`,
comparisons, `|`, `^`, `&`, `<< >>`, `+ -`, `* / %`, unary `-`, `**`.

- `+ - * / % **` are arithmetic (`**` the power), `<< >>` shifts (by 256 or
  more: 0).
- `& | ^ ~` are bitwise (`^` is xor, `~` not).
- `and`, `or`, `not` are logical: they're only printed between truth values
  (0 or 1), and give 0 or 1. A comparison gives 0 or 1.
- A condition (`if`, `while`, `require`) holds when its value isn't 0;
  `a != b` is also how a xor tested for truth is printed.

## Statements

- `require cond`: reverts (with no data) unless `cond`.
- `revert with ...`, `return ...`: end the call with that data (see Data);
  `stop`: with none; `invalid`: an invalid opcode (all the gas used).
  `revert with 'text'` is the `Error(string)` of that text, `revert with
  Panic(n)` the `Panic(uint256)` of `n`; `return memory from a len n` (or
  `from a to b`) returns those bytes of memory.
- `while cond:` loops while `cond` holds; the end of its body goes back to
  `cond`. `continue` and `break` are about the innermost loop, or the one they
  name: `loop2: while ...` ... `continue loop2`.
- `x = ...` for a name that isn't a storage variable is a local variable.
- `mem[a]` is the word of memory at `a`, `mem[a len n]` its `n` bytes at `a`;
  `mem[a len n] = v` with `v` a number writes its low `n` bytes.

## Data

A list of data - what a `return`, a `revert`, a `log`, a `sha3`, the params
of a call, a `concat(...)` or `Array(...)` holds - is the bytes of its
elements, one after the other. An element is a 32-byte word, unless it says
how wide it is:

- `Bytes(n, v)`: the low `n` bytes of `v`;
- `mem[a len n]`, `call.data[a len n]`, `ext_call.return_data[a len n]`...:
  those `n` bytes;
- `'text'`: its bytes - but when it's all the data is (after a 4-byte
  selector: `revert with 0x08c379a0, 'text'`), it's the ABI-encoded string
  (offset, length, bytes padded to 32);
- `Array(len=l, data=...)`: an ABI-encoded array;
- the 4-byte selector that starts the data of a revert or a call: printed as
  the error or the function it is when that's known (`Error(string reason)`),
  else as its 8 hex digits - a word that small is printed in decimal.

## Functions and params

- `const name = ...`: the function `name()` returns that data (read as a
  `return`'s), whatever it's called with.
- A param (`_param1`, or its name) is the word of calldata at its place: its
  type in the signature doesn't clean it. Where it's used cleaned, it's after
  the function checks it (`require _param1 == address(_param1)`), which
  proves the calldata held no more than that.
- `call.func_hash` is the selector; `calldata.size` the size of the calldata.
- An external call prints its gas as the expression it is, its selector as
  the function it calls when that's known (`unknown1234abcd(?)` when it isn't:
  no params are made up), and then its params as a list of data.
  `ext_call.success` is its result, `ext_call.return_data[a len n]` what it
  returned, `return_data.size` how much.

## Events

`log Transfer(address from=..., address indexed to=..., ...)`: the event of
that signature, `indexed` params in the topics, the others in the data in
order. An event not known is printed with its whole topic 0:
`log 0xddf2...b3ef: x, indexed y`.

## Storage

The `def storage:` header names the storage the functions use:

- `x is uint8 at storage 3 offset 8`: the bits 8 to 16 of slot 3;
- `m is mapping of uint256 at storage 5`: `m[k]` is at `keccak(k . 5)` (the
  key word then the slot word; vyper - `def storage (vyper):` - puts the slot
  first);
- `a is array of uint256 at storage 6`: `a.length` is slot 6, `a[i]` at
  `keccak(6) + i` (packed when its elements are narrower than a word);
- `.field_N` is the value from the bit `N` of the element on, up to the top of
  its word unless a cast says how many bits;
- `stor[e]` is the raw slot `e`, and `storN` the slot `N` with no name.
