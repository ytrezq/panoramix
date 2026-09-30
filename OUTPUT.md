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
  `mem[a len n] = v` with `v` a number writes its low `n` bytes (zeroes
  above its word), with a list of data (`mem[a] = Bytes(12, 0), mem[b len
  20]`, see Data) those `n` bytes.

## Data

A list of data - what a `return`, a `revert`, a `log`, a `sha3`, the params
of a call, a `concat(...)` or `Array(...)` holds - is the bytes of its
elements, one after the other. An element is a 32-byte word, unless it says
how wide it is:

- `Bytes(n, v)`: the low `n` bytes of the number `v` (zeroes above its
  word) - of bytes `v` (a range, a string...), they're `n` bytes;
- `mem[a len n]`, `call.data[a len n]`, `ext_call.return_data[a len n]`...:
  those `n` bytes;
- `'text'`: its bytes - but when it's all the data of a `return` or a
  `revert` (after a 4-byte selector: `revert with 0x08c379a0, 'text'`), it's
  the ABI-encoded string (offset, length, bytes padded to 32);
- `Array(len=l, data=...)`: an ABI-encoded array;
- the 4-byte selector that starts the data of a revert or a call: printed as
  the error or the function it is when that's known (`Error(string reason)`),
  else as its 8 hex digits - a word that small is printed in decimal.

## Functions and params

- `def f(...) payable:` runs its body whatever ether it's sent (`call.value`).
  `def f(...): # not payable` reverts with no data when it's sent any,
  before anything else - `# not payable (invalid)`: runs an invalid opcode
  then (the check of old compilers).
- `const name = ...`: the function `name()` returns that data (read as a
  `return`'s), whatever it's called with - it's not payable.
- A param (`_param1`, or its name) is the word of calldata at its place: its
  type in the signature doesn't clean it. Where it's used cleaned, it's after
  the function checks it (`require _param1 == address(_param1)`), which
  proves the calldata held no more than that.
- `call.func_hash` is the selector; `calldata.size` the size of the calldata.
- An external call prints its gas as the expression it is, its selector as
  the function it calls when that's known (`unknown1234abcd(?)` when it isn't:
  no params are made up), and then its params as a list of data. The address
  called (`x` in `call x.f(...)`, `eth.balance(x)`...) is the low 160 bits
  of `x`, as for the EVM.
  `ext_call.success` is its result, `ext_call.return_data[a len n]` what it
  returned, `return_data.size` how much.

## Events

`log Transfer(address from=..., address indexed to=..., ...)`: the event of
that signature, `indexed` params in the topics, the others in the data in
order. An event not known is printed with its whole topic 0:
`log 0xddf2...b3ef: x, indexed y`; one whose params aren't what the log has
(indexed otherwise: ERC-721's `Approval` is ERC-20's with its three params
indexed) with its signature: `log Approval(address owner, address spender,
uint256 value): indexed x, indexed y, indexed z`.

## Storage

The `def storage:` header names the storage the functions use - `def
storage (vyper):` for Vyper, whose mappings hash the slot, then the key:

- `x is uint8 at storage 3 offset 8`: the bits 8 to 16 of the slot 3. `x`
  reads as that value; a read of fewer of its bits is a cast (`uint4(x)`)
  or a shift of it.
- `m is mapping of T at storage 5`: `m[k]` is at `keccak(k . 5)`, the word
  of the key then the word of the slot. A key of bytes (a range, a
  `concat(...)`, a string) is hashed as those bytes.
- `a is array of T at storage 6`: `a.length` is the word at the slot 6,
  `a[i]` is at `keccak(6) + i` - `keccak(6) + n * i` for a `struct of n
  slots`. Values of 8, 16, 32, 64 or 128 bits are packed: `256 / width` of
  them in a slot, from its lowest bits.
- `f is fixed array of T at storage 7`: `f[i]` is at `7 + i` (packed, and
  of structs, the same way).
- `b is bytes at storage 4`: `b` is the word at the slot 4 (a short bytes or
  string holds its data and twice its length there, a long one twice its
  length plus 1), `b[i]` its data word at `keccak(4) + i`, `b.length` its
  length, `b[all]` its `b.length` bytes (from the word at the slot for a
  short one, from `keccak(4)` on for a long one) - `return Array(len=
  b.length, data=b[all])` is the getter of a `bytes` or a `string`.
- `.field_N` of a struct is its value from the bit `N` on: the slot `N /
  256` after its first one, from the bit `N % 256` of it up to the top of
  the slot, unless a cast says how many bits.
- `stor[e]` is the slot `e` itself.

A name of the header is the one of the getter that returns it when there is
one, else `storN` (`storN_O` for the bits from `O` on of the slot `N`).
