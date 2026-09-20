..
   SPDX-License-Identifier: GPL-2.0-or-later

.. _riscv-ztt:

RISC-V Attached Matrix Extension (Ztt)
======================================

QEMU implements version 0.6 of the draft RISC-V Attached Matrix Extension
(AME), named Ztt.  The implementation follows specification commit
``8d27acdcee7a189aad4e2907412e0c40a9d4b9e2`` from the
`riscv-attached-matrix-extension repository
<https://github.com/riscv/riscv-attached-matrix-extension>`_.

Ztt is experimental and disabled by default.  Its instruction encodings,
CSRs and ``mstatus`` fields are provisional.  They are not a stable guest ABI.

Configuration
-------------

Ztt is available with TCG on RV32 and RV64.  Enable it on a generic CPU with
``x-ztt=on``.  These CPU properties select the backend geometry at
initialization:

.. list-table:: CPU properties
   :header-rows: 1

   * - Property
     - Meaning
     - Default
     - Accepted values
   * - ``amenlen``
     - Matrix dimension N
     - 128
     - Powers of two from 1 to 128
   * - ``ameudsz``
     - Datapath width U, in bits
     - 64
     - Powers of two from 1 to 64
   * - ``ame-num-mregs``
     - Physical M register count NM
     - 32
     - 16 or 32
   * - ``ame-num-accregs``
     - Physical Acc register count NA
     - 16
     - 1, 2, 4, 8 or 16

The selected geometry determines the physical register storage and the
packing or grouping of each datatype.  N squared times U must be divisible
by eight: each physical M register occupies a whole number of bytes.
An Acc holds one logical square of the largest enabled datatype, rounded
up to a whole number of bytes; it need not match a physical M register's
width.  Packed M-to-Acc transfers may use several consecutive Acc registers.

The generic CPU models use QEMU's maximum implemented capability set.
There is no command-line option for selecting individual operation/datatype
combinations.  CPU and board models may define a smaller allowed set in
their source code.  Changing N or U changes register shapes, not the set
of implemented numerical operations.

See :ref:`riscv-virt-ame` for a bare-metal launch example and shared-pool
configuration.

Supported capabilities
----------------------

The following datatypes are available on generic CPUs.  An accepted
datatype is not a promise that every instruction accepts it in every
operand position or register grouping.

.. list-table:: Datatypes
   :header-rows: 1

   * - Datatype
     - Width or format
     - Rounding and overflow
   * - Signed and unsigned integer
     - 4, 8, 16, 32, 64 and 128 bits
     - RNU, RNE, RDN and ROD; saturating or non-saturating
   * - IEEE floating-point
     - FP16, BF16, FP32 and FP64
     - RNE, RTZ, RDN, RUP and RMM; provisional RNO described below
   * - OCP floating-point
     - E4M3 and E5M2
     - RNE, RTZ, RDN, RUP and RMM; E4M3 SAT and E5M2 OVF

All listed integer widths support arithmetic and conversions.  OCP RNO
and Custom datatypes are not implemented.

.. list-table:: Instruction families
   :header-rows: 1

   * - Family
     - Implemented operations
     - Operand requirements
   * - Control and state
     - Acquire/release, TRY/BOUNDED_WAIT, CSRs, datatype set/get
     - MS and ownership requirements described below
   * - Transfers and conversion
     - M/Acc and X/M transfers, broadcast, pack/unpack; integer/integer,
       integer/floating and floating/floating conversion
     - M/M and M/Acc transfers require matching datatypes;
       pack/unpack selects one packed square
   * - Memory
     - Whole-register, row/column and strided M loads/stores
     - Acc data is transferred through M; RAM accesses only
   * - Arithmetic
     - Add/subtract, absolute value/difference, mean/half-difference,
       multiply, multiply-add/subtract, exponent scaling and accumulation
     - Mixed integer signedness, widths and integer/floating sources supported
   * - Comparison and selection
     - Compare, select and conditional select
     - Comparisons produce integer masks; select conditions may be integer
       or floating, and the selected value's datatype must match the destination
   * - Bitwise, shifts and max/min
     - Integer logical operations and shifts; integer/floating max/min
     - Bitwise and max/min require matching matrix datatypes;
       shifts are described below
   * - Floating-point functions
     - exp2, log2, sin, cos, tanh, reciprocal, reciprocal square root,
       square root, four round-to-integral modes and composite logarithms
     - Floating source/destination formats may differ; logarithm bias may be integer
   * - Prefix, reduction and scatter
     - Row/column prefix and reduction operations, scatter-add and scatter-max
     - Conversion and accumulation order depends on the instruction family
   * - Matrix arithmetic
     - Six matrix multiply-add/subtract forms, including integer/floating mixtures
     - Integer and floating accumulation rules are described below
   * - Rearrangement and indices
     - Gather, row/column broadcast and shift, zip/unzip, rowid/colid
     - Some instructions require non-packed operands, as listed below

The v0.6 specification requires non-packed operands for these 14 instructions:
``mrowbcast.ew.x``/``mcolbcast.ew.x``, ``mrowgather.ew``/``mcolgather.ew``,
``mrowshift.ew.x``/``mcolshift.ew.x``, ``mrowunzip.ew``/``mcolunzip.ew``,
``mrowzip.ew``/``mcolzip.ew``, ``mrowscatadd.ew``/``mcolscatadd.ew`` and
``mrowscatmax.ew``/``mcolscatmax.ew``.  Their operands must also meet the
instruction's datatype relationships.  Packed operands report unsupported.
For example, FP8 has ``pack_factor=8`` at the default U=64, but is non-packed
at U=8.  Supporting FP8 arithmetic does not remove these instruction-specific
requirements.

For data-scalar ``.ew.x`` instructions, the X register is interpreted as
datatype S from ``amestype`` and converted to the matrix source datatype T
before arithmetic.  The operation therefore uses two sources of datatype T
and the configured destination datatype D.  Control operands, such as shift
amounts, indices and exponents, instead follow the individual instruction's
rules.

Initialization and capability probing
-------------------------------------

System software must enable MS and acquire a backend before using its
registers.  It can read N and U from ``amenlen`` and ``ameudsz``, then compute
packing, grouping and register use as specified by Ztt.

To probe a datatype, clear ``amestatus.UN``, execute ``msettyp`` or
``asettyp`` on a disposable register, and read UN.  A successful set clears
the corresponding data, so do not probe using a live register.  To probe an
operation/datatype combination, configure valid register groups, clear UN,
execute the operation, and read UN again.  There is no instruction that
enumerates all supported combinations.  UN is sticky and must be cleared
before each probe.

For example, this M-mode fragment acquires a backend and probes Int8 at M0.
The instruction words use v0.6 encodings and do not require an AME-aware
assembler.  ``no_backend`` and ``unsupported`` are application error paths::

  li      t0, 1 << 25           # MS = Initial, after reset
  csrs    mstatus, t0
  li      a1, 0                # TRY
  .word   0xa495852b            # ame.acquire a0, a1
  andi    t0, a0, 1             # Ownership granted
  beqz    t0, no_backend
  csrw    0x800, zero           # Clear amestatus.UN
  li      t0, 0x40000008        # Signed Int8, nonsaturating, RNU
  .word   0xa232802b            # msettyp m0, t0; clears M0
  csrr    t0, 0x800
  andi    t0, t0, 1
  bnez    t0, unsupported
  # Configure the remaining operands and execute AME operations here.
  .word   0xa4a0002b            # ame.release

The ``unsupported`` error path must also release the acquired backend.

Unsupported datatypes and operation/type combinations set UN without
committing other results.  This differs from an illegal instruction caused
by MS Off, missing required ownership, or an invalid register/group.
Acquire/release and reads of ``amenlen``, ``ameudsz`` and ``ameown`` do not
require existing ownership, but system-mode access still requires MS enabled.
Writing an unknown or unsupported encoding to ``amestype`` stores its raw
bits without immediately setting UN; an operation using that scalar datatype
reports unsupported.

Provisional CSR allocation
--------------------------

Ztt v0.6 places three writable CSRs in addresses whose standard CSR encoding
marks them read-only.  QEMU does not weaken the architectural CSR permission
check.  Until the specification assigns final addresses, it uses this
temporary mapping:

====================  ==================  ============
CSR                   Ztt v0.6 address    QEMU address
====================  ==================  ============
``amestatus``         ``0x800``           ``0x800``
``amestype``          ``0xcc2``           ``0x810``
``amefflags``         ``0xcc4``           ``0x811``
``amexsat``           ``0xcc5``           ``0x812``
``amenlen``           ``0xcc0``           ``0xcc0``
``ameudsz``           ``0xcc1``           ``0xcc1``
``ameown``            ``0xcc3``           ``0xcc3``
====================  ==================  ============

QEMU allocates MS to ``mstatus[26:25]`` and the corresponding ``sstatus``
field: 0 is Off, 1 Initial, 2 Clean and 3 Dirty.  Reset leaves MS Off;
AME state changes mark it Dirty.  Software must release its backend before
setting MS Off.  In virtualized execution, both the active and HS-level MS
fields must permit AME access.

The default ``AME_MAX_INT_DTYPE`` encoding is ``0x40000080``: signed Int128,
non-saturating, RNU.  ``mldexp.ew.x`` and ``mldexpacc.ew.x`` use this
datatype for the zero-extended X-register exponent, not ``amestype``.

These addresses and MS bits are part of the experimental guest ABI and may
change with a later draft.

QEMU applies the ownership requirement in section 5.2 to guest reads and
writes of ``amefflags`` and ``amexsat``, as well as to backend-context CSRs.
A hart must hold a backend and have MS enabled.  The two accrued flag values
remain hart-local; this access rule does not move them into the backend.

Backend model
-------------

Architectural matrix and accumulator state belongs to an AME backend, not to
a hart.  A hart obtains a backend with ``ame.acquire`` and relinquishes it with
``ame.release``.  TRY and BOUNDED_WAIT acquisition modes follow the draft
instruction semantics.

QEMU defines the following ``BOUNDED_WAIT`` intervals:

.. list-table:: Bounded-wait timeouts
   :header-rows: 1

   * - ``TIMEOUT_CLASS``
     - Class
     - Interval
   * - ``0x0``
     - SHORT
     - 1 ms
   * - ``0x1``
     - MEDIUM
     - 10 ms
   * - ``0x2``
     - LONG
     - 100 ms
   * - ``0x3``
     - VERY_LONG
     - 1000 ms
   * - ``0xF``
     - Implementation-defined
     - 1000 ms

These intervals use QEMU's virtual clock, not host wall-clock time.  Pausing
the VM pauses timeout progression.  Host scheduling can delay completion
after the virtual deadline.  ``TRY`` does not wait and ignores
``TIMEOUT_CLASS``.

The :ref:`virt machine <riscv-virt-ame>` provides a shared backend pool.
Acquisition selects a free slot dynamically; the guest does not choose a
backend ID.  Different harts may own different slots concurrently, but a
slot has only one owner.  The pool has no MMIO region or separate address
space; Ztt memory instructions use the executing hart's address translation.

A new grant initializes the backend context.  Acquiring a backend again
while already owning it preserves that context.

Linux-user uses a private one-slot pool for each emulated CPU.  A system CPU
not connected to a machine pool uses the same private-pool model.

Integer matrix arithmetic
-------------------------

For an integer matrix operation, QEMU visits logical squares in increasing
order and then visits each square's inner dimension in increasing order.  It
forms every product exactly in 256 bits and accumulates the complete dot
product, including an old accumulator value when required, in a signed
320-bit temporary.  The result is converted to the accumulator datatype only
once, after the complete dot product.  A saturating accumulator clamps and
sets ``amexsat.SAT``; a non-saturating accumulator retains its low bits.

Floating and mixed matrix arithmetic
------------------------------------

For a matrix operation involving a floating datatype, QEMU visits logical
squares in increasing order and then visits each square's inner dimension in
increasing order.  Each product is fused with the running destination-precision
accumulator and rounded once as one fused multiply-add.  Integer sources keep
their complete values in each term; an integer accumulator uses its integer
rounding and saturation rules for each update.  Thus the complete dot
product has deterministic QEMU ordering, but it is not accumulated with
unbounded precision or rounded only once at the end.  Exception flags from all
terms are combined and committed with the staged accumulator result.

Unary floating-point arithmetic
-------------------------------

For standard datatypes, Ztt v0.6 section 6.2.2 requires floating-point
source and destination datatypes for ``mexp2.ew``, ``mlog2.ew``, ``mcos.ew``,
``msin.ew``, ``mtanh.ew``, ``mrec.ew``, ``mrsqrt.ew``, ``msqrt.ew``,
``mfrintm.ew``, ``mfrintn.ew``, ``mfrintp.ew`` and ``mfrintz.ew``.  The
source and destination need not use the same floating-point format.

QEMU implements ``mrec``, ``mrsqrt``, ``mexp2``, ``mlog2``, ``mtanh``,
``msin`` and ``mcos`` with independently selected source and destination
formats: FP16, BF16, FP32, FP64, E4M3 and E5M2.  The source is decoded in its
own format; it is not first rounded to the destination format.  These
operations use fixed SoftFloat algorithms, including full-range argument
reduction for sine and cosine, and do not call the host math library.

``msqrt`` is correctly rounded to the destination format.  ``mfrintm``,
``mfrintn``, ``mfrintp`` and ``mfrintz`` also allow independently selected
formats; their instruction-specific rounding direction overrides the
destination rounding field, including when packing the integral result.
Removing an input fraction does not set NX; rounding the integral result
to the destination format may set NX.  Earlier accrued flags,
signaling-NaN NV, and exceptions from the final destination conversion
are retained.

For the approximate functions, the target is one destination-format ULP
relative to the exact mathematical result, not correct rounding for every
input.  For E4M3, E5M2, FP16 and BF16 sources, exhaustive interval checks
establish an error bound of one destination-format ULP for these seven
unary functions in the supported destination formats and rounding modes.
This bound applies to finite mathematical results within the destination's
finite range; overflow and special values follow their separate result
rules.  It includes subnormal results, using the subnormal spacing as ULP.
A full-domain bound has not been established for FP32/FP64 sources or for
the composite logarithm instructions.
In particular, directed rounding of values near a limiting result is not
always correctly rounded.

``mlog2sub`` and ``msublog2`` first round ``log2(abs(source))`` to the
destination format, convert the bias to that format, and then perform a
destination-format subtraction.  In ``.ew``, ``ms1`` supplies the logarithm
input and ``ms2`` supplies the bias, which may be integer or floating.  In
``.ew.x``, ``ms2`` supplies the logarithm input and ``X[rs1]`` supplies the
bias through the ``amestype`` scalar conversion.  The logarithm input and
destination must be floating under section 6.2.2's standard-datatype rules,
but their formats may differ.
Cancellation can amplify the intermediate rounding error, so
no fixed final ULP bound is claimed for the compound mathematical expression
without intermediate rounding.

Prefix and reduction arithmetic
-------------------------------

Prefix and reduction instructions convert every source element to the
destination datatype before folding.  The first converted element seeds the
fold; QEMU introduces no semantic-zero term.  QEMU then visits the remaining
elements in ascending row or column order and rounds every operation to the
destination datatype.  Conversion and fold flags are accumulated together.

OCP FP8 arithmetic
------------------

QEMU supports the OCP E4M3 and E5M2 datatypes with RNE, RTZ, RDN, RUP and
RMM.  RNO remains unsupported because Ztt v0.6 assigns its encoding without
defining the rounding algorithm.

The supported FP8 operations include conversion, arithmetic, comparison,
selection, exponent scaling, fused operations, nonlinear operations,
reductions, scatter, matrix multiplication and structural tensor operations,
including ``mhdiff`` and ``mmean`` in ``.ew`` and ``.ew.x`` forms.

Each FP8 fused operation keeps the complete product and addend until the one
destination-format rounding step.  Exponent-scaling accumulate operations
likewise scale the source before adding the old destination, with no
intermediate destination-format rounding of the scaled source.
Reductions and matrix multiplication use the deterministic traversal and
destination-precision accumulation order documented for IEEE arithmetic.

E4M3 has no infinity encoding.  An infinity-producing conversion or
arithmetic result saturates to the signed maximum finite value and records
overflow and inexact.  E5M2 preserves infinity.  OCP FP8 does not distinguish
quiet and signaling NaNs; QEMU emits the canonical NaN for each format.

The final FP8 packing uses OCP's SAT overflow policy for E4M3 and OVF for
E5M2.  These policies apply after rounding the significand to the destination
precision: an out-of-range result becomes the signed maximum finite E4M3
value or signed E5M2 infinity, independently of the rounding direction.

Datatype tuple boundaries
-------------------------

Integer arithmetic can mix signed and unsigned datatypes.  Each source is
interpreted using its own descriptor, with conversion at the boundaries
specified by the operation.  Arithmetic data sources may be wider than the
destination.  Integer elementwise expressions and matrix dot products keep
wide intermediate results until the final destination conversion; that
conversion applies destination truncation or saturation.  Reductions,
prefix operations and integer-only scatter instead convert each input to the
destination datatype before the destination-precision fold or update.
Bitwise operations still require matching data-source and destination
descriptors.

``msll``, ``msrl`` and ``msra`` allow different integer source and destination
datatypes.  The shift amount is reduced modulo the data source's width, and
the shifted value is finally saturated or wrapped to the destination width.
``msra`` requires a signed data source, but its destination may be unsigned.
``msrl`` treats the source as raw unsigned bits.

Elementwise addition, subtraction, absolute difference, mean, half-difference,
multiplication and multiply-add operations accept mixed integer/floating
sources and either kind of destination.  Each source is interpreted in its
own datatype, including all Int128 bits, without conversion to the
destination before the expression is evaluated.  The final result uses the
destination's rounding and overflow rules.  Comparisons also accept mixed
sources and produce integer masks.  Absolute value and exponent scaling
permit integer/floating source and destination combinations.

Scatter-add involving floating data rounds each binary update to the
destination datatype, without first converting the contribution.  Scatter-max
compares the current destination and contribution in their own datatypes and
then converts the selected value.  Prefix and reduction instructions retain
the per-input conversion and fold order described above.

``mmax.ew`` and ``mmin.ew`` require identical destination and source
datatype descriptors, including the rounding-mode field.  Their scalar forms
instead convert the scalar according to ``amestype`` before comparing it in
the tensor destination datatype, as required by Ztt v0.6.

For IEEE formats, QEMU provisionally interprets RNO as round-to-odd/jam,
using SoftFloat's ``float_round_to_odd_inf`` mode.  This is a QEMU choice,
not an algorithm specified by v0.6.
OCP RNO remains unsupported.

Floating-point to integer conversion
------------------------------------

QEMU supports FP16, BF16, FP32, FP64, OCP E4M3 and E5M2 conversion to
signed and unsigned integers of 4, 8, 16, 32, 64 and 128 bits.  This applies
to ``mconv.ew`` and to the conversion steps in typed scalar ingress,
``mpack.ew.x``, ``munpack.ew.x`` and scalar arithmetic.  Mixed arithmetic
with an integer destination uses the same final conversion policy.

The integer destination selects RNU (nearest, ties toward positive infinity),
RNE (nearest, ties to even), RDN (toward negative infinity), or ROD (round to
odd/jam).  RNU is not RISC-V floating-point RMM (ties away from zero).
QEMU rounds the mathematical value to an integer before testing its range.
Neither the source datatype's floating rounding field nor ``fcsr.frm``
controls this conversion.

Ztt v0.6 does not completely specify invalid-conversion results and exception
flags.  QEMU provisionally combines the destination's integer rounding and
saturation fields with the invalid-conversion conventions of
`RISC-V FCVT <https://docs.riscv.org/reference/isa/unpriv/f-st-ext.html>`_:

* An in-range, exact result sets no flags; an in-range, inexact result sets
  ``amefflags.NX``.
* A finite rounded result outside the destination range sets ``amefflags.NV``.
  A non-saturating destination retains its low destination-width bits; a
  saturating destination clamps to its range and also sets ``amexsat.SAT``.
* NaN of either sign returns the destination maximum.  Positive infinity
  returns the maximum; negative infinity returns the signed minimum or
  unsigned zero.  These cases set NV, but not SAT.
* An element that sets NV does not additionally set NX.  Flags from other
  elements and earlier instructions remain accrued.  No case sets OF or UF.

Signed zero converts exactly to integer zero.  Invalid numerical inputs do
not set ``amestatus.UN`` or trap when the backend supports the conversion
tuple.  Results and flags are committed together after the instruction
finishes.  The wrapping and integer-rounding behavior differs from scalar
FCVT; this supplementary policy may change when Ztt specifies it fully.

Memory access
-------------

Ztt memory instructions use the executing hart's address translation and
permissions.  QEMU validates the complete access before committing register
or data-memory writes.  This does not make the complete matrix transfer atomic
with respect to other harts.

Real device MMIO raises an access fault.  Device accesses can have side
effects that cannot be probed or rolled back, so QEMU cannot generally meet
Ztt's no-partial-write fault contract for them.  RAM requiring a slow TLB path
is not rejected merely because that path is flagged as MMIO internally.
This is an implementation boundary, not a general MMIO prohibition in Ztt.
RVV's completed-prefix and ``vstart`` restart semantics do not provide the
same fault contract.

Debugging and migration
-----------------------

Start system QEMU with ``-S -gdb tcp::1234`` in addition to the normal machine
and image options, then connect a RISC-V GDB with ``target remote :1234``.
The dynamic ``org.gnu.gdb.riscv.ame`` feature describes the configured AME
register sizes.  For the currently selected hart, GDB exposes raw bytes as
``m0`` ... and ``acc0`` ..., datatype encodings as ``md0`` ... and ``ad0`` ...,
and the AME CSRs.  Use, for example, ``p $m0.b`` and ``p/x $md0``.
Registers larger than 64 KiB appear in ordered chunks such as ``m0_0`` and
``m0_1`` rather than as one GDB register.

Backend register reads return zero and writes do not take effect when the
selected hart does not own a slot.  This does not grant ownership or expose
another hart's slot.  Hart-local ``amefflags`` and ``amexsat``, geometry,
ownership and ``ame_max_int_dtype`` remain readable through GDB without
ownership; the flags are writable.  This debugger access is distinct from
guest CSR access permissions.

System migration saves backend registers, ownership, waiting harts and
hart-local flags.  Source and destination must agree on pool enabled and
configured state, slot count, N/U/NM/NA, accepted datatypes, supported
operation/type combinations and ``AME_MAX_INT_DTYPE``.
An owning hart must exist on the destination and connect to the corresponding
pool.  Temporary computation buffers are not migrated.

Operating-system support and limitations
----------------------------------------

Ztt is rejected with KVM and RV128.  It is currently exposed only through
generic RISC-V CPU models whose extension set can be configured by the user.

Geometry and capabilities are frozen properties of the backend pool.  CPU
properties provide construction inputs and attachment requirements, not a
second runtime capability set.  Attached harts must match the pool geometry.
A generic CPU can use a pool exposing a subset of QEMU's implemented
capabilities; a fixed CPU's explicit capability requirements must match the
pool exactly.

In system emulation, guest firmware and the kernel must manage MS, ownership,
task context switches and signal contexts.  Enabling ``x-ztt`` does not supply
a Linux AME userspace ABI or add this support to an existing OpenSBI/kernel
image.

Linux-user does not emulate privileged MS state.  Each emulated CPU/thread
has a private pool.  A cloned thread starts without ownership or a copy of
the parent's M/Acc registers and must acquire its own backend; hart-local
flags are copied with the CPU state.  AME registers and state are not saved
in Linux-user signal frames or restored by ``rt_sigreturn``.  A signal
handler that uses AME can therefore corrupt the interrupted AME context.
