/*
 * RISC-V Ztt integer reduction and scatter test
 *
 * Run with N=4, U=8, NM=32, and NA=1.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define DTYPE_INT4       0x40000004
#define DTYPE_INT8       0x40000008
#define DTYPE_INT8_SAT   0x60000008

    .option norvc

    .macro SET_M_DTYPE mreg, encoding
    li      t0, \encoding
    .word   ZTT_MSETTYP(\mreg, ZTT_R_T0)
    .endm

    .macro FILL_SEQUENCE mreg
    li      t1, 0
    li      a2, 16
.Lfill_sequence\@:
    mv      t0, t1
    .word   ZTT_MMOVE8_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, a2, .Lfill_sequence\@
    .endm

    .macro FILL_RAW_CONST mreg, value
    li      t1, 0
    li      a2, 16
.Lfill_const\@:
    li      t0, \value
    .word   ZTT_MMOVE8_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, a2, .Lfill_const\@
    .endm

    .macro CHECK_ELEM mreg, index, value
    li      t1, \index
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, \mreg, ZTT_R_T1)
    li      t2, \value
    bne     t0, t2, fail
    .endm

    .text
    .global _start
_start:
    lla     t0, trap_handler
    csrw    mtvec, t0
    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0

    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail

    SET_M_DTYPE 0, DTYPE_INT8
    SET_M_DTYPE 1, DTYPE_INT8
    SET_M_DTYPE 2, DTYPE_INT8
    FILL_SEQUENCE 0

    /* Exercise every row/column prefix and reduction opcode. */
    .word   ZTT_MPREFIXADD_ROW(1, 0)
    CHECK_ELEM 1, 3, 6
    .word   ZTT_MPREFIXADD_COL(1, 0)
    CHECK_ELEM 1, 12, 24
    .word   ZTT_MPREFIXMAX_ROW(1, 0)
    CHECK_ELEM 1, 3, 3
    .word   ZTT_MPREFIXMAX_COL(1, 0)
    CHECK_ELEM 1, 12, 12
    .word   ZTT_MREDUCEADD_ROW(1, 0)
    CHECK_ELEM 1, 0, 6
    .word   ZTT_MREDUCEADD_COL(1, 0)
    CHECK_ELEM 1, 0, 24
    .word   ZTT_MREDUCEMAX_ROW(1, 0)
    CHECK_ELEM 1, 0, 3
    .word   ZTT_MREDUCEMAX_COL(1, 0)
    CHECK_ELEM 1, 0, 12
    .word   ZTT_MREDUCEMIN_ROW(1, 0)
    CHECK_ELEM 1, 3, 0
    .word   ZTT_MREDUCEMIN_COL(1, 0)
    CHECK_ELEM 1, 12, 0

    /* All contributions collide at index 3. */
    FILL_RAW_CONST 2, 3
    .word   ZTT_MZERO_M(1)
    .word   ZTT_MCOLSCATADD_EW(1, 0, 2)
    CHECK_ELEM 1, 3, 6
    CHECK_ELEM 1, 7, 22

    .word   ZTT_MZERO_M(1)
    .word   ZTT_MROWSCATADD_EW(1, 0, 2)
    CHECK_ELEM 1, 12, 24
    CHECK_ELEM 1, 15, 36

    .word   ZTT_MZERO_M(1)
    .word   ZTT_MCOLSCATMAX_EW(1, 0, 2)
    CHECK_ELEM 1, 3, 3
    CHECK_ELEM 1, 7, 7

    .word   ZTT_MZERO_M(1)
    .word   ZTT_MROWSCATMAX_EW(1, 0, 2)
    CHECK_ELEM 1, 12, 12
    CHECK_ELEM 1, 15, 15

    /* In-place scatter reads every contribution from the old source. */
    .word   ZTT_MCOLSCATADD_EW(0, 0, 2)
    CHECK_ELEM 0, 3, 9
    CHECK_ELEM 0, 7, 29

    /* Saturation is staged in source order and committed with the result. */
    SET_M_DTYPE 3, DTYPE_INT8_SAT
    SET_M_DTYPE 4, DTYPE_INT8
    li      t1, 0
    li      a2, 16
.Lfill_contributions:
    andi    t2, t1, 3
    li      t0, 120
    li      a3, 2
    bltu    t2, a3, .Lstore_contribution
    li      t0, -120
    beq     t2, a3, .Lstore_contribution
    li      t0, 0
.Lstore_contribution:
    .word   ZTT_MMOVE8_M_X(4, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, a2, .Lfill_contributions
    FILL_RAW_CONST 2, 1
    .word   ZTT_MZERO_M(3)
    csrwi   CSR_AMEXSAT, 0
    .word   ZTT_MCOLSCATADD_EW(3, 4, 2)
    CHECK_ELEM 3, 1, 7
    csrr    t0, CSR_AMEXSAT
    andi    t0, t0, AME_XSAT_SAT
    beqz    t0, fail

    /* Packed permutation tuples set UN and preserve the destination. */
    SET_M_DTYPE 5, DTYPE_INT4
    SET_M_DTYPE 6, DTYPE_INT4
    SET_M_DTYPE 7, DTYPE_INT8
    FILL_RAW_CONST 5, 0x5a
    FILL_RAW_CONST 6, 0x11
    FILL_RAW_CONST 7, 0
    csrwi   CSR_AMESTATUS, 0
    .word   ZTT_MROWSCATADD_EW(5, 6, 7)
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail
    CHECK_ELEM 5, 0, 0x5a

    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       semihost_exit

trap_handler:
fail:
    li      a0, 1

semihost_exit:
    lla     a1, semiargs
    li      t0, 0x20026
#if __riscv_xlen == 64
    sd      t0, 0(a1)
    sd      a0, 8(a1)
#else
    sw      t0, 0(a1)
    sw      a0, 4(a1)
#endif
    li      a0, 0x20
    .balign 16
    slli    zero, zero, 0x1f
    ebreak
    srai    zero, zero, 0x7
    j       .

    .bss
    .balign 16
semiargs:
#if __riscv_xlen == 64
    .space  16
#else
    .space  8
#endif
