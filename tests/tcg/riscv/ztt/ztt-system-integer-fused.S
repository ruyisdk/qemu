/*
 * RISC-V Ztt integer fused and scaling test
 *
 * Run with N=4, U=8, NM=32, and NA=1.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define DTYPE_INT8       0x40000008
#define DTYPE_INT8_SAT   0x60000008
#define DTYPE_INT8_ROD   0x58000008

    .option norvc

    .macro SET_M_DTYPE mreg, encoding
    li      t0, \encoding
    .word   ZTT_MSETTYP(\mreg, ZTT_R_T0)
    .endm

    .macro FILL_RAW_CONST mreg, value
    li      t1, 0
    li      a2, 16
.Lfill_raw\@:
    li      t0, \value
    .word   ZTT_MMOVE8_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, a2, .Lfill_raw\@
    .endm

    .macro CHECK_RAW_CONST mreg, value
    li      t1, 0
    li      a2, 16
.Lcheck_raw\@:
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, \mreg, ZTT_R_T1)
    li      t2, \value
    bne     t0, t2, fail
    addi    t1, t1, 1
    bne     t1, a2, .Lcheck_raw\@
    .endm

    .macro RUN_CURRENT insn, initial, expected
    FILL_RAW_CONST 0, \initial
    csrwi   CSR_AMESTATUS, 0
    .word   \insn
    CHECK_RAW_CONST 0, \expected
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    bnez    t0, fail
    .endm

    .macro RUN_SCALAR_CURRENT insn, scalar, initial, expected
    FILL_RAW_CONST 0, \initial
    li      t0, \scalar
    csrwi   CSR_AMESTATUS, 0
    .word   \insn
    CHECK_RAW_CONST 0, \expected
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    bnez    t0, fail
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
    FILL_RAW_CONST 1, 5
    FILL_RAW_CONST 2, 2
    .word   ZTT_MHDIFF_EW(0, 1, 2)
    CHECK_RAW_CONST 0, 0xff
    .word   ZTT_MMEAN_EW(0, 1, 2)
    CHECK_RAW_CONST 0, 4

    li      t0, DTYPE_INT8
    csrw    CSR_AMESTYPE, t0
    li      t0, 5
    .word   ZTT_MHDIFF_EW_X(0, ZTT_R_T0, 2)
    CHECK_RAW_CONST 0, 0xff
    li      t0, 5
    .word   ZTT_MMEAN_EW_X(0, ZTT_R_T0, 2)
    CHECK_RAW_CONST 0, 4

    FILL_RAW_CONST 1, -3
    FILL_RAW_CONST 2, 4
    RUN_CURRENT ZTT_MMULACC_EW(0, 1, 2), 10, 0xfe
    RUN_CURRENT ZTT_MMULACCNEG_EW(0, 1, 2), 10, 22
    RUN_CURRENT ZTT_MMULADD_EW(0, 1, 2), 10, 37
    RUN_CURRENT ZTT_MMULSUB_EW(0, 1, 2), 10, 0xd5
    RUN_CURRENT ZTT_MCMOVGE_EW(0, 1, 2), 10, 10
    RUN_CURRENT ZTT_MCMOVLT_EW(0, 1, 2), 10, 4

    RUN_SCALAR_CURRENT ZTT_MMULACC_EW_X(0, ZTT_R_T0, 2), 5, 10, 30
    RUN_SCALAR_CURRENT ZTT_MMULACCNEG_EW_X(0, ZTT_R_T0, 2), 5, 10, 0xf6
    RUN_SCALAR_CURRENT ZTT_MMULADD_EW_X(0, ZTT_R_T0, 2), 5, 10, 45
    RUN_SCALAR_CURRENT ZTT_MMULSUB_EW_X(0, ZTT_R_T0, 2), 5, 10, 0xdd

    /* Matrix exponents use their declared integer datatype. */
    FILL_RAW_CONST 1, 3
    FILL_RAW_CONST 2, 1
    .word   ZTT_MLDEXP_EW(0, 1, 2)
    CHECK_RAW_CONST 0, 6
    .word   ZTT_MRDEXP_EW(0, 1, 2)
    CHECK_RAW_CONST 0, 2
    RUN_CURRENT ZTT_MLDEXPACC_EW(0, 1, 2), 10, 16
    RUN_CURRENT ZTT_MRDEXPACC_EW(0, 1, 2), 10, 12

    /* Scalar exponents are fixed controls and ignore amestype. */
    csrwi   CSR_AMESTYPE, 0
    li      t0, 1
    .word   ZTT_MLDEXP_EW_X(0, ZTT_R_T0, 1)
    CHECK_RAW_CONST 0, 6
    RUN_SCALAR_CURRENT ZTT_MLDEXPACC_EW_X(0, ZTT_R_T0, 1), 1, 10, 16

    /* Round-to-odd applies to the complete current + scaled expression. */
    SET_M_DTYPE 0, DTYPE_INT8_ROD
    SET_M_DTYPE 1, DTYPE_INT8_ROD
    FILL_RAW_CONST 1, 3
    FILL_RAW_CONST 2, 1
    RUN_CURRENT ZTT_MRDEXPACC_EW(0, 1, 2), 10, 11
    RUN_CURRENT ZTT_MRDEXPACC_EW(0, 1, 2), 11, 13

    /* Saturation and amexsat commit with the destination. */
    SET_M_DTYPE 0, DTYPE_INT8_SAT
    SET_M_DTYPE 1, DTYPE_INT8_SAT
    SET_M_DTYPE 2, DTYPE_INT8_SAT
    FILL_RAW_CONST 1, 20
    FILL_RAW_CONST 2, 2
    csrwi   CSR_AMEXSAT, 0
    RUN_CURRENT ZTT_MMULACC_EW(0, 1, 2), 100, 127
    csrr    t0, CSR_AMEXSAT
    andi    t0, t0, AME_XSAT_SAT
    beqz    t0, fail

    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       semihost_exit

trap_handler:
    j       fail

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
