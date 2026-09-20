/*
 * RISC-V Ztt linux-user narrowing integer arithmetic test
 *
 * Run with N=4, U=8, NM=16, and NA=1.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define DTYPE_INT8_SAT  0x60000008
#define DTYPE_INT16     0x40000010
#define DTYPE_UINT8     0x00000008

    .option norvc

    .macro SET_M_DTYPE mreg, encoding
    li      t0, \encoding
    .word   ZTT_MSETTYP(\mreg, ZTT_R_T0)
    .endm

    .macro FILL8 mreg, value
    li      t1, 0
    li      a2, 16
.Lfill8\@:
    li      t0, \value
    .word   ZTT_MMOVE8_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, a2, .Lfill8\@
    .endm

    .macro FILL16 mreg, value
    li      t1, 0
    li      a2, 8
.Lfill16\@:
    li      t0, \value
    .word   ZTT_MMOVE16_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, a2, .Lfill16\@
    .endm

    .macro FILL_WIDE mreg, value
    FILL16  \mreg, \value
    FILL16  (\mreg + 1), \value
    .endm

    .macro FILL_ALTERNATING mreg
    li      t1, 0
    li      a2, 8
.Lfill_alternating\@:
    li      t0, 1000
    andi    t2, t1, 1
    beqz    t2, .Lstore_alternating\@
    li      t0, -990
.Lstore_alternating\@:
    .word   ZTT_MMOVE16_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, a2, .Lfill_alternating\@
    .endm

    .macro CHECK8 mreg, value
    li      t1, 0
    li      a2, 16
.Lcheck8\@:
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, \mreg, ZTT_R_T1)
    li      t2, \value
    bne     t0, t2, fail
    addi    t1, t1, 1
    bne     t1, a2, .Lcheck8\@
    .endm

    .macro CHECK_SAT value
    csrr    t0, CSR_AMEXSAT
    li      t1, \value
    bne     t0, t1, fail
    .endm

    .text
    .global _start
_start:
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail
    SET_M_DTYPE 0, DTYPE_INT8_SAT
    SET_M_DTYPE 4, DTYPE_INT16
    SET_M_DTYPE 8, DTYPE_INT16
    csrwi   CSR_AMESTATUS, 0
    csrwi   CSR_AMEXSAT, 0

    /* Complete the wide expression before narrowing to Int8_Sat. */
    FILL_WIDE 4, 1000
    FILL_WIDE 8, -990
    .word   ZTT_MADD_EW(0, 4, 8)
    CHECK8  0, 10
    .word   ZTT_MMEAN_EW(0, 4, 8)
    CHECK8  0, 5
    CHECK_SAT 0
    .word   ZTT_MSUB_EW(0, 4, 8)
    CHECK8  0, 128
    CHECK_SAT 1
    csrwi   CSR_AMEXSAT, 0

    /* Data scalar conversion is to Int16; only the sum narrows to D. */
    li      t0, DTYPE_INT16
    csrw    CSR_AMESTYPE, t0
    li      t0, -990
    .word   ZTT_MADD_EW_X(0, ZTT_R_T0, 4)
    CHECK8  0, 10
    CHECK_SAT 0

    /* Do not saturate the product before adding the old destination. */
    FILL_WIDE 4, 128
    FILL_WIDE 8, 1
    FILL8   0, 128
    .word   ZTT_MMULACC_EW(0, 4, 8)
    CHECK8  0, 0
    FILL8   0, 128
    .word   ZTT_MMULADD_EW(0, 4, 8)
    CHECK8  0, 0
    FILL_WIDE 4, -128
    FILL8   0, 128
    .word   ZTT_MMULACCNEG_EW(0, 4, 8)
    CHECK8  0, 0
    FILL8   0, 128
    .word   ZTT_MMULSUB_EW(0, 4, 8)
    CHECK8  0, 0
    CHECK_SAT 0

    FILL_WIDE 4, 1024
    FILL_WIDE 8, 4
    .word   ZTT_MRDEXP_EW(0, 4, 8)
    CHECK8  0, 64
    FILL_WIDE 4, 64
    FILL8   0, 128
    li      t0, 1
    .word   ZTT_MLDEXPACC_EW_X(0, ZTT_R_T0, 4)
    CHECK8  0, 0
    CHECK_SAT 0

    /* Reduce converts each leaf first: 127 - 128 + 127 - 128 = -2. */
    FILL_ALTERNATING 4
    FILL_ALTERNATING 5
    .word   ZTT_MREDUCEADD_ROW(0, 4)
    CHECK8  0, 254
    CHECK_SAT 1

    /* Scatter has the same per-contribution conversion boundary. */
    SET_M_DTYPE 12, DTYPE_UINT8
    FILL8   12, 1
    FILL8   0, 0
    csrwi   CSR_AMEXSAT, 0
    .word   ZTT_MCOLSCATADD_EW(0, 4, 12)
    li      t1, 0
    li      a2, 16
.Lcheck_scatter:
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 0, ZTT_R_T1)
    andi    t2, t1, 3
    li      t3, 1
    li      t4, 0
    bne     t2, t3, .Lcompare_scatter
    li      t4, 254
.Lcompare_scatter:
    bne     t0, t4, fail
    addi    t1, t1, 1
    bne     t1, a2, .Lcheck_scatter
    CHECK_SAT 1

    /* Matrix dot products keep wide products until the final Acc conversion. */
    FILL_WIDE 8, 1
    FILL8   0, 0
    li      t0, DTYPE_INT8_SAT
    .word   ZTT_ASETTYP(0, ZTT_R_T0)
    .word   ZTT_MMOV_A_M(0, 0)
    csrwi   CSR_AMEXSAT, 0
    .word   ZTT_MMULACC_2D(0, 4, 8)
    .word   ZTT_MMOV_M_A(0, 0)
    CHECK8  0, 20
    CHECK_SAT 0
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    bnez    t0, fail

    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       exit
fail:
    li      a0, 1
exit:
    li      a7, 93
    ecall
