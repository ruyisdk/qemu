/*
 * RISC-V Ztt linux-user integer elementwise test
 *
 * Run with N=4, U=8, and NM=16.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define NR_EXIT          93
#define DTYPE_INT8       0x40000008
#define DTYPE_INT16      0x40000010
#define DTYPE_UINT8      0x00000008
#define DTYPE_UINT8_SAT  0x20000008

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

    .text
    .global _start
_start:
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail

    SET_M_DTYPE 0, DTYPE_INT8
    SET_M_DTYPE 1, DTYPE_INT8
    SET_M_DTYPE 2, DTYPE_INT8
    FILL_RAW_CONST 1, -3
    FILL_RAW_CONST 2, 4

    .word   ZTT_MABS_EW(0, 1)
    CHECK_RAW_CONST 0, 3
    .word   ZTT_MADD_EW(0, 1, 2)
    CHECK_RAW_CONST 0, 1
    .word   ZTT_MANDNOT_EW(0, 1, 2)
    CHECK_RAW_CONST 0, 0xf9
    .word   ZTT_MCMPLT_EW(0, 1, 2)
    CHECK_RAW_CONST 0, 0xff
    .word   ZTT_MSELLT_EW(0, 1, 2)
    CHECK_RAW_CONST 0, 4
    .word   ZTT_MSRA_EW(0, 1, 2)
    CHECK_RAW_CONST 0, 0xff

    li      t0, DTYPE_INT8
    csrw    CSR_AMESTYPE, t0
    li      t0, 5
    .word   ZTT_MSUB_EW_X(0, ZTT_R_T0, 2)
    CHECK_RAW_CONST 0, 0xff

    /* Fixed control scalar does not depend on amestype. */
    csrwi   CSR_AMESTYPE, 0
    li      t0, 9
    .word   ZTT_MSLL_EW_X(0, ZTT_R_T0, 2)
    CHECK_RAW_CONST 0, 8

    /* Interpret mixed integers before the one destination conversion. */
    SET_M_DTYPE 0, DTYPE_UINT8_SAT
    SET_M_DTYPE 2, DTYPE_UINT8
    FILL_RAW_CONST 1, -1
    FILL_RAW_CONST 2, 1
    csrwi   CSR_AMEXSAT, 0
    .word   ZTT_MABSDIFF_EW(0, 1, 2)
    CHECK_RAW_CONST 0, 2
    .word   ZTT_MCMPGE_EW(0, 2, 1)
    CHECK_RAW_CONST 0, 0xff
    csrr    t0, CSR_AMEXSAT
    bnez    t0, fail

    FILL_RAW_CONST 2, 255
    .word   ZTT_MADD_EW(0, 1, 2)
    CHECK_RAW_CONST 0, 254
    .word   ZTT_MMUL_EW(0, 1, 2)
    CHECK_RAW_CONST 0, 0
    csrr    t0, CSR_AMEXSAT
    li      t1, 1
    bne     t0, t1, fail
    csrwi   CSR_AMEXSAT, 0
    FILL_RAW_CONST 0, 255
    .word   ZTT_MMULACC_EW(0, 1, 2)
    CHECK_RAW_CONST 0, 0
    csrr    t0, CSR_AMEXSAT
    bnez    t0, fail

    /* Scalar Uint8(255) must convert to Int8(-1), then add to Int8(2). */
    SET_M_DTYPE 2, DTYPE_INT8
    FILL_RAW_CONST 2, 2
    li      t0, DTYPE_UINT8
    csrw    CSR_AMESTYPE, t0
    li      t0, 255
    .word   ZTT_MADD_EW_X(0, ZTT_R_T0, 2)
    CHECK_RAW_CONST 0, 1
    csrr    t0, CSR_AMEXSAT
    bnez    t0, fail
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    bnez    t0, fail

    /* An unsupported bitwise tuple sets UN and preserves destination. */
    SET_M_DTYPE 4, DTYPE_INT16
    FILL_RAW_CONST 4, 0x5a
    FILL_RAW_CONST 5, 0x5a
    csrwi   CSR_AMESTATUS, 0
    .word   ZTT_MAND_EW(4, 1, 2)
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail
    CHECK_RAW_CONST 4, 0x5a
    CHECK_RAW_CONST 5, 0x5a

    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       exit

fail:
    li      a0, 1
exit:
    li      a7, NR_EXIT
    ecall
