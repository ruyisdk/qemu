/*
 * RISC-V Ztt linux-user structural tensor operation test
 *
 * Run with N=4, U=8, and NM=16.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define NR_EXIT          93
#define DTYPE_INT4       0x40000004
#define DTYPE_INT4_SAT   0x60000004
#define DTYPE_INT8       0x40000008

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

    .macro FILL_SEQUENCE mreg
    li      t1, 0
    li      a2, 16
.Lfill_sequence\@:
    mv      t0, t1
    .word   ZTT_MMOVE8_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, a2, .Lfill_sequence\@
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

    /* Packed coordinate generation is available in user mode. */
    SET_M_DTYPE 0, DTYPE_INT4
    .word   ZTT_MCOLID_EW(0)
    li      t1, 0
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 0, ZTT_R_T1)
    li      t2, 0x10
    bne     t0, t2, fail
    li      t1, 1
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 0, ZTT_R_T1)
    li      t2, 0x32
    bne     t0, t2, fail
    li      t1, 8
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 0, ZTT_R_T1)
    li      t2, 0x10
    bne     t0, t2, fail

    SET_M_DTYPE 1, DTYPE_INT8
    li      t0, -3
    li      t1, DTYPE_INT8
    .word   ZTT_MBCAST_M_X(1, ZTT_R_T0, ZTT_R_T1)
    CHECK_RAW_CONST 1, 0xfd

    SET_M_DTYPE 2, DTYPE_INT8
    SET_M_DTYPE 3, DTYPE_INT8
    SET_M_DTYPE 9, DTYPE_INT8
    FILL_SEQUENCE 2
    li      t0, 1
    .word   ZTT_MCOLSHIFT_EW_X(3, ZTT_R_T0, 2)
    li      t1, 0
.Lcheck_shift:
    andi    t2, t1, 3
    li      a2, 3
    beq     t2, a2, .Lshift_zero
    addi    t2, t1, 1
    j       .Lshift_compare
.Lshift_zero:
    li      t2, 0
.Lshift_compare:
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 3, ZTT_R_T1)
    bne     t0, t2, fail
    addi    t1, t1, 1
    li      t2, 16
    bne     t1, t2, .Lcheck_shift

    FILL_RAW_CONST 9, 3
    .word   ZTT_MCOLGATHER_EW(3, 2, 9)
    li      t1, 0
.Lcheck_gather:
    srli    t2, t1, 2
    slli    t2, t2, 2
    addi    t2, t2, 3
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 3, ZTT_R_T1)
    bne     t0, t2, fail
    addi    t1, t1, 1
    li      t2, 16
    bne     t1, t2, .Lcheck_gather

    SET_M_DTYPE 4, DTYPE_INT4_SAT
    SET_M_DTYPE 5, DTYPE_INT8
    SET_M_DTYPE 6, DTYPE_INT8
    FILL_RAW_CONST 4, 0x11
    FILL_RAW_CONST 5, 9
    csrwi   CSR_AMEXSAT, 0
    li      t0, 3
    .word   ZTT_MPACK_EW_X(4, ZTT_R_T0, 5)
    csrr    t0, CSR_AMEXSAT
    andi    t0, t0, AME_XSAT_SAT
    beqz    t0, fail
    li      t0, 3
    .word   ZTT_MUNPACK_EW_X(6, ZTT_R_T0, 4)
    CHECK_RAW_CONST 6, 7

    /* Zip consumes both pre-instruction sources before either write. */
    SET_M_DTYPE 7, DTYPE_INT8
    SET_M_DTYPE 8, DTYPE_INT8
    FILL_RAW_CONST 7, 0x11
    FILL_RAW_CONST 8, 0x22
    .word   ZTT_MCOLZIP_EW(7, 8)
    li      t1, 0
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 7, ZTT_R_T1)
    li      t2, 0x11
    bne     t0, t2, fail
    li      t1, 1
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 7, ZTT_R_T1)
    li      t2, 0x22
    bne     t0, t2, fail
    .word   ZTT_MCOLUNZIP_EW(7, 8)
    CHECK_RAW_CONST 7, 0x11
    CHECK_RAW_CONST 8, 0x22

    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       exit

fail:
    li      a0, 1
exit:
    li      a7, NR_EXIT
    ecall
