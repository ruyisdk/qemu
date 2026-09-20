/*
 * RISC-V Ztt zip and unzip test
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
#define DTYPE_INT16      0x40000010

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

    .macro FILL_SEQUENCE mreg, base
    li      t1, 0
    li      a2, 16
.Lfill_sequence\@:
    addi    t0, t1, \base
    .word   ZTT_MMOVE8_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, a2, .Lfill_sequence\@
    .endm

    .macro CHECK_SEQUENCE mreg, base
    li      t1, 0
    li      a2, 16
.Lcheck_sequence\@:
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, \mreg, ZTT_R_T1)
    addi    t2, t1, \base
    bne     t0, t2, fail
    addi    t1, t1, 1
    bne     t1, a2, .Lcheck_sequence\@
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
    FILL_SEQUENCE 0, 0
    FILL_SEQUENCE 1, 16

    /* Check the column zip mapping before checking its inverse. */
    .word   ZTT_MCOLZIP_EW(0, 1)
    li      t1, 0
.Lcheck_colzip:
    srli    a2, t1, 2
    andi    a3, t1, 3
    andi    a4, a3, 1
    srli    a3, a3, 1
    slli    t2, a2, 2
    add     t2, t2, a3
    slli    a4, a4, 4
    add     t2, t2, a4
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 0, ZTT_R_T1)
    bne     t0, t2, fail

    addi    a3, a3, 2
    slli    t2, a2, 2
    add     t2, t2, a3
    add     t2, t2, a4
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 1, ZTT_R_T1)
    bne     t0, t2, fail
    addi    t1, t1, 1
    li      t2, 16
    bne     t1, t2, .Lcheck_colzip
    .word   ZTT_MCOLUNZIP_EW(0, 1)
    CHECK_SEQUENCE 0, 0
    CHECK_SEQUENCE 1, 16

    /* Check the row zip mapping before checking its inverse. */
    .word   ZTT_MROWZIP_EW(0, 1)
    li      t1, 0
.Lcheck_rowzip:
    srli    a2, t1, 2
    andi    a3, t1, 3
    andi    a4, a2, 1
    srli    a2, a2, 1
    slli    t2, a2, 2
    add     t2, t2, a3
    slli    a4, a4, 4
    add     t2, t2, a4
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 0, ZTT_R_T1)
    bne     t0, t2, fail

    addi    a2, a2, 2
    slli    t2, a2, 2
    add     t2, t2, a3
    add     t2, t2, a4
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 1, ZTT_R_T1)
    bne     t0, t2, fail
    addi    t1, t1, 1
    li      t2, 16
    bne     t1, t2, .Lcheck_rowzip
    .word   ZTT_MROWUNZIP_EW(0, 1)
    CHECK_SEQUENCE 0, 0
    CHECK_SEQUENCE 1, 16

    /* Wide datatypes use two disjoint physical groups atomically. */
    SET_M_DTYPE 8, DTYPE_INT16
    SET_M_DTYPE 10, DTYPE_INT16
    FILL_RAW_CONST 8, 0x11
    FILL_RAW_CONST 9, 0x22
    FILL_RAW_CONST 10, 0x33
    FILL_RAW_CONST 11, 0x44
    .word   ZTT_MROWZIP_EW(8, 10)
    .word   ZTT_MROWUNZIP_EW(8, 10)
    CHECK_RAW_CONST 8, 0x11
    CHECK_RAW_CONST 9, 0x22
    CHECK_RAW_CONST 10, 0x33
    CHECK_RAW_CONST 11, 0x44

    /* Mismatched datatypes set UN and preserve both destinations. */
    SET_M_DTYPE 0, DTYPE_INT8
    SET_M_DTYPE 2, DTYPE_INT16
    FILL_RAW_CONST 0, 0x51
    FILL_RAW_CONST 2, 0x62
    FILL_RAW_CONST 3, 0x73
    csrwi   CSR_AMESTATUS, 0
    .word   ZTT_MROWZIP_EW(0, 2)
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail
    CHECK_RAW_CONST 0, 0x51
    CHECK_RAW_CONST 2, 0x62
    CHECK_RAW_CONST 3, 0x73

    /* Packed tuples are unsupported and preserve both raw registers. */
    SET_M_DTYPE 4, DTYPE_INT4
    SET_M_DTYPE 5, DTYPE_INT4
    FILL_RAW_CONST 4, 0x64
    FILL_RAW_CONST 5, 0x75
    csrwi   CSR_AMESTATUS, 0
    .word   ZTT_MCOLZIP_EW(4, 5)
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail
    CHECK_RAW_CONST 4, 0x64
    CHECK_RAW_CONST 5, 0x75

    /* Equal operands are Illegal before datatype or register state is read. */
    SET_M_DTYPE 0, DTYPE_INT8
    FILL_RAW_CONST 0, 0x5a
    csrwi   CSR_AMESTATUS, 0
    li      s0, 1
    lla     s1, .Lafter_equal
    lla     s2, .Lequal_fault
.Lequal_fault:
    .word   ZTT_MROWZIP_EW(0, 0)
    j       fail
.Lafter_equal:
    li      t0, 2
    bne     s0, t0, fail
    csrr    t0, CSR_AMESTATUS
    bnez    t0, fail
    CHECK_RAW_CONST 0, 0x5a

    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       semihost_exit

trap_handler:
    csrr    t0, mcause
    li      t1, 2
    bne     t0, t1, fail
    csrr    t0, mepc
    bne     t0, s2, fail
    li      t0, 1
    bne     s0, t0, fail
    li      s0, 2
    csrw    mepc, s1
    mret

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
