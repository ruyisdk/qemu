/*
 * RISC-V Ztt matrix and accumulator move test
 *
 * Run with N=4, U=8, NM=16, and NA=4.  The fixed-width debug moves used to
 * seed and inspect raw M storage are implemented before this test is enabled.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define DTYPE_INT4  0x40000004
#define DTYPE_INT8  0x40000008
#define DTYPE_INT16 0x40000010

    .option norvc

    .macro SET_M_DTYPE mreg, encoding
    li      t0, \encoding
    .word   ZTT_MSETTYP(\mreg, ZTT_R_T0)
    .endm

    .macro SET_A_DTYPE acc, encoding
    li      t0, \encoding
    .word   ZTT_ASETTYP(\acc, ZTT_R_T0)
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

    .macro FILL_RAW_SEQUENCE mreg
    li      t1, 0
    li      a2, 16
.Lfill_seq\@:
    addi    t0, t1, 1
    .word   ZTT_MMOVE8_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, a2, .Lfill_seq\@
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

    .macro CHECK_RAW_SEQUENCE mreg
    li      t1, 0
    li      a2, 16
.Lcheck_seq\@:
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, \mreg, ZTT_R_T1)
    addi    t2, t1, 1
    bne     t0, t2, fail
    addi    t1, t1, 1
    bne     t1, a2, .Lcheck_seq\@
    .endm

    .macro EXPECT_UN insn
    csrwi   CSR_AMESTATUS, 0
    .word   \insn
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail
    .endm

    .macro EXPECT_ILLEGAL insn
    li      s0, 1
    lla     s1, .Lresume\@
    lla     s2, .Lfault\@
.Lfault\@:
    .word   \insn
    j       fail
.Lresume\@:
    li      t0, 2
    bne     s0, t0, fail
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

    /* Unit datatype: M-to-M copy and exact source/destination overlap. */
    SET_M_DTYPE 0, DTYPE_INT8
    SET_M_DTYPE 1, DTYPE_INT8
    FILL_RAW_SEQUENCE 1
    .word   ZTT_MMOV_M_M(0, 1)
    CHECK_RAW_SEQUENCE 0
    .word   ZTT_MMOV_M_M(0, 0)
    CHECK_RAW_SEQUENCE 0

    /* Packed Int4 copies both logical squares in one physical M register. */
    SET_M_DTYPE 2, DTYPE_INT4
    SET_M_DTYPE 3, DTYPE_INT4
    FILL_RAW_CONST 2, 0xa5
    .word   ZTT_MMOV_M_M(3, 2)
    CHECK_RAW_CONST 3, 0xa5

    /* Packed M-to-Acc uses two Acc squares and ignores the derived Ad. */
    SET_A_DTYPE 0, DTYPE_INT4
    SET_A_DTYPE 1, DTYPE_INT16
    .word   ZTT_MMOV_A_M(0, 2)
    .word   ZTT_MZERO_M(3)
    .word   ZTT_MMOV_M_A(3, 0)
    CHECK_RAW_CONST 3, 0xa5
    .word   ZTT_AGETTYP(ZTT_R_T0, 1)
    li      t1, DTYPE_INT16
    bne     t0, t1, fail

    /* A derived Acc-span failure is precise and cannot partially update. */
    SET_M_DTYPE 4, DTYPE_INT4
    SET_M_DTYPE 5, DTYPE_INT4
    SET_M_DTYPE 6, DTYPE_INT4
    SET_A_DTYPE 2, DTYPE_INT4
    SET_A_DTYPE 3, DTYPE_INT4
    FILL_RAW_CONST 4, 0x3c
    .word   ZTT_MMOV_A_M(2, 2)
    EXPECT_ILLEGAL ZTT_MMOV_A_M(3, 4)
    .word   ZTT_MMOV_M_A(5, 2)
    CHECK_RAW_CONST 5, 0xa5

    FILL_RAW_CONST 6, 0x69
    EXPECT_ILLEGAL ZTT_MMOV_M_A(6, 3)
    CHECK_RAW_CONST 6, 0x69

    /* A unit datatype maps one M square to exactly one Acc square. */
    SET_M_DTYPE 8, DTYPE_INT8
    SET_M_DTYPE 9, DTYPE_INT8
    SET_A_DTYPE 0, DTYPE_INT8
    FILL_RAW_SEQUENCE 8
    .word   ZTT_MMOV_A_M(0, 8)
    .word   ZTT_MMOV_M_A(9, 0)
    CHECK_RAW_SEQUENCE 9

    /* A wide Int16 square spans two M registers but still one Acc. */
    SET_M_DTYPE 10, DTYPE_INT16
    SET_M_DTYPE 12, DTYPE_INT16
    SET_A_DTYPE 1, DTYPE_INT16
    FILL_RAW_CONST 10, 0x12
    FILL_RAW_CONST 11, 0x34
    .word   ZTT_MMOV_M_M(12, 10)
    CHECK_RAW_CONST 12, 0x12
    CHECK_RAW_CONST 13, 0x34
    .word   ZTT_MMOV_A_M(1, 10)
    .word   ZTT_MZERO_M(12)
    .word   ZTT_MMOV_M_A(12, 1)
    CHECK_RAW_CONST 12, 0x12
    CHECK_RAW_CONST 13, 0x34

    /* Datatype mismatch sets UN and preserves the complete destination. */
    SET_M_DTYPE 14, DTYPE_INT4
    SET_M_DTYPE 15, DTYPE_INT8
    FILL_RAW_CONST 14, 0x5a
    FILL_RAW_CONST 15, 0x77
    EXPECT_UN ZTT_MMOV_M_A(14, 0)
    CHECK_RAW_CONST 14, 0x5a
    EXPECT_UN ZTT_MMOV_M_M(15, 14)
    CHECK_RAW_CONST 15, 0x77

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
