/*
 * RISC-V Ztt decoder and operand validation test
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

    .option norvc

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

    /* Debug element 5 is [1,1] in either 4x4 memory order. */
    .macro EXPECT_M0 value
    li      t2, 5
    .word   ZTT_MMOVE16_X_M(5, 0, 7)
    li      t1, \value
    bne     t0, t1, fail
    .endm

    .text
    .global _start
_start:
    lla     t0, trap_handler
    csrw    mtvec, t0
    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0

    li      a1, 0
    .word   ZTT_AME_ACQUIRE(10, 11)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail

    /* Cover each decoder argument shape with a valid tensor index. */
    EXPECT_UN ZTT_MADD_EW(0, 0, 0)
    EXPECT_UN ZTT_MABS_EW(0, 0)
    EXPECT_UN ZTT_MCOLID_EW(0)
    EXPECT_UN ZTT_MCOLZIP_EW(0, 1)

    /* X-register 31 is a scalar operand, not a matrix-register index. */
    EXPECT_UN ZTT_MADD_EW_X(0, 31, 0)

    /* This test runs with NM=16 and NA=1. */
    EXPECT_ILLEGAL ZTT_MABS_EW(0, 31)
    EXPECT_ILLEGAL ZTT_MZERO_2D_ACC(1)
    EXPECT_ILLEGAL ZTT_MADD_EW(16, 0, 0)
    EXPECT_ILLEGAL ZTT_MADD_EW_X(0, 31, 16)
    EXPECT_ILLEGAL ZTT_MLDEXP_EW_X(0, 31, 16)
    EXPECT_ILLEGAL ZTT_MMULACC_2D(1, 0, 0)

    /* Encoded M index precedes tuple and address validation. */
    EXPECT_ILLEGAL ZTT_MLS_RM(16, 0)
    EXPECT_UN ZTT_MLS_RM(0, 0)

    /* The representative families must consume distinct decoded operands. */
    li      t0, 0x40000010       /* signed Int16 */
    .word   ZTT_MSETTYP(0, 5)
    .word   ZTT_MSETTYP(1, 5)
    .word   ZTT_MSETTYP(2, 5)
    .word   ZTT_ASETTYP(0, 5)
    csrw    CSR_AMESTYPE, t0
    lla     t6, input_a
    .word   ZTT_MLS_RM(1, 31)
    lla     t6, input_b
    .word   ZTT_MLS_RM(2, 31)
    .word   ZTT_MADD_EW(0, 1, 2)
    EXPECT_M0 8

    li      t6, 7
    .word   ZTT_MADD_EW_X(0, 31, 1)
    EXPECT_M0 13

    /* An invalid amestype blocks data scalar, not control scalar. */
    li      t0, -1
    csrw    CSR_AMESTYPE, t0
    csrwi   CSR_AMEFFLAGS, AME_FFLAG_NV
    csrwi   CSR_AMEXSAT, 1
    EXPECT_UN ZTT_MADD_EW_X(0, 31, 1)
    EXPECT_M0 13
    csrr    t0, CSR_AMEFFLAGS
    li      t1, AME_FFLAG_NV
    bne     t0, t1, fail
    csrr    t0, CSR_AMEXSAT
    li      t1, 1
    bne     t0, t1, fail

    csrwi   CSR_AMESTATUS, 0
    li      t6, 1
    .word   ZTT_MLDEXP_EW_X(0, 31, 1)
    EXPECT_M0 12
    csrr    t0, CSR_AMESTATUS
    bnez    t0, fail

    .word   ZTT_MZERO_2D_ACC(0)
    .word   ZTT_MMULACC_2D(0, 1, 2)
    .word   ZTT_MMOV_M_A(0, 0)
    EXPECT_M0 52

    /* Unassigned permutation encodings must remain illegal. */
    EXPECT_ILLEGAL 0xa2a0002b
    EXPECT_ILLEGAL 0xc20000ab

    /* Reserved overwrite matrix multiply and AME fence encodings. */
    EXPECT_ILLEGAL 0x9200002b
    EXPECT_ILLEGAL 0x9800002b
    EXPECT_ILLEGAL 0xa620002b

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

    .section .rodata
    .balign 16
input_a:
    .hword  1, 2, 3, 4, 5, 6, 7, 8
    .hword  9, 10, 11, 12, 13, 14, 15, 16
input_b:
    .hword  2, 2, 2, 2, 2, 2, 2, 2
    .hword  2, 2, 2, 2, 2, 2, 2, 2
