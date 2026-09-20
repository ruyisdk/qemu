/*
 * RISC-V Ztt integer conversion test
 *
 * Run with N=4, U=8, NM=32, and NA=1.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define DTYPE_INT4       0x40000004
#define DTYPE_INT4_SAT   0x60000004
#define DTYPE_INT8       0x40000008
#define DTYPE_UINT8_SAT  0x20000008
#define DTYPE_INT16      0x40000010
#define DTYPE_INT128     0x40000080
#define DTYPE_FP32       0x20300120

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

    .macro CHECK_RAW_PAIR mreg, even, odd
    li      t1, 0
    li      a2, 16
.Lcheck_pair\@:
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, \mreg, ZTT_R_T1)
    andi    t2, t1, 1
    beqz    t2, .Leven\@
    li      t2, \odd
    j       .Lcompare\@
.Leven\@:
    li      t2, \even
.Lcompare\@:
    bne     t0, t2, fail
    addi    t1, t1, 1
    bne     t1, a2, .Lcheck_pair\@
    .endm

    .macro CHECK_RAW_SPLIT mreg, low, high
    li      t1, 0
    li      a2, 8
    li      a3, 16
.Lcheck_split\@:
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, \mreg, ZTT_R_T1)
    bltu    t1, a2, .Llow\@
    li      t2, \high
    j       .Lsplit_compare\@
.Llow\@:
    li      t2, \low
.Lsplit_compare\@:
    bne     t0, t2, fail
    addi    t1, t1, 1
    bne     t1, a3, .Lcheck_split\@
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

    /* Packed Int4 expands to two Int8 squares with sign extension. */
    SET_M_DTYPE 0, DTYPE_INT4
    SET_M_DTYPE 2, DTYPE_INT8
    FILL_RAW_CONST 0, 0x8f
    .word   ZTT_MCONV_EW(2, 0)
    CHECK_RAW_PAIR 2, 0xff, 0xf8
    CHECK_RAW_PAIR 3, 0xff, 0xf8

    /* Two Int8 squares compress into one saturating Int4 register. */
    SET_M_DTYPE 4, DTYPE_INT8
    SET_M_DTYPE 6, DTYPE_INT4_SAT
    FILL_RAW_CONST 4, 0xf7
    FILL_RAW_CONST 5, 0x08
    csrwi   CSR_AMEXSAT, 0
    .word   ZTT_MCONV_EW(6, 4)
    CHECK_RAW_SPLIT 6, 0x88, 0x77
    csrr    t0, CSR_AMEXSAT
    andi    t0, t0, AME_XSAT_SAT
    beqz    t0, fail

    /* The source may overlap the complete wide destination span. */
    SET_M_DTYPE 8, DTYPE_INT16
    SET_M_DTYPE 9, DTYPE_INT8
    FILL_RAW_CONST 9, 1
    .word   ZTT_MCONV_EW(8, 9)
    CHECK_RAW_PAIR 8, 1, 0
    CHECK_RAW_PAIR 9, 1, 0

    /* Floating sqrt requires a floating destination; UN preserves it. */
    SET_M_DTYPE 10, DTYPE_INT8
    SET_M_DTYPE 12, DTYPE_FP32
    FILL_RAW_CONST 10, 0x5a
    EXPECT_UN ZTT_MSQRT_EW(10, 12)
    CHECK_RAW_CONST 10, 0x5a

    /* Derived spans are checked before any source read or destination write. */
    SET_M_DTYPE 15, DTYPE_INT4
    SET_M_DTYPE 31, DTYPE_INT8
    FILL_RAW_CONST 15, 0x69
    EXPECT_ILLEGAL ZTT_MCONV_EW(15, 31)
    CHECK_RAW_CONST 15, 0x69

    /* Int128 signed negatives saturate to zero in Uint8_Sat. */
    SET_M_DTYPE 7, DTYPE_UINT8_SAT
    SET_M_DTYPE 16, DTYPE_INT128
    FILL_RAW_CONST 16, 0xff
    FILL_RAW_CONST 17, 0xff
    FILL_RAW_CONST 18, 0xff
    FILL_RAW_CONST 19, 0xff
    FILL_RAW_CONST 20, 0xff
    FILL_RAW_CONST 21, 0xff
    FILL_RAW_CONST 22, 0xff
    FILL_RAW_CONST 23, 0xff
    FILL_RAW_CONST 24, 0xff
    FILL_RAW_CONST 25, 0xff
    FILL_RAW_CONST 26, 0xff
    FILL_RAW_CONST 27, 0xff
    FILL_RAW_CONST 28, 0xff
    FILL_RAW_CONST 29, 0xff
    FILL_RAW_CONST 30, 0xff
    FILL_RAW_CONST 31, 0xff
    csrwi   CSR_AMEXSAT, 0
    .word   ZTT_MCONV_EW(7, 16)
    CHECK_RAW_CONST 7, 0
    csrr    t0, CSR_AMEXSAT
    andi    t0, t0, AME_XSAT_SAT
    beqz    t0, fail

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
