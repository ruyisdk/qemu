/*
 * RISC-V Ztt helper calls preserve live scalar registers and precise traps.
 * Run with N=4, U=8, NM=16 and NA=1.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define DTYPE_INT4       0x40000004
#define DTYPE_INT8       0x40000008
#define DTYPE_INT8_SAT   0x60000008
#define DTYPE_FP32       0x20300120

    .option norvc

    .macro SET_M_DTYPE mreg, encoding
    li      t0, \encoding
    .word   ZTT_MSETTYP(\mreg, ZTT_R_T0)
    .endm

    .macro FILL8 mreg, value
    li      t0, \value
    li      t1, 0
    li      t2, 16
.Lfill\@:
    .word   ZTT_MMOVE8_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, t2, .Lfill\@
    .endm

    /* Memory inputs keep these updates live across the following helper. */
    .macro LOAD_LIVE
    lla     t5, live_values
    lw      t3, 0(t5)
    addi    t3, t3, 1
    lw      t4, 4(t5)
    ori     t4, t4, 1
    flw     ft0, 8(t5)
    fsgnjn.s ft0, ft0, ft0
    flw     ft1, 12(t5)
    fsgnjn.s ft1, ft1, ft1
    .endm

    .macro CHECK_LIVE
    li      t0, 0x12345679
    bne     t3, t0, fail
    li      t0, 0x2468ace1
    bne     t4, t0, fail
    fmv.x.w t0, ft0
    li      t1, 0x3fc00000
    bne     t0, t1, fail
    fmv.x.w t0, ft1
    li      t1, 0x40100000
    bne     t0, t1, fail
    .endm

    .macro CHECK_STATE flags, ms
    csrr    t0, CSR_AMEFFLAGS
    li      t1, \flags
    bne     t0, t1, fail
    csrr    t0, CSR_AMEXSAT
    li      t1, AME_XSAT_SAT
    bne     t0, t1, fail
    csrr    t0, mstatus
    li      t1, MSTATUS_AME_MS_MASK
    and     t0, t0, t1
    li      t1, \ms
    bne     t0, t1, fail
    .endm

    .macro EXPECT_ILLEGAL insn
    li      s0, 1
    lla     s1, .Lresume\@
    lla     s2, .Lfault\@
    LOAD_LIVE
.Lfault\@:
    .word   \insn
    j       fail
.Lresume\@:
    li      t0, 2
    bne     s0, t0, fail
    CHECK_LIVE
    .endm

    .text
    .global _start
_start:
    lla     t0, trap_handler
    csrw    mtvec, t0
    li      t0, MSTATUS_AME_MS_INITIAL | MSTATUS_FS_INITIAL
    csrs    mstatus, t0
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail

    SET_M_DTYPE 0, DTYPE_INT8_SAT
    SET_M_DTYPE 1, DTYPE_INT8
    SET_M_DTYPE 2, DTYPE_INT8
    SET_M_DTYPE 3, DTYPE_INT8
    FILL8   1, 127
    FILL8   2, 1
    csrwi   CSR_AMEFFLAGS, 0
    csrwi   CSR_AMEXSAT, 0

    /* Several successful helpers leave live GPR/FPR values unchanged. */
    LOAD_LIVE
    .word   ZTT_MADD_EW(0, 1, 2)
    .word   ZTT_MABS_EW(0, 1)
    .word   ZTT_MCONV_EW(3, 1)
    .word   ZTT_MZERO_M(3)
    .word   ZTT_MMOV_M_M(3, 1)
    .word   ZTT_MPREFIXADD_ROW(0, 1)
    CHECK_LIVE
    CHECK_STATE 0, MSTATUS_AME_MS_DIRTY
    li      t1, 0
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 0, ZTT_R_T1)
    li      t2, 127
    bne     t0, t2, fail

    /* AME overflow updates only AME flags, not the scalar FP registers. */
    SET_M_DTYPE 4, DTYPE_FP32
    SET_M_DTYPE 8, DTYPE_FP32
    lla     t0, fp_max
    .word   ZTT_MLS_RM(8, ZTT_R_T0)
    LOAD_LIVE
    .word   ZTT_MADD_EW(4, 8, 8)
    CHECK_LIVE
    CHECK_STATE (AME_FFLAG_OF | AME_FFLAG_NX), MSTATUS_AME_MS_DIRTY
    li      t1, 0
    .word   ZTT_MMOVE32_X_M(ZTT_R_T0, 4, ZTT_R_T1)
    li      t2, 0x7f800000
    bne     t0, t2, fail

    /* An unsupported destination sets UN without replacing the result. */
    LOAD_LIVE
    .word   ZTT_MSQRT_EW(0, 8)
    CHECK_LIVE
    CHECK_STATE (AME_FFLAG_OF | AME_FFLAG_NX), MSTATUS_AME_MS_DIRTY
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail
    li      t1, 0
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 0, ZTT_R_T1)
    li      t2, 127
    bne     t0, t2, fail

    /* Bounds and derived spans trap before writing AME state. */
    SET_M_DTYPE 3, DTYPE_INT4
    SET_M_DTYPE 15, DTYPE_INT8
    FILL8   15, 0x5a
    csrr    t0, mstatus
    li      t1, ~MSTATUS_AME_MS_MASK
    and     t0, t0, t1
    li      t1, MSTATUS_AME_MS_CLEAN
    or      t0, t0, t1
    csrw    mstatus, t0
    EXPECT_ILLEGAL ZTT_MADD_EW(16, 1, 2)
    CHECK_STATE (AME_FFLAG_OF | AME_FFLAG_NX), MSTATUS_AME_MS_CLEAN
    EXPECT_ILLEGAL ZTT_MCONV_EW(15, 3)
    CHECK_STATE (AME_FFLAG_OF | AME_FFLAG_NX), MSTATUS_AME_MS_CLEAN
    li      t1, 0
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 15, ZTT_R_T1)
    li      t2, 0x5a
    bne     t0, t2, fail

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
    CHECK_LIVE
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

    .section .rodata
    .balign 4
live_values:
    .word   0x12345678, 0x2468ace0, 0xbfc00000, 0xc0100000
fp_max:
    .rept 16
    .word   0x7f7fffff
    .endr

    .bss
    .balign 16
semiargs:
#if __riscv_xlen == 64
    .space  16
#else
    .space  8
#endif
