/*
 * RISC-V Ztt nonlinear floating-point execution test
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

    .option norvc

    .macro SET_M_DTYPE mreg, encoding
    li      t0, \encoding
    .word   ZTT_MSETTYP(\mreg, ZTT_R_T0)
    .endm

    .macro FILL16 mreg, value
    li      t0, \value
    li      t1, 0
    li      t2, 32
1:
    .word   ZTT_MMOVE16_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, t2, 1b
    .endm

    .macro FILL32 mreg, value
    li      t0, \value
    li      t1, 0
    li      t2, 16
1:
    .word   ZTT_MMOVE32_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, t2, 1b
    .endm

    .macro FILL64_HALF mreg, high
    li      t0, \high
    li      t1, 1
    li      t2, 17
1:
    .word   ZTT_MMOVE32_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 2
    bne     t1, t2, 1b
    .endm

    .macro FILL64 mreg0, mreg1, high
    FILL32  \mreg0, 0
    FILL32  \mreg1, 0
    FILL64_HALF \mreg0, \high
    FILL64_HALF \mreg1, \high
    .endm

    .macro CHECK16 mreg, position, expected
    li      t1, \position
    .word   ZTT_MMOVE16_X_M(ZTT_R_T0, \mreg, ZTT_R_T1)
    li      t2, \expected
    bne     t0, t2, fail
    .endm

    .macro CHECK32 mreg, position, expected
    li      t1, \position
    .word   ZTT_MMOVE32_X_M(ZTT_R_T0, \mreg, ZTT_R_T1)
    li      t2, \expected
    bne     t0, t2, fail
    .endm

    .macro CHECK64 mreg, low, high
    CHECK32 \mreg, 0, \low
    CHECK32 \mreg, 1, \high
    .endm

    .macro CLEAR_RESULT_STATE
    csrw    CSR_AMESTATUS, zero
    csrw    CSR_AMEFFLAGS, zero
    .endm

    .macro CHECK_FLAGS expected
    csrr    t0, CSR_AMEFFLAGS
    andi    t0, t0, AME_FFLAGS_MASK
    li      t1, \expected
    bne     t0, t1, fail
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    bnez    t0, fail
    .endm

    .text
    .global _start
_start:
    li      s0, 1
    lla     t0, trap_handler
    csrw    mtvec, t0
    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail

    /* FP32 exact points and one approximate tanh result. */
    li      s0, 2
    SET_M_DTYPE 0, 0x20300120
    SET_M_DTYPE 1, 0x20300120
    SET_M_DTYPE 2, 0x20300120
    FILL32  1, 0x40800000
    CLEAR_RESULT_STATE
    .word   ZTT_MRSQRT_EW(0, 1)
    CHECK32 0, 0, 0x3f000000
    CHECK_FLAGS 0

    FILL32  1, 0x41000000
    CLEAR_RESULT_STATE
    .word   ZTT_MLOG2_EW(0, 1)
    CHECK32 0, 0, 0x40400000
    CHECK_FLAGS 0

    FILL32  1, 0x40400000
    CLEAR_RESULT_STATE
    .word   ZTT_MEXP2_EW(0, 1)
    CHECK32 0, 0, 0x41000000
    CHECK_FLAGS 0

    FILL32  1, 0x3f800000
    CLEAR_RESULT_STATE
    .word   ZTT_MTANH_EW(0, 1)
    CHECK32 0, 0, 0x3f42f7d6
    CHECK_FLAGS AME_FFLAG_NX

    /* Compound log2 operations follow the architected operand order. */
    li      s0, 3
    FILL32  1, 0x41000000
    FILL32  2, 0x3f800000
    CLEAR_RESULT_STATE
    .word   ZTT_MLOG2SUB_EW(0, 1, 2)
    CHECK32 0, 0, 0x40000000
    CHECK_FLAGS 0
    CLEAR_RESULT_STATE
    .word   ZTT_MSUBLOG2_EW(0, 1, 2)
    CHECK32 0, 0, 0xc0000000
    CHECK_FLAGS 0

    li      t0, 0x20300120
    csrw    CSR_AMESTYPE, t0
    li      t0, 0x3f800000
    CLEAR_RESULT_STATE
    .word   ZTT_MLOG2SUB_EW_X(0, ZTT_R_T0, 1)
    CHECK32 0, 0, 0x40000000
    CHECK_FLAGS 0
    li      t0, 0x3f800000
    CLEAR_RESULT_STATE
    .word   ZTT_MSUBLOG2_EW_X(0, ZTT_R_T0, 1)
    CHECK32 0, 0, 0xc0000000
    CHECK_FLAGS 0

    /* Compound log2 converts both intermediate terms to FP16. */
    li      s0, 31
    SET_M_DTYPE 0, 0x14300110       /* FP16 destination */
    SET_M_DTYPE 2, 0x20300120       /* FP32 logarithm source */
    SET_M_DTYPE 4, 0x20300110       /* BF16 bias */
    FILL32  2, 0x40000000           /* 2.0 */
    FILL32  3, 0x40000000
    FILL16  4, 0x3f00               /* 0.5 */
    CLEAR_RESULT_STATE
    li      s0, 311
    .word   ZTT_MLOG2SUB_EW(0, 2, 4)
    CHECK16 0, 0, 0x3800            /* 1.0 - 0.5 */
    CHECK16 0, 31, 0x3800
    .word   ZTT_MSUBLOG2_EW(0, 2, 4)
    CHECK16 0, 0, 0xb800            /* 0.5 - 1.0 */
    CHECK_FLAGS 0

    /* log2(1.5) is rounded to FP16 before subtracting the bias. */
    SET_M_DTYPE 4, 0x20300120
    FILL32  2, 0x3fc00000
    FILL32  3, 0x3fc00000
    FILL32  4, 0x3f15c000
    FILL32  5, 0x3f15c000
    CLEAR_RESULT_STATE
    li      s0, 312
    .word   ZTT_MLOG2SUB_EW(0, 2, 4)
    CHECK16 0, 0, 0
    CHECK16 0, 31, 0
    CHECK_FLAGS AME_FFLAG_NX

    FILL32  2, 0x40000000
    FILL32  3, 0x40000000
    li      s0, 313
    CLEAR_RESULT_STATE
    li      t0, 0x20300110
    csrw    CSR_AMESTYPE, t0
    li      t0, 0x3f00
    .word   ZTT_MLOG2SUB_EW_X(0, ZTT_R_T0, 2)
    CHECK16 0, 31, 0x3800
    li      s0, 314
    li      t0, 0x3f00
    .word   ZTT_MSUBLOG2_EW_X(0, ZTT_R_T0, 2)
    CHECK16 0, 0, 0xb800
    CHECK_FLAGS 0

    /* FP64 reaches the binary128 intermediate path through real decode. */
    li      s0, 41
    SET_M_DTYPE 0, 0x2c300140
    SET_M_DTYPE 2, 0x2c300140
    FILL64  2, 3, 0x40100000
    CLEAR_RESULT_STATE
    .word   ZTT_MRSQRT_EW(0, 2)
    CHECK64 0, 0, 0x3fe00000
    CHECK_FLAGS 0
    li      s0, 42
    FILL64  2, 3, 0x3ff00000
    CLEAR_RESULT_STATE
    .word   ZTT_MTANH_EW(0, 2)
    CHECK64 0, 0xb514f394, 0x3fe85efa
    CHECK_FLAGS AME_FFLAG_NX

    /* Trigonometric tuples are deliberately limited to FP16. */
    li      s0, 5
    SET_M_DTYPE 0, 0x14300110
    SET_M_DTYPE 1, 0x14300110
    FILL16  1, 0x3c00
    CLEAR_RESULT_STATE
    .word   ZTT_MSIN_EW(0, 1)
    CHECK16 0, 0, 0x3abb
    CHECK_FLAGS AME_FFLAG_NX
    CLEAR_RESULT_STATE
    .word   ZTT_MCOS_EW(0, 1)
    CHECK16 0, 31, 0x3853
    CHECK_FLAGS AME_FFLAG_NX

    /* Wider trig domains, with the old flags preserved. */
    li      s0, 6
    SET_M_DTYPE 0, 0x20300110
    SET_M_DTYPE 1, 0x20300110
    FILL16  0, 0x55aa
    FILL16  1, 0x3f80
    li      t0, AME_FFLAGS_MASK
    csrw    CSR_AMEFFLAGS, t0
    csrw    CSR_AMESTATUS, zero
    .word   ZTT_MSIN_EW(0, 1)
    CHECK16 0, 0, 0x3f57
    CHECK16 0, 31, 0x3f57
    CHECK_FLAGS AME_FFLAGS_MASK

    /* One packed BF16 register feeds two FP32 destination registers. */
    SET_M_DTYPE 0, 0x20300120
    SET_M_DTYPE 2, 0x20300110
    FILL16  2, 0x3f80
    csrw    CSR_AMESTATUS, zero
    .word   ZTT_MCOS_EW(0, 2)
    CHECK32 0, 0, 0x3f0a5140
    CHECK32 1, 15, 0x3f0a5140
    CHECK_FLAGS AME_FFLAGS_MASK

    /* Integer trig tuples remain unsupported and cannot change the data. */
    SET_M_DTYPE 2, 0x40000020
    FILL32  0, 0x55aa55aa
    FILL32  2, 1
    csrw    CSR_AMESTATUS, zero
    .word   ZTT_MCOS_EW(0, 2)
    CHECK32 0, 0, 0x55aa55aa
    CHECK32 0, 15, 0x55aa55aa
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail

    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       semihost_exit

trap_handler:
fail:
    mv      a0, s0

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
