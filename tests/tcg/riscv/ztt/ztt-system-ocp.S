/*
 * RISC-V Ztt OCP FP8 execution and amefflags test
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

    .macro SET_A_DTYPE acc, encoding
    li      t0, \encoding
    .word   ZTT_ASETTYP(\acc, ZTT_R_T0)
    .endm

    .macro FILL8 mreg, value
    li      t0, \value
    li      t1, 0
    li      t2, 64
1:
    .word   ZTT_MMOVE8_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, t2, 1b
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

    .macro CHECK8 mreg, position, expected
    li      t1, \position
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, \mreg, ZTT_R_T1)
    li      t2, \expected
    bne     t0, t2, fail
    .endm

    .macro CHECK16 mreg, position, expected
    li      t1, \position
    .word   ZTT_MMOVE16_X_M(ZTT_R_T0, \mreg, ZTT_R_T1)
    li      t2, \expected
    bne     t0, t2, fail
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
    li      t0, MSTATUS_AME_MS_INITIAL
    csrs    mstatus, t0
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail

    /* E4M3 elementwise arithmetic uses the destination rounding mode. */
    li      s0, 2
    SET_M_DTYPE 0, 0x10100108       /* E4M3, RNE */
    SET_M_DTYPE 1, 0x10100108
    SET_M_DTYPE 2, 0x10100108
    FILL8   1, 0x38                 /* 1.0 */
    FILL8   2, 0x40                 /* 2.0 */
    CLEAR_RESULT_STATE
    .word   ZTT_MADD_EW(0, 1, 2)
    CHECK8  0, 0, 0x44              /* 3.0 */
    CHECK8  0, 63, 0x44
    CHECK_FLAGS 0

    /* Tininess is detected after rounding. */
    li      s0, 3
    FILL8   1, 0x01                 /* minimum subnormal */
    FILL8   2, 0x4f                 /* 7.5 */
    CLEAR_RESULT_STATE
    .word   ZTT_MMUL_EW(0, 1, 2)
    CHECK8  0, 0, 0x08              /* tie -> minimum normal */
    CHECK_FLAGS (AME_FFLAG_UF | AME_FFLAG_NX)
    FILL8   2, 0x30                 /* 0.5 */
    CLEAR_RESULT_STATE
    .word   ZTT_MMUL_EW(0, 1, 2)
    CHECK8  0, 0, 0x00
    CHECK_FLAGS (AME_FFLAG_UF | AME_FFLAG_NX)

    /* E4M3 has no infinity and therefore saturates required infinities. */
    li      s0, 4
    FILL8   1, 0x7e                 /* maximum finite */
    FILL8   2, 0x40                 /* 2.0 */
    CLEAR_RESULT_STATE
    .word   ZTT_MMUL_EW(0, 1, 2)
    CHECK8  0, 0, 0x7e
    CHECK_FLAGS (AME_FFLAG_OF | AME_FFLAG_NX)
    FILL8   1, 0
    CLEAR_RESULT_STATE
    .word   ZTT_MREC_EW(0, 1)
    CHECK8  0, 0, 0x7e
    CHECK_FLAGS (AME_FFLAG_DZ | AME_FFLAG_OF | AME_FFLAG_NX)

    /* E5M2 represents infinity and treats every NaN encoding as quiet. */
    li      s0, 5
    SET_M_DTYPE 0, 0x14300108       /* E5M2, RNE */
    SET_M_DTYPE 1, 0x14300108
    SET_M_DTYPE 2, 0x14300108
    FILL8   1, 0
    FILL8   2, 0x7c                 /* +infinity */
    CLEAR_RESULT_STATE
    .word   ZTT_MMUL_EW(0, 1, 2)
    CHECK8  0, 0, 0x7e              /* canonical NaN */
    CHECK_FLAGS AME_FFLAG_NV
    FILL8   1, 0x7d                 /* an OCP NaN encoding */
    FILL8   2, 0x3c                 /* 1.0 */
    CLEAR_RESULT_STATE
    .word   ZTT_MADD_EW(0, 1, 2)
    CHECK8  0, 0, 0x7e
    CHECK_FLAGS 0

    /* E5M2 uses OVF after rounding, including directed rounding modes. */
    li      s0, 6
    FILL8   1, 0x7b                 /* maximum finite */
    FILL8   2, 0x40                 /* 2.0 */
    CLEAR_RESULT_STATE
    .word   ZTT_MMUL_EW(0, 1, 2)
    CHECK8  0, 0, 0x7c
    CHECK_FLAGS (AME_FFLAG_OF | AME_FFLAG_NX)
    SET_M_DTYPE 0, 0x14b00108       /* E5M2, RDN */
    SET_M_DTYPE 1, 0x14b00108
    SET_M_DTYPE 2, 0x14b00108
    FILL8   1, 0x7b
    FILL8   2, 0x40
    CLEAR_RESULT_STATE
    .word   ZTT_MMUL_EW(0, 1, 2)
    CHECK8  0, 0, 0x7c
    CHECK_FLAGS (AME_FFLAG_OF | AME_FFLAG_NX)

    /* OCP converts exactly when widening and once when narrowing. */
    li      s0, 7
    SET_M_DTYPE 0, 0x14300110       /* FP16, RNE; spans M0:M1 */
    SET_M_DTYPE 4, 0x10100108       /* E4M3, RNE */
    FILL8   4, 0x38
    CLEAR_RESULT_STATE
    .word   ZTT_MCONV_EW(0, 4)
    CHECK16 0, 0, 0x3c00
    CHECK16 1, 31, 0x3c00
    CHECK_FLAGS 0
    SET_M_DTYPE 0, 0x10100108
    SET_M_DTYPE 4, 0x14300110
    SET_M_DTYPE 5, 0x14300110
    FILL16  4, 0x6000               /* FP16 512.0 */
    FILL16  5, 0x6000
    CLEAR_RESULT_STATE
    .word   ZTT_MCONV_EW(0, 4)
    CHECK8  0, 0, 0x7e
    CHECK_FLAGS (AME_FFLAG_OF | AME_FFLAG_NX)

    /* Fused accumulate and exponent scaling round only the final result. */
    li      s0, 8
    SET_M_DTYPE 0, 0x10100108
    SET_M_DTYPE 1, 0x10100108
    SET_M_DTYPE 2, 0x10100108
    FILL8   0, 0x38
    FILL8   1, 0x38
    FILL8   2, 0x38
    CLEAR_RESULT_STATE
    .word   ZTT_MMULACC_EW(0, 1, 2)
    CHECK8  0, 0, 0x40              /* 1 + 1 * 1 = 2 */
    CHECK_FLAGS 0
    SET_M_DTYPE 2, 0x40000008       /* signed Int8 */
    FILL8   0, 0x38
    FILL8   1, 0x38
    FILL8   2, 1
    CLEAR_RESULT_STATE
    .word   ZTT_MLDEXPACC_EW(0, 1, 2)
    CHECK8  0, 0, 0x44              /* 1 + 1 * 2^1 = 3 */
    CHECK_FLAGS 0

    /* Nonlinear operations execute through the same OCP tuple model. */
    li      s0, 9
    SET_M_DTYPE 0, 0x10100108
    SET_M_DTYPE 1, 0x10100108
    FILL8   1, 0x38
    CLEAR_RESULT_STATE
    .word   ZTT_MRSQRT_EW(0, 1)
    CHECK8  0, 0, 0x38
    .word   ZTT_MEXP2_EW(0, 1)
    CHECK8  0, 0, 0x40
    FILL8   1, 0x40
    .word   ZTT_MLOG2_EW(0, 1)
    CHECK8  0, 0, 0x38
    FILL8   1, 0
    .word   ZTT_MSIN_EW(0, 1)
    CHECK8  0, 0, 0
    .word   ZTT_MCOS_EW(0, 1)
    CHECK8  0, 0, 0x38
    .word   ZTT_MTANH_EW(0, 1)
    CHECK8  0, 0, 0
    CHECK_FLAGS 0

    /* Reductions and matrix multiplication preserve the OCP datatype. */
    li      s0, 10
    SET_M_DTYPE 0, 0x10100108
    SET_M_DTYPE 1, 0x10100108
    FILL8   1, 0x38
    CLEAR_RESULT_STATE
    .word   ZTT_MREDUCEADD_ROW(0, 1)
    CHECK8  0, 0, 0x48              /* four 1.0 elements */
    CHECK8  0, 63, 0x48
    CHECK_FLAGS 0
    li      s0, 101
    SET_M_DTYPE 4, 0x10100108
    SET_M_DTYPE 5, 0x10100108
    SET_A_DTYPE 0, 0x10100108
    FILL8   4, 0x38
    FILL8   5, 0x38
    CLEAR_RESULT_STATE
    .word   ZTT_MZERO_ACC(0)
    .word   ZTT_MMULACC_2D(0, 4, 5)
    CHECK_FLAGS 0
    .word   ZTT_MMOV_M_A(0, 0)
    li      s0, 102
    CHECK8  0, 0, 0x58              /* four squares, four terms each */
    li      s0, 103
    CHECK8  0, 15, 0x58
    li      s0, 104
    CHECK_FLAGS 0

    /* Scalar operands use the configured OCP scalar datatype. */
    li      s0, 11
    SET_M_DTYPE 0, 0x10100108
    SET_M_DTYPE 1, 0x10100108
    FILL8   1, 0x38
    li      t0, 0x10100108
    csrw    CSR_AMESTYPE, t0
    li      t0, 0x38
    CLEAR_RESULT_STATE
    .word   ZTT_MADD_EW_X(0, ZTT_R_T0, 1)
    CHECK8  0, 0, 0x40
    CHECK_FLAGS 0

    /* OCP half operations cover tensor and scalar instruction forms. */
    li      s0, 12
    SET_M_DTYPE 0, 0x10100108
    SET_M_DTYPE 1, 0x10100108
    SET_M_DTYPE 2, 0x10100108
    FILL8   1, 0x38                 /* 1.0 */
    FILL8   2, 0x44                 /* 3.0 */
    CLEAR_RESULT_STATE
    .word   ZTT_MMEAN_EW(0, 1, 2)
    CHECK8  0, 0, 0x40              /* 2.0 */
    .word   ZTT_MHDIFF_EW(0, 1, 2)
    CHECK8  0, 63, 0x38             /* (3.0 - 1.0) / 2 */
    li      t0, 0x10100108
    csrw    CSR_AMESTYPE, t0
    li      t0, 0x38
    .word   ZTT_MMEAN_EW_X(0, ZTT_R_T0, 2)
    CHECK8  0, 0, 0x40
    li      t0, 0x38
    .word   ZTT_MHDIFF_EW_X(0, ZTT_R_T0, 2)
    CHECK8  0, 63, 0x38
    CHECK_FLAGS 0

    /* Prefix/reduction converts each OCP source leaf to the destination. */
    li      s0, 13
    SET_M_DTYPE 0, 0x10100108       /* E4M3 destination */
    SET_M_DTYPE 1, 0x14300108       /* E5M2 source */
    FILL8   1, 0x3c                 /* 1.0 */
    CLEAR_RESULT_STATE
    .word   ZTT_MREDUCEADD_ROW(0, 1)
    CHECK8  0, 0, 0x48              /* four 1.0 elements */
    CHECK8  0, 63, 0x48
    CHECK_FLAGS 0

    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       semihost_exit

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
