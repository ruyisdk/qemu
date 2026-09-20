/*
 * RISC-V Ztt IEEE execution and amefflags test
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

    .macro WRITE32 mreg, position, value
    li      t0, \value
    li      t1, \position
    .word   ZTT_MMOVE32_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    .endm

    .macro WRITE16 mreg, position, value
    li      t0, \value
    li      t1, \position
    .word   ZTT_MMOVE16_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
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

    .macro FILL16 mreg, value
    li      t0, \value
    li      t1, 0
    li      t2, 32
1:
    .word   ZTT_MMOVE16_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, t2, 1b
    .endm

    .macro CHECK32 mreg, position, expected
    li      t1, \position
    .word   ZTT_MMOVE32_X_M(ZTT_R_T0, \mreg, ZTT_R_T1)
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
#ifdef ZTT_TEST_WITH_F
    li      t0, MSTATUS_AME_MS_INITIAL | MSTATUS_FS_INITIAL
#else
    li      t0, MSTATUS_AME_MS_INITIAL
#endif
    csrs    mstatus, t0
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail
#ifdef ZTT_TEST_WITH_F
    li      t0, AME_FFLAGS_MASK
    csrw    CSR_FFLAGS, t0
#endif

    /* FP32 add, full source snapshot on overlap, and exact flags. */
    li      s0, 2
    SET_M_DTYPE 0, 0x20300120
    SET_M_DTYPE 1, 0x20300120
    SET_M_DTYPE 2, 0x20300120
    FILL32  1, 0x3f800000
    FILL32  2, 0x40000000
    CLEAR_RESULT_STATE
    .word   ZTT_MADD_EW(0, 1, 2)
    CHECK32 0, 0, 0x40400000
    CHECK_FLAGS 0
    .word   ZTT_MADD_EW(1, 1, 1)
    CHECK32 1, 15, 0x40000000
    CHECK_FLAGS 0

    /* Overflow is OF|NX and commits only with the destination. */
    li      s0, 3
    FILL32  1, 0x7f7fffff
    CLEAR_RESULT_STATE
    .word   ZTT_MADD_EW(0, 1, 1)
    CHECK32 0, 0, 0x7f800000
    CHECK_FLAGS (AME_FFLAG_OF | AME_FFLAG_NX)

    /* Tininess is detected after rounding and inexact underflow sets NX. */
    li      s0, 31
    FILL32  1, 1
    FILL32  2, 0x3f000000
    CLEAR_RESULT_STATE
    .word   ZTT_MMUL_EW(0, 1, 2)
    CHECK32 0, 0, 0
    CHECK_FLAGS (AME_FFLAG_UF | AME_FFLAG_NX)

    /* Invalid sqrt and reciprocal pole use canonical results. */
    li      s0, 4
    FILL32  1, 0xbf800000
    CLEAR_RESULT_STATE
    .word   ZTT_MSQRT_EW(0, 1)
    CHECK32 0, 0, 0x7fc00000
    CHECK_FLAGS AME_FFLAG_NV
    FILL32  1, 0
    CLEAR_RESULT_STATE
    .word   ZTT_MREC_EW(0, 1)
    CHECK32 0, 0, 0x7f800000
    CHECK_FLAGS AME_FFLAG_DZ

    /* Element flags accrue locally across the complete instruction. */
    li      s0, 41
    FILL32  1, 0x3f800000
    WRITE32 1, 0, 0x7f812345
    WRITE32 1, 1, 0
    CLEAR_RESULT_STATE
    .word   ZTT_MREC_EW(0, 1)
    CHECK32 0, 0, 0x7fc00000
    CHECK32 0, 1, 0x7f800000
    CHECK_FLAGS (AME_FFLAG_NV | AME_FFLAG_DZ)

    /* Fixed frint modes do not depend on the datatype rounding field. */
    li      s0, 5
    FILL32  1, 0x3fc00000
    CLEAR_RESULT_STATE
    .word   ZTT_MFRINTN_EW(0, 1)
    CHECK32 0, 0, 0x40000000
    CHECK_FLAGS 0
    CLEAR_RESULT_STATE
    .word   ZTT_MFRINTM_EW(0, 1)
    CHECK32 0, 0, 0x3f800000
    CHECK_FLAGS 0
    CLEAR_RESULT_STATE
    .word   ZTT_MFRINTP_EW(0, 1)
    CHECK32 0, 0, 0x40000000
    CHECK_FLAGS 0
    CLEAR_RESULT_STATE
    .word   ZTT_MFRINTZ_EW(0, 1)
    CHECK32 0, 0, 0x3f800000
    CHECK_FLAGS 0

    /* Positive and negative inputs uniquely identify all four fixed modes. */
    FILL32  1, 0xbfc00000
    CLEAR_RESULT_STATE
    .word   ZTT_MFRINTN_EW(0, 1)
    CHECK32 0, 0, 0xc0000000
    CHECK_FLAGS 0
    CLEAR_RESULT_STATE
    .word   ZTT_MFRINTM_EW(0, 1)
    CHECK32 0, 0, 0xc0000000
    CHECK_FLAGS 0
    CLEAR_RESULT_STATE
    .word   ZTT_MFRINTP_EW(0, 1)
    CHECK32 0, 0, 0xbf800000
    CHECK_FLAGS 0
    CLEAR_RESULT_STATE
    .word   ZTT_MFRINTZ_EW(0, 1)
    CHECK32 0, 0, 0xbf800000
    CHECK_FLAGS 0

    /* Integer conversion takes the destination rounding mode. */
    li      s0, 6
    SET_M_DTYPE 0, 0x20f00120 /* FP32, RUP */
    SET_M_DTYPE 1, 0x40000020 /* signed Int32 */
    FILL32  1, 16777217
    CLEAR_RESULT_STATE
    .word   ZTT_MCONV_EW(0, 1)
    CHECK32 0, 0, 0x4b800001
    CHECK_FLAGS AME_FFLAG_NX

    /* NaNs are canonicalized; only a signaling NaN sets NV. */
    li      s0, 7
    SET_M_DTYPE 0, 0x20300120
    SET_M_DTYPE 1, 0x20300120
    FILL32  1, 0x7fc12345
    CLEAR_RESULT_STATE
    .word   ZTT_MCONV_EW(0, 1)
    CHECK32 0, 0, 0x7fc00000
    CHECK_FLAGS 0
    FILL32  1, 0x7f812345
    CLEAR_RESULT_STATE
    .word   ZTT_MCONV_EW(0, 1)
    CHECK32 0, 0, 0x7fc00000
    CHECK_FLAGS AME_FFLAG_NV

    /* Ordered compare and numeric selection accept mixed predicate data. */
    li      s0, 8
    SET_M_DTYPE 0, 0x40000020
    SET_M_DTYPE 1, 0x20300120
    SET_M_DTYPE 2, 0x20300120
    FILL32  1, 0x7fc12345
    FILL32  2, 0x3f800000
    CLEAR_RESULT_STATE
    .word   ZTT_MCMPLT_EW(0, 1, 2)
    CHECK32 0, 0, 0
    CHECK_FLAGS 0
    FILL32  1, 0x7f812345
    CLEAR_RESULT_STATE
    .word   ZTT_MCMPGE_EW(0, 1, 2)
    CHECK32 0, 0, 0
    CHECK_FLAGS AME_FFLAG_NV

    SET_M_DTYPE 0, 0x20300120
    SET_M_DTYPE 1, 0x40000020
    SET_M_DTYPE 2, 0x20300120
    FILL32  0, 0x40000000
    FILL32  1, -1
    FILL32  2, 0x40400000
    CLEAR_RESULT_STATE
    .word   ZTT_MCMOVLT_EW(0, 1, 2)
    CHECK32 0, 0, 0x40400000
    CHECK_FLAGS 0

    /* Every unsigned integer is non-negative. */
    li      s0, 81
    SET_M_DTYPE 1, 0x00000020
    FILL32  0, 0x40000000
    FILL32  1, -1
    li      t0, AME_FFLAGS_MASK
    csrw    CSR_AMEFFLAGS, t0
    csrw    CSR_AMESTATUS, zero
    .word   ZTT_MCMOVLT_EW(0, 1, 2)
    CHECK32 0, 0, 0x40000000
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    bnez    t0, fail
    csrr    t0, CSR_AMEFFLAGS
    andi    t0, t0, AME_FFLAGS_MASK
    li      t1, AME_FFLAGS_MASK
    bne     t0, t1, fail

    /* IEEE predicates use ordered comparison: -0 equals zero. */
    li      s0, 82
    SET_M_DTYPE 0, 0x20300120
    SET_M_DTYPE 1, 0x20300120
    FILL32  0, 0x40000000
    FILL32  1, 0x80000000
    FILL32  2, 0x40400000
    CLEAR_RESULT_STATE
    .word   ZTT_MCMOVLT_EW(0, 1, 2)
    CHECK32 0, 0, 0x40000000
    CHECK_FLAGS 0
    .word   ZTT_MCMOVGE_EW(0, 1, 2)
    CHECK32 0, 0, 0x40400000
    CHECK_FLAGS 0

    /* Unordered NaN predicates select neither relation; sNaN raises NV. */
    FILL32  0, 0x40000000
    FILL32  1, 0xffc12345
    CLEAR_RESULT_STATE
    .word   ZTT_MCMOVLT_EW(0, 1, 2)
    CHECK32 0, 0, 0x40000000
    CHECK_FLAGS 0
    FILL32  1, 0xff812345
    CLEAR_RESULT_STATE
    .word   ZTT_MSELGE_EW(0, 1, 2)
    CHECK32 0, 0, 0
    CHECK_FLAGS AME_FFLAG_NV

    /* Fused multiply-accumulate has no intermediate rounding. */
    li      s0, 9
    SET_M_DTYPE 1, 0x20300120
    FILL32  0, 0xbf800000
    FILL32  1, 0x3f800001
    FILL32  2, 0x3f7fffff
    CLEAR_RESULT_STATE
    .word   ZTT_MMULACC_EW(0, 1, 2)
    CHECK32 0, 0, 0x337ffffe
    CHECK_FLAGS 0

    /* IEEE min/max preserve the required signed-zero ordering. */
    li      s0, 10
    FILL32  1, 0x80000000
    FILL32  2, 0
    CLEAR_RESULT_STATE
    .word   ZTT_MMIN_EW(0, 1, 2)
    CHECK32 0, 0, 0x80000000
    li      s0, 101
    .word   ZTT_MMAX_EW(0, 1, 2)
    CHECK32 0, 0, 0
    li      s0, 102
    CHECK_FLAGS 0

    /* Tensor min/max requires identical datatype descriptors. */
    li      s0, 1021
    SET_M_DTYPE 0, 0x20300120       /* FP32, RNE */
    SET_M_DTYPE 1, 0x20300120
    SET_M_DTYPE 2, 0x20f00120       /* FP32, RUP */
    FILL32  0, 0x55aa55aa
    FILL32  1, 0x3f800000
    FILL32  2, 0x40000000
    csrw    CSR_AMESTATUS, zero
    .word   ZTT_MMAX_EW(0, 1, 2)
    CHECK32 0, 0, 0x55aa55aa
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail

    /* Scalar min/max converts amestype before comparing with Md[ms2]. */
    li      s0, 1022
    SET_M_DTYPE 0, 0x20300120
    SET_M_DTYPE 2, 0x20300120
    FILL32  2, 0x3f800000
    li      t0, 0x14300110           /* FP16 scalar datatype */
    csrw    CSR_AMESTYPE, t0
    li      t0, 0x4000               /* FP16 2.0 */
    CLEAR_RESULT_STATE
    .word   ZTT_MMAX_EW_X(0, ZTT_R_T0, 2)
    CHECK32 0, 0, 0x40000000
    CHECK_FLAGS 0

    /* Negation and absdiff precede the single destination rounding. */
    li      s0, 103
    SET_M_DTYPE 0, 0x20b00120 /* FP32, RDN */
    FILL32  1, 0x33000000     /* 2^-25 */
    FILL32  2, 0x3f800000     /* 1 */
    CLEAR_RESULT_STATE
    .word   ZTT_MABSDIFF_EW(0, 1, 2)
    CHECK32 0, 0, 0x3f7fffff
    CHECK_FLAGS AME_FFLAG_NX

    SET_M_DTYPE 0, 0x20f00120 /* FP32, RUP */
    CLEAR_RESULT_STATE
    .word   ZTT_MABSDIFF_EW(0, 1, 2)
    CHECK32 0, 0, 0x3f800000
    CHECK_FLAGS AME_FFLAG_NX

    li      s0, 104
    SET_M_DTYPE 0, 0x20b00120 /* FP32, RDN */
    FILL32  1, 0x3f800001
    FILL32  2, 0x3f800001
    CLEAR_RESULT_STATE
    .word   ZTT_MMULNEG_EW(0, 1, 2)
    CHECK32 0, 0, 0xbf800003
    CHECK_FLAGS AME_FFLAG_NX

    SET_M_DTYPE 0, 0x20f00120 /* FP32, RUP */
    CLEAR_RESULT_STATE
    .word   ZTT_MMULNEG_EW(0, 1, 2)
    CHECK32 0, 0, 0xbf800002
    CHECK_FLAGS AME_FFLAG_NX

    /* Ztt subtraction is B-A; scalar subtraction is B-c. */
    li      s0, 105
    SET_M_DTYPE 0, 0x20300120
    FILL32  1, 0x3f800000
    FILL32  2, 0x40400000
    CLEAR_RESULT_STATE
    .word   ZTT_MSUB_EW(0, 1, 2)
    CHECK32 0, 0, 0x40000000
    CHECK_FLAGS 0
    li      t0, 0x20300120
    csrw    CSR_AMESTYPE, t0
    li      t0, 0x40400000
    CLEAR_RESULT_STATE
    .word   ZTT_MSUB_EW_X(0, ZTT_R_T0, 1)
    CHECK32 0, 0, 0xc0000000
    CHECK_FLAGS 0

    /* Matrix and fixed-type scalar exponents scale by exact powers of two. */
    li      s0, 11
    SET_M_DTYPE 2, 0x40000020
    FILL32  1, 0x3f800000
    FILL32  2, -1
    CLEAR_RESULT_STATE
    .word   ZTT_MLDEXP_EW(0, 1, 2)
    CHECK32 0, 0, 0x3f000000
    .word   ZTT_MRDEXP_EW(0, 1, 2)
    CHECK32 0, 0, 0x40000000
    li      t0, 1
    .word   ZTT_MLDEXP_EW_X(0, ZTT_R_T0, 1)
    CHECK32 0, 0, 0x40000000
    CHECK_FLAGS 0

    /* Scale-accumulate scales only the unrounded product before addition. */
    li      s0, 111
    FILL32  0, 0x40400000
    FILL32  1, 0x41200000
    FILL32  2, 1
    CLEAR_RESULT_STATE
    .word   ZTT_MLDEXPACC_EW(0, 1, 2)
    CHECK32 0, 0, 0x41b80000
    CHECK_FLAGS 0

    FILL32  0, 0x40400000
    FILL32  1, 0x41800000
    FILL32  2, 2
    .word   ZTT_MRDEXPACC_EW(0, 1, 2)
    CHECK32 0, 0, 0x40e00000
    CHECK_FLAGS 0

    FILL32  0, 0x40400000
    FILL32  1, 0x41200000
    li      t0, 1
    .word   ZTT_MLDEXPACC_EW_X(0, ZTT_R_T0, 1)
    CHECK32 0, 0, 0x41b80000
    CHECK_FLAGS 0

    /* md == ms1 uses the destination value as both pre-state operands. */
    FILL32  0, 0x40400000
    FILL32  2, 1
    .word   ZTT_MLDEXPACC_EW(0, 0, 2)
    CHECK32 0, 0, 0x41100000
    CHECK_FLAGS 0

    /* A rounded scale would overflow here, but the fused result is finite. */
    FILL32  0, 0xff7fffff
    FILL32  1, 0x7f7fffff
    FILL32  2, 1
    CLEAR_RESULT_STATE
    .word   ZTT_MLDEXPACC_EW(0, 1, 2)
    CHECK32 0, 0, 0x7f7fffff
    CHECK_FLAGS 0

    /* minsub + minsub/2 is rounded once to two minsubnormals. */
    FILL32  0, 1
    FILL32  1, 1
    FILL32  2, -1
    CLEAR_RESULT_STATE
    .word   ZTT_MLDEXPACC_EW(0, 1, 2)
    CHECK32 0, 0, 2
    CHECK_FLAGS (AME_FFLAG_UF | AME_FFLAG_NX)

    /* A non-integer exponent tuple is UN and has no side effects. */
    SET_M_DTYPE 2, 0x20300120
    FILL32  0, 0x40400000
    FILL32  2, 0x3f800000
    li      t0, AME_FFLAGS_MASK
    csrw    CSR_AMEFFLAGS, t0
    csrw    CSR_AMESTATUS, zero
    .word   ZTT_MLDEXPACC_EW(0, 1, 2)
    CHECK32 0, 0, 0x40400000
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail
    csrr    t0, CSR_AMEFFLAGS
    andi    t0, t0, AME_FFLAGS_MASK
    li      t1, AME_FFLAGS_MASK
    bne     t0, t1, fail
    csrw    CSR_AMESTATUS, zero
    csrw    CSR_AMEFFLAGS, zero

    /* Floating abs converts its final value to integer and accrues flags. */
    li      s0, 12
    SET_M_DTYPE 0, 0x40000020
    SET_M_DTYPE 1, 0x20300120
    FILL32  0, 0x12345678
    FILL32  1, 0x3f800000
    li      t0, AME_FFLAGS_MASK
    csrw    CSR_AMEFFLAGS, t0
    csrw    CSR_AMESTATUS, zero
    .word   ZTT_MABS_EW(0, 1)
    CHECK32 0, 0, 1
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    bnez    t0, fail
    csrr    t0, CSR_AMEFFLAGS
    andi    t0, t0, AME_FFLAGS_MASK
    li      t1, AME_FFLAGS_MASK
    bne     t0, t1, fail

    /* All four IEEE storage formats execute through the same model. */
    li      s0, 13
    SET_M_DTYPE 0, 0x14300110 /* FP16 */
    SET_M_DTYPE 1, 0x14300110
    SET_M_DTYPE 2, 0x14300110
    WRITE16 1, 0, 0x3c00
    WRITE16 2, 0, 0x4000
    CLEAR_RESULT_STATE
    .word   ZTT_MADD_EW(0, 1, 2)
    CHECK16 0, 0, 0x4200
    CHECK_FLAGS 0
    li      t0, 1
    .word   ZTT_MLDEXPACC_EW_X(0, ZTT_R_T0, 1)
    CHECK16 0, 0, 0x4500
    CHECK_FLAGS 0

    SET_M_DTYPE 0, 0x20300110 /* BF16 */
    SET_M_DTYPE 1, 0x20300110
    SET_M_DTYPE 2, 0x20300110
    WRITE16 1, 0, 0x3f80
    WRITE16 2, 0, 0x4000
    CLEAR_RESULT_STATE
    .word   ZTT_MADD_EW(0, 1, 2)
    CHECK16 0, 0, 0x4040
    CHECK_FLAGS 0
    li      t0, 1
    .word   ZTT_MLDEXPACC_EW_X(0, ZTT_R_T0, 1)
    CHECK16 0, 0, 0x40a0
    CHECK_FLAGS 0

#if __riscv_xlen == 64
    SET_M_DTYPE 0, 0x2c300140 /* FP64 */
    SET_M_DTYPE 2, 0x2c300140
    SET_M_DTYPE 4, 0x2c300140
    li      t0, 0x3ff0000000000000
    li      t1, 0
    .word   ZTT_MMOVE64_M_X(2, ZTT_R_T0, ZTT_R_T1)
    li      t0, 0x4000000000000000
    .word   ZTT_MMOVE64_M_X(4, ZTT_R_T0, ZTT_R_T1)
    CLEAR_RESULT_STATE
    .word   ZTT_MADD_EW(0, 2, 4)
    li      t1, 0
    .word   ZTT_MMOVE64_X_M(ZTT_R_T0, 0, ZTT_R_T1)
    li      t2, 0x4008000000000000
    bne     t0, t2, fail
    CHECK_FLAGS 0
    li      t0, 1
    .word   ZTT_MLDEXPACC_EW_X(0, ZTT_R_T0, 2)
    li      t1, 0
    .word   ZTT_MMOVE64_X_M(ZTT_R_T0, 0, ZTT_R_T1)
    li      t2, 0x4014000000000000
    bne     t0, t2, fail
    CHECK_FLAGS 0
#endif

    /* Prefix/reduction use a fixed ascending order in this implementation. */
    li      s0, 14
    SET_M_DTYPE 0, 0x20300120
    SET_M_DTYPE 1, 0x20300120
    FILL32  1, 0x3f800000
    CLEAR_RESULT_STATE
    .word   ZTT_MPREFIXADD_ROW(0, 1)
    CHECK32 0, 0, 0x3f800000
    CHECK32 0, 3, 0x40800000
    .word   ZTT_MREDUCEADD_ROW(0, 1)
    CHECK32 0, 0, 0x40800000
    CHECK32 0, 3, 0x40800000
    CHECK_FLAGS 0
    FILL32  1, 0x7f7fffff
    CLEAR_RESULT_STATE
    .word   ZTT_MPREFIXADD_ROW(0, 1)
    CHECK32 0, 1, 0x7f800000
    CHECK_FLAGS (AME_FFLAG_OF | AME_FFLAG_NX)

    /* Reduction itself accrues flags and preserves IEEE zero ordering. */
    li      s0, 141
    CLEAR_RESULT_STATE
    .word   ZTT_MREDUCEADD_ROW(0, 1)
    CHECK32 0, 0, 0x7f800000
    CHECK_FLAGS (AME_FFLAG_OF | AME_FFLAG_NX)
    FILL32  1, 0
    WRITE32 1, 0, 0x80000000
    CLEAR_RESULT_STATE
    .word   ZTT_MREDUCEMIN_ROW(0, 1)
    CHECK32 0, 0, 0x80000000
    .word   ZTT_MREDUCEMAX_ROW(0, 1)
    CHECK32 0, 0, 0
    CHECK_FLAGS 0

    /* Mixed prefix converts each FP32 leaf before the FP16 fold. */
    li      s0, 142
    SET_M_DTYPE 0, 0x14300110
    SET_M_DTYPE 2, 0x20300120
    FILL32  2, 0
    FILL32  3, 0
    WRITE32 2, 0, 0x3f801001
    WRITE32 2, 1, 0xbf800000
    WRITE32 3, 0, 0x3f801001
    WRITE32 3, 1, 0xbf800000
    CLEAR_RESULT_STATE
    .word   ZTT_MPREFIXADD_ROW(0, 2)
    CHECK16 0, 1, 0x1400
    CHECK16 0, 17, 0x1400
    CHECK_FLAGS AME_FFLAG_NX

    /* Scatter updates collide in ascending source-element order. */
    li      s0, 15
    SET_M_DTYPE 0, 0x20300120
    SET_M_DTYPE 1, 0x20300120
    SET_M_DTYPE 2, 0x40000020
    FILL32  0, 0
    FILL32  1, 0x3f800000
    FILL32  2, 0
    CLEAR_RESULT_STATE
    .word   ZTT_MCOLSCATADD_EW(0, 1, 2)
    CHECK32 0, 0, 0x40800000
    CHECK32 0, 1, 0
    CHECK32 0, 4, 0x40800000
    CHECK_FLAGS 0

    /* Scatter arithmetic reports aggregate overflow and signaling NaNs. */
    li      s0, 153
    FILL32  0, 0
    FILL32  1, 0x7f7fffff
    CLEAR_RESULT_STATE
    .word   ZTT_MCOLSCATADD_EW(0, 1, 2)
    li      s0, 154
    CHECK32 0, 0, 0x7f800000
    li      s0, 155
    CHECK_FLAGS (AME_FFLAG_OF | AME_FFLAG_NX)
    li      s0, 156
    FILL32  0, 0x3f800000
    FILL32  1, 0x7f812345
    CLEAR_RESULT_STATE
    .word   ZTT_MCOLSCATMAX_EW(0, 1, 2)
    li      s0, 157
    CHECK32 0, 0, 0x7fc00000
    li      s0, 158
    CHECK_FLAGS AME_FFLAG_NV

    /* Gather copies IEEE payloads but requires an integer index datatype. */
    li      s0, 151
    FILL32  0, 0x12345678
    FILL32  1, 0x3f800000
    FILL32  2, 0
    CLEAR_RESULT_STATE
    .word   ZTT_MCOLGATHER_EW(0, 1, 2)
    CHECK32 0, 0, 0x3f800000
    CHECK32 0, 15, 0x3f800000
    CHECK_FLAGS 0

    li      s0, 152
    SET_M_DTYPE 2, 0x20300120
    FILL32  0, 0x12345678
    li      t0, AME_FFLAGS_MASK
    csrw    CSR_AMEFFLAGS, t0
    csrw    CSR_AMESTATUS, zero
    .word   ZTT_MCOLGATHER_EW(0, 1, 2)
    CHECK32 0, 0, 0x12345678
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail
    csrr    t0, CSR_AMEFFLAGS
    andi    t0, t0, AME_FFLAGS_MASK
    li      t1, AME_FFLAGS_MASK
    bne     t0, t1, fail

    /* Matrix multiply walks square, then inner index, and fuses each term. */
    li      s0, 16
    SET_M_DTYPE 0, 0x20300120
    SET_M_DTYPE 1, 0x20300120
    SET_M_DTYPE 2, 0x20300120
    SET_A_DTYPE 0, 0x20300120
    FILL32  1, 0x3f800000
    FILL32  2, 0x3f800000
    CLEAR_RESULT_STATE
    .word   ZTT_MZERO_ACC(0)
    .word   ZTT_MMULACC_2D(0, 1, 2)
    .word   ZTT_MMOV_M_A(0, 0)
    CHECK32 0, 0, 0x40800000
    CHECK32 0, 15, 0x40800000
    CHECK_FLAGS 0

    /* Matrix terms report exceptions through the independent AME status. */
    li      s0, 161
    FILL32  1, 0x7f800000
    FILL32  2, 0
    CLEAR_RESULT_STATE
    .word   ZTT_MZERO_ACC(0)
    .word   ZTT_MMULACC_2D(0, 1, 2)
    .word   ZTT_MMOV_M_A(0, 0)
    CHECK32 0, 0, 0x7fc00000
    CHECK_FLAGS AME_FFLAG_NV

    /* Each matrix term is fused with the running sum before one rounding. */
    li      s0, 162
    FILL32  0, 0xbf800000
    .word   ZTT_MMOV_A_M(0, 0)
    FILL32  1, 0
    FILL32  2, 0
    WRITE32 1, 0, 0x3f800001
    WRITE32 2, 0, 0x3f7fffff
    CLEAR_RESULT_STATE
    .word   ZTT_MMULACC_2D(0, 1, 2)
    .word   ZTT_MMOV_M_A(0, 0)
    CHECK32 0, 0, 0x337ffffe
    CHECK_FLAGS 0

    /* FP16 and BF16 widen exactly before one FP32 operation. */
    li      s0, 163
    SET_M_DTYPE 0, 0x20300120
    SET_M_DTYPE 4, 0x14300110
    SET_M_DTYPE 5, 0x20300110
    FILL16  4, 0x3e00       /* FP16 1.5 */
    FILL16  5, 0x4000       /* BF16 2.0 */
    CLEAR_RESULT_STATE
    .word   ZTT_MADD_EW(0, 4, 5)
    CHECK32 0, 0, 0x40600000
    CHECK32 1, 15, 0x40600000
    CHECK_FLAGS 0

    FILL32  0, 0x3e800000   /* FP32 0.25 */
    CLEAR_RESULT_STATE
    .word   ZTT_MMULACC_EW(0, 4, 5)
    CHECK32 0, 0, 0x40500000
    CHECK_FLAGS 0

    FILL16  4, 0x3c00       /* FP16 1.0 */
    FILL16  5, 0x4000       /* BF16 2.0 */
    SET_A_DTYPE 0, 0x20300120
    CLEAR_RESULT_STATE
    .word   ZTT_MZERO_ACC(0)
    .word   ZTT_MMULACC_2D(0, 4, 5)
    SET_M_DTYPE 0, 0x20300120
    .word   ZTT_MMOV_M_A(0, 0)
    CHECK32 0, 0, 0x41800000
    CHECK32 0, 15, 0x41800000
    CHECK_FLAGS 0

    /* Mixed arithmetic decodes each source before one FP16 rounding. */
    li      s0, 164
    SET_M_DTYPE 0, 0x14300110 /* FP16 */
    SET_M_DTYPE 2, 0x20300120 /* FP32 */
    SET_M_DTYPE 4, 0x14300110 /* FP16 */
    FILL32  2, 0x3f801001
    FILL32  3, 0x3f801001
    FILL16  4, 0xbc00
    CLEAR_RESULT_STATE
    .word   ZTT_MADD_EW(0, 2, 4)
    CHECK16 0, 0, 0x1000
    CHECK16 0, 31, 0x1000
    CHECK_FLAGS AME_FFLAG_NX

    /* A mixed multiply-accumulate remains fused until FP16 writeback. */
    li      s0, 165
    SET_M_DTYPE 4, 0x20300120 /* FP32 */
    FILL16  0, 0xbc00
    FILL32  2, 0x3f801000
    FILL32  3, 0x3f801000
    FILL32  4, 0x3f7fe000
    FILL32  5, 0x3f7fe000
    CLEAR_RESULT_STATE
    .word   ZTT_MMULACC_EW(0, 2, 4)
    CHECK16 0, 0, 0x8004
    CHECK16 0, 31, 0x8004
    CHECK_FLAGS 0

    /* UInt128 converts directly, without an intermediate FP64 rounding. */
    li      s0, 166
    SET_M_DTYPE 0, 0x2c300140 /* FP64 */
    SET_M_DTYPE 4, 0x00000080 /* UInt128 */
    lla     t0, uint128_input
    .word   ZTT_MLS_RM(4, ZTT_R_T0)
    CLEAR_RESULT_STATE
    .word   ZTT_MCONV_EW(0, 4)
    lla     t0, fp64_output
    .word   ZTT_MSS_RM(0, ZTT_R_T0)
    fence   rw, rw
    lla     t3, fp64_output
    li      t4, 16
1:
    li      s0, 167
    lw      t0, 0(t3)
    li      t1, 1
    bne     t0, t1, fail
    li      s0, 168
    lw      t0, 4(t3)
    li      t1, 0x47e00000
    bne     t0, t1, fail
    addi    t3, t3, 8
    addi    t4, t4, -1
    bnez    t4, 1b
    li      s0, 169
    CHECK_FLAGS AME_FFLAG_NX

    /* Unsupported integer trig sources have the same no-write rule. */
    li      s0, 18
    SET_M_DTYPE 0, 0x20300120
    SET_M_DTYPE 1, 0x20300120
    SET_M_DTYPE 2, 0x40000020
    FILL32  0, 0x12345678
    FILL32  1, 0x3f800000
    FILL32  2, 1
    li      t0, AME_FFLAGS_MASK
    csrw    CSR_AMEFFLAGS, t0
    csrw    CSR_AMESTATUS, zero
    .word   ZTT_MCOS_EW(0, 2)
    CHECK32 0, 0, 0x12345678
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail
    csrr    t0, CSR_AMEFFLAGS
    andi    t0, t0, AME_FFLAGS_MASK
    li      t1, AME_FFLAGS_MASK
    bne     t0, t1, fail

    /* AME uses an independent status and never modifies F-extension fflags. */
    li      s0, 19
#ifdef ZTT_TEST_WITH_F
    csrr    t0, CSR_FFLAGS
    li      t1, AME_FFLAGS_MASK
    bne     t0, t1, fail
#endif

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

    .section .rodata
    .balign 256
uint128_input:
    .rept   16
    .8byte  1, 0x8000000000000400
    .endr

    .bss
    .balign 256
fp64_output:
    .space  128
    .balign 16
semiargs:
#if __riscv_xlen == 64
    .space  16
#else
    .space  8
#endif
