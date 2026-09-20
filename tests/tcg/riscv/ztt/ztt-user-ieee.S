/*
 * RISC-V Ztt linux-user IEEE execution and amefflags test
 *
 * Run with N=4, U=32, NM=16, and NA=1.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define NR_EXIT          93
#define DTYPE_INT32      0x40000020
#define DTYPE_UINT128    0x00000080
#define DTYPE_FP16       0x14300110
#define DTYPE_FP32       0x20300120
#define DTYPE_FP32_RUP   0x20f00120
#define DTYPE_FP64       0x2c300140

    .option norvc

    .macro SET_M_DTYPE mreg, encoding
    li      t0, \encoding
    .word   ZTT_MSETTYP(\mreg, ZTT_R_T0)
    .endm

    .macro SET_A_DTYPE acc, encoding
    li      t0, \encoding
    .word   ZTT_ASETTYP(\acc, ZTT_R_T0)
    .endm

    .macro FILL32 mreg, value
    li      t0, \value
    li      t1, 0
    li      t2, 16
.Lfill32\@:
    .word   ZTT_MMOVE32_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, t2, .Lfill32\@
    .endm

    .macro CHECK32 mreg, position, expected
    li      t1, \position
    .word   ZTT_MMOVE32_X_M(ZTT_R_T0, \mreg, ZTT_R_T1)
    li      t2, \expected
    bne     t0, t2, fail
    .endm

    .macro FILL16 mreg, value
    li      t0, \value
    li      t1, 0
    li      t2, 32
.Lfill16\@:
    .word   ZTT_MMOVE16_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, t2, .Lfill16\@
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
    li      a1, 0
    .word   ZTT_AME_ACQUIRE(ZTT_R_A0, ZTT_R_A1)
    li      t0, AME_ACQUIRE_GRANTED
    bne     a0, t0, fail

    /* Basic IEEE arithmetic and instruction-wide exception flags. */
    SET_M_DTYPE 0, DTYPE_FP32
    SET_M_DTYPE 1, DTYPE_FP32
    SET_M_DTYPE 2, DTYPE_FP32
    FILL32  1, 0x3f800000
    FILL32  2, 0x40000000
    CLEAR_RESULT_STATE
    .word   ZTT_MADD_EW(0, 1, 2)
    CHECK32 0, 0, 0x40400000
    CHECK32 0, 15, 0x40400000
    CHECK_FLAGS 0

    FILL32  1, 0x7f7fffff
    CLEAR_RESULT_STATE
    .word   ZTT_MADD_EW(0, 1, 1)
    CHECK32 0, 0, 0x7f800000
    CHECK_FLAGS (AME_FFLAG_OF | AME_FFLAG_NX)

    FILL32  1, 0
    CLEAR_RESULT_STATE
    .word   ZTT_MREC_EW(0, 1)
    CHECK32 0, 0, 0x7f800000
    CHECK_FLAGS AME_FFLAG_DZ

    FILL32  1, 0x3fc00000
    CLEAR_RESULT_STATE
    .word   ZTT_MFRINTN_EW(0, 1)
    CHECK32 0, 0, 0x40000000
    CHECK_FLAGS 0

    /* Exact zero lanes must not inherit underflow from an earlier NX. */
    .irp position, 0, 15
    FILL32  1, 0
    li      t0, 0x3f800000
    li      t1, \position
    .word   ZTT_MMOVE32_M_X(1, ZTT_R_T0, ZTT_R_T1)
    CLEAR_RESULT_STATE
    li      t0, AME_FFLAG_DZ
    csrw    fflags, t0
    li      t0, AME_FFLAG_NV
    csrw    CSR_AMEFFLAGS, t0
    .word   ZTT_MTANH_EW(0, 1)
    CHECK32 0, \position, 0x3f42f7d6
    CHECK_FLAGS (AME_FFLAG_NV | AME_FFLAG_NX)
    csrr    t0, fflags
    li      t1, AME_FFLAG_DZ
    bne     t0, t1, fail
    .endr

    /* Unary FP64 -> packed FP16 must not first narrow the input. */
    SET_M_DTYPE 0, DTYPE_FP16
    SET_M_DTYPE 4, DTYPE_FP64
    .irp reg, 4, 5, 6, 7
    FILL32  \reg, 0
    li      t0, 0x40f00000          /* FP64 65536 */
    li      t1, 1
    li      t2, 17
1:
    .word   ZTT_MMOVE32_M_X(\reg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 2
    bne     t1, t2, 1b
    .endr
    CLEAR_RESULT_STATE
    .word   ZTT_MSQRT_EW(0, 4)
    CHECK16 0, 0, 0x5c00
    CHECK16 0, 31, 0x5c00
    CHECK_FLAGS 0
    CLEAR_RESULT_STATE
    .word   ZTT_MREC_EW(0, 4)
    CHECK16 0, 0, 0x0100
    CHECK16 0, 31, 0x0100
    CHECK_FLAGS 0

    /* Packed FP16 -> grouped FP64, followed by equal-width mixed formats. */
    SET_M_DTYPE 0, DTYPE_FP64
    SET_M_DTYPE 8, DTYPE_FP16
    FILL16  8, 0x3e00
    CLEAR_RESULT_STATE
    .word   ZTT_MFRINTN_EW(0, 8)
    CHECK32 0, 0, 0
    CHECK32 0, 1, 0x40000000
    CHECK32 3, 14, 0
    CHECK32 3, 15, 0x40000000
    CHECK_FLAGS 0
    SET_M_DTYPE 0, 0x20300110       /* BF16 */
    FILL16  8, 0x4400              /* FP16 4 */
    CLEAR_RESULT_STATE
    .word   ZTT_MSQRT_EW(0, 8)
    CHECK16 0, 0, 0x4000
    CHECK16 0, 31, 0x4000
    CHECK_FLAGS 0
    SET_M_DTYPE 0, DTYPE_FP16
    SET_M_DTYPE 8, 0x20300110
    FILL16  8, 0x3f80              /* BF16 1 */
    CLEAR_RESULT_STATE
    .word   ZTT_MREC_EW(0, 8)
    CHECK16 0, 0, 0x3c00
    CHECK16 0, 31, 0x3c00
    CHECK_FLAGS 0

    /* Integer-to-IEEE conversion uses the destination rounding mode. */
    SET_M_DTYPE 0, DTYPE_FP32_RUP
    SET_M_DTYPE 1, DTYPE_INT32
    FILL32  1, 16777217
    CLEAR_RESULT_STATE
    .word   ZTT_MCONV_EW(0, 1)
    CHECK32 0, 0, 0x4b800001
    CHECK_FLAGS AME_FFLAG_NX

    /* Mixed arithmetic decodes each source before one FP16 rounding. */
    SET_M_DTYPE 0, DTYPE_FP16
    SET_M_DTYPE 2, DTYPE_FP32
    SET_M_DTYPE 4, DTYPE_FP16
    FILL32  2, 0x3f801001
    FILL32  3, 0x3f801001
    FILL16  4, 0xbc00
    CLEAR_RESULT_STATE
    .word   ZTT_MADD_EW(0, 2, 4)
    CHECK16 0, 0, 0x1000
    CHECK16 0, 31, 0x1000
    CHECK_FLAGS AME_FFLAG_NX

    /* A mixed multiply-accumulate remains fused until FP16 writeback. */
    SET_M_DTYPE 4, DTYPE_FP32
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
    SET_M_DTYPE 0, DTYPE_FP64
    SET_M_DTYPE 4, DTYPE_UINT128
    lla     t0, uint128_input
    .word   ZTT_MLS_RM(4, ZTT_R_T0)
    CLEAR_RESULT_STATE
    .word   ZTT_MCONV_EW(0, 4)
    lla     t0, fp64_output
    .word   ZTT_MSS_RM(0, ZTT_R_T0)
    fence   rw, rw
    lla     t3, fp64_output
    li      t4, 16
.Lcheck_fp64:
    lw      t0, 0(t3)
    li      t1, 1
    bne     t0, t1, fail
    lw      t0, 4(t3)
    li      t1, 0x47e00000
    bne     t0, t1, fail
    addi    t3, t3, 8
    addi    t4, t4, -1
    bnez    t4, .Lcheck_fp64
    CHECK_FLAGS AME_FFLAG_NX

    /* Fused multiply-accumulate rounds only the final result. */
    SET_M_DTYPE 0, DTYPE_FP32
    SET_M_DTYPE 1, DTYPE_FP32
    SET_M_DTYPE 2, DTYPE_FP32
    FILL32  0, 0xbf800000
    FILL32  1, 0x3f800001
    FILL32  2, 0x3f7fffff
    CLEAR_RESULT_STATE
    .word   ZTT_MMULACC_EW(0, 1, 2)
    CHECK32 0, 0, 0x337ffffe
    CHECK_FLAGS 0

    /* Integer exponents scale IEEE values by an exact power of two. */
    SET_M_DTYPE 2, DTYPE_INT32
    FILL32  1, 0x3f800000
    FILL32  2, -1
    CLEAR_RESULT_STATE
    .word   ZTT_MLDEXP_EW(0, 1, 2)
    CHECK32 0, 0, 0x3f000000
    CHECK_FLAGS 0

    /* Matrix multiply uses the same IEEE executor and flag state. */
    SET_M_DTYPE 2, DTYPE_FP32
    SET_A_DTYPE 0, DTYPE_FP32
    FILL32  1, 0x3f800000
    FILL32  2, 0x3f800000
    CLEAR_RESULT_STATE
    .word   ZTT_MZERO_ACC(0)
    .word   ZTT_MMULACC_2D(0, 1, 2)
    .word   ZTT_MMOV_M_A(0, 0)
    CHECK32 0, 0, 0x40800000
    CHECK32 0, 15, 0x40800000
    CHECK_FLAGS 0

    /* Floating abs converts its final value to integer and accrues flags. */
    SET_M_DTYPE 0, DTYPE_INT32
    SET_M_DTYPE 1, DTYPE_FP32
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

    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       exit

fail:
    li      a0, 1
exit:
    li      a7, NR_EXIT
    ecall

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
