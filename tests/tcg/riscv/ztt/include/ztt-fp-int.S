/*
 * Shared user/system FP-to-integer instruction cases.
 * Geometry: N=2, U=64, NM=32, NA=1. No scalar F/D instructions are used.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

    .macro F2I_TYPE mreg, encoding
    li      t0, \encoding
    .word   ZTT_MSETTYP(\mreg, ZTT_R_T0)
    .endm

    .macro F2I_FILL32 mreg, value
    li      t0, \value
    li      t1, 0
    li      t2, 8
.Lf2i_fill\@:
    .word   ZTT_MMOVE32_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, t2, .Lf2i_fill\@
    .endm

    .macro F2I_CHECK32 mreg, position, expected
    li      t1, \position
    .word   ZTT_MMOVE32_X_M(ZTT_R_T0, \mreg, ZTT_R_T1)
    li      t2, \expected
    bne     t0, t2, fail
    .endm

    .macro F2I_CLEAR
    csrw    CSR_AMESTATUS, zero
    csrw    CSR_AMEFFLAGS, zero
    csrw    CSR_AMEXSAT, zero
    .endm

    .macro F2I_FLAGS flags, sat=0
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    bnez    t0, fail
    csrr    t0, CSR_AMEFFLAGS
    li      t1, \flags
    bne     t0, t1, fail
    csrr    t0, CSR_AMEXSAT
    li      t1, \sat
    bne     t0, t1, fail
    .endm

    .macro F2I_CASE dtype, raw, expected, flags, sat=0
    .set    f2i_case, f2i_case + 1
    li      s0, f2i_case
    F2I_TYPE 0, \dtype
    .irp reg, 16, 17, 18, 19, 20, 21, 22, 23
    F2I_FILL32 \reg, \raw
    .endr
    F2I_CLEAR
    .word   ZTT_MCONV_EW(0, 16)
    F2I_CHECK32 0, 0, \expected
    F2I_CHECK32 0, 7, \expected
    F2I_FLAGS \flags, \sat
    .endm

    .macro F2I_FILL64 mreg, high
    F2I_FILL32 \mreg, 0
    li      t0, \high
    li      t1, 1
    li      t2, 9
.Lf2i_fill64\@:
    .word   ZTT_MMOVE32_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 2
    bne     t1, t2, .Lf2i_fill64\@
    .endm

    .macro F2I_TESTS
    .set    f2i_case, 0
    /* FP32 RUP does not override any destination integer rounding mode. */
    F2I_TYPE 16, 0x20f00120
    F2I_CASE 0x40000004, 0x3fc00000, 0x22222222, AME_FFLAG_NX
    F2I_CASE 0x40000004, 0xbfc00000, 0xffffffff, AME_FFLAG_NX
    F2I_CASE 0x48000004, 0xbfc00000, 0xeeeeeeee, AME_FFLAG_NX
    F2I_CASE 0x50000004, 0xbfc00000, 0xeeeeeeee, AME_FFLAG_NX
    F2I_CASE 0x58000004, 0xbfc00000, 0xffffffff, AME_FFLAG_NX
    F2I_CASE 0x40000004, 0x40f00000, 0x88888888, AME_FFLAG_NV
    F2I_CASE 0x60000004, 0x40f00000, 0x77777777, AME_FFLAG_NV, 1
    F2I_CASE 0x40000004, 0xc1080000, 0x88888888, AME_FFLAG_NX
    F2I_CASE 0x50000004, 0xc1080000, 0x77777777, AME_FFLAG_NV
    F2I_CASE 0x00000004, 0xbf000000, 0, AME_FFLAG_NX
    F2I_CASE 0x10000004, 0xbf000000, 0xffffffff, AME_FFLAG_NV
    F2I_CASE 0x30000004, 0xbf000000, 0, AME_FFLAG_NV, 1
    F2I_CASE 0x60000004, 0x7fc00000, 0x77777777, AME_FFLAG_NV
    F2I_CASE 0x60000004, 0xff800001, 0x77777777, AME_FFLAG_NV
    F2I_CASE 0x60000004, 0x7f800000, 0x77777777, AME_FFLAG_NV
    F2I_CASE 0x60000004, 0xff800000, 0x88888888, AME_FFLAG_NV
    F2I_CASE 0x20000004, 0xff800000, 0, AME_FFLAG_NV
    F2I_CASE 0x40000004, 0x7f7fffff, 0, AME_FFLAG_NV
    F2I_CASE 0x60000004, 0x7f7fffff, 0x77777777, AME_FFLAG_NV, 1
    F2I_CASE 0x00000004, 0x41780000, 0, AME_FFLAG_NV
    F2I_CASE 0x20000004, 0x41780000, 0xffffffff, AME_FFLAG_NV, 1
    F2I_CASE 0x40000004, 0x00000001, 0, AME_FFLAG_NX
    F2I_CASE 0x58000004, 0x80000001, 0xffffffff, AME_FFLAG_NX

    /* NV must neither create NX on its lane nor erase another lane's NX. */
    F2I_CASE 0x40000004, 0x7fc00000, 0x77777777, AME_FFLAG_NV
    li      t0, 0x3fc00000
    li      t1, 0
    .word   ZTT_MMOVE32_M_X(16, ZTT_R_T0, ZTT_R_T1)
    F2I_CLEAR
    .word   ZTT_MCONV_EW(0, 16)
    F2I_CHECK32 0, 0, 0x77777772
    F2I_FLAGS (AME_FFLAG_NV | AME_FFLAG_NX)
    F2I_CASE 0x40000004, 0x3fc00000, 0x22222222, AME_FFLAG_NX
    li      t0, 0x7fc00000
    li      t1, 0
    .word   ZTT_MMOVE32_M_X(16, ZTT_R_T0, ZTT_R_T1)
    F2I_CLEAR
    .word   ZTT_MCONV_EW(0, 16)
    F2I_CHECK32 0, 0, 0x22222227
    F2I_FLAGS (AME_FFLAG_NV | AME_FFLAG_NX)

    /* One converted scalar is then used by integer binary/ternary kernels. */
    li      s0, 26
    F2I_TYPE 0, 0x48000008
    F2I_TYPE 1, 0x48000008
    F2I_FILL32 1, 0x01010101
    li      t0, 0x14300110
    csrw    CSR_AMESTYPE, t0
    li      s3, 0xbe00             /* FP16 -1.5 -> Int8 RNE -2 */
    F2I_CLEAR
    .word   ZTT_MADD_EW_X(0, ZTT_R_S3, 1)
    F2I_CHECK32 0, 0, 0xffffffff
    F2I_FLAGS AME_FFLAG_NX
    F2I_FILL32 0, 0x01010101
    F2I_CLEAR
    .word   ZTT_MMULACC_EW_X(0, ZTT_R_S3, 1)
    F2I_CHECK32 0, 7, 0xffffffff
    F2I_FLAGS AME_FFLAG_NX

    /* Typed broadcast uses its explicit source datatype, not amestype. */
    li      s0, 27
    li      s4, 0x14300110
    F2I_CLEAR
    .word   ZTT_MBCAST_M_X(0, ZTT_R_S3, ZTT_R_S4)
    F2I_CHECK32 0, 0, 0xfefefefe
    F2I_CHECK32 0, 7, 0xfefefefe
    F2I_FLAGS AME_FFLAG_NX

    /* Pack converts one square and leaves the other packed slots alone. */
    li      s0, 28
    F2I_TYPE 16, 0x2c300140
    F2I_FILL64 16, 0xbff80000     /* FP64 -1.5 */
    F2I_FILL32 0, 0x5a5a5a5a
    li      s3, 3
    F2I_CLEAR
    .word   ZTT_MPACK_EW_X(0, ZTT_R_S3, 16)
    F2I_CHECK32 0, 2, 0x5a5a5a5a
    F2I_CHECK32 0, 3, 0xfefefefe
    F2I_CHECK32 0, 4, 0x5a5a5a5a
    F2I_FLAGS AME_FFLAG_NX

    /* Packed FP16 expands to grouped Int128; check both halves. */
    li      s0, 29
    F2I_TYPE 0, 0x40000080
    F2I_TYPE 16, 0x14300110
    F2I_FILL32 16, 0x3e003e00
    F2I_CLEAR
    .word   ZTT_MCONV_EW(0, 16)
    .irp reg, 0, 1, 2, 3, 4, 5, 6, 7
    F2I_CHECK32 \reg, 0, 2
    F2I_CHECK32 \reg, 3, 0
    F2I_CHECK32 \reg, 4, 2
    F2I_CHECK32 \reg, 7, 0
    .endr
    F2I_FLAGS AME_FFLAG_NX
    li      s3, 1
    F2I_CLEAR
    .word   ZTT_MUNPACK_EW_X(0, ZTT_R_S3, 16)
    F2I_CHECK32 1, 4, 2
    F2I_CHECK32 1, 7, 0
    F2I_FLAGS AME_FFLAG_NX

    /* +2^127 is valid Uint128, but invalid for either signed variant. */
    li      s0, 30
    F2I_TYPE 16, 0x2c300140
    F2I_FILL64 16, 0x47e00000
    F2I_TYPE 0, 0x00000080
    F2I_CLEAR
    .word   ZTT_MCONV_EW(0, 16)
    F2I_CHECK32 0, 0, 0
    F2I_CHECK32 0, 3, 0x80000000
    F2I_CHECK32 1, 7, 0x80000000
    F2I_FLAGS 0
    F2I_TYPE 0, 0x40000080
    F2I_CLEAR
    .word   ZTT_MCONV_EW(0, 16)
    F2I_CHECK32 1, 7, 0x80000000
    F2I_FLAGS AME_FFLAG_NV
    F2I_TYPE 0, 0x60000080
    F2I_CLEAR
    .word   ZTT_MCONV_EW(0, 16)
    F2I_CHECK32 0, 0, 0xffffffff
    F2I_CHECK32 0, 3, 0x7fffffff
    F2I_CHECK32 1, 7, 0x7fffffff
    F2I_FLAGS AME_FFLAG_NV, 1
    F2I_FILL64 16, 0xc7e00000    /* -2^127 is exactly representable. */
    F2I_CLEAR
    .word   ZTT_MCONV_EW(0, 16)
    F2I_CHECK32 0, 0, 0
    F2I_CHECK32 0, 3, 0x80000000
    F2I_FLAGS 0

    /* Previously accrued architectural flags survive exact conversion. */
    li      s0, 31
    li      t0, AME_FFLAG_DZ | AME_FFLAG_NX
    csrw    CSR_AMEFFLAGS, t0
    li      t0, 1
    csrw    CSR_AMEXSAT, t0
    .word   ZTT_MCONV_EW(0, 16)
    F2I_FLAGS (AME_FFLAG_DZ | AME_FFLAG_NX), 1
    .endm
