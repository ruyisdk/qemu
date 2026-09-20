/*
 * Shared user/system mixed numeric cases. N=2, U=64, NM=32, NA=1.
 * Values are written in 32-bit pieces so RV32 and RV64 run identical cases.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

    .macro MIX_TYPE mreg, encoding
    li      t0, \encoding
    .word   ZTT_MSETTYP(\mreg, ZTT_R_T0)
    .endm

    .macro MIX_FILL32 mreg, raw
    li      t0, \raw
    li      t1, 0
    li      t2, 8
.Lmix_fill\@:
    .word   ZTT_MMOVE32_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, t2, .Lmix_fill\@
    .endm

    .macro MIX_FILL64 mreg, low, high
    li      t1, 0
    li      t2, 8
.Lmix_fill64\@:
    li      t0, \low
    .word   ZTT_MMOVE32_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    li      t0, \high
    .word   ZTT_MMOVE32_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, t2, .Lmix_fill64\@
    .endm

    .macro MIX_LARGE128 mreg, high=0x80000000
    .set    mix_reg, \mreg
    .rept   2
    MIX_FILL32 mix_reg, 0
    .irp pos, 0, 4
    li      t1, \pos
    li      t0, 1
    .word   ZTT_MMOVE32_M_X(mix_reg, ZTT_R_T0, ZTT_R_T1)
    .endr
    .irp pos, 3, 7
    li      t1, \pos
    li      t0, \high
    .word   ZTT_MMOVE32_M_X(mix_reg, ZTT_R_T0, ZTT_R_T1)
    .endr
    .set    mix_reg, mix_reg + 1
    .endr
    .endm

    .macro MIX_WRITE128 mreg, position, word0, word1, word2, word3
    li      t1, \position
    li      t0, \word0
    .word   ZTT_MMOVE32_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    li      t0, \word1
    .word   ZTT_MMOVE32_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    li      t0, \word2
    .word   ZTT_MMOVE32_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    li      t0, \word3
    .word   ZTT_MMOVE32_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    .endm

    .macro MIX_FILL128 mreg, word0, word1, word2, word3
    .set    mix_reg, \mreg
    .rept   2
    MIX_WRITE128 mix_reg, 0, \word0, \word1, \word2, \word3
    MIX_WRITE128 mix_reg, 4, \word0, \word1, \word2, \word3
    .set    mix_reg, mix_reg + 1
    .endr
    .endm

    .macro MIX_CHECK mreg, position, expected
    li      t1, \position
    .word   ZTT_MMOVE32_X_M(ZTT_R_T0, \mreg, ZTT_R_T1)
    li      t2, (\expected & 0xffffffff)
    bne     t0, t2, fail
    .endm

    .macro MIX_CLEAR
    csrw    CSR_AMESTATUS, zero
    csrw    CSR_AMEFFLAGS, zero
    csrw    CSR_AMEXSAT, zero
    .endm

    .macro MIX_FLAGS flags=0, sat=0
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

    .macro MIX_TESTS
    /* Full Int128 precision survives cancellation with a binary64 source. */
    li      s0, 1
    MIX_TYPE 0, 0x2c300140
    MIX_TYPE 4, 0x00000080
    MIX_TYPE 8, 0x2c300140
    MIX_LARGE128 4
    MIX_FILL64 8, 0, 0xc7e00000
    MIX_CLEAR
    .word   ZTT_MADD_EW(0, 4, 8)
    MIX_CHECK 0, 0, 0
    MIX_CHECK 0, 1, 0x3ff00000
    MIX_CHECK 0, 7, 0x3ff00000
    MIX_FLAGS

    li      s0, 2
    MIX_TYPE 0, 0x00000080
    MIX_CLEAR
    .word   ZTT_MADD_EW(0, 4, 8)
    .irp reg, 0, 1
    MIX_CHECK \reg, 0, 1
    MIX_CHECK \reg, 1, 0
    MIX_CHECK \reg, 3, 0
    MIX_CHECK \reg, 4, 1
    MIX_CHECK \reg, 7, 0
    .endr
    MIX_FLAGS

    /* The Int128 product is not pre-rounded before the old FP destination. */
    li      s0, 3
    MIX_TYPE 0, 0x2c300140
    MIX_FILL64 0, 0, 0xc7e00000
    MIX_FILL64 8, 0, 0x3ff00000
    MIX_CLEAR
    .word   ZTT_MMULACC_EW(0, 4, 8)
    MIX_CHECK 0, 1, 0x3ff00000
    MIX_FLAGS

    /* Uint128 RNE: (2^127+1)+0.5 rounds to 2^127+2, not 2^127. */
    li      s0, 4
    MIX_TYPE 0, 0x08000080
    MIX_FILL64 8, 0, 0x3fe00000
    MIX_CLEAR
    .word   ZTT_MADD_EW(0, 4, 8)
    MIX_CHECK 0, 0, 2
    MIX_CHECK 0, 3, 0x80000000
    MIX_CHECK 1, 4, 2
    MIX_FLAGS AME_FFLAG_NX

    /* Scalar ingress converts FP16 -1.5 to Int8 RNE -2 before addition. */
    li      s0, 5
    MIX_TYPE 0, 0x2c300140
    MIX_TYPE 16, 0x48000008
    MIX_FILL32 16, 0x03030303
    li      t0, 0x14300110
    csrw    CSR_AMESTYPE, t0
    li      s3, 0xbe00
    MIX_CLEAR
    .word   ZTT_MADD_EW_X(0, ZTT_R_S3, 16)
    MIX_CHECK 0, 1, 0x3ff00000
    MIX_CHECK 7, 7, 0x3ff00000
    MIX_FLAGS AME_FFLAG_NX

    /* In .ew, ms1 is the log input; ms2 is the signed integer bias. */
    li      s0, 6
    MIX_TYPE 4, 0x2c300140
    MIX_TYPE 8, 0x40000040
    MIX_FILL64 4, 0, 0x40200000
    MIX_FILL64 8, 0xfffffffe, 0xffffffff
    MIX_CLEAR
    .word   ZTT_MLOG2SUB_EW(0, 4, 8)
    MIX_CHECK 0, 1, 0x40140000
    MIX_FLAGS
    MIX_CLEAR
    .word   ZTT_MSUBLOG2_EW(0, 4, 8)
    MIX_CHECK 0, 1, 0xc0140000
    MIX_FLAGS

    /* In .ew.x, ms2 is the log input and X[rs1] supplies the bias. */
    li      s0, 7
    li      t0, 0x40000008
    csrw    CSR_AMESTYPE, t0
    li      s3, -2
    MIX_CLEAR
    .word   ZTT_MLOG2SUB_EW_X(0, ZTT_R_S3, 4)
    MIX_CHECK 0, 1, 0x40140000
    MIX_FLAGS

    /* SRA keeps a signed data source but permits an unsigned destination. */
    li      s0, 8
    MIX_TYPE 0, 0x20000040
    MIX_TYPE 4, 0x40000080
    MIX_TYPE 8, 0x00000040
    MIX_FILL32 4, -1
    MIX_FILL32 5, -1
    MIX_FILL64 8, 128, 0       /* modulo 128: shift by zero */
    MIX_CLEAR
    .word   ZTT_MSRA_EW(0, 4, 8)
    MIX_CHECK 0, 0, 0
    MIX_CHECK 0, 7, 0
    MIX_FLAGS 0, 1

    li      s0, 9
    MIX_TYPE 0, 0x00000040
    li      s3, 127
    MIX_CLEAR
    .word   ZTT_MSRA_EW_X(0, ZTT_R_S3, 4)
    MIX_CHECK 0, 0, -1
    MIX_CHECK 0, 7, -1
    MIX_FLAGS
    MIX_CLEAR
    .word   ZTT_MSRL_EW_X(0, ZTT_R_S3, 4)
    MIX_CHECK 0, 0, 1
    MIX_CHECK 0, 1, 0
    MIX_FLAGS

    /* SLL preserves the widened result; it does not wrap at source width. */
    li      s0, 10
    MIX_TYPE 0, 0x00000080
    MIX_TYPE 4, 0x00000040
    MIX_FILL32 4, -1
    li      s3, 63
    MIX_CLEAR
    .word   ZTT_MSLL_EW_X(0, ZTT_R_S3, 4)
    MIX_CHECK 0, 0, 0
    MIX_CHECK 0, 1, 0x80000000
    MIX_CHECK 0, 2, -1
    MIX_CHECK 0, 3, 0x7fffffff
    MIX_FLAGS

    /* Mixed FP16/Int8 cannot enter the homogeneous FP16 raw-bit loop. */
    li      s0, 11
    MIX_TYPE 0, 0x14300110
    MIX_TYPE 16, 0x40000008
    MIX_TYPE 8, 0x14300110
    MIX_FILL32 16, 0x03030303
    MIX_FILL32 8, 0x3c003c00
    MIX_FILL32 9, 0x3c003c00
    MIX_CLEAR
    .word   ZTT_MADD_EW(0, 16, 8)
    MIX_CHECK 0, 0, 0x44004400
    MIX_CHECK 1, 7, 0x44004400
    MIX_FLAGS

    /* Matrix multiplication also accepts Int128/FP64 sources. */
    li      s0, 12
    MIX_TYPE 0, 0x00000080
    MIX_TYPE 4, 0x00000080
    MIX_TYPE 8, 0x2c300140
    MIX_LARGE128 4, 0
    MIX_FILL64 8, 0, 0x3ff00000
    li      t0, 0x00000080
    .word   ZTT_ASETTYP(0, ZTT_R_T0)
    .word   ZTT_MZERO_ACC(0)
    MIX_CLEAR
    .word   ZTT_MMULACC_2D(0, 4, 8)
    .word   ZTT_MMOV_M_A(0, 0)
    MIX_CHECK 0, 0, 2
    MIX_CHECK 0, 3, 0
    MIX_CHECK 1, 4, 2
    MIX_FLAGS

    MIX_TYPE 0, 0x2c300140
    li      t0, 0x2c300140
    .word   ZTT_ASETTYP(0, ZTT_R_T0)
    .word   ZTT_MZERO_ACC(0)
    MIX_CLEAR
    .word   ZTT_MMULACC_2D(0, 4, 8)
    .word   ZTT_MMOV_M_A(0, 0)
    MIX_CHECK 0, 0, 0
    MIX_CHECK 0, 1, 0x40000000
    MIX_CHECK 0, 7, 0x40000000
    MIX_FLAGS

    /* (2^128-1)+0.5 stays below 2^128, and is inexact in both modes. */
    li      s0, 13
    MIX_TYPE 4, 0x00000080
    MIX_TYPE 8, 0x2c300140
    MIX_FILL32 4, -1
    MIX_FILL32 5, -1
    MIX_FILL64 8, 0, 0x3fe00000
    MIX_TYPE 0, 0x2cb00140      /* FP64 RDN */
    MIX_CLEAR
    .word   ZTT_MADD_EW(0, 4, 8)
    MIX_CHECK 0, 0, -1
    MIX_CHECK 0, 1, 0x47efffff
    MIX_CHECK 0, 7, 0x47efffff
    MIX_FLAGS AME_FFLAG_NX
    MIX_TYPE 0, 0x2c300140      /* FP64 RNE */
    MIX_CLEAR
    .word   ZTT_MADD_EW(0, 4, 8)
    MIX_CHECK 0, 0, 0
    MIX_CHECK 0, 1, 0x47f00000
    MIX_FLAGS AME_FFLAG_NX

    /* FP32 RDN yields max finite without OF: the rounded exponent is 127. */
    li      s0, 14
    MIX_TYPE 0, 0x20b00120
    /* Packed D selects two squares, so each source spans two squares too. */
    MIX_FILL32 6, -1
    MIX_FILL32 7, -1
    MIX_FILL64 9, 0, 0x3fe00000
    MIX_CLEAR
    .word   ZTT_MADD_EW(0, 4, 8)
    MIX_CHECK 0, 0, 0x7f7fffff
    MIX_CHECK 0, 7, 0x7f7fffff
    MIX_FLAGS AME_FFLAG_NX

    /* Full product plus 8.5 is 9*2^252-0.5, not the exact upper boundary. */
    li      s0, 15
    MIX_TYPE 8, 0x00000080
    MIX_FILL128 4, 0xfffffffd, -1, -1, 0xbfffffff
    MIX_FILL128 8, 3, 0, 0, 0xc0000000
    .irp encoding, 0x2cb00140, 0x2c300140
    MIX_TYPE 0, \encoding
    MIX_FILL64 0, 0, 0x40210000
    MIX_CLEAR
    .word   ZTT_MMULACC_EW(0, 4, 8)
    .if \encoding == 0x2cb00140
    MIX_CHECK 0, 0, -1
    MIX_CHECK 0, 1, 0x4fe1ffff
    .else
    MIX_CHECK 0, 0, 0
    MIX_CHECK 0, 1, 0x4fe20000
    .endif
    MIX_FLAGS AME_FFLAG_NX
    .endr

    li      s0, 16
    MIX_TYPE 0, 0x2cf00140      /* FP64 RUP */
    MIX_FILL64 0, 0, 0xc0210000
    MIX_CLEAR
    .word   ZTT_MMULACCNEG_EW(0, 4, 8)
    MIX_CHECK 0, 0, -1
    MIX_CHECK 0, 1, 0xcfe1ffff
    MIX_FLAGS AME_FFLAG_NX

    /* One nonzero matrix product isolates the same fused-addition boundary. */
    li      s0, 17
    MIX_FILL32 4, 0
    MIX_FILL32 5, 0
    MIX_FILL32 8, 0
    MIX_FILL32 9, 0
    MIX_WRITE128 4, 0, 0xfffffffd, -1, -1, 0xbfffffff
    MIX_WRITE128 8, 0, 3, 0, 0, 0xc0000000
    .irp encoding, 0x2cb00140, 0x2c300140
    MIX_TYPE 0, \encoding
    li      t0, \encoding
    .word   ZTT_ASETTYP(0, ZTT_R_T0)
    MIX_FILL64 0, 0, 0x40210000
    .word   ZTT_MMOV_A_M(0, 0)
    MIX_CLEAR
    .word   ZTT_MMULACC_2D(0, 4, 8)
    .word   ZTT_MMOV_M_A(0, 0)
    .if \encoding == 0x2cb00140
    MIX_CHECK 0, 0, -1
    MIX_CHECK 0, 1, 0x4fe1ffff
    .else
    MIX_CHECK 0, 0, 0
    MIX_CHECK 0, 1, 0x4fe20000
    .endif
    MIX_CHECK 0, 3, 0x40210000
    MIX_FLAGS AME_FFLAG_NX
    .endr

    /* Floor suppresses only its own NX, not destination rounding or old NX. */
    li      s0, 18
    MIX_TYPE 0, 0x14f00110      /* FP16 RUP; instruction overrides it. */
    MIX_TYPE 8, 0x2c300140
    .irp reg, 8, 9, 10, 11
    MIX_FILL64 \reg, 0, 0x40a00300 /* 2049.5 -> 2049 -> FP16 2048 */
    .endr
    MIX_CLEAR
    .word   ZTT_MFRINTM_EW(0, 8)
    MIX_CHECK 0, 0, 0x68006800
    MIX_CHECK 0, 7, 0x68006800
    MIX_FLAGS AME_FFLAG_NX
    .irp reg, 8, 9, 10, 11
    MIX_FILL64 \reg, 0, 0x40a00100 /* 2048.5 -> exact FP16 2048 */
    .endr
    MIX_CLEAR
    .word   ZTT_MFRINTM_EW(0, 8)
    MIX_CHECK 0, 0, 0x68006800
    MIX_FLAGS
    li      t0, AME_FFLAG_NX | AME_FFLAG_DZ
    csrw    CSR_AMEFFLAGS, t0
    .word   ZTT_MFRINTM_EW(0, 8)
    MIX_FLAGS AME_FFLAG_NX | AME_FFLAG_DZ

    /* Standard FRINT rejects an integer destination without numeric flags. */
    li      s0, 19
    MIX_TYPE 0, 0x40000020
    MIX_FILL32 0, 0x12345678
    MIX_FILL64 8, 0, 0x7ff00000
    MIX_CLEAR
    .word   ZTT_MFRINTM_EW(0, 8)
    MIX_CHECK 0, 0, 0x12345678
    csrr    t0, CSR_AMESTATUS
    li      t1, AME_STATUS_UN
    bne     t0, t1, fail
    csrr    t0, CSR_AMEFFLAGS
    bnez    t0, fail
    csrr    t0, CSR_AMEXSAT
    bnez    t0, fail
    .endm
