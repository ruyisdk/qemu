/*
 * RISC-V Ztt structural tensor operation test
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
#define DTYPE_INT16      0x40000010
#define DTYPE_UINT64     0x00000040
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

    .macro FILL_I8_SEQUENCE mreg
    li      t1, 0
    li      a2, 16
.Lfill_sequence\@:
    mv      t0, t1
    .word   ZTT_MMOVE8_M_X(\mreg, ZTT_R_T0, ZTT_R_T1)
    addi    t1, t1, 1
    bne     t1, a2, .Lfill_sequence\@
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

    SET_M_DTYPE 0, DTYPE_INT8
    SET_M_DTYPE 1, DTYPE_INT8
    FILL_I8_SEQUENCE 0

    /* Broadcast controls use the unsigned low 32 bits. */
    li      t0, 2
    .word   ZTT_MCOLBCAST_EW_X(1, ZTT_R_T0, 0)
    li      t1, 0
.Lcheck_colbcast:
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 1, ZTT_R_T1)
    andi    t2, t1, -4
    addi    t2, t2, 2
    bne     t0, t2, fail
    addi    t1, t1, 1
    li      t2, 16
    bne     t1, t2, .Lcheck_colbcast

    li      t0, 1
    .word   ZTT_MROWBCAST_EW_X(1, ZTT_R_T0, 0)
    li      t1, 0
.Lcheck_rowbcast:
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 1, ZTT_R_T1)
    andi    t2, t1, 3
    addi    t2, t2, 4
    bne     t0, t2, fail
    addi    t1, t1, 1
    li      t2, 16
    bne     t1, t2, .Lcheck_rowbcast

    /* In-place shifts read the complete pre-instruction source snapshot. */
    li      t0, 1
    .word   ZTT_MCOLSHIFT_EW_X(0, ZTT_R_T0, 0)
    li      t1, 0
.Lcheck_colshift:
    andi    t2, t1, 3
    li      a2, 3
    beq     t2, a2, .Lcolshift_zero
    addi    t2, t1, 1
    j       .Lcolshift_compare
.Lcolshift_zero:
    li      t2, 0
.Lcolshift_compare:
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 0, ZTT_R_T1)
    bne     t0, t2, fail
    addi    t1, t1, 1
    li      t2, 16
    bne     t1, t2, .Lcheck_colshift

    FILL_I8_SEQUENCE 0
    li      t0, -1
    .word   ZTT_MROWSHIFT_EW_X(0, ZTT_R_T0, 0)
    li      t1, 0
.Lcheck_rowshift:
    srli    t2, t1, 2
    beqz    t2, .Lrowshift_zero
    addi    t2, t1, -4
    j       .Lrowshift_compare
.Lrowshift_zero:
    li      t2, 0
.Lrowshift_compare:
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 0, ZTT_R_T1)
    bne     t0, t2, fail
    addi    t1, t1, 1
    li      t2, 16
    bne     t1, t2, .Lcheck_rowshift

    /* Packed structural permutation is unsupported and writes nothing. */
    SET_M_DTYPE 2, DTYPE_INT4
    FILL_RAW_CONST 2, 0x5a
    li      t0, 0
    EXPECT_UN ZTT_MCOLBCAST_EW_X(2, ZTT_R_T0, 2)
    CHECK_RAW_CONST 2, 0x5a

    /* Out-of-range broadcast is Illegal after a supported tuple is formed. */
    FILL_RAW_CONST 1, 0x6b
    li      t0, 4
    EXPECT_ILLEGAL ZTT_MCOLBCAST_EW_X(1, ZTT_R_T0, 0)
    CHECK_RAW_CONST 1, 0x6b

    /* Packed index writes the same local coordinates into both squares. */
    .word   ZTT_MCOLID_EW(2)
    li      t1, 0
.Lcheck_colid:
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 2, ZTT_R_T1)
    andi    t2, t1, 1
    beqz    t2, .Lcolid_low
    li      t2, 0x32
    j       .Lcolid_compare
.Lcolid_low:
    li      t2, 0x10
.Lcolid_compare:
    bne     t0, t2, fail
    addi    t1, t1, 1
    li      t2, 16
    bne     t1, t2, .Lcheck_colid

    .word   ZTT_MROWID_EW(2)
    li      t1, 0
.Lcheck_rowid:
    andi    t2, t1, 7
    srli    t2, t2, 1
    li      a2, 17
    mul     t2, t2, a2
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 2, ZTT_R_T1)
    bne     t0, t2, fail
    addi    t1, t1, 1
    li      t2, 16
    bne     t1, t2, .Lcheck_rowid

    /* Gather consumes only low32(index), then low log2(N). */
    SET_M_DTYPE 0, DTYPE_INT8
    SET_M_DTYPE 1, DTYPE_INT8
    SET_M_DTYPE 8, DTYPE_UINT64
    FILL_I8_SEQUENCE 0
    lla     t0, indices64
    .word   ZTT_MLS_RM(8, ZTT_R_T0)
    .word   ZTT_MCOLGATHER_EW(1, 0, 8)
    li      t1, 0
.Lcheck_colgather:
    srli    t2, t1, 2
    slli    t2, t2, 2
    addi    t2, t2, 3
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 1, ZTT_R_T1)
    bne     t0, t2, fail
    addi    t1, t1, 1
    li      t2, 16
    bne     t1, t2, .Lcheck_colgather

    .word   ZTT_MROWGATHER_EW(1, 0, 8)
    li      t1, 0
.Lcheck_rowgather:
    andi    t2, t1, 3
    addi    t2, t2, 12
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 1, ZTT_R_T1)
    bne     t0, t2, fail
    addi    t1, t1, 1
    li      t2, 16
    bne     t1, t2, .Lcheck_rowgather

    FILL_RAW_CONST 2, 0xa5
    EXPECT_UN ZTT_MCOLGATHER_EW(2, 2, 8)
    CHECK_RAW_CONST 2, 0xa5

    /* Explicit scalar datatype conversion is shared with mconv.ew. */
    SET_M_DTYPE 0, DTYPE_INT8
    li      t0, -3
    li      t1, DTYPE_INT8
    .word   ZTT_MBCAST_M_X(0, ZTT_R_T0, ZTT_R_T1)
    CHECK_RAW_CONST 0, 0xfd

    SET_M_DTYPE 2, DTYPE_INT4_SAT
    csrwi   CSR_AMEXSAT, 0
    li      t0, 9
    li      t1, DTYPE_INT8
    .word   ZTT_MBCAST_M_X(2, ZTT_R_T0, ZTT_R_T1)
    CHECK_RAW_CONST 2, 0x77
    csrr    t0, CSR_AMEXSAT
    andi    t0, t0, AME_XSAT_SAT
    beqz    t0, fail

    FILL_RAW_CONST 0, 0x5c
    li      t0, 1
    li      t1, DTYPE_FP32
    csrwi   CSR_AMESTATUS, 0
    csrwi   CSR_AMEFFLAGS, 0
    .word   ZTT_MBCAST_M_X(0, ZTT_R_T0, ZTT_R_T1)
    CHECK_RAW_CONST 0, 0
    csrr    t0, CSR_AMEFFLAGS
    li      t1, AME_FFLAG_NX
    bne     t0, t1, fail
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    bnez    t0, fail

    /* Pack preserves unselected slots; unpack masks high index bits. */
    SET_M_DTYPE 2, DTYPE_INT4_SAT
    SET_M_DTYPE 3, DTYPE_INT8
    SET_M_DTYPE 4, DTYPE_INT8
    FILL_RAW_CONST 2, 0x11
    FILL_RAW_CONST 3, 9
    csrwi   CSR_AMEXSAT, 0
    li      t0, 5
    .word   ZTT_MPACK_EW_X(2, ZTT_R_T0, 3)
    li      t1, 0
.Lcheck_pack:
    .word   ZTT_MMOVE8_X_M(ZTT_R_T0, 2, ZTT_R_T1)
    li      t2, 8
    bltu    t1, t2, .Lpack_low
    li      t2, 0x77
    j       .Lpack_compare
.Lpack_low:
    li      t2, 0x11
.Lpack_compare:
    bne     t0, t2, fail
    addi    t1, t1, 1
    li      t2, 16
    bne     t1, t2, .Lcheck_pack
    csrr    t0, CSR_AMEXSAT
    andi    t0, t0, AME_XSAT_SAT
    beqz    t0, fail

    li      t0, 5
    .word   ZTT_MUNPACK_EW_X(4, ZTT_R_T0, 2)
    CHECK_RAW_CONST 4, 7

    FILL_RAW_CONST 4, 0x4d
    li      t0, 0
    EXPECT_UN ZTT_MPACK_EW_X(4, ZTT_R_T0, 3)
    CHECK_RAW_CONST 4, 0x4d

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

    .section .rodata
    .balign 128
indices64:
    .rept   16
    .8byte  0xffffffff00000003
    .endr

    .bss
    .balign 16
semiargs:
#if __riscv_xlen == 64
    .space  16
#else
    .space  8
#endif
