/*
 * RISC-V Ztt integer elementwise test
 *
 * Run with N=4, U=8, NM=32, and NA=1.
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "include/ztt-csr.h"

#define DTYPE_INT4       0x40000004
#define DTYPE_INT8       0x40000008
#define DTYPE_INT8_SAT   0x60000008
#define DTYPE_INT16      0x40000010
#define DTYPE_INT128     0x40000080
#define DTYPE_UINT8      0x00000008
#define DTYPE_UINT8_SAT  0x20000008

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

    .macro CHECK_NO_UN
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    bnez    t0, fail
    .endm

    .macro RUN_BINARY insn, expected
    csrwi   CSR_AMESTATUS, 0
    .word   \insn
    CHECK_RAW_CONST 0, \expected
    CHECK_NO_UN
    .endm

    .macro RUN_SCALAR insn, scalar, expected
    li      t0, \scalar
    csrwi   CSR_AMESTATUS, 0
    .word   \insn
    CHECK_RAW_CONST 0, \expected
    CHECK_NO_UN
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
    SET_M_DTYPE 2, DTYPE_INT8
    FILL_RAW_CONST 1, -3
    FILL_RAW_CONST 2, 4

    csrwi   CSR_AMESTATUS, 0
    .word   ZTT_MABS_EW(0, 1)
    CHECK_RAW_CONST 0, 3
    CHECK_NO_UN
    RUN_BINARY ZTT_MABSDIFF_EW(0, 1, 2), 7
    RUN_BINARY ZTT_MADD_EW(0, 1, 2), 1
    RUN_BINARY ZTT_MAND_EW(0, 1, 2), 4
    RUN_BINARY ZTT_MANDNOT_EW(0, 1, 2), 0xf9
    RUN_BINARY ZTT_MCMPGE_EW(0, 1, 2), 0
    RUN_BINARY ZTT_MCMPLT_EW(0, 1, 2), 0xff
    RUN_BINARY ZTT_MMAX_EW(0, 1, 2), 4
    RUN_BINARY ZTT_MMIN_EW(0, 1, 2), 0xfd
    RUN_BINARY ZTT_MMUL_EW(0, 1, 2), 0xf4
    RUN_BINARY ZTT_MMULNEG_EW(0, 1, 2), 12
    RUN_BINARY ZTT_MOR_EW(0, 1, 2), 0xfd
    RUN_BINARY ZTT_MORNOT_EW(0, 1, 2), 0xff
    RUN_BINARY ZTT_MSELGE_EW(0, 1, 2), 0
    RUN_BINARY ZTT_MSELLT_EW(0, 1, 2), 4
    RUN_BINARY ZTT_MSLL_EW(0, 1, 2), 0xd0
    RUN_BINARY ZTT_MSRA_EW(0, 1, 2), 0xff
    RUN_BINARY ZTT_MSRL_EW(0, 1, 2), 0x0f
    RUN_BINARY ZTT_MSUB_EW(0, 1, 2), 7
    RUN_BINARY ZTT_MXOR_EW(0, 1, 2), 0xf9

    /* Mixed-source arithmetic and comparison do not reinterpret negatives. */
    SET_M_DTYPE 0, DTYPE_UINT8_SAT
    SET_M_DTYPE 2, DTYPE_UINT8
    FILL_RAW_CONST 1, -1
    FILL_RAW_CONST 2, 1
    csrwi   CSR_AMEXSAT, 0
    RUN_BINARY ZTT_MABSDIFF_EW(0, 1, 2), 2
    RUN_BINARY ZTT_MCMPGE_EW(0, 2, 1), 0xff
    csrr    t0, CSR_AMEXSAT
    bnez    t0, fail
    FILL_RAW_CONST 2, 255
    RUN_BINARY ZTT_MADD_EW(0, 1, 2), 254
    RUN_BINARY ZTT_MMUL_EW(0, 1, 2), 0
    csrr    t0, CSR_AMEXSAT
    li      t1, 1
    bne     t0, t1, fail
    csrwi   CSR_AMEXSAT, 0
    FILL_RAW_CONST 0, 255
    RUN_BINARY ZTT_MMULACC_EW(0, 1, 2), 0
    csrr    t0, CSR_AMEXSAT
    bnez    t0, fail

    /* Preserve the mandatory scalar-to-matrix conversion. */
    SET_M_DTYPE 2, DTYPE_INT8
    FILL_RAW_CONST 2, 2
    li      t0, DTYPE_UINT8
    csrw    CSR_AMESTYPE, t0
    RUN_SCALAR ZTT_MADD_EW_X(0, ZTT_R_T0, 2), 255, 1
    csrr    t0, CSR_AMEXSAT
    bnez    t0, fail

    SET_M_DTYPE 0, DTYPE_INT8
    FILL_RAW_CONST 1, -3
    FILL_RAW_CONST 2, 4
    li      t0, DTYPE_INT8
    csrw    CSR_AMESTYPE, t0
    RUN_SCALAR ZTT_MABSDIFF_EW_X(0, ZTT_R_T0, 2), 5, 1
    RUN_SCALAR ZTT_MADD_EW_X(0, ZTT_R_T0, 2), 5, 9
    RUN_SCALAR ZTT_MAND_EW_X(0, ZTT_R_T0, 2), 5, 4
    RUN_SCALAR ZTT_MANDNOT_EW_X(0, ZTT_R_T0, 2), 5, 1
    RUN_SCALAR ZTT_MCMPGE_EW_X(0, ZTT_R_T0, 2), 5, 0xff
    RUN_SCALAR ZTT_MCMPLT_EW_X(0, ZTT_R_T0, 2), 5, 0
    RUN_SCALAR ZTT_MMAX_EW_X(0, ZTT_R_T0, 2), 5, 5
    RUN_SCALAR ZTT_MMIN_EW_X(0, ZTT_R_T0, 2), 5, 4
    RUN_SCALAR ZTT_MMUL_EW_X(0, ZTT_R_T0, 2), 5, 20
    RUN_SCALAR ZTT_MMULNEG_EW_X(0, ZTT_R_T0, 2), 5, 0xec
    RUN_SCALAR ZTT_MOR_EW_X(0, ZTT_R_T0, 2), 5, 5
    RUN_SCALAR ZTT_MORNOT_EW_X(0, ZTT_R_T0, 2), 5, 0xff
    RUN_SCALAR ZTT_MSUB_EW_X(0, ZTT_R_T0, 2), 5, 0xff
    RUN_SCALAR ZTT_MXOR_EW_X(0, ZTT_R_T0, 2), 5, 1

    /* Shift scalars are fixed controls and ignore amestype. */
    csrwi   CSR_AMESTYPE, 0
    RUN_SCALAR ZTT_MSLL_EW_X(0, ZTT_R_T0, 2), 9, 8
    RUN_SCALAR ZTT_MSRA_EW_X(0, ZTT_R_T0, 2), 9, 2
    RUN_SCALAR ZTT_MSRL_EW_X(0, ZTT_R_T0, 2), 9, 2

    /* A data scalar with an uninitialized datatype sets UN and writes none. */
    FILL_RAW_CONST 0, 0x5a
    csrwi   CSR_AMESTATUS, 0
    li      t0, 1
    .word   ZTT_MADD_EW_X(0, ZTT_R_T0, 2)
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail
    CHECK_RAW_CONST 0, 0x5a

    /* Saturation is accumulated and committed with the destination. */
    SET_M_DTYPE 3, DTYPE_INT8_SAT
    FILL_RAW_CONST 1, 127
    FILL_RAW_CONST 2, 1
    csrwi   CSR_AMEXSAT, 0
    .word   ZTT_MADD_EW(3, 1, 2)
    CHECK_RAW_CONST 3, 127
    csrr    t0, CSR_AMEXSAT
    andi    t0, t0, AME_XSAT_SAT
    beqz    t0, fail

    /* Mixed-width logical squares use N_sq=max(pack_factor). */
    SET_M_DTYPE 4, DTYPE_INT8
    SET_M_DTYPE 6, DTYPE_INT4
    SET_M_DTYPE 8, DTYPE_INT8
    SET_M_DTYPE 9, DTYPE_INT8
    FILL_RAW_CONST 6, 0x11
    FILL_RAW_CONST 8, 2
    FILL_RAW_CONST 9, 2
    .word   ZTT_MADD_EW(4, 6, 8)
    CHECK_RAW_CONST 4, 3
    CHECK_RAW_CONST 5, 3

    /* Unsupported exact tuples set UN and preserve the full destination. */
    SET_M_DTYPE 10, DTYPE_INT16
    FILL_RAW_CONST 10, 0x5a
    FILL_RAW_CONST 11, 0x5a
    csrwi   CSR_AMESTATUS, 0
    .word   ZTT_MAND_EW(10, 1, 2)
    csrr    t0, CSR_AMESTATUS
    andi    t0, t0, AME_STATUS_UN
    beqz    t0, fail
    CHECK_RAW_CONST 10, 0x5a
    CHECK_RAW_CONST 11, 0x5a

    /* Int128 is executed through the same staged scalar path. */
    SET_M_DTYPE 0, DTYPE_INT128
    FILL_RAW_CONST 0, 0xff
    FILL_RAW_CONST 1, 0xff
    FILL_RAW_CONST 2, 0xff
    FILL_RAW_CONST 3, 0xff
    FILL_RAW_CONST 4, 0xff
    FILL_RAW_CONST 5, 0xff
    FILL_RAW_CONST 6, 0xff
    FILL_RAW_CONST 7, 0xff
    FILL_RAW_CONST 8, 0xff
    FILL_RAW_CONST 9, 0xff
    FILL_RAW_CONST 10, 0xff
    FILL_RAW_CONST 11, 0xff
    FILL_RAW_CONST 12, 0xff
    FILL_RAW_CONST 13, 0xff
    FILL_RAW_CONST 14, 0xff
    FILL_RAW_CONST 15, 0xff
    li      t0, DTYPE_INT8
    csrw    CSR_AMESTYPE, t0
    li      t0, 1
    .word   ZTT_MADD_EW_X(0, ZTT_R_T0, 0)
    CHECK_RAW_CONST 0, 0
    CHECK_RAW_CONST 1, 0
    CHECK_RAW_CONST 2, 0
    CHECK_RAW_CONST 3, 0
    CHECK_RAW_CONST 4, 0
    CHECK_RAW_CONST 5, 0
    CHECK_RAW_CONST 6, 0
    CHECK_RAW_CONST 7, 0
    CHECK_RAW_CONST 8, 0
    CHECK_RAW_CONST 9, 0
    CHECK_RAW_CONST 10, 0
    CHECK_RAW_CONST 11, 0
    CHECK_RAW_CONST 12, 0
    CHECK_RAW_CONST 13, 0
    CHECK_RAW_CONST 14, 0
    CHECK_RAW_CONST 15, 0

    .word   ZTT_AME_RELEASE
    li      a0, 0
    j       semihost_exit

trap_handler:
    j       fail

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
