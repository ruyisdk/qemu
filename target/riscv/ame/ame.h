/*
 * RISC-V Attached Matrix Extension (AME) common definitions
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef TARGET_RISCV_AME_H
#define TARGET_RISCV_AME_H

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>

/* Bounds: N is the square side, U is bits per M cell, NM/NA count registers. */
#define RV_AME_N_MAX         128
#define RV_AME_UDSZ_MAX      64
#define RV_AME_M_REGS_MAX    32
#define RV_AME_ACC_REGS_MAX  16
#define AME_DTYPE_BITS_MIN   4

/* amestatus.UN: the requested operation/datatype combination is unsupported. */
#define AME_STATUS_UN        (1U << 0)
/* amefflags: inexact, underflow, overflow, divide-by-zero and invalid. */
#define AME_FFLAGS_MASK      0x1f
#define AME_FFLAG_NX         (1U << 0)
#define AME_FFLAG_UF         (1U << 1)
#define AME_FFLAG_OF         (1U << 2)
#define AME_FFLAG_DZ         (1U << 3)
#define AME_FFLAG_NV         (1U << 4)
#define AME_XSAT_SAT         (1U << 0)

/* Numerical or structural operation used by AME instructions. */
typedef enum AmeOperation {
    AME_OP_ABS,
    AME_OP_ABSDIFF,
    AME_OP_ADD,
    AME_OP_AND,
    AME_OP_ANDNOT,
    AME_OP_CMPGE,
    AME_OP_CMPLT,
    AME_OP_COLBCAST,
    AME_OP_COLID,
    AME_OP_COLUNZIP,
    AME_OP_COLZIP,
    AME_OP_CONV,
    AME_OP_COS,
    AME_OP_EXP2,
    AME_OP_FRINTM,
    AME_OP_FRINTN,
    AME_OP_FRINTP,
    AME_OP_FRINTZ,
    AME_OP_GATHER,
    AME_OP_GATHERROW,
    AME_OP_HDIFF,
    AME_OP_LDEXP,
    AME_OP_LDEXPACC,
    AME_OP_LOG2,
    AME_OP_LOG2SUB,
    AME_OP_MAX,
    AME_OP_MEAN,
    AME_OP_MIN,
    AME_OP_MMACC,
    AME_OP_MMACCNEG,
    AME_OP_MMATACC,
    AME_OP_MMATACCNEG,
    AME_OP_MMBTACC,
    AME_OP_MMBTACCNEG,
    AME_OP_MUL,
    AME_OP_MULACC,
    AME_OP_MULADD,
    AME_OP_MULNACC,
    AME_OP_MULSUB,
    AME_OP_MUXGE,
    AME_OP_MUXLT,
    AME_OP_NMUL,
    AME_OP_OR,
    AME_OP_ORNOT,
    AME_OP_PREFIXADD,
    AME_OP_PREFIXADDCOL,
    AME_OP_PREFIXMAX,
    AME_OP_PREFIXMAXCOL,
    AME_OP_RDEXP,
    AME_OP_RDEXPACC,
    AME_OP_REC,
    AME_OP_REDUCEADD,
    AME_OP_REDUCEADDCOL,
    AME_OP_REDUCEMAX,
    AME_OP_REDUCEMAXCOL,
    AME_OP_REDUCEMIN,
    AME_OP_REDUCEMINCOL,
    AME_OP_ROWBCAST,
    AME_OP_ROWID,
    AME_OP_ROWSHIFT,
    AME_OP_ROWUNZIP,
    AME_OP_ROWZIP,
    AME_OP_RSQRT,
    AME_OP_COLSCATADD,
    AME_OP_COLSCATMAX,
    AME_OP_ROWSCATADD,
    AME_OP_ROWSCATMAX,
    AME_OP_SEL,
    AME_OP_SELGE,
    AME_OP_SHIFT,
    AME_OP_SIN,
    AME_OP_SLL,
    AME_OP_SQRT,
    AME_OP_SRA,
    AME_OP_SRL,
    AME_OP_SUB,
    AME_OP_SUBLOG2,
    AME_OP_TANH,
    AME_OP_XOR,
    AME_OP_ZERO,
    AME_OP_COUNT,
} AmeOperation;

/*
 * Raw bits of one tensor element, with bits 0..63 in lo and 64..127 in hi.
 * The operand's AmeDTypeDesc supplies its width and numerical interpretation.
 */
typedef struct AmeValue {
    uint64_t lo;
    uint64_t hi;
} AmeValue;

#define AME_VALUE_WORD_BITS  (sizeof(uint64_t) * CHAR_BIT)
#define AME_VALUE_BITS       (sizeof(AmeValue) * CHAR_BIT)
#define AME_VALUE_WORDS      (AME_VALUE_BITS / AME_VALUE_WORD_BITS)

/*
 * One computed element and returned exception/saturation flags. fflags uses
 * AME_FFLAG_* and may include earlier exceptions in a shared float_status;
 * saturated records integer clamping. The caller writes the value and
 * accrues the flags into architectural state.
 */
typedef struct AmeExecResult {
    AmeValue value;
    uint8_t fflags;
    bool saturated;
} AmeExecResult;

/* Flags combined across the elements of one tensor operation. */
typedef struct AmeExecFlags {
    uint8_t fflags;
    bool saturated;
} AmeExecFlags;

#endif /* TARGET_RISCV_AME_H */
