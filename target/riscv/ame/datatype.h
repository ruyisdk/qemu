/*
 * RISC-V Attached Matrix Extension (AME) datatype definitions
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef TARGET_RISCV_AME_DATATYPE_H
#define TARGET_RISCV_AME_DATATYPE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "qemu/host-utils.h"

/* Default AME_MAX_INT_DTYPE: signed Int128, nonsaturating, RNU. */
#define RV_AME_MAX_INT_DTYPE UINT32_C(0x40000080)
/* Marks an absent datatype, including an unused source in a tuple. */
#define AME_DTYPE_ID_INVALID UINT16_MAX

/* Selects the interpretation of a datatype's bits and descriptor union. */
typedef enum AmeDTypeClass {
    AME_DTYPE_CLASS_INTEGER,
    AME_DTYPE_CLASS_IEEE,
    AME_DTYPE_CLASS_OCP,
    AME_DTYPE_CLASS_CUSTOM,
    AME_DTYPE_CLASS_MAX,
} AmeDTypeClass;

/* Floating-point storage format, independent of the rounding mode. */
typedef enum AmeFPFormat {
    AME_FP_FORMAT_NONE,
    AME_FP_FORMAT_FP16,
    AME_FP_FORMAT_BF16,
    AME_FP_FORMAT_FP32,
    AME_FP_FORMAT_FP64,
    AME_FP_FORMAT_OCP_E4M3,
    AME_FP_FORMAT_OCP_E5M2,
    AME_FP_FORMAT_MAX,
} AmeFPFormat;

/* AME integer rounding encodings, used when an operation discards low bits. */
typedef enum AmeIntRoundMode {
    AME_INT_RNU,
    AME_INT_RNE,
    AME_INT_RDN,
    AME_INT_ROD,
    AME_INT_ROUND_MAX,
} AmeIntRoundMode;

/* AME floating-point rounding encodings. */
typedef enum AmeFPRoundMode {
    AME_FP_RNE,
    AME_FP_RTZ,
    AME_FP_RDN,
    AME_FP_RUP,
    AME_FP_RMM,
    AME_FP_RNO,
    AME_FP_ROUND_MAX,
} AmeFPRoundMode;

/* Array/bitmap sizing; integer variants include sign, saturation, rounding. */
enum {
    AME_INT_WIDTH_INDEX_4,
    AME_INT_WIDTH_INDEX_8,
    AME_INT_WIDTH_INDEX_16,
    AME_INT_WIDTH_INDEX_32,
    AME_INT_WIDTH_INDEX_64,
    AME_INT_WIDTH_INDEX_128,
    AME_INT_WIDTH_COUNT,
    AME_INT_VARIANTS_PER_WIDTH = 2 * 2 * AME_INT_ROUND_MAX,
    AME_INT_DTYPE_COUNT = AME_INT_WIDTH_COUNT * AME_INT_VARIANTS_PER_WIDTH,
    AME_FP_FORMAT_COUNT = AME_FP_FORMAT_MAX - AME_FP_FORMAT_FP16,
    AME_FP_DTYPE_COUNT = AME_FP_FORMAT_COUNT * AME_FP_ROUND_MAX,
    AME_STANDARD_DTYPE_COUNT = AME_INT_DTYPE_COUNT + AME_FP_DTYPE_COUNT,
    /* Capacity of the descriptor array and datatype bitmaps. */
    AME_DTYPE_COUNT = AME_STANDARD_DTYPE_COUNT,
    AME_DTYPE_WORD_COUNT = (AME_DTYPE_COUNT + 63) / 64,
};

/* Storage width and special-value support shared by all rounding variants. */
typedef struct AmeFPFormatDesc {
    uint16_t bits;
    AmeDTypeClass class;
    bool has_inf;
    bool has_denorm; /* The format can represent subnormal values. */
} AmeFPFormatDesc;

/*
 * One complete AME datatype: its encoding, element width and numerical rules.
 * id is its index in riscv_ame_dtypes[], used by datatype sets and tuples.
 */
typedef struct AmeDTypeDesc {
    uint32_t encoding;
    uint16_t id;
    uint16_t bits;
    AmeDTypeClass class;
    /* Only INTEGER uses integer; only IEEE/OCP use fp. CUSTOM uses neither. */
    union {
        struct {
            AmeIntRoundMode rounding;
            bool is_signed;
            bool saturating; /* Clamp integer results instead of wrapping. */
        } integer;
        struct {
            AmeFPFormat format;
            AmeFPRoundMode rounding;
        } fp;
    };
} AmeDTypeDesc;

/**
 * riscv_ame_dtype_is_integer: Check the standard integer class and width
 * @dtype: Descriptor to inspect; may be NULL
 *
 * Returns: true for integer widths 4, 8, 16, 32, 64 and 128.
 */
static inline bool riscv_ame_dtype_is_integer(const AmeDTypeDesc *dtype)
{
    return dtype && dtype->class == AME_DTYPE_CLASS_INTEGER &&
           dtype->bits >= 4 && dtype->bits <= 128 &&
           is_power_of_2(dtype->bits);
}

/**
 * riscv_ame_dtype_is_fp: Identify an IEEE or OCP floating-point datatype
 * @dtype: Descriptor to inspect; may be NULL
 *
 * Returns: true for IEEE/OCP classes; false for NULL or other classes.
 */
static inline bool riscv_ame_dtype_is_fp(const AmeDTypeDesc *dtype)
{
    return dtype && (dtype->class == AME_DTYPE_CLASS_IEEE ||
                     dtype->class == AME_DTYPE_CLASS_OCP);
}

/**
 * riscv_ame_dtype_is_standard: Identify a standard integer or floating datatype
 * @dtype: Descriptor to inspect; may be NULL
 *
 * Returns: true for INTEGER, IEEE or OCP; false for NULL or CUSTOM.
 */
static inline bool riscv_ame_dtype_is_standard(const AmeDTypeDesc *dtype)
{
    return dtype && (dtype->class == AME_DTYPE_CLASS_INTEGER ||
                     riscv_ame_dtype_is_fp(dtype));
}

/* Immutable datatype descriptions indexed by id. */
extern const AmeDTypeDesc riscv_ame_dtypes[AME_DTYPE_COUNT];

/**
 * riscv_ame_fp_format_desc: Look up floating-point format properties
 * @format: Storage format to look up
 *
 * Returns: a borrowed, static descriptor; NULL for NONE or an invalid format.
 */
const AmeFPFormatDesc *riscv_ame_fp_format_desc(AmeFPFormat format);

/**
 * riscv_ame_dtype_by_encoding: Decode a complete AME datatype encoding
 * @encoding: 32-bit datatype encoding from an AME type register
 *
 * Check all encoding fields, including reserved bits.
 *
 * Returns: a borrowed, static descriptor; NULL for an unrecognized encoding.
 */
const AmeDTypeDesc *riscv_ame_dtype_by_encoding(uint32_t encoding);

/**
 * riscv_ame_dtype_by_id: Look up a datatype by its internal array index
 * @id: Index in riscv_ame_dtypes[]
 *
 * Returns: a borrowed, static descriptor; NULL if @id is out of range.
 */
const AmeDTypeDesc *riscv_ame_dtype_by_id(uint16_t id);

#endif /* TARGET_RISCV_AME_DATATYPE_H */
