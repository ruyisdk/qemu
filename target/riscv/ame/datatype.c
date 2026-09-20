/*
 * RISC-V Attached Matrix Extension (AME) datatype definitions
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/bitops.h"
#include "target/riscv/ame/ame.h"
#include "target/riscv/ame/datatype.h"

G_STATIC_ASSERT(AME_DTYPE_COUNT <= AME_DTYPE_ID_INVALID);

#define AME_INT_WIDTH_LIST(_) \
    _(AME_INT_WIDTH_INDEX_4, 4) \
    _(AME_INT_WIDTH_INDEX_8, 8) \
    _(AME_INT_WIDTH_INDEX_16, 16) \
    _(AME_INT_WIDTH_INDEX_32, 32) \
    _(AME_INT_WIDTH_INDEX_64, 64) \
    _(AME_INT_WIDTH_INDEX_128, 128)

#define AME_FP_FORMAT_LIST(_) \
    _(AME_FP_FORMAT_FP16, 16, AME_DTYPE_CLASS_IEEE, 5, true, true) \
    _(AME_FP_FORMAT_BF16, 16, AME_DTYPE_CLASS_IEEE, 8, true, true) \
    _(AME_FP_FORMAT_FP32, 32, AME_DTYPE_CLASS_IEEE, 8, true, true) \
    _(AME_FP_FORMAT_FP64, 64, AME_DTYPE_CLASS_IEEE, 11, true, true) \
    _(AME_FP_FORMAT_OCP_E4M3, 8, AME_DTYPE_CLASS_OCP, 4, false, true) \
    _(AME_FP_FORMAT_OCP_E5M2, 8, AME_DTYPE_CLASS_OCP, 5, true, true)

/* Dense datatype indexes for the descriptor array and bitmaps. */
#define AME_INT_ID(width_index, sign, sat, round) \
    ((width_index) * AME_INT_VARIANTS_PER_WIDTH + \
     (sign) * 2 * AME_INT_ROUND_MAX + (sat) * AME_INT_ROUND_MAX + (round))
#define AME_INT_ENCODING(bits, sign, sat, round) \
    (((uint32_t)(sign) << 30) | ((uint32_t)(sat) << 29) | \
     ((uint32_t)(round) << 27) | (bits))
#define AME_INT_DESC(width_index_, bits_, sign_, sat_, round_) \
    [AME_INT_ID(width_index_, sign_, sat_, round_)] = { \
        .encoding = AME_INT_ENCODING(bits_, sign_, sat_, round_), \
        .id = AME_INT_ID(width_index_, sign_, sat_, round_), \
        .bits = bits_, \
        .class = AME_DTYPE_CLASS_INTEGER, \
        .integer = { \
            .rounding = round_, \
            .is_signed = sign_, \
            .saturating = sat_, \
        }, \
    },
#define AME_INT_ROUND_DESCS(width_index_, bits_, sign_, sat_) \
    AME_INT_DESC(width_index_, bits_, sign_, sat_, AME_INT_RNU) \
    AME_INT_DESC(width_index_, bits_, sign_, sat_, AME_INT_RNE) \
    AME_INT_DESC(width_index_, bits_, sign_, sat_, AME_INT_RDN) \
    AME_INT_DESC(width_index_, bits_, sign_, sat_, AME_INT_ROD)
#define AME_INT_WIDTH_DESCS(width_index_, bits_) \
    AME_INT_ROUND_DESCS(width_index_, bits_, false, false) \
    AME_INT_ROUND_DESCS(width_index_, bits_, false, true) \
    AME_INT_ROUND_DESCS(width_index_, bits_, true, false) \
    AME_INT_ROUND_DESCS(width_index_, bits_, true, true)

#define AME_FP_FORMAT_INDEX(format) ((format) - AME_FP_FORMAT_FP16)
#define AME_FP_ID(format, round) \
    (AME_INT_DTYPE_COUNT + AME_FP_FORMAT_INDEX(format) * AME_FP_ROUND_MAX + \
     (round))
#define AME_FP_ENCODING(bits, exp, inf, denorm, round) \
    (((uint32_t)(exp) << 26) | ((uint32_t)(round) << 22) | \
     ((uint32_t)(inf) << 21) | ((uint32_t)(denorm) << 20) | \
     BIT(8) | (bits))
#define AME_FP_DESC(format_, bits_, class_, exp_, inf_, denorm_, \
                    round_) \
    [AME_FP_ID(format_, round_)] = { \
        .encoding = AME_FP_ENCODING(bits_, exp_, inf_, denorm_, round_), \
        .id = AME_FP_ID(format_, round_), \
        .bits = bits_, \
        .class = class_, \
        .fp = { .format = format_, .rounding = round_ }, \
    },
#define AME_FP_FORMAT_DESCS(format_, bits_, class_, exp_, inf_, denorm_) \
    AME_FP_DESC(format_, bits_, class_, exp_, inf_, denorm_, AME_FP_RNE) \
    AME_FP_DESC(format_, bits_, class_, exp_, inf_, denorm_, AME_FP_RTZ) \
    AME_FP_DESC(format_, bits_, class_, exp_, inf_, denorm_, AME_FP_RDN) \
    AME_FP_DESC(format_, bits_, class_, exp_, inf_, denorm_, AME_FP_RUP) \
    AME_FP_DESC(format_, bits_, class_, exp_, inf_, denorm_, AME_FP_RMM) \
    AME_FP_DESC(format_, bits_, class_, exp_, inf_, denorm_, AME_FP_RNO)

const AmeDTypeDesc riscv_ame_dtypes[AME_DTYPE_COUNT] = {
    AME_INT_WIDTH_LIST(AME_INT_WIDTH_DESCS)
    AME_FP_FORMAT_LIST(AME_FP_FORMAT_DESCS)
};

#define AME_FP_FORMAT_DESC(format_, bits_, class_, exp_, inf_, denorm_) \
    [format_] = { \
        .bits = bits_, \
        .class = class_, \
        .has_inf = inf_, \
        .has_denorm = denorm_, \
    },

static const AmeFPFormatDesc ame_fp_formats[AME_FP_FORMAT_MAX] = {
    AME_FP_FORMAT_LIST(AME_FP_FORMAT_DESC)
};
#undef AME_FP_FORMAT_DESC

const AmeFPFormatDesc *riscv_ame_fp_format_desc(AmeFPFormat format)
{
    if (format < AME_FP_FORMAT_FP16 || format >= AME_FP_FORMAT_MAX) {
        return NULL;
    }
    return &ame_fp_formats[format];
}

/*
 * Source-level entry for custom descriptors. A concrete implementation must
 * also extend by-ID lookup and the descriptor array capacity, and supply
 * explicit capability schemas and execution paths. Custom encodings are
 * opaque apart from the size byte; never reinterpret their standard bits.
 */
static const AmeDTypeDesc *ame_custom_dtype_by_encoding(uint32_t encoding)
{
    return NULL;
}

const AmeDTypeDesc *riscv_ame_dtype_by_encoding(uint32_t encoding)
{
    unsigned bits = extract32(encoding, 0, 8);
    unsigned round, id;
    const AmeDTypeDesc *dtype;

    if (encoding & BIT(31)) {
        return ame_custom_dtype_by_encoding(encoding);
    }

    if (extract32(encoding, 8, 1)) {
        AmeFPFormat format;

        round = extract32(encoding, 22, 4);
        if (round >= AME_FP_ROUND_MAX) {
            return NULL;
        }
        /* Select the candidate by storage size and exponent width. */
        switch (bits) {
        case 8:
            format = extract32(encoding, 26, 5) == 4 ?
                     AME_FP_FORMAT_OCP_E4M3 : AME_FP_FORMAT_OCP_E5M2;
            break;
        case 16:
            format = extract32(encoding, 26, 5) == 5 ?
                     AME_FP_FORMAT_FP16 : AME_FP_FORMAT_BF16;
            break;
        case 32:
            format = AME_FP_FORMAT_FP32;
            break;
        case 64:
            format = AME_FP_FORMAT_FP64;
            break;
        default:
            return NULL;
        }
        id = AME_FP_ID(format, round);
    } else {
        unsigned width_index;

        if (bits < AME_DTYPE_BITS_MIN || !is_power_of_2(bits)) {
            return NULL;
        }
        width_index = ctz32(bits) - ctz32(AME_DTYPE_BITS_MIN);
        if (width_index >= AME_INT_WIDTH_COUNT) {
            return NULL;
        }
        id = AME_INT_ID(width_index, extract32(encoding, 30, 1),
                        extract32(encoding, 29, 1),
                        extract32(encoding, 27, 2));
    }

    /* Reject reserved bits and nonstandard format attributes. */
    dtype = riscv_ame_dtype_by_id(id);
    return dtype && dtype->encoding == encoding ? dtype : NULL;
}

const AmeDTypeDesc *riscv_ame_dtype_by_id(uint16_t id)
{
    if (id >= AME_DTYPE_COUNT) {
        return NULL;
    }
    return &riscv_ame_dtypes[id];
}
