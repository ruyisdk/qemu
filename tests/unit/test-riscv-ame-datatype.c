/*
 * RISC-V Attached Matrix Extension datatype and capability tests
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qapi/error.h"
#include "qemu/bitops.h"
#include "target/riscv/ame/cap-set.h"

static const AmeDTypeDesc *dtype(uint32_t encoding)
{
    const AmeDTypeDesc *desc = riscv_ame_dtype_by_encoding(encoding);

    g_assert_nonnull(desc);
    return desc;
}

static const AmeDTypeDesc *fp_dtype(AmeFPFormat format, AmeFPRoundMode round)
{
    static const uint32_t base[] = {
        [AME_FP_FORMAT_FP16] = UINT32_C(0x14300110),
        [AME_FP_FORMAT_BF16] = UINT32_C(0x20300110),
        [AME_FP_FORMAT_FP32] = UINT32_C(0x20300120),
        [AME_FP_FORMAT_FP64] = UINT32_C(0x2c300140),
        [AME_FP_FORMAT_OCP_E4M3] = UINT32_C(0x10100108),
        [AME_FP_FORMAT_OCP_E5M2] = UINT32_C(0x14300108),
    };

    g_assert_cmpuint(format, >=, AME_FP_FORMAT_FP16);
    g_assert_cmpuint(format, <=, AME_FP_FORMAT_OCP_E5M2);
    return dtype(base[format] | ((uint32_t)round << 22));
}

static void test_dtype_descriptors(void)
{
    const AmeDTypeDesc *int4;
    const AmeDTypeDesc *uint64_sat_rod;
    const AmeDTypeDesc *fp16;
    const AmeDTypeDesc *bf16;
    const AmeDTypeDesc *e4m3;
    const AmeDTypeDesc *e5m2;

    g_assert_cmpuint(AME_DTYPE_COUNT, ==, 132);
    for (size_t i = 0; i < AME_DTYPE_COUNT; i++) {
        const AmeDTypeDesc *desc = &riscv_ame_dtypes[i];

        g_assert_cmpuint(desc->id, ==, i);
        g_assert_true(riscv_ame_dtype_by_id(i) == desc);
        g_assert_true(riscv_ame_dtype_by_encoding(desc->encoding) == desc);
        g_assert_cmpuint(desc->bits, >, 0);
        g_assert_cmpuint(desc->bits, <=, 128);

        for (size_t j = i + 1; j < AME_DTYPE_COUNT; j++) {
            g_assert_cmphex(desc->encoding, !=,
                            riscv_ame_dtypes[j].encoding);
        }
    }

    int4 = dtype(UINT32_C(0x40000004));
    g_assert_cmpuint(int4->class, ==, AME_DTYPE_CLASS_INTEGER);
    g_assert_cmpuint(int4->bits, ==, 4);
    g_assert_true(int4->integer.is_signed);
    g_assert_false(int4->integer.saturating);
    g_assert_cmpuint(int4->integer.rounding, ==, AME_INT_RNU);

    uint64_sat_rod = dtype(UINT32_C(0x38000040));
    g_assert_cmpuint(uint64_sat_rod->bits, ==, 64);
    g_assert_false(uint64_sat_rod->integer.is_signed);
    g_assert_true(uint64_sat_rod->integer.saturating);
    g_assert_cmpuint(uint64_sat_rod->integer.rounding, ==, AME_INT_ROD);

    fp16 = dtype(UINT32_C(0x14300110));
    bf16 = dtype(UINT32_C(0x20300110));
    e4m3 = dtype(UINT32_C(0x10100108));
    e5m2 = dtype(UINT32_C(0x14300108));
    g_assert_cmpuint(fp16->fp.format, ==, AME_FP_FORMAT_FP16);
    g_assert_cmpuint(bf16->fp.format, ==, AME_FP_FORMAT_BF16);
    g_assert_cmpuint(fp16->bits, ==, bf16->bits);
    g_assert_cmpuint(fp16->class, ==, AME_DTYPE_CLASS_IEEE);
    g_assert_cmpuint(e4m3->class, ==, AME_DTYPE_CLASS_OCP);
    g_assert_cmpuint(e4m3->fp.format, ==, AME_FP_FORMAT_OCP_E4M3);
    g_assert_cmpuint(e5m2->fp.format, ==, AME_FP_FORMAT_OCP_E5M2);
    g_assert_false(riscv_ame_fp_format_desc(e4m3->fp.format)->has_inf);
    g_assert_true(riscv_ame_fp_format_desc(e5m2->fp.format)->has_inf);

    g_assert_null(riscv_ame_dtype_by_encoding(0));
    g_assert_null(riscv_ame_dtype_by_encoding(UINT32_C(0x80000004)));
    g_assert_null(riscv_ame_dtype_by_encoding(UINT32_C(0x40000204)));
    g_assert_null(riscv_ame_dtype_by_encoding(UINT32_C(0x40000003)));
    g_assert_null(riscv_ame_dtype_by_encoding(UINT32_C(0x15b00110)));
    g_assert_null(riscv_ame_dtype_by_id(AME_DTYPE_ID_INVALID));
}

static void test_dtype_classes(void)
{
    AmeDTypeDesc custom = { .class = AME_DTYPE_CLASS_CUSTOM };
    AmeDTypeDesc unknown = { .class = AME_DTYPE_CLASS_MAX };

    g_assert_false(riscv_ame_dtype_is_fp(NULL));
    g_assert_false(riscv_ame_dtype_is_standard(NULL));
    g_assert_false(riscv_ame_dtype_is_fp(&custom));
    g_assert_false(riscv_ame_dtype_is_standard(&custom));
    g_assert_false(riscv_ame_dtype_is_fp(&unknown));
    g_assert_false(riscv_ame_dtype_is_standard(&unknown));
    for (size_t i = 0; i < AME_STANDARD_DTYPE_COUNT; i++) {
        const AmeDTypeDesc *desc = &riscv_ame_dtypes[i];

        g_assert_true(riscv_ame_dtype_is_standard(desc));
        g_assert_cmpint(riscv_ame_dtype_is_fp(desc), ==,
                        (desc->encoding & BIT(8)) != 0);
        g_assert_null(riscv_ame_dtype_by_encoding(desc->encoding | BIT(31)));
    }
}

/* The v0.6 encoding tables, independently of the descriptor array and IDs. */
static bool standard_dtype_encoding(uint32_t encoding)
{
    unsigned bits = encoding & 0xff;

    if (!(encoding & BIT(8))) {
        if (encoding & UINT32_C(0x87ffff00)) {
            return false;
        }
        return bits == 4 || bits == 8 || bits == 16 || bits == 32 ||
               bits == 64 || bits == 128;
    }

    if (((encoding >> 22) & 15) > 5) {
        return false;
    }
    switch (encoding & ~UINT32_C(0x03c00000)) {
    case UINT32_C(0x14300110): /* FP16 */
    case UINT32_C(0x20300110): /* BF16 */
    case UINT32_C(0x20300120): /* FP32 */
    case UINT32_C(0x2c300140): /* FP64 */
    case UINT32_C(0x10100108): /* OCP E4M3 */
    case UINT32_C(0x14300108): /* OCP E5M2 */
        return true;
    default:
        return false;
    }
}

static void check_dtype_encoding(uint32_t encoding)
{
    const AmeDTypeDesc *desc = riscv_ame_dtype_by_encoding(encoding);

    g_assert_cmpint(desc != NULL, ==, standard_dtype_encoding(encoding));
    if (desc) {
        g_assert_cmphex(desc->encoding, ==, encoding);
        g_assert_true(riscv_ame_dtype_by_id(desc->id) == desc);
        g_assert_cmpuint(desc->bits, ==, encoding & 0xff);
        if (!(encoding & BIT(8))) {
            g_assert_cmpuint(desc->class, ==, AME_DTYPE_CLASS_INTEGER);
            g_assert_cmpuint(desc->integer.rounding, ==,
                             (encoding >> 27) & 3);
            g_assert_cmpint(desc->integer.is_signed, ==,
                            (encoding >> 30) & 1);
            g_assert_cmpint(desc->integer.saturating, ==,
                            (encoding >> 29) & 1);
        } else {
            const AmeFPFormatDesc *format =
                riscv_ame_fp_format_desc(desc->fp.format);

            g_assert_nonnull(format);
            g_assert_cmpuint(desc->class, ==,
                             desc->bits == 8 ? AME_DTYPE_CLASS_OCP :
                                               AME_DTYPE_CLASS_IEEE);
            g_assert_cmpuint(desc->fp.rounding, ==, (encoding >> 22) & 15);
            g_assert_cmpuint(format->bits, ==, desc->bits);
            g_assert_cmpuint(format->class, ==, desc->class);
            g_assert_cmpint(format->has_inf, ==, (encoding >> 21) & 1);
            g_assert_cmpint(format->has_denorm, ==, (encoding >> 20) & 1);
        }
    }
}

static void test_dtype_encoding_space(void)
{
    unsigned fields, bits, reserved;

    g_assert_null(riscv_ame_fp_format_desc(AME_FP_FORMAT_NONE));
    g_assert_null(riscv_ame_fp_format_desc(AME_FP_FORMAT_MAX));
    g_assert_null(riscv_ame_fp_format_desc(-1));

    /* All size bytes and integer sign/saturation/rounding combinations. */
    for (fields = 0; fields < 16; fields++) {
        for (bits = 0; bits < 256; bits++) {
            uint32_t encoding = (fields << 27) | bits;

            check_dtype_encoding(encoding);
            check_dtype_encoding(encoding | BIT(31));
            if (standard_dtype_encoding(encoding)) {
                for (reserved = 9; reserved <= 26; reserved++) {
                    check_dtype_encoding(encoding | BIT(reserved));
                }
            }
        }
    }

    /* All exponent, rounding, infinity, denormal and size field values. */
    for (fields = 0; fields < (1 << 11); fields++) {
        for (bits = 0; bits < 256; bits++) {
            uint32_t encoding = (fields << 20) | BIT(8) | bits;

            check_dtype_encoding(encoding);
            if (standard_dtype_encoding(encoding)) {
                check_dtype_encoding(encoding | BIT(31));
                for (reserved = 9; reserved <= 19; reserved++) {
                    check_dtype_encoding(encoding | BIT(reserved));
                }
            }
        }
    }
}

static bool dtype_supported(uint16_t id)
{
    const AmeDTypeDesc *desc = &riscv_ame_dtypes[id];

    /* Implementation boundary, independent of the capability bitmap. */
    return desc->class != AME_DTYPE_CLASS_OCP ||
           desc->fp.rounding != AME_FP_RNO;
}

static bool has(AmeOperation op, uint16_t dest, uint16_t a, uint16_t b)
{
    return riscv_ame_capabilities_has(riscv_ame_maximum_capabilities(),
                                      op, AME_FORM_BASE, dest, a, b);
}

static void test_maximum_closed_world(void)
{
    const AmeCapabilities *max = riscv_ame_maximum_capabilities();
    size_t supported = 0;

    g_assert_cmphex(max->max_int_dtype, ==, RV_AME_MAX_INT_DTYPE);
    for (size_t i = 0; i < AME_DTYPE_COUNT; i++) {
        bool enabled = riscv_ame_dtype_set_has(&max->dtypes, i);

        g_assert_cmpint(enabled, ==, dtype_supported(i));
        g_assert_cmpint(has(AME_OP_ZERO, i, AME_DTYPE_ID_INVALID,
                            AME_DTYPE_ID_INVALID), ==, enabled);
        supported += enabled;
    }
    g_assert_cmpuint(supported, ==, 130);
    for (unsigned op = 0; op < AME_OP_COUNT; op++) {
        g_assert_cmpuint(max->by_op[op][AME_FORM_BASE].count, >, 0);
        g_assert_false(has(op, AME_DTYPE_ID_INVALID,
                           AME_DTYPE_ID_INVALID, AME_DTYPE_ID_INVALID));
    }
    g_assert_false(has(AME_OP_COUNT, 0, 0, 0));
    g_assert_false(has(-1, 0, 0, 0));
}

static void test_explicit_allowlist(void)
{
    static const uint32_t encodings[] = { 0x40000008, 0x40000040 };
    AmeDTypeSet public = { 0 };
    AmeCapRequest request = {
        .key = { AME_OP_CONV, AME_FORM_BASE, AME_CAP_CATEGORY_INTEGER },
        .kind = AME_TUPLE_SET_RULE,
    };
    AmeCapabilities explicit = { 0 }, equal = { 0 };
    const AmeCapabilities *max = riscv_ame_maximum_capabilities();
    uint16_t int64 = dtype(0x40000040)->id;
    uint16_t int128 = dtype(0x40000080)->id;

    g_assert_true(riscv_ame_dtype_set_from_encodings(
        &public, encodings, ARRAY_SIZE(encodings), &error_abort));
    request.operand_types[0] = public;
    request.operand_types[1] = public;
    g_assert_true(riscv_ame_capabilities_init(
        &explicit, 0x40000040, &public, &request, 1, &error_abort));
    g_assert_false(riscv_ame_capabilities_equal(max, &explicit));
    g_assert_true(riscv_ame_capabilities_has(
        &explicit, AME_OP_CONV, AME_FORM_BASE,
        int64, int64, AME_DTYPE_ID_INVALID));
    g_assert_false(riscv_ame_capabilities_has(
        &explicit, AME_OP_CONV, AME_FORM_BASE,
        int64, int128, AME_DTYPE_ID_INVALID));
    riscv_ame_capabilities_copy(&equal, max);
    g_assert_true(riscv_ame_capabilities_equal(max, &equal));
    riscv_ame_capabilities_clear(&equal);
    riscv_ame_capabilities_clear(&explicit);
}

static void test_numeric_conversion_matrix(void)
{
    size_t destination;

    for (destination = 0; destination < AME_DTYPE_COUNT; destination++) {
        const AmeDTypeDesc *dest_dtype = &riscv_ame_dtypes[destination];
        size_t source;

        for (source = 0; source < AME_DTYPE_COUNT; source++) {
            const AmeDTypeDesc *source_dtype = &riscv_ame_dtypes[source];
            bool dest_supported = dtype_supported(dest_dtype->id);
            bool source_supported = dtype_supported(source_dtype->id);
            bool expected = dest_supported && source_supported;

            g_assert_cmpint(has(AME_OP_CONV,
                                 dest_dtype->id, source_dtype->id,
                                 AME_DTYPE_ID_INVALID), ==, expected);
        }
    }
}

static bool fp_mixed_tuple(const AmeDTypeDesc *dest,
                           const AmeDTypeDesc *source)
{
    bool dest_fp = dest->class == AME_DTYPE_CLASS_IEEE ||
                   dest->class == AME_DTYPE_CLASS_OCP;
    bool source_fp = source->class == AME_DTYPE_CLASS_IEEE ||
                     source->class == AME_DTYPE_CLASS_OCP;

    return dest_fp && source_fp;
}

static void test_integer_elementwise_matrix(void)
{
    static const AmeOperation arithmetic_ops[] = {
        AME_OP_ABSDIFF, AME_OP_ADD, AME_OP_HDIFF,
        AME_OP_MEAN, AME_OP_MUL, AME_OP_NMUL,
        AME_OP_SUB,
    };
    static const AmeOperation fused_ops[] = {
        AME_OP_MULACC, AME_OP_MULADD, AME_OP_MULNACC,
        AME_OP_MULSUB,
    };
    static const AmeOperation scale_ops[] = {
        AME_OP_LDEXP, AME_OP_RDEXP,
    };
    static const AmeOperation scale_acc_ops[] = {
        AME_OP_LDEXPACC, AME_OP_RDEXPACC,
    };
    static const AmeOperation bitwise_ops[] = {
        AME_OP_AND, AME_OP_ANDNOT, AME_OP_OR,
        AME_OP_ORNOT, AME_OP_XOR,
    };
    size_t destination;

    for (destination = 0; destination < AME_DTYPE_COUNT; destination++) {
        const AmeDTypeDesc *dest = &riscv_ame_dtypes[destination];
        size_t source1;

        if (dest->class != AME_DTYPE_CLASS_INTEGER) {
            continue;
        }
        for (source1 = 0; source1 < AME_DTYPE_COUNT; source1++) {
            const AmeDTypeDesc *src1 = &riscv_ame_dtypes[source1];
            size_t source2;

            if (src1->class != AME_DTYPE_CLASS_INTEGER) {
                continue;
            }
            g_assert_true(has(AME_OP_ABS, dest->id, src1->id,
                               AME_DTYPE_ID_INVALID));

            for (source2 = 0; source2 < AME_DTYPE_COUNT; source2++) {
                const AmeDTypeDesc *src2 = &riscv_ame_dtypes[source2];
                size_t i;

                if (src2->class != AME_DTYPE_CLASS_INTEGER) {
                    continue;
                }
                for (i = 0; i < ARRAY_SIZE(arithmetic_ops); i++) {
                    g_assert_true(has(arithmetic_ops[i], dest->id,
                                       src1->id, src2->id));
                }
                for (i = 0; i < ARRAY_SIZE(fused_ops); i++) {
                    g_assert_true(has(fused_ops[i], dest->id,
                                       src1->id, src2->id));
                }
                for (i = 0; i < ARRAY_SIZE(scale_ops); i++) {
                    g_assert_true(has(scale_ops[i], dest->id,
                                       src1->id, src2->id));
                }
                for (i = 0; i < ARRAY_SIZE(scale_acc_ops); i++) {
                    g_assert_true(has(scale_acc_ops[i], dest->id,
                                       src1->id, src2->id));
                }
                for (i = 0; i < ARRAY_SIZE(bitwise_ops); i++) {
                    g_assert_cmpint(has(bitwise_ops[i], dest->id,
                                         src1->id, src2->id), ==,
                                     dest->id == src1->id &&
                                     dest->id == src2->id);
                }
                g_assert_true(has(AME_OP_CMPGE, dest->id,
                                   src1->id, src2->id));
                g_assert_true(has(AME_OP_CMPLT, dest->id,
                                   src1->id, src2->id));
                g_assert_cmpint(has(AME_OP_MAX, dest->id,
                                     src1->id, src2->id), ==,
                                 dest->id == src1->id &&
                                 dest->id == src2->id);
                g_assert_cmpint(has(AME_OP_MIN, dest->id,
                                     src1->id, src2->id), ==,
                                 dest->id == src1->id &&
                                 dest->id == src2->id);
                g_assert_cmpint(has(AME_OP_SEL, dest->id,
                                     src1->id, src2->id), ==,
                                 dest->id == src2->id);
                g_assert_cmpint(has(AME_OP_SELGE, dest->id,
                                     src1->id, src2->id), ==,
                                 dest->id == src2->id);
                g_assert_cmpint(has(AME_OP_MUXLT, dest->id,
                                     src1->id, src2->id), ==,
                                 dest->id == src2->id);
                g_assert_cmpint(has(AME_OP_MUXGE, dest->id,
                                     src1->id, src2->id), ==,
                                 dest->id == src2->id);
                g_assert_true(has(AME_OP_SLL, dest->id, src1->id, src2->id));
                g_assert_cmpint(has(AME_OP_SRA, dest->id,
                                     src1->id, src2->id), ==,
                                 src1->integer.is_signed);
                g_assert_true(has(AME_OP_SRL, dest->id, src1->id, src2->id));
            }
        }
    }
}

static void test_structural_capability_matrix(void)
{
    static const AmeOperation permutation_ops[] = {
        AME_OP_COLBCAST, AME_OP_ROWBCAST,
        AME_OP_SHIFT, AME_OP_ROWSHIFT,
    };
    static const AmeOperation zip_ops[] = {
        AME_OP_COLZIP, AME_OP_COLUNZIP,
        AME_OP_ROWZIP, AME_OP_ROWUNZIP,
    };
    static const AmeOperation gather_ops[] = {
        AME_OP_GATHER, AME_OP_GATHERROW,
    };
    static const AmeOperation index_ops[] = {
        AME_OP_COLID, AME_OP_ROWID,
    };
    uint16_t int8 = dtype(UINT32_C(0x40000008))->id;
    uint16_t fp16 = dtype(UINT32_C(0x14300110))->id;
    size_t datatype, operation;

    for (datatype = 0; datatype < AME_DTYPE_COUNT; datatype++) {
        const AmeDTypeDesc *desc = &riscv_ame_dtypes[datatype];
        bool data_supported = dtype_supported(desc->id);

        for (operation = 0; operation < ARRAY_SIZE(permutation_ops);
             operation++) {
            g_assert_cmpint(has(permutation_ops[operation], desc->id,
                                 desc->id, AME_DTYPE_ID_INVALID), ==,
                             data_supported);
        }
        for (operation = 0; operation < ARRAY_SIZE(zip_ops); operation++) {
            g_assert_cmpint(has(zip_ops[operation], desc->id, desc->id,
                                 AME_DTYPE_ID_INVALID), ==,
                             data_supported);
        }
        for (operation = 0; operation < ARRAY_SIZE(gather_ops);
             operation++) {
            g_assert_cmpint(has(gather_ops[operation], desc->id,
                                 desc->id, int8), ==,
                             data_supported);
        }
        for (operation = 0; operation < ARRAY_SIZE(index_ops); operation++) {
            g_assert_cmpint(has(index_ops[operation],
                                 desc->id, AME_DTYPE_ID_INVALID,
                                 AME_DTYPE_ID_INVALID), ==,
                             desc->class == AME_DTYPE_CLASS_INTEGER);
        }
    }

    g_assert_cmpint(has(AME_OP_COLBCAST,
                         int8, fp16, AME_DTYPE_ID_INVALID), ==,
                     false);
    g_assert_false(has(AME_OP_COLZIP, int8, fp16,
                       AME_DTYPE_ID_INVALID));
    g_assert_cmpint(has(AME_OP_GATHER,
                         fp16, fp16, fp16), ==, false);
}

static void test_reduction_scatter_capability_matrix(void)
{
    static const AmeOperation reduction_ops[] = {
        AME_OP_PREFIXADD, AME_OP_PREFIXADDCOL,
        AME_OP_PREFIXMAX, AME_OP_PREFIXMAXCOL,
        AME_OP_REDUCEADD, AME_OP_REDUCEADDCOL,
        AME_OP_REDUCEMAX, AME_OP_REDUCEMAXCOL,
        AME_OP_REDUCEMIN, AME_OP_REDUCEMINCOL,
    };
    static const AmeOperation scatter_ops[] = {
        AME_OP_COLSCATADD, AME_OP_COLSCATMAX,
        AME_OP_ROWSCATADD, AME_OP_ROWSCATMAX,
    };
    uint16_t index_dtype = dtype(UINT32_C(0x00000040))->id;
    size_t operation, destination, source, index;

    for (destination = 0; destination < AME_DTYPE_COUNT; destination++) {
        const AmeDTypeDesc *dest = &riscv_ame_dtypes[destination];
        bool dest_supported = dtype_supported(dest->id);
        bool can_reduce = dest_supported &&
                          (dest->class == AME_DTYPE_CLASS_INTEGER ||
                           dest->class == AME_DTYPE_CLASS_IEEE ||
                           dest->class == AME_DTYPE_CLASS_OCP);

        for (source = 0; source < AME_DTYPE_COUNT; source++) {
            const AmeDTypeDesc *src = &riscv_ame_dtypes[source];
            bool src_supported = dtype_supported(src->id);
            bool expected_reduce = can_reduce && src_supported;
            bool expected_scatter = dest_supported && src_supported;
            for (operation = 0; operation < ARRAY_SIZE(reduction_ops);
                 operation++) {
                g_assert_cmpint(has(reduction_ops[operation], dest->id,
                                     src->id, AME_DTYPE_ID_INVALID), ==,
                                 expected_reduce);
            }

            for (operation = 0; operation < ARRAY_SIZE(scatter_ops);
                 operation++) {
                g_assert_cmpint(has(scatter_ops[operation], dest->id,
                                     src->id, index_dtype), ==,
                                 expected_scatter);
            }
        }
    }

    for (index = 0; index < AME_DTYPE_COUNT; index++) {
        const AmeDTypeDesc *index_desc = &riscv_ame_dtypes[index];
        uint16_t int8 = dtype(UINT32_C(0x40000008))->id;

        g_assert_cmpint(has(AME_OP_COLSCATADD, int8, int8,
                             index_desc->id), ==,
                         index_desc->class == AME_DTYPE_CLASS_INTEGER);
    }

    g_assert_cmpint(has(AME_OP_REDUCEADD,
                         dtype(UINT32_C(0x40000010))->id,
                         dtype(UINT32_C(0x40000008))->id,
                         AME_DTYPE_ID_INVALID), ==, true);
}

static void test_numeric_matmul_capability_matrix(void)
{
    static const AmeOperation operations[] = {
        AME_OP_MMACC, AME_OP_MMACCNEG,
        AME_OP_MMATACC, AME_OP_MMATACCNEG,
        AME_OP_MMBTACC, AME_OP_MMBTACCNEG,
    };
    size_t destination, source1, source2, operation;

    for (destination = 0; destination < AME_DTYPE_COUNT; destination++) {
        const AmeDTypeDesc *dest = &riscv_ame_dtypes[destination];

        for (source1 = 0; source1 < AME_DTYPE_COUNT; source1++) {
            const AmeDTypeDesc *src1 = &riscv_ame_dtypes[source1];

            for (source2 = 0; source2 < AME_DTYPE_COUNT; source2++) {
                const AmeDTypeDesc *src2 = &riscv_ame_dtypes[source2];
                bool supported = dtype_supported(dest->id) &&
                    dtype_supported(src1->id) &&
                    dtype_supported(src2->id);

                for (operation = 0; operation < ARRAY_SIZE(operations);
                     operation++) {
                    g_assert_cmpint(has(operations[operation], dest->id,
                                         src1->id, src2->id), ==,
                                     supported);
                }
            }
        }
    }
}

static void test_ieee_elementwise_capability_matrix(void)
{
    static const AmeOperation mixed_ops[] = {
        AME_OP_ABSDIFF, AME_OP_ADD, AME_OP_HDIFF,
        AME_OP_LOG2SUB, AME_OP_MEAN, AME_OP_MUL,
        AME_OP_NMUL, AME_OP_SUB, AME_OP_SUBLOG2,
    };
    static const AmeOperation fused_ops[] = {
        AME_OP_MULACC, AME_OP_MULADD, AME_OP_MULNACC,
        AME_OP_MULSUB,
    };
    static const AmeOperation same_dtype_ops[] = {
        AME_OP_MAX, AME_OP_MIN,
    };
    static const AmeOperation unary_ops[] = {
        AME_OP_EXP2, AME_OP_FRINTM, AME_OP_FRINTN,
        AME_OP_FRINTP, AME_OP_FRINTZ, AME_OP_LOG2,
        AME_OP_REC, AME_OP_RSQRT, AME_OP_SQRT,
        AME_OP_TANH, AME_OP_COS, AME_OP_SIN,
    };
    static const AmeOperation scale_ops[] = {
        AME_OP_LDEXP, AME_OP_RDEXP,
    };
    static const AmeOperation scale_acc_ops[] = {
        AME_OP_LDEXPACC, AME_OP_RDEXPACC,
    };
    uint16_t exponent = dtype(UINT32_C(0x40000008))->id;
    uint16_t signed_predicate = dtype(UINT32_C(0x40000008))->id;
    uint16_t unsigned_predicate = dtype(UINT32_C(0x00000008))->id;
    uint16_t fp16 = dtype(UINT32_C(0x14300110))->id;
    uint16_t int128 = dtype(UINT32_C(0x40000080))->id;
    size_t destination;

    for (destination = 0; destination < AME_DTYPE_COUNT; destination++) {
        const AmeDTypeDesc *dest = &riscv_ame_dtypes[destination];
        size_t source1;

        if (dest->class != AME_DTYPE_CLASS_IEEE) {
            continue;
        }
        for (source1 = 0; source1 < AME_DTYPE_COUNT; source1++) {
            const AmeDTypeDesc *src1 = &riscv_ame_dtypes[source1];
            bool source1_supported = dtype_supported(src1->id);
            bool mixed = source1_supported && fp_mixed_tuple(dest, src1);
            size_t operation;
            size_t source2;

            g_assert_cmpint(has(AME_OP_ABS,
                                 dest->id, src1->id,
                                 AME_DTYPE_ID_INVALID), ==,
                             source1_supported);
            for (operation = 0;
                 operation < ARRAY_SIZE(unary_ops);
                 operation++) {
                g_assert_cmpint(has(unary_ops[operation],
                                     dest->id, src1->id,
                                     AME_DTYPE_ID_INVALID), ==,
                                 mixed);
            }
            for (operation = 0; operation < ARRAY_SIZE(scale_ops);
                 operation++) {
                g_assert_cmpint(has(scale_ops[operation], dest->id,
                                     src1->id, exponent), ==,
                                 source1_supported);
            }
            for (operation = 0; operation < ARRAY_SIZE(scale_acc_ops);
                 operation++) {
                g_assert_cmpint(has(scale_acc_ops[operation], dest->id,
                                     src1->id, exponent), ==,
                                 source1_supported);
            }

            for (source2 = 0; source2 < AME_DTYPE_COUNT; source2++) {
                const AmeDTypeDesc *src2 = &riscv_ame_dtypes[source2];
                bool source2_supported = dtype_supported(src2->id);
                bool both_supported = source1_supported && source2_supported;
                bool all_same_dtype = dest->id == src1->id &&
                                      dest->id == src2->id;

                for (operation = 0; operation < ARRAY_SIZE(mixed_ops);
                     operation++) {
                    bool logarithm = mixed_ops[operation] == AME_OP_LOG2SUB ||
                                     mixed_ops[operation] == AME_OP_SUBLOG2;

                    g_assert_cmpint(has(mixed_ops[operation], dest->id,
                                         src1->id, src2->id), ==,
                                     both_supported && (!logarithm || mixed));
                }
                for (operation = 0; operation < ARRAY_SIZE(fused_ops);
                     operation++) {
                    g_assert_cmpint(has(fused_ops[operation], dest->id,
                                         src1->id, src2->id), ==,
                                     both_supported);
                }
                for (operation = 0;
                     operation < ARRAY_SIZE(same_dtype_ops);
                     operation++) {
                    g_assert_cmpint(has(same_dtype_ops[operation],
                                         dest->id, src1->id, src2->id), ==,
                                     all_same_dtype);
                }
            }
        }
    }

    for (size_t source1 = 0; source1 < AME_DTYPE_COUNT; source1++) {
        const AmeDTypeDesc *src1 = &riscv_ame_dtypes[source1];

        for (size_t source2 = 0; source2 < AME_DTYPE_COUNT; source2++) {
            const AmeDTypeDesc *src2 = &riscv_ame_dtypes[source2];
            bool expected = dtype_supported(src1->id) &&
                            dtype_supported(src2->id);

            g_assert_cmpint(has(AME_OP_CMPGE,
                                 int128, src1->id, src2->id), ==, expected);
            g_assert_cmpint(has(AME_OP_CMPLT,
                                 int128, src1->id, src2->id), ==, expected);
        }
    }

    g_assert_cmpint(has(AME_OP_SEL,
                         fp16, signed_predicate, fp16), ==,
                     true);
    g_assert_cmpint(has(AME_OP_SEL,
                         fp16, fp16, fp16), ==, true);
    g_assert_cmpint(has(AME_OP_SEL,
                         fp16, unsigned_predicate, fp16), ==,
                     true);
    g_assert_cmpint(has(AME_OP_COS,
                         fp16, fp16, AME_DTYPE_ID_INVALID), ==,
                     true);
}

static void test_ocp_capability_matrix(void)
{
    static const AmeOperation unary_ops[] = {
        AME_OP_COS, AME_OP_EXP2,
        AME_OP_FRINTM, AME_OP_FRINTN, AME_OP_FRINTP,
        AME_OP_FRINTZ, AME_OP_LOG2, AME_OP_REC,
        AME_OP_RSQRT, AME_OP_SIN, AME_OP_SQRT, AME_OP_TANH,
    };
    static const AmeOperation binary_ops[] = {
        AME_OP_ABSDIFF, AME_OP_ADD, AME_OP_HDIFF,
        AME_OP_LOG2SUB, AME_OP_MEAN,
        AME_OP_MAX, AME_OP_MIN, AME_OP_MUL,
        AME_OP_NMUL, AME_OP_SUB, AME_OP_SUBLOG2,
    };
    static const AmeOperation ternary_ops[] = {
        AME_OP_MULACC, AME_OP_MULADD, AME_OP_MULNACC,
        AME_OP_MULSUB,
    };
    static const AmeOperation reduce_ops[] = {
        AME_OP_PREFIXADD, AME_OP_PREFIXADDCOL,
        AME_OP_PREFIXMAX, AME_OP_PREFIXMAXCOL,
        AME_OP_REDUCEADD, AME_OP_REDUCEADDCOL,
        AME_OP_REDUCEMAX, AME_OP_REDUCEMAXCOL,
        AME_OP_REDUCEMIN, AME_OP_REDUCEMINCOL,
    };
    static const AmeOperation scatter_ops[] = {
        AME_OP_COLSCATADD, AME_OP_COLSCATMAX,
        AME_OP_ROWSCATADD, AME_OP_ROWSCATMAX,
    };
    static const AmeOperation matmul_ops[] = {
        AME_OP_MMACC, AME_OP_MMACCNEG,
        AME_OP_MMATACC, AME_OP_MMATACCNEG,
        AME_OP_MMBTACC, AME_OP_MMBTACCNEG,
    };
    static const AmeOperation structural_ops[] = {
        AME_OP_COLBCAST, AME_OP_ROWBCAST,
        AME_OP_ROWSHIFT, AME_OP_SHIFT,
    };
    const AmeDTypeDesc *int8 = dtype(UINT32_C(0x40000008));
    const AmeDTypeDesc *fp16 = fp_dtype(AME_FP_FORMAT_FP16, AME_FP_RNE);
    size_t format, round, operation;

    for (format = AME_FP_FORMAT_OCP_E4M3;
         format <= AME_FP_FORMAT_OCP_E5M2; format++) {
        for (round = AME_FP_RNE; round <= AME_FP_RNO; round++) {
            const AmeDTypeDesc *fp8 = fp_dtype(format, round);
            bool supported = round != AME_FP_RNO;

            g_assert_cmpint(dtype_supported(fp8->id), ==,
                            supported);
            g_assert_cmpint(has(AME_OP_ABS,
                                 fp8->id, fp8->id,
                                 AME_DTYPE_ID_INVALID), ==,
                             supported);
            for (operation = 0; operation < ARRAY_SIZE(unary_ops);
                 operation++) {
                for (unsigned source = 0; source < AME_DTYPE_COUNT; source++) {
                    const AmeDTypeDesc *src = &riscv_ame_dtypes[source];
                    bool expected = supported && dtype_supported(src->id) &&
                                    fp_mixed_tuple(fp8, src);

                    g_assert_cmpint(has(unary_ops[operation], fp8->id,
                                         src->id, AME_DTYPE_ID_INVALID), ==,
                                     expected);
                }
            }
            for (operation = 0; operation < ARRAY_SIZE(binary_ops);
                 operation++) {
                g_assert_cmpint(has(binary_ops[operation], fp8->id,
                                     fp8->id, fp8->id), ==,
                                 supported);
            }
            for (operation = 0; operation < ARRAY_SIZE(ternary_ops);
                 operation++) {
                g_assert_cmpint(has(ternary_ops[operation], fp8->id,
                                     fp8->id, fp8->id), ==,
                                 supported);
            }
            for (operation = 0; operation < ARRAY_SIZE(reduce_ops);
                 operation++) {
                g_assert_cmpint(has(reduce_ops[operation], fp8->id,
                                     fp8->id, AME_DTYPE_ID_INVALID), ==,
                                 supported);
            }
            for (operation = 0; operation < ARRAY_SIZE(scatter_ops);
                 operation++) {
                g_assert_cmpint(has(scatter_ops[operation], fp8->id,
                                     fp8->id, int8->id), ==,
                                 supported);
            }
            for (operation = 0; operation < ARRAY_SIZE(matmul_ops);
                 operation++) {
                g_assert_cmpint(has(matmul_ops[operation], fp8->id,
                                     fp8->id, fp8->id), ==,
                                 supported);
            }
            for (operation = 0; operation < ARRAY_SIZE(structural_ops);
                 operation++) {
                g_assert_cmpint(has(structural_ops[operation], fp8->id,
                                     fp8->id, AME_DTYPE_ID_INVALID), ==,
                                 supported);
            }
            g_assert_cmpint(has(AME_OP_LDEXP,
                                 fp8->id, fp8->id, int8->id), ==,
                             supported);
            g_assert_cmpint(has(AME_OP_LDEXPACC, fp8->id, fp8->id,
                                 int8->id), ==,
                             supported);
            g_assert_cmpint(has(AME_OP_CMPGE,
                                 int8->id, fp8->id, fp8->id), ==,
                             supported);
            g_assert_cmpint(has(AME_OP_SEL,
                                 fp8->id, fp8->id, fp8->id), ==,
                             supported);
            g_assert_cmpint(has(AME_OP_ZERO,
                                 fp8->id, AME_DTYPE_ID_INVALID,
                                 AME_DTYPE_ID_INVALID), ==,
                             supported);
        }
    }

    g_assert_cmpint(has(AME_OP_ADD,
                         fp16->id,
                         fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE)->id,
                         fp_dtype(AME_FP_FORMAT_OCP_E5M2, AME_FP_RNE)->id),
                     ==, true);
    g_assert_cmpint(has(AME_OP_ADD,
                         fp_dtype(AME_FP_FORMAT_OCP_E4M3, AME_FP_RNE)->id,
                         fp_dtype(AME_FP_FORMAT_OCP_E5M2, AME_FP_RNE)->id,
                         fp_dtype(AME_FP_FORMAT_OCP_E5M2, AME_FP_RNE)->id),
                     ==, true);
}

static void test_fp_mixed_capability_matrix(void)
{
    static const AmeOperation binary_ops[] = {
        AME_OP_ABSDIFF, AME_OP_ADD, AME_OP_HDIFF,
        AME_OP_MEAN, AME_OP_MUL, AME_OP_NMUL, AME_OP_SUB,
    };
    static const AmeOperation fused_ops[] = {
        AME_OP_MULACC, AME_OP_MULADD,
        AME_OP_MULNACC, AME_OP_MULSUB,
    };
    static const AmeOperation matmul_ops[] = {
        AME_OP_MMACC, AME_OP_MMACCNEG,
        AME_OP_MMATACC, AME_OP_MMATACCNEG,
        AME_OP_MMBTACC, AME_OP_MMBTACCNEG,
    };
    static const AmeOperation scatter_ops[] = {
        AME_OP_COLSCATADD, AME_OP_COLSCATMAX,
        AME_OP_ROWSCATADD, AME_OP_ROWSCATMAX,
    };
    uint16_t int8 = dtype(UINT32_C(0x40000008))->id;

    for (size_t d = 0; d < AME_DTYPE_COUNT; d++) {
        const AmeDTypeDesc *dest = &riscv_ame_dtypes[d];

        if (!dtype_supported(dest->id) ||
            (dest->class != AME_DTYPE_CLASS_IEEE &&
             dest->class != AME_DTYPE_CLASS_OCP)) {
            continue;
        }
        for (size_t s1 = 0; s1 < AME_DTYPE_COUNT; s1++) {
            const AmeDTypeDesc *src1 = &riscv_ame_dtypes[s1];

            if (!dtype_supported(src1->id) ||
                (src1->class != AME_DTYPE_CLASS_IEEE &&
                 src1->class != AME_DTYPE_CLASS_OCP)) {
                continue;
            }
            g_assert_cmpint(has(AME_OP_ABS,
                                 dest->id, src1->id,
                                 AME_DTYPE_ID_INVALID), ==,
                             true);
            g_assert_cmpint(has(AME_OP_LDEXP,
                                 dest->id, src1->id, int8), ==,
                             true);
            g_assert_cmpint(has(AME_OP_LDEXPACC,
                                 dest->id, src1->id, int8), ==,
                             true);
            for (size_t op = 0; op < ARRAY_SIZE(scatter_ops); op++) {
                g_assert_cmpint(has(scatter_ops[op], dest->id, src1->id,
                                     int8), ==, true);
            }

            for (size_t s2 = 0; s2 < AME_DTYPE_COUNT; s2++) {
                const AmeDTypeDesc *src2 = &riscv_ame_dtypes[s2];

                if (!dtype_supported(src2->id) ||
                    (src2->class != AME_DTYPE_CLASS_IEEE &&
                     src2->class != AME_DTYPE_CLASS_OCP)) {
                    continue;
                }
                for (size_t op = 0; op < ARRAY_SIZE(binary_ops); op++) {
                    g_assert_cmpint(has(binary_ops[op], dest->id,
                                         src1->id, src2->id), ==,
                                     true);
                }
                for (size_t op = 0; op < ARRAY_SIZE(fused_ops); op++) {
                    g_assert_cmpint(has(fused_ops[op], dest->id,
                                         src1->id, src2->id), ==,
                                     true);
                }
                for (size_t op = 0; op < ARRAY_SIZE(matmul_ops); op++) {
                    g_assert_cmpint(has(matmul_ops[op], dest->id,
                                         src1->id, src2->id), ==,
                                     true);
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);
    g_test_add_func("/riscv/ame/datatype/descriptors",
                    test_dtype_descriptors);
    g_test_add_func("/riscv/ame/datatype/classes", test_dtype_classes);
    g_test_add_func("/riscv/ame/datatype/encoding-space",
                    test_dtype_encoding_space);
    g_test_add_func("/riscv/ame/capability/maximum-closed-world",
                    test_maximum_closed_world);
    g_test_add_func("/riscv/ame/capability/explicit-allowlist",
                    test_explicit_allowlist);
    g_test_add_func("/riscv/ame/capability/numeric-conversion-matrix",
                    test_numeric_conversion_matrix);
    g_test_add_func("/riscv/ame/capability/integer-elementwise-matrix",
                    test_integer_elementwise_matrix);
    g_test_add_func("/riscv/ame/capability/structural-matrix",
                    test_structural_capability_matrix);
    g_test_add_func("/riscv/ame/capability/reduction-scatter-matrix",
                    test_reduction_scatter_capability_matrix);
    g_test_add_func("/riscv/ame/capability/numeric-matmul-matrix",
                    test_numeric_matmul_capability_matrix);
    g_test_add_func("/riscv/ame/capability/ieee-elementwise-matrix",
                    test_ieee_elementwise_capability_matrix);
    g_test_add_func("/riscv/ame/capability/ocp-matrix",
                    test_ocp_capability_matrix);
    g_test_add_func("/riscv/ame/capability/fp-mixed-matrix",
                    test_fp_mixed_capability_matrix);

    return g_test_run();
}
