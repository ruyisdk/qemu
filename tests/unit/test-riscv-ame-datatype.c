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
#include "target/riscv/ame/datatype.h"

static const AmeDTypeDesc *dtype(uint32_t encoding)
{
    const AmeDTypeDesc *desc = riscv_ame_dtype_by_encoding(encoding);

    g_assert_nonnull(desc);
    return desc;
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

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);
    g_test_add_func("/riscv/ame/datatype/descriptors",
                    test_dtype_descriptors);
    g_test_add_func("/riscv/ame/datatype/classes", test_dtype_classes);
    g_test_add_func("/riscv/ame/datatype/encoding-space",
                    test_dtype_encoding_space);

    return g_test_run();
}
