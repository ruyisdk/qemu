/*
 * RISC-V Ztt standard decodetree tests
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/bitops.h"

typedef struct DisasContext {
    const char *name;
    int rd;
    int rs1;
    int rs2;
} DisasContext;

#include "decode-ztt.c.inc"

#define DECODE_R3(name, a) \
    ((DisasContext) { name, (a)->rd, (a)->rs1, (a)->rs2 })
#define DECODE_R2(name, a) \
    ((DisasContext) { name, (a)->rd, (a)->rs1, -1 })
#define DECODE_R1(name, a) \
    ((DisasContext) { name, (a)->rd, -1, -1 })
#define DECODE_ZIP(name, a) \
    ((DisasContext) { name, -1, (a)->rs1, (a)->rs2 })
#define DECODE_FIXED(name, a) \
    ((DisasContext) { name, -1, -1, -1 })

#define AME_TEST_INSN(name, shape, match, mnemonic, operands) \
static bool trans_##name(DisasContext *ctx, arg_##name *a) \
{ \
    *ctx = DECODE_##shape(#name, a); \
    return true; \
}
#include "riscv-ame-insns.inc"
#undef AME_TEST_INSN

#define MASK_R3       0xfe00707f
#define MASK_R2       0xfff0707f
#define MASK_R1       0xfffff07f
#define MASK_ZIP      0xfe007fff
#define MASK_FIXED    0xffffffff

typedef struct AmeDecodeCase {
    const char *name;
    uint32_t match;
    uint32_t mask;
} AmeDecodeCase;

static const AmeDecodeCase cases[] = {
#define AME_TEST_INSN(name, shape, match, mnemonic, operands) \
    { #name, match, MASK_##shape },
#include "riscv-ame-insns.inc"
#undef AME_TEST_INSN
};

static void check_decode(uint32_t word, const AmeDecodeCase *expected)
{
    static const unsigned int shifts[] = { 7, 15, 20 };
    DisasContext ctx = { 0 };
    bool decoded = decode_ztt(&ctx, word);
    int fields[3];

    g_assert_cmpint(decoded, ==, expected != NULL);
    if (!decoded) {
        g_assert_null(ctx.name);
        return;
    }

    g_assert_cmpstr(ctx.name, ==, expected->name);
    fields[0] = ctx.rd;
    fields[1] = ctx.rs1;
    fields[2] = ctx.rs2;
    for (size_t i = 0; i < ARRAY_SIZE(shifts); i++) {
        int value = expected->mask & (31 << shifts[i]) ? -1 :
                    (word >> shifts[i]) & 31;

        g_assert_cmpint(fields[i], ==, value);
    }
}

static void test_all_operand_fields(void)
{
    size_t checked = 0;

    g_assert_cmpuint(ARRAY_SIZE(cases), ==, 138);
    for (size_t i = 0; i < ARRAY_SIZE(cases); i++) {
        const AmeDecodeCase *c = &cases[i];
        unsigned int nd = c->mask & 0x00000f80 ? 1 : 32;
        unsigned int ns1 = c->mask & 0x000f8000 ? 1 : 32;
        unsigned int ns2 = c->mask & 0x01f00000 ? 1 : 32;

        for (unsigned int d = 0; d < nd; d++) {
            for (unsigned int s1 = 0; s1 < ns1; s1++) {
                for (unsigned int s2 = 0; s2 < ns2; s2++) {
                    uint32_t word = c->match | (d << 7) | (s1 << 15) |
                                    (s2 << 20);

                    check_decode(word, c);
                    checked++;
                }
            }
        }
    }
    g_assert_cmpuint(checked, ==, 3025025);
}

/* Include reserved selectors and fixed fields in all function banks. */
static void test_selector_space(void)
{
    for (uint32_t f7 = 0; f7 < 128; f7++) {
        for (uint32_t s2 = 0; s2 < 32; s2++) {
            for (uint32_t s1 = 0; s1 < 32; s1++) {
                for (uint32_t d = 0; d < 32; d += 31) {
                    uint32_t word = (f7 << 25) | (s2 << 20) | (s1 << 15) |
                                    (d << 7) | 0x2b;
                    const AmeDecodeCase *expected = NULL;

                    for (size_t i = 0; i < ARRAY_SIZE(cases); i++) {
                        if ((word & cases[i].mask) == cases[i].match) {
                            g_assert_null(expected);
                            expected = &cases[i];
                        }
                    }
                    check_decode(word, expected);
                }
            }
        }
    }
}

static void test_other_opcodes(void)
{
    for (size_t i = 0; i < ARRAY_SIZE(cases); i++) {
        uint32_t word = cases[i].match;

        for (uint32_t f3 = 1; f3 < 8; f3++) {
            check_decode(word | (f3 << 12), NULL);
        }
        for (uint32_t opcode = 0; opcode < 128; opcode++) {
            if (opcode != 0x2b) {
                check_decode((word & ~UINT32_C(127)) | opcode, NULL);
            }
        }
    }
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);
    g_test_add_func("/riscv/ame/decode/all-operand-fields",
                    test_all_operand_fields);
    g_test_add_func("/riscv/ame/decode/selector-space", test_selector_space);
    g_test_add_func("/riscv/ame/decode/other-opcodes", test_other_opcodes);
    return g_test_run();
}
