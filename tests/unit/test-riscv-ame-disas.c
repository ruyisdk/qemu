/*
 * RISC-V Attached Matrix Extension disassembler tests
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/bitops.h"
#include "qemu/bswap.h"
#include "disas/dis-asm.h"
#include "disas/riscv-ame.h"
#include "target/riscv/cpu_cfg.h"

#define MASK_R3       0xfe00707f
#define MASK_R2       0xfff0707f
#define MASK_R1       0xfffff07f
#define MASK_ZIP      0xfe007fff
#define MASK_FIXED    0xffffffff

typedef struct AmeDisasCase {
    uint32_t match;
    uint32_t mask;
    const char *mnemonic;
    const char *operands;
} AmeDisasCase;

static const AmeDisasCase ame_disas_cases[] = {
#define AME_TEST_INSN(name, shape, match, mnemonic, operands) \
    { match, MASK_##shape, mnemonic, operands },
#include "riscv-ame-insns.inc"
#undef AME_TEST_INSN
};

static int ame_test_read_memory(bfd_vma address, bfd_byte *buffer, int length,
                                disassemble_info *info)
{
    bfd_vma offset;

    if (address < info->buffer_vma) {
        return 1;
    }
    offset = address - info->buffer_vma;
    if (offset > info->buffer_length ||
        length > info->buffer_length - offset) {
        return 1;
    }
    memcpy(buffer, info->buffer + offset, length);
    return 0;
}

static void ame_test_memory_error(int status, bfd_vma address,
                                  disassemble_info *info)
{
    (void)info;
    g_error("unexpected disassembler read error %d at 0x%" PRIx64,
            status, (uint64_t)address);
}

static char *ame_disassemble(uint32_t insn, bool rv64,
                             const RISCVCPUConfig *cfg)
{
    uint8_t code[4];
    char *text = NULL;
    size_t text_size = 0;
    FILE *stream = open_memstream(&text, &text_size);
    disassemble_info info = {
        .fprintf_func = fprintf,
        .stream = stream,
        .read_memory_func = ame_test_read_memory,
        .memory_error_func = ame_test_memory_error,
        .buffer = code,
        .buffer_vma = 0,
        .buffer_length = sizeof(code),
        .target_info = cfg,
    };
    int length;

    g_assert_nonnull(stream);
    stl_le_p(code, insn);
    length = rv64 ? print_insn_riscv64(0, &info) :
                    print_insn_riscv32(0, &info);
    g_assert_cmpint(length, ==, (insn & 3) == 3 ? 4 : 2);
    g_assert_cmpint(fclose(stream), ==, 0);
    g_assert_cmpuint(strlen(text), ==, text_size);
    return text;
}

static void test_all_instructions(void)
{
    const RISCVCPUConfig cfg = { .ext_ztt = true };

    g_assert_cmpuint(ARRAY_SIZE(ame_disas_cases), ==, 138);
    for (size_t i = 0; i < ARRAY_SIZE(ame_disas_cases); i++) {
        const AmeDisasCase *test = &ame_disas_cases[i];
        uint32_t fields = (UINT32_C(3) << 7) | (UINT32_C(4) << 15) |
                          (UINT32_C(5) << 20);
        uint32_t encoded = test->match | (fields & ~test->mask);
        g_autofree char *expected = g_strdup_printf("%-24s%s",
                                                    test->mnemonic,
                                                    test->operands);

        for (int rv64 = 0; rv64 <= 1; rv64++) {
            g_autofree char *actual = ame_disassemble(encoded, rv64, &cfg);

            g_assert_cmpstr(actual, ==, expected);
        }
    }
}

static void test_extension_gating(void)
{
    const RISCVCPUConfig enabled = { .ext_ztt = true };
    const RISCVCPUConfig disabled = { 0 };
    g_autofree char *rv32 = ame_disassemble(UINT32_C(0x0400002b),
                                            false, &enabled);
    g_autofree char *off = ame_disassemble(UINT32_C(0x0400002b),
                                           true, &disabled);
    g_autofree char *no_cpu = ame_disassemble(UINT32_C(0x0400002b),
                                              true, NULL);
    g_autofree char *old_r2 = ame_disassemble(UINT32_C(0xa2a0002b),
                                              true, &enabled);
    g_autofree char *old_zip = ame_disassemble(UINT32_C(0xc20000ab),
                                               true, &enabled);
    g_autofree char *madd = g_strdup_printf("%-24s%s", "madd.ew",
                                            "m0,m0,m0");
    g_autofree char *illegal = g_strdup_printf("%-24s", "illegal");

    g_assert_cmpstr(rv32, ==, madd);
    g_assert_cmpstr(off, ==, illegal);
    g_assert_cmpstr(no_cpu, ==, illegal);
    g_assert_cmpstr(old_r2, ==, illegal);
    g_assert_cmpstr(old_zip, ==, illegal);
}

static const AmeDisasCase *reference_decode(uint32_t insn)
{
    const AmeDisasCase *match = NULL;

    /* The fixture comes from the spec, not the production opcode table. */
    for (size_t i = 0; i < ARRAY_SIZE(ame_disas_cases); i++) {
        const AmeDisasCase *test = &ame_disas_cases[i];

        if ((insn & test->mask) == test->match) {
            g_assert_null(match);
            match = test;
        }
    }
    return match;
}

static void check_decode(uint32_t insn, const AmeDisasCase *expected)
{
    for (rv_isa isa = rv32; isa <= rv64; isa++) {
        rv_decode dec = {
            .inst = insn,
            .op = UINT16_MAX,
            .opcode_data = ame_opcode_data,
        };

        decode_ame(&dec, isa);
        g_assert_true(dec.opcode_data == ame_opcode_data);
        g_assert_cmpuint(dec.op, <=, ARRAY_SIZE(ame_disas_cases));
        if (!expected) {
            if (dec.op != rv_op_illegal) {
                g_error("%08x: expected illegal, got %s (RV%d)", insn,
                        ame_opcode_data[dec.op].name, isa == rv32 ? 32 : 64);
            }
        } else if (strcmp(ame_opcode_data[dec.op].name, expected->mnemonic)) {
            g_error("%08x: expected %s, got %s (RV%d)", insn,
                    expected->mnemonic, ame_opcode_data[dec.op].name,
                    isa == rv32 ? 32 : 64);
        }
    }
}

static void test_encoding_space(void)
{
    /*
     * All 2^22 encodings with opcode=custom-1 and funct3=0, in both XLENs.
     * The fixture constrains rd either to zero or not at all. Therefore
     * reference lookups for rd=0 and rd=31 suffice for each source pair;
     * the actual decoder still runs for every value of all three fields.
     */
    for (size_t i = 0; i < ARRAY_SIZE(ame_disas_cases); i++) {
        const AmeDisasCase *test = &ame_disas_cases[i];
        uint32_t rd_mask = test->mask & 0xf80;

        g_assert_true(rd_mask == 0 || rd_mask == 0xf80);
        g_assert_cmphex(test->match & 0xf80, ==, 0);
    }
    for (unsigned int funct7 = 0; funct7 < 128; funct7++) {
        for (unsigned int rs2 = 0; rs2 < 32; rs2++) {
            for (unsigned int rs1 = 0; rs1 < 32; rs1++) {
                uint32_t insn = (funct7 << 25) | (rs2 << 20) |
                                (rs1 << 15) | 0x2b;
                const AmeDisasCase *zero = reference_decode(insn);
                const AmeDisasCase *nonzero = reference_decode(insn | 0xf80);

                for (unsigned int rd = 0; rd < 32; rd++) {
                    check_decode(insn | (rd << 7), rd ? nonzero : zero);
                }
            }
        }
    }
}

static void test_other_opcode_and_funct3(void)
{
    for (size_t i = 0; i < ARRAY_SIZE(ame_disas_cases); i++) {
        const AmeDisasCase *test = &ame_disas_cases[i];

        /* Include both extremes of the unconstrained operand fields. */
        for (unsigned int ones = 0; ones <= 1; ones++) {
            uint32_t insn = test->match | (ones ? ~test->mask : 0);

            for (unsigned int opcode = 0; opcode < 128; opcode++) {
                if (opcode != 0x2b) {
                    check_decode((insn & ~0x7f) | opcode, NULL);
                }
            }
            for (unsigned int funct3 = 1; funct3 < 8; funct3++) {
                check_decode(insn | (funct3 << 12), NULL);
            }
        }
    }
}

static void test_operand_boundaries(void)
{
    const RISCVCPUConfig cfg = { .ext_ztt = true };
    static const struct {
        uint32_t insn;
        const char *mnemonic;
        const char *operands;
    } cases[] = {
        { 0x0400002b, "madd.ew", "m0,m0,m0" },
        { 0x05ff8fab, "madd.ew", "m31,m31,m31" },
        { 0x07ff8fab, "madd.ew.x", "m31,t6,m31" },
        { 0x95ff8fab, "mmulacc.2d", "acc31,m31,m31" },
        { 0xa20f8fab, "agettyp", "t6,acc31" },
        { 0xa21f8fab, "asettyp", "acc31,t6" },
        { 0xa49f8fab, "ame.acquire", "t6,t6" },
        { 0xa6000fab, "mzero.2d.acc", "acc31" },
        { 0xa6018fab, "mrowid.ew", "m31" },
        { 0xa9ff8fab, "mls.st", "m31,(t6),t6" },
        { 0xc3ff802b, "mcolunzip.ew", "m31,m31" },
    };

    for (size_t i = 0; i < ARRAY_SIZE(cases); i++) {
        g_autofree char *expected = g_strdup_printf("%-24s%s",
                                                    cases[i].mnemonic,
                                                    cases[i].operands);

        for (int rv64 = 0; rv64 <= 1; rv64++) {
            g_autofree char *actual = ame_disassemble(cases[i].insn,
                                                    rv64, &cfg);

            g_assert_cmpstr(actual, ==, expected);
        }
    }
}

static void test_other_decoders(void)
{
    const RISCVCPUConfig cfg = {
        .ext_ztt = true,
        .ext_xtheadsync = true,
        .ext_XVentanaCondOps = true,
        .ext_xlrbr = true,
    };
    static const struct {
        uint32_t insn;
        const char *mnemonic;
    } cases[] = {
        { 0x00000013, "nop" },
        { 0x00000001, "nop" },
        { 0x00000073, "ecall" },
        { 0x0180000b, "th.sync" },
        { 0x0000607b, "vt.maskc" },
        { 0x61001013, "crc32.b" },
    };

    for (size_t i = 0; i < ARRAY_SIZE(cases); i++) {
        g_autofree char *prefix = g_strdup_printf("%-24s", cases[i].mnemonic);

        for (int rv64 = 0; rv64 <= 1; rv64++) {
            g_autofree char *actual = ame_disassemble(cases[i].insn,
                                                    rv64, &cfg);

            g_assert_true(g_str_has_prefix(actual, prefix));
        }
    }
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);
    g_test_add_func("/riscv/ame/disas/all-instructions",
                    test_all_instructions);
    g_test_add_func("/riscv/ame/disas/extension-gating",
                    test_extension_gating);
    g_test_add_func("/riscv/ame/disas/encoding-space", test_encoding_space);
    g_test_add_func("/riscv/ame/disas/other-opcode-funct3",
                    test_other_opcode_and_funct3);
    g_test_add_func("/riscv/ame/disas/operand-boundaries",
                    test_operand_boundaries);
    g_test_add_func("/riscv/ame/disas/other-decoders", test_other_decoders);
    return g_test_run();
}
