/*
 * RISC-V Ztt migration tests
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "libqtest.h"
#include "qobject/qdict.h"
#include "qemu/bswap.h"
#include "qemu/cutils.h"

#define ZTT_MIGRATION_CODE_ADDR UINT64_C(0x80000000)
#define ZTT_MIGRATION_MAILBOX   UINT64_C(0x80000700)
#define ZTT_SHARED_HART0_ADDR   UINT64_C(0x80000000)
#define ZTT_SHARED_HART1_ADDR   UINT64_C(0x80000200)
#define ZTT_SHARED_HART2_ADDR   UINT64_C(0x80000400)
#define ZTT_VM_SUBSECTION       0x05

enum {
    ZTT_SHARED_CONTROL,
    ZTT_SHARED_HART0_READY = 4,
    ZTT_SHARED_HART1_READY = 8,
    ZTT_SHARED_HART2_READY = 12,
    ZTT_SHARED_HART0_SCALAR = 16,
    ZTT_SHARED_HART0_VALUE = 20,
    ZTT_SHARED_HART0_OWNED = 24,
    ZTT_SHARED_HART0_DONE = 28,
    ZTT_SHARED_HART1_SCALAR = 32,
    ZTT_SHARED_HART1_VALUE = 36,
    ZTT_SHARED_HART1_OWNED = 40,
    ZTT_SHARED_HART1_DONE = 44,
    ZTT_SHARED_HART2_RESULT = 48,
    ZTT_SHARED_HART2_SCALAR = 52,
    ZTT_SHARED_HART2_VALUE = 56,
    ZTT_SHARED_HART2_OWNED = 60,
    ZTT_SHARED_START_WAITER = 64,
    ZTT_SHARED_HART2_DONE = 68,
    ZTT_SHARED_HART0_ACQUIRE = 76,
    ZTT_SHARED_HART1_ACQUIRE = 80,
    ZTT_RESET_RESULT = 84,
    ZTT_RESET_SCALAR = 88,
    ZTT_RESET_VALUE = 92,
    ZTT_RESET_OWNED = 96,
    ZTT_RESET_DONE = 100,
    ZTT_RESET_FFLAGS = 104,
    ZTT_RESET_XSAT = 108,
};

static const uint8_t ztt_manifest_prefix[] = {
    0, 0, 0, 0, 0, 0, 0, 4, /* N */
    0, 0, 0, 0, 0, 0, 0, 8, /* U */
    16, 1,                   /* NM, NA */
    0x41, 0x4d, 0x45, 0x43, /* AMEC */
};

#define ZTT_CAPABILITY_MANIFEST_VERSION 1

enum {
    X_ZERO,
    X_RA,
    X_SP,
    X_GP,
    X_TP,
    X_T0,
    X_T1,
    X_T2,
    X_S0,
    X_S1,
    X_A0,
    X_A1,
    X_A2,
    X_A3,
    X_A4,
    X_A5,
    X_A6,
    X_A7,
    X_S2,
    X_S3,
    X_S4,
    X_S5,
    X_S6,
    X_S7,
    X_S8,
    X_S9,
    X_S10,
    X_S11,
    X_T3,
    X_T4,
};

static uint32_t ztt_lui(unsigned rd, uint32_t upper)
{
    return (upper << 12) | (rd << 7) | 0x37;
}
static uint32_t ztt_i(unsigned rd, unsigned rs1, unsigned funct3,
                      int32_t imm)
{
    g_assert_cmpint(imm, >=, -2048);
    g_assert_cmpint(imm, <=, 2047);
    return ((uint32_t)imm & 0xfff) << 20 | rs1 << 15 | funct3 << 12 |
           rd << 7 | 0x13;
}

static uint32_t ztt_load(unsigned rd, unsigned rs1, unsigned funct3,
                         int32_t imm)
{
    return (uint32_t)imm << 20 | rs1 << 15 | funct3 << 12 | rd << 7 |
           0x03;
}

static uint32_t ztt_store(unsigned rs2, unsigned rs1, unsigned funct3,
                          int32_t imm)
{
    uint32_t uimm = imm;

    return (uimm >> 5) << 25 | rs2 << 20 | rs1 << 15 | funct3 << 12 |
           (uimm & 0x1f) << 7 | 0x23;
}

static uint32_t ztt_branch(unsigned rs1, unsigned rs2, unsigned funct3,
                           int32_t imm)
{
    uint32_t uimm = imm;

    g_assert_cmpint(imm & 1, ==, 0);
    g_assert_cmpint(imm, >=, -4096);
    g_assert_cmpint(imm, <=, 4094);
    return ((uimm >> 12) & 1) << 31 | ((uimm >> 5) & 0x3f) << 25 |
           rs2 << 20 | rs1 << 15 | funct3 << 12 |
           ((uimm >> 1) & 0xf) << 8 | ((uimm >> 11) & 1) << 7 | 0x63;
}

static uint32_t ztt_csr(unsigned rd, unsigned rs1, unsigned funct3,
                        unsigned csr)
{
    return csr << 20 | rs1 << 15 | funct3 << 12 | rd << 7 | 0x73;
}

static uint32_t ztt_r3(unsigned funct7, unsigned rd, unsigned rs1,
                       unsigned rs2)
{
    return funct7 << 25 | rs2 << 20 | rs1 << 15 | rd << 7 | 0x2b;
}

static void ztt_emit(uint32_t *code, size_t *count, uint32_t insn)
{
    g_assert_cmpuint(*count, <, 96);
    code[(*count)++] = cpu_to_le32(insn);
}

static void ztt_emit_mailbox_address(uint32_t *code, size_t *count)
{
    ztt_emit(code, count, ztt_i(X_T4, X_ZERO, 0, 1));
    ztt_emit(code, count, ztt_i(X_T4, X_T4, 1, 31));
    ztt_emit(code, count, ztt_i(X_T4, X_T4, 0, 0x700));
}

static size_t ztt_shared_owner_program(uint32_t code[96], uint32_t scalar,
                                       uint32_t value, unsigned ready_offset,
                                       unsigned result_offset, bool release)
{
    size_t count = 0;

#define EMIT(_insn) ztt_emit(code, &count, (_insn))
    ztt_emit_mailbox_address(code, &count);
    EMIT(ztt_lui(X_T0, 0x2000));
    EMIT(ztt_csr(X_ZERO, X_T0, 2, 0x300));
    EMIT(ztt_i(X_A1, X_ZERO, 0, 0));
    EMIT(UINT32_C(0xa490002b) | X_A0 << 7 | X_A1 << 15);
    EMIT(ztt_store(X_A0, X_T4, 2, ready_offset + 72));

    EMIT(ztt_lui(X_T0, scalar >> 12));
    EMIT(ztt_i(X_T0, X_T0, 0, scalar & 0xfff));
    EMIT(ztt_csr(X_ZERO, X_T0, 1, 0x810));
    EMIT(ztt_i(X_T0, X_ZERO, 0, value));
    EMIT(ztt_i(X_T1, X_ZERO, 0, 0));
    EMIT(ztt_r3(0x5b, 0, X_T0, X_T1));
    EMIT(ztt_i(X_T0, X_ZERO, 0, 0x1b));
    EMIT(ztt_csr(X_ZERO, X_T0, 1, 0x811));
    EMIT(ztt_i(X_T0, X_ZERO, 0, 1));
    EMIT(ztt_csr(X_ZERO, X_T0, 1, 0x812));
    EMIT(ztt_i(X_T0, X_ZERO, 0, 1));
    EMIT(ztt_store(X_T0, X_T4, 2, ready_offset));

    EMIT(ztt_load(X_T0, X_T4, 2, ZTT_SHARED_CONTROL));
    EMIT(ztt_branch(X_T0, X_ZERO, 0, -4));
    EMIT(ztt_csr(X_A0, X_ZERO, 2, 0x810));
    EMIT(ztt_store(X_A0, X_T4, 2, result_offset));
    EMIT(ztt_i(X_T1, X_ZERO, 0, 0));
    EMIT(ztt_r3(0x5f, X_A0, 0, X_T1));
    EMIT(ztt_store(X_A0, X_T4, 2, result_offset + 4));
    EMIT(ztt_csr(X_A0, X_ZERO, 2, 0xcc3));
    EMIT(ztt_store(X_A0, X_T4, 2, result_offset + 8));
    if (release) {
        EMIT(UINT32_C(0xa4a0002b));
    }
    EMIT(ztt_i(X_T0, X_ZERO, 0, 1));
    EMIT(ztt_store(X_T0, X_T4, 2, result_offset + 12));
    EMIT(UINT32_C(0x0000006f));
#undef EMIT
    return count;
}

static size_t ztt_shared_waiter_program(uint32_t code[96], uint64_t *wait_pc)
{
    size_t count = 0;

#define EMIT(_insn) ztt_emit(code, &count, (_insn))
    ztt_emit_mailbox_address(code, &count);
    EMIT(ztt_lui(X_T0, 0x2000));
    EMIT(ztt_csr(X_ZERO, X_T0, 2, 0x300));
    EMIT(ztt_load(X_T0, X_T4, 2, ZTT_SHARED_START_WAITER));
    EMIT(ztt_branch(X_T0, X_ZERO, 0, -4));
    EMIT(ztt_i(X_T0, X_ZERO, 0, 1));
    EMIT(ztt_store(X_T0, X_T4, 2, ZTT_SHARED_HART2_READY));
    EMIT(ztt_i(X_A1, X_ZERO, 0, 13));
    *wait_pc = ZTT_SHARED_HART2_ADDR + count * sizeof(code[0]);
    EMIT(UINT32_C(0xa490002b) | X_A0 << 7 | X_A1 << 15);
    EMIT(ztt_store(X_A0, X_T4, 2, ZTT_SHARED_HART2_RESULT));
    EMIT(ztt_csr(X_A0, X_ZERO, 2, 0x810));
    EMIT(ztt_store(X_A0, X_T4, 2, ZTT_SHARED_HART2_SCALAR));
    EMIT(ztt_i(X_T1, X_ZERO, 0, 0));
    EMIT(ztt_r3(0x5f, X_A0, 0, X_T1));
    EMIT(ztt_store(X_A0, X_T4, 2, ZTT_SHARED_HART2_VALUE));
    EMIT(ztt_csr(X_A0, X_ZERO, 2, 0xcc3));
    EMIT(ztt_store(X_A0, X_T4, 2, ZTT_SHARED_HART2_OWNED));
    EMIT(UINT32_C(0xa4a0002b));
    EMIT(ztt_i(X_T0, X_ZERO, 0, 1));
    EMIT(ztt_store(X_T0, X_T4, 2, ZTT_SHARED_HART2_DONE));
    EMIT(UINT32_C(0x0000006f));
#undef EMIT
    return count;
}

static size_t ztt_park_program(uint32_t code[96])
{
    code[0] = cpu_to_le32(UINT32_C(0x0000006f));
    return 1;
}

static size_t ztt_reset_checker_program(uint32_t code[96])
{
    size_t count = 0;

#define EMIT(_insn) ztt_emit(code, &count, (_insn))
    ztt_emit_mailbox_address(code, &count);
    EMIT(ztt_lui(X_T0, 0x2000));
    EMIT(ztt_csr(X_ZERO, X_T0, 2, 0x300));
    EMIT(ztt_i(X_A1, X_ZERO, 0, 0));
    EMIT(UINT32_C(0xa490002b) | X_A0 << 7 | X_A1 << 15);
    EMIT(ztt_store(X_A0, X_T4, 2, ZTT_RESET_RESULT));
    /* Flag CSRs require a grant; raw CPU reset flags are checked by GDB. */
    EMIT(ztt_csr(X_A0, X_ZERO, 2, 0x811));
    EMIT(ztt_store(X_A0, X_T4, 2, ZTT_RESET_FFLAGS));
    EMIT(ztt_csr(X_A0, X_ZERO, 2, 0x812));
    EMIT(ztt_store(X_A0, X_T4, 2, ZTT_RESET_XSAT));
    EMIT(ztt_csr(X_A0, X_ZERO, 2, 0x810));
    EMIT(ztt_store(X_A0, X_T4, 2, ZTT_RESET_SCALAR));
    EMIT(ztt_i(X_T1, X_ZERO, 0, 0));
    EMIT(ztt_r3(0x5f, X_A0, 0, X_T1));
    EMIT(ztt_store(X_A0, X_T4, 2, ZTT_RESET_VALUE));
    EMIT(ztt_csr(X_A0, X_ZERO, 2, 0xcc3));
    EMIT(ztt_store(X_A0, X_T4, 2, ZTT_RESET_OWNED));
    EMIT(ztt_i(X_T0, X_ZERO, 0, 1));
    EMIT(ztt_store(X_T0, X_T4, 2, ZTT_RESET_DONE));
    EMIT(UINT32_C(0x0000006f));
#undef EMIT
    return count;
}

static size_t ztt_migration_program(uint32_t code[96])
{
    size_t count = 0;

#define EMIT(_insn) ztt_emit(code, &count, (_insn))
    /* t4 = zero-extended 0x80000700 on both RV32 and RV64. */
    EMIT(ztt_i(X_T4, X_ZERO, 0, 1));
    EMIT(ztt_i(X_T4, X_T4, 1, 31));
    EMIT(ztt_i(X_T4, X_T4, 0, 0x700));

    EMIT(ztt_lui(X_T0, 0x2000));       /* mstatus.AME_MS = Initial */
    EMIT(ztt_csr(X_ZERO, X_T0, 2, 0x300));
    EMIT(ztt_i(X_A1, X_ZERO, 0, 0));
    EMIT(UINT32_C(0xa490002b) | X_A0 << 7 | X_A1 << 15);

    EMIT(ztt_lui(X_T0, 0x20300));      /* FP32 RNE */
    EMIT(ztt_i(X_T0, X_T0, 0, 0x120));
    EMIT(UINT32_C(0xa230002b) | X_T0 << 15); /* msettyp m0,t0 */
    EMIT(UINT32_C(0xa230012b) | X_T0 << 15); /* msettyp m2,t0 */
    EMIT(UINT32_C(0xa210002b) | X_T0 << 15); /* asettyp acc0,t0 */

    EMIT(ztt_lui(X_T0, 0x12345));
    EMIT(ztt_i(X_T0, X_T0, 0, 0x678));
    EMIT(ztt_i(X_T1, X_ZERO, 0, 0));
    EMIT(ztt_r3(0x5b, 0, X_T0, X_T1));
    EMIT(ztt_lui(X_T0, 0xdeadc));
    EMIT(ztt_i(X_T0, X_T0, 0, -0x111));
    EMIT(ztt_i(X_T1, X_ZERO, 0, 15));
    EMIT(ztt_r3(0x5b, 0, X_T0, X_T1));
    EMIT(UINT32_C(0xa3d0002b));        /* mmov.2d.a.m acc0,m0 */

    EMIT(ztt_i(X_T0, X_ZERO, 0, -1)); /* Unsupported raw scalar dtype. */
    EMIT(ztt_csr(X_ZERO, X_T0, 1, 0x810));
    EMIT(ztt_i(X_T0, X_ZERO, 0, 0x1b));
    EMIT(ztt_csr(X_ZERO, X_T0, 1, 0x811));
    EMIT(ztt_i(X_T0, X_ZERO, 0, 1));
    EMIT(ztt_csr(X_ZERO, X_T0, 1, 0x812));
    EMIT(UINT32_C(0xa2e000ab));        /* unsupported mcos.ew m1,m0 */

    EMIT(ztt_i(X_T0, X_ZERO, 0, 1));
    EMIT(ztt_store(X_T0, X_T4, 2, 0));
    EMIT(ztt_load(X_T0, X_T4, 2, 4));
    EMIT(ztt_branch(X_T0, X_ZERO, 0, -4));

    EMIT(UINT32_C(0xa220052b));        /* mgettyp a0,m0 */
    EMIT(ztt_store(X_A0, X_T4, 2, 8));
    EMIT(UINT32_C(0xa200052b));        /* agettyp a0,acc0 */
    EMIT(ztt_store(X_A0, X_T4, 2, 12));
    EMIT(ztt_i(X_T1, X_ZERO, 0, 0));
    EMIT(ztt_r3(0x5f, X_A0, 0, X_T1));
    EMIT(ztt_store(X_A0, X_T4, 2, 16));
    EMIT(ztt_i(X_T1, X_ZERO, 0, 15));
    EMIT(ztt_r3(0x5f, X_A0, 0, X_T1));
    EMIT(ztt_store(X_A0, X_T4, 2, 20));
    EMIT(UINT32_C(0xa3c0012b));        /* mmov.2d.m.a m2,acc0 */
    EMIT(ztt_i(X_T1, X_ZERO, 0, 0));
    EMIT(ztt_r3(0x5f, X_A0, 2, X_T1));
    EMIT(ztt_store(X_A0, X_T4, 2, 24));
    EMIT(ztt_i(X_T1, X_ZERO, 0, 15));
    EMIT(ztt_r3(0x5f, X_A0, 2, X_T1));
    EMIT(ztt_store(X_A0, X_T4, 2, 28));

    EMIT(ztt_csr(X_A0, X_ZERO, 2, 0x810));
    EMIT(ztt_store(X_A0, X_T4, 2, 32));
    EMIT(ztt_csr(X_A0, X_ZERO, 2, 0x800));
    EMIT(ztt_store(X_A0, X_T4, 2, 36));
    EMIT(ztt_csr(X_A0, X_ZERO, 2, 0x811));
    EMIT(ztt_store(X_A0, X_T4, 2, 40));
    EMIT(ztt_csr(X_A0, X_ZERO, 2, 0x812));
    EMIT(ztt_store(X_A0, X_T4, 2, 44));
    EMIT(ztt_csr(X_A0, X_ZERO, 2, 0xcc3));
    EMIT(ztt_store(X_A0, X_T4, 2, 48));
    EMIT(ztt_csr(X_A0, X_ZERO, 2, 0x300));
    EMIT(ztt_store(X_A0, X_T4, 2, 52));
    EMIT(ztt_csr(X_ZERO, X_ZERO, 1, 0x800));
    EMIT(ztt_i(X_T0, X_ZERO, 0, 1));
    EMIT(ztt_i(X_T1, X_ZERO, 0, 0));
    EMIT(ztt_r3(0x03, 0, X_T0, 2)); /* madd.ew.x m0,t0,m2 */
    EMIT(ztt_r3(0x5f, X_A0, 0, X_T1));
    EMIT(ztt_store(X_A0, X_T4, 2, 56));
    EMIT(ztt_csr(X_A0, X_ZERO, 2, 0x800));
    EMIT(ztt_store(X_A0, X_T4, 2, 60));
    EMIT(ztt_i(X_T0, X_ZERO, 0, 2));
    EMIT(ztt_store(X_T0, X_T4, 2, 0));
    EMIT(UINT32_C(0x0000006f));        /* jal zero,0 */
#undef EMIT
    return count;
}
static const char *ztt_cpu_model(void)
{
    return g_str_equal(qtest_get_arch(), "riscv32") ? "rv32" : "rv64";
}

static QTestState *ztt_start(const char *options)
{
    return qtest_initf("-machine virt -bios none -nodefaults -S -accel tcg "
                       "-cpu %s,%s", ztt_cpu_model(), options);
}
static void wait_for_migration(QTestState *qts)
{
    int64_t deadline = g_get_monotonic_time() + 20 * G_TIME_SPAN_SECOND;

    while (true) {
        QDict *response = qtest_qmp(qts,
                                    "{ 'execute': 'query-migrate' }");
        QDict *migration;
        const char *status;

        g_assert_false(qdict_haskey(response, "error"));
        migration = qdict_get_qdict(response, "return");
        status = qdict_get_str(migration, "status");
        if (g_str_equal(status, "completed")) {
            qobject_unref(response);
            return;
        }
        g_assert_true(g_str_equal(status, "setup") ||
                      g_str_equal(status, "active") ||
                      g_str_equal(status, "device"));
        qobject_unref(response);
        g_assert_cmpint(g_get_monotonic_time(), <, deadline);
        g_usleep(1000);
    }
}

static char *wait_for_incoming_migration_failure(QTestState *qts)
{
    int64_t deadline = g_get_monotonic_time() + 5 * G_TIME_SPAN_SECOND;

    while (true) {
        QDict *response = qtest_qmp(qts,
                                    "{ 'execute': 'query-migrate' }");
        QDict *migration;
        const char *status;
        char *error_desc;

        g_assert_false(qdict_haskey(response, "error"));
        migration = qdict_get_qdict(response, "return");
        status = qdict_get_str(migration, "status");
        if (g_str_equal(status, "failed")) {
            error_desc = g_strdup(qdict_get_str(migration, "error-desc"));
            qobject_unref(response);
            return error_desc;
        }
        g_assert_cmpstr(status, !=, "completed");
        qobject_unref(response);
        if (g_get_monotonic_time() >= deadline) {
            g_error("Timed out waiting for incoming migration failure");
        }
        g_usleep(1000);
    }
}

static char *save_qtest_state(QTestState *qts)
{
    g_autofree char *template = g_strdup_printf(
        "%s/riscv-ztt-migration-XXXXXX", g_get_tmp_dir());
    g_autofree char *uri = NULL;
    QDict *response;
    int fd;

    fd = g_mkstemp(template);
    g_assert_cmpint(fd, >=, 0);
    close(fd);
    g_assert_cmpint(unlink(template), ==, 0);
    uri = g_strdup_printf("file:%s", template);
    response = qtest_qmp(qts,
                         "{ 'execute': 'migrate', 'arguments': { "
                         "  'uri': %s } }", uri);
    g_assert_false(qdict_haskey(response, "error"));
    qobject_unref(response);
    wait_for_migration(qts);
    return g_steal_pointer(&template);
}

static char *save_ztt_state(const char *options)
{
    QTestState *qts = ztt_start(options);
    char *filename = save_qtest_state(qts);

    qtest_quit(qts);
    return filename;
}

static void ztt_wait_mailbox_offset(QTestState *qts, unsigned offset,
                                    uint32_t expected)
{
    int64_t deadline = g_get_monotonic_time() + 5 * G_TIME_SPAN_SECOND;
    uint32_t value;

    while ((value = qtest_readl(qts, ZTT_MIGRATION_MAILBOX + offset)) !=
           expected) {
        if (g_get_monotonic_time() >= deadline) {
            g_error("Timed out waiting for Ztt mailbox offset 0x%x: "
                    "expected 0x%x, got 0x%x", offset, expected, value);
        }
        g_usleep(1000);
    }
}

static void ztt_wait_mailbox(QTestState *qts, uint32_t expected)
{
    ztt_wait_mailbox_offset(qts, 0, expected);
}

static void ztt_stop(QTestState *qts)
{
    QDict *response = qtest_qmp(qts, "{ 'execute': 'stop' }");

    g_assert_false(qdict_haskey(response, "error"));
    qobject_unref(response);
    qtest_qmp_eventwait(qts, "STOP");
}

static QTestState *ztt_start_migration_program_machine(const char *machine,
                                                       const char *options)
{
    uint32_t code[96];
    size_t count = ztt_migration_program(code);
    QTestState *qts = qtest_initf(
        "-machine %s -bios none -nodefaults -S -accel tcg "
        "-cpu %s,%s -device loader,addr=0x80000000,cpu-num=0",
        machine, ztt_cpu_model(), options);
    QDict *response;

    qtest_memwrite(qts, ZTT_MIGRATION_CODE_ADDR, code,
                   count * sizeof(code[0]));
    response = qtest_qmp(qts, "{ 'execute': 'cont' }");
    g_assert_false(qdict_haskey(response, "error"));
    qobject_unref(response);
    ztt_wait_mailbox(qts, 1);
    ztt_stop(qts);
    return qts;
}

static QTestState *ztt_start_migration_program(const char *options)
{
    return ztt_start_migration_program_machine("virt", options);
}

static QTestState *ztt_start_shared_migration_program(bool with_waiter)
{
    uint32_t hart0[96], hart1[96], hart2[96];
    size_t count0, count1, count2;
    uint64_t waiter_pc;
    QTestState *qts;
    QDict *response;

    count0 = ztt_shared_owner_program(
        hart0, UINT32_C(0x40000008), 0x11, ZTT_SHARED_HART0_READY,
        ZTT_SHARED_HART0_SCALAR, true);
    count1 = with_waiter ? ztt_shared_owner_program(
        hart1, UINT32_C(0x40000010), 0x22, ZTT_SHARED_HART1_READY,
        ZTT_SHARED_HART1_SCALAR, false) : ztt_park_program(hart1);
    count2 = ztt_shared_waiter_program(hart2, &waiter_pc);

    if (with_waiter) {
        qts = qtest_initf(
            "-machine virt,x-ame-backends=2 -bios none -nodefaults -S "
            "-accel tcg -smp 3 -cpu %s,x-ztt=true,amenlen=4,ameudsz=8,"
            "ame-num-mregs=16,ame-num-accregs=1 "
            "-device loader,addr=0x80000000,cpu-num=0 "
            "-device loader,addr=0x80000200,cpu-num=1 "
            "-device loader,addr=0x80000400,cpu-num=2",
            ztt_cpu_model());
    } else {
        qts = qtest_initf(
            "-machine virt,x-ame-backends=2 -bios none -nodefaults -S "
            "-accel tcg -smp 2 -cpu %s,x-ztt=true,amenlen=4,ameudsz=8,"
            "ame-num-mregs=16,ame-num-accregs=1 "
            "-device loader,addr=0x80000000,cpu-num=0 "
            "-device loader,addr=0x80000200,cpu-num=1",
            ztt_cpu_model());
    }
    qtest_memwrite(qts, ZTT_SHARED_HART0_ADDR, hart0,
                   count0 * sizeof(hart0[0]));
    qtest_memwrite(qts, ZTT_SHARED_HART1_ADDR, hart1,
                   count1 * sizeof(hart1[0]));
    if (with_waiter) {
        qtest_memwrite(qts, ZTT_SHARED_HART2_ADDR, hart2,
                       count2 * sizeof(hart2[0]));
    }
    response = qtest_qmp(qts, "{ 'execute': 'cont' }");
    g_assert_false(qdict_haskey(response, "error"));
    qobject_unref(response);
    ztt_wait_mailbox_offset(qts, ZTT_SHARED_HART0_READY, 1);
    if (with_waiter) {
        ztt_wait_mailbox_offset(qts, ZTT_SHARED_HART1_READY, 1);
        qtest_writel(qts, ZTT_MIGRATION_MAILBOX +
                     ZTT_SHARED_START_WAITER, 1);
        ztt_wait_mailbox_offset(qts, ZTT_SHARED_HART2_READY, 1);
    }
    ztt_stop(qts);
    if (with_waiter) {
        g_autoptr(QDict) hmp = qtest_qmp(
            qts,
            "{'execute': 'human-monitor-command', "
            " 'arguments': {'command-line': 'info registers', "
            "               'cpu-index': 2}}");
        const char *registers = qdict_get_str(hmp, "return");
        const char *pc_line = strstr(registers, " pc");
        uint64_t pc;

        g_assert_nonnull(pc_line);
        g_assert_cmpint(sscanf(pc_line, " pc %" SCNx64, &pc), ==, 1);
        if (g_str_equal(qtest_get_arch(), "riscv32")) {
            pc = (uint32_t)pc;
        }
        g_assert_cmphex(pc, ==, waiter_pc);
    }
    return qts;
}

static QTestState *ztt_restore_shared_migration(const char *filename,
                                                bool with_waiter)
{
    if (with_waiter) {
        return qtest_initf(
            "-machine virt,x-ame-backends=2 -bios none -nodefaults -S "
            "-accel tcg -smp 3 -cpu %s,x-ztt=true,amenlen=4,ameudsz=8,"
            "ame-num-mregs=16,ame-num-accregs=1 "
            "-device loader,addr=0x80000000,cpu-num=0 "
            "-device loader,addr=0x80000200,cpu-num=1 "
            "-device loader,addr=0x80000400,cpu-num=2 "
            "-incoming file:%s", ztt_cpu_model(), filename);
    }
    return qtest_initf(
        "-machine virt,x-ame-backends=2 -bios none -nodefaults -S "
        "-accel tcg -smp 2 -cpu %s,x-ztt=true,amenlen=4,ameudsz=8,"
        "ame-num-mregs=16,ame-num-accregs=1 "
        "-device loader,addr=0x80000000,cpu-num=0 "
        "-device loader,addr=0x80000200,cpu-num=1 "
        "-incoming file:%s", ztt_cpu_model(), filename);
}

static void test_ztt_migration_round_trip(void)
{
    static const char *options =
        "x-ztt=true,amenlen=4,ameudsz=8,ame-num-mregs=16,"
        "ame-num-accregs=1";
    g_autofree char *filename = save_ztt_state(options);
    QTestState *qts = qtest_initf(
        "-machine virt -bios none -nodefaults -S -accel tcg "
        "-cpu %s,%s -incoming file:%s", ztt_cpu_model(), options, filename);

    wait_for_migration(qts);
    qtest_quit(qts);
    g_assert_cmpint(unlink(filename), ==, 0);
}

static void test_ztt_migration_disabled(void)
{
    static const char section_name[] = "riscv-ame-backend-pool";
    g_autofree char *filename = save_ztt_state("x-ztt=false");
    g_autofree char *contents = NULL;
    g_autoptr(GError) error = NULL;
    gsize length;
    QTestState *qts = qtest_initf(
        "-machine virt -bios none -nodefaults -S -accel tcg "
        "-cpu %s,x-ztt=false -incoming file:%s", ztt_cpu_model(),
        filename);

    wait_for_migration(qts);
    g_assert_true(g_file_get_contents(filename, &contents, &length, &error));
    g_assert_null(memmem(contents, length, section_name,
                        sizeof(section_name) - 1));
    qtest_quit(qts);
    g_assert_cmpint(unlink(filename), ==, 0);
}

static void test_ztt_migration_extension_enabled_to_disabled(void)
{
    const char *binary = g_getenv("QTEST_QEMU_BINARY");
    g_autofree char *filename = save_ztt_state("x-ztt=true");
    g_autofree char *cpu = g_strdup_printf("%s,x-ztt=false",
                                           ztt_cpu_model());
    g_autofree char *incoming = g_strdup_printf("file:%s", filename);
    g_autofree char *error_output = NULL;
    g_autoptr(GError) error = NULL;
    const char *argv[] = {
        binary, "-machine", "virt", "-bios", "none", "-nodefaults",
        "-display", "none", "-accel", "tcg", "-cpu", cpu,
        "-incoming", incoming, NULL,
    };
    int status;

    g_assert_true(g_spawn_sync(NULL, (char **)argv, NULL,
                               G_SPAWN_STDOUT_TO_DEV_NULL,
                               NULL, NULL, NULL, &error_output, &status,
                               &error));
    g_assert_true(WIFEXITED(status));
    g_assert_cmpint(WEXITSTATUS(status), !=, 0);
    g_assert_nonnull(strstr(error_output,
                            "RISC-V Ztt hart migration state received by "
                            "disabled CPU"));
    g_assert_cmpint(unlink(filename), ==, 0);
}

static void test_ztt_migration_extension_disabled_to_enabled(void)
{
    g_autofree char *filename = save_ztt_state("x-ztt=false");
    g_autofree char *uri = g_strdup_printf("file:%s", filename);
    g_autofree char *error_desc = NULL;
    QTestState *qts = qtest_initf(
        "-machine virt -bios none -nodefaults -S -accel tcg "
        "-cpu %s,x-ztt=true -incoming defer", ztt_cpu_model());
    QDict *response;

    response = qtest_qmp(qts,
                         "{ 'execute': 'migrate-incoming', "
                         "  'arguments': { 'uri': %s, "
                         "                 'exit-on-error': false } }", uri);
    g_assert_false(qdict_haskey(response, "error"));
    qobject_unref(response);
    error_desc = wait_for_incoming_migration_failure(qts);
    g_assert_cmpstr(error_desc, !=, "");
    qtest_quit(qts);
    g_assert_cmpint(unlink(filename), ==, 0);
}

static void test_ztt_migration_backend_disabled(void)
{
    QTestState *source = qtest_initf(
        "-machine virt,x-ame-backend-enabled=off -bios none "
        "-nodefaults -S -accel tcg -cpu %s,x-ztt=true",
        ztt_cpu_model());
    g_autofree char *filename = save_qtest_state(source);
    QTestState *dest;

    qtest_quit(source);
    dest = qtest_initf(
        "-machine virt,x-ame-backend-enabled=off -bios none "
        "-nodefaults -S -accel tcg -cpu %s,x-ztt=true "
        "-incoming file:%s", ztt_cpu_model(), filename);
    wait_for_migration(dest);
    qtest_quit(dest);
    g_assert_cmpint(unlink(filename), ==, 0);
}

static void test_ztt_migration_backend_enabled_to_disabled(void)
{
    const char *binary = g_getenv("QTEST_QEMU_BINARY");
    g_autofree char *filename = save_ztt_state("x-ztt=true");
    g_autofree char *cpu = g_strdup_printf("%s,x-ztt=true",
                                           ztt_cpu_model());
    g_autofree char *incoming = g_strdup_printf("file:%s", filename);
    g_autofree char *error_output = NULL;
    g_autoptr(GError) error = NULL;
    const char *argv[] = {
        binary, "-machine", "virt,x-ame-backend-enabled=off",
        "-bios", "none", "-nodefaults", "-display", "none",
        "-accel", "tcg", "-cpu", cpu, "-incoming", incoming, NULL,
    };
    int status;

    g_assert_true(g_spawn_sync(NULL, (char **)argv, NULL,
                               G_SPAWN_STDOUT_TO_DEV_NULL,
                               NULL, NULL, NULL, &error_output, &status,
                               &error));
    g_assert_true(WIFEXITED(status));
    g_assert_cmpint(WEXITSTATUS(status), !=, 0);
    g_assert_nonnull(strstr(error_output,
                            "RISC-V Ztt backend pool configuration "
                            "mismatch"));
    g_assert_cmpint(unlink(filename), ==, 0);
}

static void test_ztt_migration_max_backing(void)
{
    g_autofree char *filename = save_ztt_state("x-ztt=true");
    QTestState *qts = qtest_initf(
        "-machine virt -bios none -nodefaults -S -accel tcg "
        "-cpu %s,x-ztt=true -incoming file:%s", ztt_cpu_model(),
        filename);

    wait_for_migration(qts);
    qtest_quit(qts);
    g_assert_cmpint(unlink(filename), ==, 0);
}

static void test_ztt_migration_dirty_state(void)
{
    static const char *options =
        "x-ztt=true,amenlen=4,ameudsz=32,ame-num-mregs=16,"
        "ame-num-accregs=1";
    QTestState *source = ztt_start_migration_program(options);
    g_autofree char *filename = save_qtest_state(source);
    QTestState *dest;
    QDict *response;

    qtest_quit(source);
    dest = qtest_initf(
        "-machine virt -bios none -nodefaults -S -accel tcg "
        "-cpu %s,%s -incoming file:%s", ztt_cpu_model(), options,
        filename);
    /* Monitor readiness does not imply that incoming RAM has been loaded. */
    wait_for_migration(dest);
    qtest_writel(dest, ZTT_MIGRATION_MAILBOX + 4, 1);
    response = qtest_qmp(dest, "{ 'execute': 'cont' }");
    g_assert_false(qdict_haskey(response, "error"));
    qobject_unref(response);
    ztt_wait_mailbox(dest, 2);
    ztt_stop(dest);

    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX + 8), ==,
                    UINT32_C(0x20300120));
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX + 12), ==,
                    UINT32_C(0x20300120));
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX + 16), ==,
                    UINT32_C(0x12345678));
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX + 20), ==,
                    UINT32_C(0xdeadbeef));
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX + 24), ==,
                    UINT32_C(0x12345678));
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX + 28), ==,
                    UINT32_C(0xdeadbeef));
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX + 32), ==,
                    UINT32_MAX);
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX + 36), ==, 1);
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX + 40), ==,
                    0x1b);
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX + 44), ==, 1);
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX + 48), ==, 1);
    g_assert_cmphex((qtest_readl(dest, ZTT_MIGRATION_MAILBOX + 52) >> 25) &
                    3, ==, 3);
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX + 56), ==,
                    UINT32_C(0x12345678));
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX + 60), ==, 1);
    qtest_quit(dest);
    g_assert_cmpint(unlink(filename), ==, 0);
}

static void test_ztt_migration_ram_dirty(void)
{
    const uint64_t data_addr = UINT64_C(0x80400000);
    const size_t data_size = 128 * 4096;
    static const char *options =
        "x-ztt=true,amenlen=4,ameudsz=8,ame-num-mregs=16,"
        "ame-num-accregs=1";
    g_autofree uint8_t *expected = g_malloc(data_size);
    g_autofree uint8_t *actual = g_malloc(data_size);
    g_autofree char *filename = g_strdup_printf(
        "%s/riscv-ztt-ram-XXXXXX", g_get_tmp_dir());
    g_autofree char *uri = NULL;
    uint32_t code[96];
    size_t count = 0, outer, inner;
    QTestState *source, *dest;
    int64_t deadline;
    int fd;

    /* Only AME writes the data pages; scalar stores touch only the mailbox. */
#define EMIT(_insn) ztt_emit(code, &count, (_insn))
    ztt_emit_mailbox_address(code, &count);
    EMIT(ztt_lui(X_T0, 0x2000));
    EMIT(ztt_csr(X_ZERO, X_T0, 2, 0x300));
    EMIT(UINT32_C(0xa490002b) | X_A0 << 7); /* acquire, TRY */
    EMIT(ztt_i(X_A2, X_ZERO, 0, 1));
    EMIT(ztt_i(X_A2, X_A2, 1, 31));
    EMIT(ztt_lui(X_T0, 0x400));
    EMIT(UINT32_C(0x00000033) | X_A2 << 7 | X_A2 << 15 | X_T0 << 20);
    EMIT(ztt_lui(X_T3, 1)); /* page stride */
    EMIT(ztt_i(X_S0, X_ZERO, 0, 0));
    outer = count;
    EMIT(ztt_i(X_S0, X_S0, 0, 1));
    EMIT(ztt_r3(0x5b, 0, X_S0, X_ZERO)); /* mmove32.m.x */
    EMIT(ztt_i(X_A3, X_A2, 0, 0));
    EMIT(ztt_i(X_A4, X_ZERO, 0, 128));
    inner = count;
    EMIT(UINT32_C(0xa390002b) | X_A3 << 15); /* mss.1r */
    EMIT(UINT32_C(0x00000033) | X_A3 << 7 | X_A3 << 15 | X_T3 << 20);
    EMIT(ztt_i(X_A4, X_A4, 0, -1));
    EMIT(ztt_branch(X_A4, X_ZERO, 1,
                    ((int32_t)inner - (int32_t)count) * 4));
    EMIT(ztt_store(X_S0, X_T4, 2, 0));
    EMIT(ztt_branch(X_ZERO, X_ZERO, 0,
                    ((int32_t)outer - (int32_t)count) * 4));
#undef EMIT

    source = qtest_initf(
        "-machine virt -bios none -nodefaults -S -accel tcg "
        "-cpu %s,%s -device loader,addr=0x80000000,cpu-num=0",
        ztt_cpu_model(), options);
    qtest_memwrite(source, ZTT_MIGRATION_CODE_ADDR, code, count * 4);
    qtest_qmp_assert_success(source, "{ 'execute': 'cont' }");
    deadline = g_get_monotonic_time() + 10 * G_TIME_SPAN_SECOND;
    while (!qtest_readl(source, ZTT_MIGRATION_MAILBOX)) {
        g_assert_cmpint(g_get_monotonic_time(), <, deadline);
        g_usleep(1000);
    }
    /* Keep the writer running through multiple real dirty-bitmap syncs. */
    qtest_qmp_assert_success(source,
        "{ 'execute': 'migrate-set-parameters', 'arguments': {"
        "'max-bandwidth': 3000000, 'downtime-limit': 1 } }");
    fd = g_mkstemp(filename);
    g_assert_cmpint(fd, >=, 0);
    close(fd);
    g_assert_cmpint(unlink(filename), ==, 0);
    uri = g_strdup_printf("file:%s", filename);
    qtest_qmp_assert_success(source,
        "{ 'execute': 'migrate', 'arguments': { 'uri': %s } }", uri);
    deadline = g_get_monotonic_time() + 20 * G_TIME_SPAN_SECOND;
    while (true) {
        QDict *migration = qtest_qmp_assert_success_ref(
            source, "{ 'execute': 'query-migrate' }");
        QDict *ram = qdict_get_qdict(migration, "ram");
        const char *status = qdict_get_str(migration, "status");
        bool synced = ram && qdict_get_int(ram, "dirty-sync-count") >= 3;

        g_assert_true(g_str_equal(status, "setup") ||
                      g_str_equal(status, "active"));
        qobject_unref(migration);
        if (synced) {
            break;
        }
        g_assert_cmpint(g_get_monotonic_time(), <, deadline);
        g_usleep(1000);
    }
    ztt_stop(source);
    qtest_memread(source, data_addr, expected, data_size);
    g_assert_cmpuint(ldl_le_p(expected), >, 0);
    qtest_qmp_assert_success(source,
        "{ 'execute': 'migrate-set-parameters', 'arguments': {"
        "'max-bandwidth': 1000000000, 'downtime-limit': 30000 } }");
    wait_for_migration(source);
    qtest_quit(source);

    dest = qtest_initf(
        "-machine virt -bios none -nodefaults -S -accel tcg "
        "-cpu %s,%s -incoming file:%s", ztt_cpu_model(), options, filename);
    wait_for_migration(dest);
    qtest_memread(dest, data_addr, actual, data_size);
    for (size_t i = 0; i < data_size; i += 4096) {
        if (memcmp(actual + i, expected + i, 16)) {
            g_test_message("RAM page %zu: expected generation %u, got %u",
                           i / 4096, ldl_le_p(expected + i),
                           ldl_le_p(actual + i));
            break;
        }
    }
    g_assert_cmpmem(actual, data_size, expected, data_size);
    qtest_quit(dest);
    g_assert_cmpint(unlink(filename), ==, 0);
}

static void test_ztt_migration_private_system_pool(void)
{
    static const char *options =
        "x-ztt=true,amenlen=4,ameudsz=32,ame-num-mregs=16,"
        "ame-num-accregs=1";
    QTestState *source = ztt_start_migration_program_machine("spike",
                                                             options);
    g_autofree char *filename = save_qtest_state(source);
    QTestState *dest;
    QDict *response;

    qtest_quit(source);
    dest = qtest_initf(
        "-machine spike -bios none -nodefaults -S -accel tcg "
        "-cpu %s,%s -incoming file:%s", ztt_cpu_model(), options,
        filename);
    wait_for_migration(dest);
    qtest_writel(dest, ZTT_MIGRATION_MAILBOX + 4, 1);
    response = qtest_qmp(dest, "{ 'execute': 'cont' }");
    g_assert_false(qdict_haskey(response, "error"));
    qobject_unref(response);
    ztt_wait_mailbox(dest, 2);
    ztt_stop(dest);

    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX + 8), ==,
                    UINT32_C(0x20300120));
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX + 16), ==,
                    UINT32_C(0x12345678));
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX + 48), ==, 1);
    qtest_quit(dest);
    g_assert_cmpint(unlink(filename), ==, 0);
}

static void test_ztt_migration_partial_pool(void)
{
    QTestState *source = ztt_start_shared_migration_program(false);
    g_autofree char *filename = save_qtest_state(source);
    QTestState *dest;
    QDict *response;

    g_assert_cmphex(qtest_readl(source, ZTT_MIGRATION_MAILBOX +
                               ZTT_SHARED_HART0_ACQUIRE), ==, 1);
    qtest_quit(source);
    dest = ztt_restore_shared_migration(filename, false);
    wait_for_migration(dest);
    qtest_writel(dest, ZTT_MIGRATION_MAILBOX + ZTT_SHARED_CONTROL, 1);
    response = qtest_qmp(dest, "{ 'execute': 'cont' }");
    g_assert_false(qdict_haskey(response, "error"));
    qobject_unref(response);
    ztt_wait_mailbox_offset(dest, ZTT_SHARED_HART0_DONE, 1);

    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX +
                               ZTT_SHARED_HART0_SCALAR), ==,
                    UINT32_C(0x40000008));
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX +
                               ZTT_SHARED_HART0_VALUE), ==, 0x11);
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX +
                               ZTT_SHARED_HART0_OWNED), ==, 1);
    qtest_quit(dest);
    g_assert_cmpint(unlink(filename), ==, 0);
}

static void test_ztt_migration_full_pool_with_waiter(void)
{
    QTestState *source = ztt_start_shared_migration_program(true);
    g_autofree char *filename = save_qtest_state(source);
    QTestState *dest;
    QDict *response;

    g_assert_cmphex(qtest_readl(source, ZTT_MIGRATION_MAILBOX +
                               ZTT_SHARED_HART0_ACQUIRE), ==, 1);
    g_assert_cmphex(qtest_readl(source, ZTT_MIGRATION_MAILBOX +
                               ZTT_SHARED_HART1_ACQUIRE), ==, 1);
    qtest_quit(source);
    dest = ztt_restore_shared_migration(filename, true);
    wait_for_migration(dest);
    /* A paused destination must not consume the migrated virtual deadline. */
    g_usleep(1100 * 1000);
    qtest_writel(dest, ZTT_MIGRATION_MAILBOX + ZTT_SHARED_CONTROL, 1);
    response = qtest_qmp(dest, "{ 'execute': 'cont' }");
    g_assert_false(qdict_haskey(response, "error"));
    qobject_unref(response);
    ztt_wait_mailbox_offset(dest, ZTT_SHARED_HART0_DONE, 1);
    ztt_wait_mailbox_offset(dest, ZTT_SHARED_HART1_DONE, 1);
    ztt_wait_mailbox_offset(dest, ZTT_SHARED_HART2_DONE, 1);

    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX +
                               ZTT_SHARED_HART0_SCALAR), ==,
                    UINT32_C(0x40000008));
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX +
                               ZTT_SHARED_HART0_VALUE), ==, 0x11);
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX +
                               ZTT_SHARED_HART0_OWNED), ==, 1);
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX +
                               ZTT_SHARED_HART1_SCALAR), ==,
                    UINT32_C(0x40000010));
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX +
                               ZTT_SHARED_HART1_VALUE), ==, 0x22);
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX +
                               ZTT_SHARED_HART1_OWNED), ==, 1);
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX +
                               ZTT_SHARED_HART2_RESULT), ==, 1);
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX +
                               ZTT_SHARED_HART2_SCALAR), ==, 0);
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX +
                               ZTT_SHARED_HART2_VALUE), ==, 0);
    g_assert_cmphex(qtest_readl(dest, ZTT_MIGRATION_MAILBOX +
                               ZTT_SHARED_HART2_OWNED), ==, 1);
    qtest_quit(dest);
    g_assert_cmpint(unlink(filename), ==, 0);
}

static void test_ztt_migration_pool_count_mismatch(void)
{
    const char *binary = g_getenv("QTEST_QEMU_BINARY");
    QTestState *source = qtest_initf(
        "-machine virt,x-ame-backends=1 -bios none -nodefaults -S "
        "-accel tcg -smp 2 -cpu %s,x-ztt=true,amenlen=4,ameudsz=8,"
        "ame-num-mregs=16,ame-num-accregs=1", ztt_cpu_model());
    g_autofree char *filename = save_qtest_state(source);
    g_autofree char *cpu = g_strdup_printf(
        "%s,x-ztt=true,amenlen=4,ameudsz=8,ame-num-mregs=16,"
        "ame-num-accregs=1", ztt_cpu_model());
    g_autofree char *incoming = g_strdup_printf("file:%s", filename);
    g_autofree char *error_output = NULL;
    g_autoptr(GError) error = NULL;
    const char *argv[] = {
        binary, "-machine", "virt,x-ame-backends=2", "-bios", "none",
        "-nodefaults", "-display", "none", "-accel", "tcg", "-smp",
        "2", "-cpu", cpu, "-incoming", incoming, NULL,
    };
    int status;

    qtest_quit(source);
    g_assert_true(g_spawn_sync(NULL, (char **)argv, NULL,
                               G_SPAWN_STDOUT_TO_DEV_NULL,
                               NULL, NULL, NULL, &error_output, &status,
                               &error));
    g_assert_true(WIFEXITED(status));
    g_assert_cmpint(WEXITSTATUS(status), !=, 0);
    g_assert_nonnull(strstr(error_output,
                            "RISC-V Ztt backend pool configuration "
                            "mismatch"));
    g_assert_cmpint(unlink(filename), ==, 0);
}

static void test_ztt_shared_pool_system_reset(void)
{
    QTestState *qts = ztt_start_shared_migration_program(true);
    uint32_t checker[96], park[96];
    size_t checker_count = ztt_reset_checker_program(checker);
    size_t park_count = ztt_park_program(park);
    QDict *response;

    qtest_memwrite(qts, ZTT_SHARED_HART0_ADDR, checker,
                   checker_count * sizeof(checker[0]));
    qtest_memwrite(qts, ZTT_SHARED_HART1_ADDR, park,
                   park_count * sizeof(park[0]));
    qtest_memwrite(qts, ZTT_SHARED_HART2_ADDR, park,
                   park_count * sizeof(park[0]));
    qtest_writel(qts, ZTT_MIGRATION_MAILBOX + ZTT_RESET_DONE, 0);

    qtest_system_reset(qts);
    response = qtest_qmp(qts, "{ 'execute': 'cont' }");
    g_assert_false(qdict_haskey(response, "error"));
    qobject_unref(response);
    ztt_wait_mailbox_offset(qts, ZTT_RESET_DONE, 1);

    g_assert_cmphex(qtest_readl(qts, ZTT_MIGRATION_MAILBOX +
                               ZTT_RESET_RESULT), ==, 1);
    g_assert_cmphex(qtest_readl(qts, ZTT_MIGRATION_MAILBOX +
                               ZTT_RESET_SCALAR), ==, 0);
    g_assert_cmphex(qtest_readl(qts, ZTT_MIGRATION_MAILBOX +
                               ZTT_RESET_VALUE), ==, 0);
    g_assert_cmphex(qtest_readl(qts, ZTT_MIGRATION_MAILBOX +
                               ZTT_RESET_OWNED), ==, 1);
    g_assert_cmphex(qtest_readl(qts, ZTT_MIGRATION_MAILBOX +
                               ZTT_RESET_FFLAGS), ==, 0);
    g_assert_cmphex(qtest_readl(qts, ZTT_MIGRATION_MAILBOX +
                               ZTT_RESET_XSAT), ==, 0);
    qtest_quit(qts);
}

static void test_ztt_migration_config_mismatch(void)
{
    static const char *source_options =
        "x-ztt=true,amenlen=4,ameudsz=8,ame-num-mregs=16,"
        "ame-num-accregs=1";
    const char *binary = g_getenv("QTEST_QEMU_BINARY");
    g_autofree char *filename = save_ztt_state(source_options);
    g_autofree char *cpu = g_strdup_printf(
        "%s,x-ztt=true,amenlen=8,ameudsz=8,ame-num-mregs=16,"
        "ame-num-accregs=1", ztt_cpu_model());
    g_autofree char *incoming = g_strdup_printf("file:%s", filename);
    g_autofree char *error_output = NULL;
    g_autoptr(GError) error = NULL;
    const char *argv[] = {
        binary, "-machine", "virt", "-bios", "none", "-nodefaults",
        "-display", "none", "-accel", "tcg", "-cpu", cpu,
        "-incoming", incoming, NULL,
    };
    int status;

    g_assert_true(g_spawn_sync(NULL, (char **)argv, NULL,
                               G_SPAWN_STDOUT_TO_DEV_NULL,
                               NULL, NULL, NULL, &error_output, &status,
                               &error));
    g_assert_true(WIFEXITED(status));
    g_assert_cmpint(WEXITSTATUS(status), !=, 0);
    g_assert_nonnull(strstr(error_output,
                            "RISC-V Ztt backend layout mismatch"));
    g_assert_cmpint(unlink(filename), ==, 0);
}

static void ztt_migration_payload_offsets(const char *contents, gsize length,
                                          uint64_t u, gsize *pool_header,
                                          gsize *capability_version,
                                          gsize *state)
{
    uint8_t manifest_prefix[sizeof(ztt_manifest_prefix)];
    gsize offset;
    uint16_t dtype_count, entry_count;

    memcpy(manifest_prefix, ztt_manifest_prefix, sizeof(manifest_prefix));
    stq_be_p(manifest_prefix + sizeof(uint64_t), u);
    for (offset = 0;
         offset + sizeof(manifest_prefix) <= length; offset++) {
        if (memcmp(contents + offset, manifest_prefix,
                   sizeof(manifest_prefix)) == 0) {
            break;
        }
    }
    g_assert_cmpuint(offset + sizeof(manifest_prefix), <=, length);
    g_assert_cmpuint(offset, >=, 6);
    if (pool_header != NULL) {
        /* enabled, configured, and num_slots precede N. */
        *pool_header = offset - 6;
    }
    offset += sizeof(manifest_prefix);
    g_assert_cmpuint(offset + 4, <=, length);
    *capability_version = offset;
    g_assert_cmpuint(ldl_be_p(contents + offset), ==,
                     ZTT_CAPABILITY_MANIFEST_VERSION);
    offset += 4;
    g_assert_cmpuint(offset + 4, <=, length);
    g_assert_cmphex(ldl_be_p(contents + offset), ==, 0x40000080);
    offset += 4;
    g_assert_cmpuint(offset + 2, <=, length);
    dtype_count = lduw_be_p(contents + offset);
    offset += 2;
    g_assert_cmpuint(dtype_count, <=, (length - offset) / sizeof(uint32_t));
    offset += dtype_count * sizeof(uint32_t);
    g_assert_cmpuint(offset + 2, <=, length);
    entry_count = lduw_be_p(contents + offset);
    offset += 2;
    for (unsigned int i = 0; i < entry_count; i++) {
        uint8_t mode;

        g_assert_cmpuint(offset + 6, <=, length);
        g_assert_cmpuint(lduw_be_p(contents + offset), <, 80);
        g_assert_cmpuint((uint8_t)contents[offset + 2], <, 3);
        g_assert_cmpuint((uint8_t)contents[offset + 3], <, 7);
        mode = contents[offset + 4];
        g_assert_cmpuint((uint8_t)contents[offset + 5], ==, 0);
        offset += 6;
        if (mode == 1) {
            for (unsigned int p = 0; p < 3; p++) {
                g_assert_cmpuint(offset + 2, <=, length);
                dtype_count = lduw_be_p(contents + offset);
                offset += 2;
                g_assert_cmpuint(dtype_count, <=, (length - offset) / 4);
                offset += dtype_count * 4;
            }
        } else {
            uint32_t tuple_count;

            g_assert_cmpuint(mode, ==, 2);
            g_assert_cmpuint(offset + 4, <=, length);
            tuple_count = ldl_be_p(contents + offset);
            offset += 4;
            g_assert_cmpuint(tuple_count, <=, (length - offset) / 12);
            offset += tuple_count * 12;
        }
    }
    *state = offset;
}

static void test_ztt_migration_disabled_pool_occupied(void)
{
    static const char *options =
        "x-ztt=true,amenlen=4,ameudsz=32,ame-num-mregs=16,"
        "ame-num-accregs=1";
    const char *binary = g_getenv("QTEST_QEMU_BINARY");
    QTestState *source = ztt_start_migration_program(options);
    g_autofree char *filename = save_qtest_state(source);
    g_autofree char *contents = NULL;
    g_autofree char *cpu = g_strdup_printf("%s,%s", ztt_cpu_model(),
                                           options);
    g_autofree char *incoming = g_strdup_printf("file:%s", filename);
    g_autofree char *error_output = NULL;
    g_autoptr(GError) error = NULL;
    gsize length, pool_header, capability_version, state_offset;
    const char *argv[] = {
        binary, "-machine", "virt,x-ame-backend-enabled=off",
        "-bios", "none", "-nodefaults", "-display", "none",
        "-accel", "tcg", "-cpu", cpu, "-incoming", incoming, NULL,
    };
    int status;

    qtest_quit(source);
    g_assert_true(g_file_get_contents(filename, &contents, &length, &error));
    ztt_migration_payload_offsets(contents, length, 32, &pool_header,
                                  &capability_version, &state_offset);
    g_assert_cmphex((uint8_t)contents[pool_header], ==, 1);
    contents[pool_header] = 0;
    g_clear_error(&error);
    g_assert_true(g_file_set_contents(filename, contents, length, &error));

    g_assert_true(g_spawn_sync(NULL, (char **)argv, NULL,
                               G_SPAWN_STDOUT_TO_DEV_NULL,
                               NULL, NULL, NULL, &error_output, &status,
                               &error));
    g_assert_true(WIFEXITED(status));
    g_assert_cmpint(WEXITSTATUS(status), !=, 0);
    g_assert_nonnull(strstr(error_output,
                            "Invalid RISC-V Ztt backend pool migration "
                            "state"));
    g_assert_cmpint(unlink(filename), ==, 0);
}

static void test_ztt_migration_manifest_mismatch(void)
{
    static const char *options =
        "x-ztt=true,amenlen=4,ameudsz=8,ame-num-mregs=16,"
        "ame-num-accregs=1";
    const char *binary = g_getenv("QTEST_QEMU_BINARY");
    g_autofree char *filename = save_ztt_state(options);
    g_autofree char *contents = NULL;
    g_autofree char *cpu = g_strdup_printf("%s,%s", ztt_cpu_model(),
                                           options);
    g_autofree char *incoming = g_strdup_printf("file:%s", filename);
    g_autofree char *error_output = NULL;
    g_autoptr(GError) error = NULL;
    gsize length, capability_version, state_offset;
    const char *argv[] = {
        binary, "-machine", "virt", "-bios", "none", "-nodefaults",
        "-display", "none", "-accel", "tcg", "-cpu", cpu,
        "-incoming", incoming, NULL,
    };
    int status;

    g_assert_true(g_file_get_contents(filename, &contents, &length, &error));
    ztt_migration_payload_offsets(contents, length, 8, NULL,
                                  &capability_version, &state_offset);
    g_assert_cmpuint(capability_version + 4, <=, length);
    contents[capability_version + 3] ^= 1;
    g_clear_error(&error);
    g_assert_true(g_file_set_contents(filename, contents, length, &error));

    g_assert_true(g_spawn_sync(NULL, (char **)argv, NULL,
                               G_SPAWN_STDOUT_TO_DEV_NULL,
                               NULL, NULL, NULL, &error_output, &status,
                               &error));
    g_assert_true(WIFEXITED(status));
    g_assert_cmpint(WEXITSTATUS(status), !=, 0);
    g_assert_nonnull(strstr(error_output,
                            "RISC-V Ztt migration capability manifest "
                            "mismatch"));
    g_assert_cmpint(unlink(filename), ==, 0);
}

static void test_ztt_migration_manifest_corrupt(void)
{
    static const char *options =
        "x-ztt=true,amenlen=4,ameudsz=8,ame-num-mregs=16,"
        "ame-num-accregs=1";
    const char *binary = g_getenv("QTEST_QEMU_BINARY");
    g_autofree char *filename = save_ztt_state(options);
    g_autofree char *contents = NULL;
    g_autofree char *cpu = g_strdup_printf("%s,%s", ztt_cpu_model(), options);
    g_autofree char *incoming = g_strdup_printf("file:%s", filename);
    g_autoptr(GError) error = NULL;
    gsize length, version, state, dtypes, entries, entry;
    const char *argv[] = {
        binary, "-machine", "virt", "-bios", "none", "-nodefaults",
        "-display", "none", "-accel", "tcg", "-cpu", cpu,
        "-incoming", incoming, NULL,
    };

    g_assert_true(g_file_get_contents(filename, &contents, &length, &error));
    ztt_migration_payload_offsets(contents, length, 8, NULL, &version, &state);
    /* Wire fields, deliberately independent of the production serializer. */
    dtypes = version + 10;
    entries = dtypes + lduw_be_p(contents + dtypes - 2) * 4;
    entry = entries + 2;
    g_assert_cmpuint(entry + 12, <, state);
    g_assert_cmpuint((uint8_t)contents[entry + 4], ==, 1);
    const gsize fields[] = {
        version - 4, /* magic; rejects the previous development layout */
        version,     /* version */
        version + 4, /* maximum integer encoding */
        dtypes - 2,  /* public datatype count */
        dtypes,      /* public encoding */
        entries,     /* entry count */
        entry,       /* operation */
        entry + 2,   /* form */
        entry + 3,   /* branch */
        entry + 4,   /* mode */
        entry + 5,   /* reserved */
        entry + 6,   /* position count */
        entry + 8,   /* position encoding */
        state - 1,   /* last manifest byte, beyond the first read chunk */
    };

    for (size_t i = 0; i < ARRAY_SIZE(fields); i++) {
        g_autofree char *error_output = NULL;
        int status;

        g_test_message("corrupt manifest field at offset %zu", fields[i]);
        contents[fields[i]] ^= 0x80;
        g_assert_true(g_file_set_contents(filename, contents, length, &error));
        g_assert_true(g_spawn_sync(NULL, (char **)argv, NULL,
                                   G_SPAWN_STDOUT_TO_DEV_NULL,
                                   NULL, NULL, NULL, &error_output, &status,
                                   &error));
        g_assert_true(WIFEXITED(status));
        g_assert_cmpint(WEXITSTATUS(status), !=, 0);
        g_assert_nonnull(strstr(error_output,
                                "Ztt migration capability manifest mismatch"));
        contents[fields[i]] ^= 0x80;
    }
    g_assert_cmpint(unlink(filename), ==, 0);
}

static void test_ztt_migration_invalid_state(void)
{
    static const char *options =
        "x-ztt=true,amenlen=4,ameudsz=32,ame-num-mregs=16,"
        "ame-num-accregs=1";
    const char *binary = g_getenv("QTEST_QEMU_BINARY");
    QTestState *source = ztt_start_migration_program(options);
    g_autofree char *filename = save_qtest_state(source);
    g_autofree char *contents = NULL;
    g_autofree char *cpu = g_strdup_printf("%s,%s", ztt_cpu_model(),
                                           options);
    g_autofree char *incoming = g_strdup_printf("file:%s", filename);
    g_autofree char *error_output = NULL;
    g_autoptr(GError) error = NULL;
    gsize length, capability_version, state_offset;
    const char *argv[] = {
        binary, "-machine", "virt", "-bios", "none", "-nodefaults",
        "-display", "none", "-accel", "tcg", "-cpu", cpu,
        "-incoming", incoming, NULL,
    };
    int status;

    qtest_quit(source);

    g_assert_true(g_file_get_contents(filename, &contents, &length, &error));
    ztt_migration_payload_offsets(contents, length, 32, NULL,
                                  &capability_version, &state_offset);
    g_assert_cmpuint(state_offset + 1 + sizeof(uint64_t) +
                     2 * sizeof(uint32_t), <=, length);
    stl_be_p(contents + state_offset + 1 + sizeof(uint64_t) +
             sizeof(uint32_t),
             UINT32_C(0x40000080)); /* Int128 Md1 is not group-aligned. */
    g_clear_error(&error);
    g_assert_true(g_file_set_contents(filename, contents, length, &error));

    g_assert_true(g_spawn_sync(NULL, (char **)argv, NULL,
                               G_SPAWN_STDOUT_TO_DEV_NULL,
                               NULL, NULL, NULL, &error_output, &status,
                               &error));
    g_assert_true(WIFEXITED(status));
    g_assert_cmpint(WEXITSTATUS(status), !=, 0);
    g_assert_nonnull(strstr(error_output,
                            "Invalid RISC-V Ztt backend pool migration "
                            "state"));
    g_assert_cmpint(unlink(filename), ==, 0);
}

static void test_ztt_migration_truncated_state(void)
{
    static const char *options =
        "x-ztt=true,amenlen=4,ameudsz=32,ame-num-mregs=16,"
        "ame-num-accregs=1";
    const char *binary = g_getenv("QTEST_QEMU_BINARY");
    QTestState *source = ztt_start_migration_program(options);
    g_autofree char *filename = save_qtest_state(source);
    g_autofree char *contents = NULL;
    g_autofree char *cpu = g_strdup_printf("%s,%s", ztt_cpu_model(),
                                           options);
    g_autofree char *incoming = g_strdup_printf("file:%s", filename);
    g_autofree char *error_output = NULL;
    g_autoptr(GError) error = NULL;
    gsize length, capability_version, state_offset, truncated_length;
    const char *argv[] = {
        binary, "-machine", "virt", "-bios", "none", "-nodefaults",
        "-display", "none", "-accel", "tcg", "-cpu", cpu,
        "-incoming", incoming, NULL,
    };
    int status;

    qtest_quit(source);

    g_assert_true(g_file_get_contents(filename, &contents, &length, &error));
    ztt_migration_payload_offsets(contents, length, 32, NULL,
                                  &capability_version, &state_offset);
    /*
     * Keep all scalar state and the complete 256-byte M backing, then cut
     * the 256-byte Acc backing in half.  The destination must reject the
     * stream before installing any staged AME state.
     */
    truncated_length = state_offset + 1 + sizeof(uint64_t) +
                       16 * sizeof(uint32_t) + sizeof(uint32_t) +
                       2 * sizeof(uint32_t) + 256 + 128;
    g_assert_cmpuint(truncated_length, <, length);
    g_clear_error(&error);
    g_assert_true(g_file_set_contents(filename, contents, truncated_length,
                                      &error));

    g_assert_true(g_spawn_sync(NULL, (char **)argv, NULL,
                               G_SPAWN_STDOUT_TO_DEV_NULL,
                               NULL, NULL, NULL, &error_output, &status,
                               &error));
    g_assert_true(WIFEXITED(status));
    g_assert_cmpint(WEXITSTATUS(status), !=, 0);
    g_assert_nonnull(strstr(error_output,
                            "Invalid RISC-V Ztt backend pool migration "
                            "state"));
    g_assert_cmpint(unlink(filename), ==, 0);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);
    g_test_add_func("/riscv/ztt/reset/shared-pool-system",
                    test_ztt_shared_pool_system_reset);
    g_test_add_func("/riscv/ztt/migration/round-trip",
                    test_ztt_migration_round_trip);
    g_test_add_func("/riscv/ztt/migration/disabled",
                    test_ztt_migration_disabled);
    g_test_add_func("/riscv/ztt/migration/extension-enabled-to-disabled",
                    test_ztt_migration_extension_enabled_to_disabled);
    g_test_add_func("/riscv/ztt/migration/extension-disabled-to-enabled",
                    test_ztt_migration_extension_disabled_to_enabled);
    g_test_add_func("/riscv/ztt/migration/backend-disabled",
                    test_ztt_migration_backend_disabled);
    g_test_add_func("/riscv/ztt/migration/backend-enabled-to-disabled",
                    test_ztt_migration_backend_enabled_to_disabled);
    g_test_add_func("/riscv/ztt/migration/disabled-pool-occupied",
                    test_ztt_migration_disabled_pool_occupied);
    g_test_add_func("/riscv/ztt/migration/max-backing",
                    test_ztt_migration_max_backing);
    g_test_add_func("/riscv/ztt/migration/dirty-state",
                    test_ztt_migration_dirty_state);
    g_test_add_func("/riscv/ztt/migration/ram-dirty",
                    test_ztt_migration_ram_dirty);
    g_test_add_func("/riscv/ztt/migration/private-system-pool",
                    test_ztt_migration_private_system_pool);
    g_test_add_func("/riscv/ztt/migration/partial-pool",
                    test_ztt_migration_partial_pool);
    g_test_add_func("/riscv/ztt/migration/full-pool-with-waiter",
                    test_ztt_migration_full_pool_with_waiter);
    g_test_add_func("/riscv/ztt/migration/pool-count-mismatch",
                    test_ztt_migration_pool_count_mismatch);
    g_test_add_func("/riscv/ztt/migration/config-mismatch",
                    test_ztt_migration_config_mismatch);
    g_test_add_func("/riscv/ztt/migration/manifest-mismatch",
                    test_ztt_migration_manifest_mismatch);
    g_test_add_func("/riscv/ztt/migration/manifest-corrupt",
                    test_ztt_migration_manifest_corrupt);
    g_test_add_func("/riscv/ztt/migration/invalid-state",
                    test_ztt_migration_invalid_state);
    g_test_add_func("/riscv/ztt/migration/truncated-state",
                    test_ztt_migration_truncated_state);
    return g_test_run();
}
