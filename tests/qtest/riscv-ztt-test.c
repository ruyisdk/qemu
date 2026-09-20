/*
 * RISC-V Ztt CPU configuration tests
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "libqtest.h"
#include "qobject/qbool.h"
#include "qobject/qdict.h"
#include "qobject/qlist.h"
#include "qobject/qnum.h"

static const char *ztt_cpu_model(void)
{
    return g_str_equal(qtest_get_arch(), "riscv32") ? "rv32" : "rv64";
}

static QTestState *ztt_start(const char *options)
{
    return qtest_initf("-machine virt -bios none -nodefaults -S -accel tcg "
                       "-cpu %s,%s", ztt_cpu_model(), options);
}

static char *cpu_qom_path(QTestState *qts)
{
    QDict *response;
    QList *cpus;
    QDict *cpu;
    char *path;

    response = qtest_qmp(qts, "{ 'execute': 'query-cpus-fast' }");
    g_assert_false(qdict_haskey(response, "error"));
    cpus = qdict_get_qlist(response, "return");
    g_assert_cmpuint(qlist_size(cpus), ==, 1);
    cpu = qobject_to(QDict, qlist_peek(cpus));
    path = g_strdup(qdict_get_str(cpu, "qom-path"));
    qobject_unref(response);
    return path;
}

static QObject *qom_get(QTestState *qts, const char *path,
                        const char *property)
{
    QDict *response;
    QObject *value;

    response = qtest_qmp(qts,
                         "{ 'execute': 'qom-get', 'arguments': { "
                         "  'path': %s, 'property': %s } }",
                         path, property);
    g_assert_false(qdict_haskey(response, "error"));
    value = qdict_get(response, "return");
    qobject_ref(value);
    qobject_unref(response);
    return value;
}

static int64_t qom_get_int(QTestState *qts, const char *path,
                           const char *property)
{
    QObject *object = qom_get(qts, path, property);
    QNum *number = qobject_to(QNum, object);
    int64_t value = qnum_get_int(number);

    qobject_unref(object);
    return value;
}

static bool qom_get_bool(QTestState *qts, const char *path,
                         const char *property)
{
    QObject *object = qom_get(qts, path, property);
    QBool *boolean = qobject_to(QBool, object);
    bool value = qbool_get_bool(boolean);

    qobject_unref(object);
    return value;
}

static void test_ztt_default_properties(void)
{
    QTestState *qts;
    g_autofree char *path = NULL;

    if (!qtest_has_accel("tcg")) {
        g_test_skip("TCG acceleration is unavailable");
        return;
    }

    qts = ztt_start("x-ztt=true");
    path = cpu_qom_path(qts);

    g_assert_true(qom_get_bool(qts, path, "x-ztt"));
    g_assert_cmpint(qom_get_int(qts, path, "amenlen"), ==, 128);
    g_assert_cmpint(qom_get_int(qts, path, "ameudsz"), ==, 64);
    g_assert_cmpint(qom_get_int(qts, path, "ame-num-mregs"), ==, 32);
    g_assert_cmpint(qom_get_int(qts, path, "ame-num-accregs"), ==, 16);
    qtest_quit(qts);
}

static void test_ztt_small_properties(void)
{
    QTestState *qts;
    g_autofree char *path = NULL;

    if (!qtest_has_accel("tcg")) {
        g_test_skip("TCG acceleration is unavailable");
        return;
    }

    qts = ztt_start("x-ztt=true,amenlen=4,ameudsz=8,"
                    "ame-num-mregs=16,ame-num-accregs=8");
    path = cpu_qom_path(qts);

    g_assert_cmpint(qom_get_int(qts, path, "amenlen"), ==, 4);
    g_assert_cmpint(qom_get_int(qts, path, "ameudsz"), ==, 8);
    g_assert_cmpint(qom_get_int(qts, path, "ame-num-mregs"), ==, 16);
    g_assert_cmpint(qom_get_int(qts, path, "ame-num-accregs"), ==, 8);
    qtest_quit(qts);
}

static void assert_cpu_realize_fails(const char *model, const char *options,
                                     const char *message)
{
    const char *binary = g_getenv("QTEST_QEMU_BINARY");
    g_autofree char *cpu = g_strdup_printf("%s,%s", model, options);
    g_autofree char *error_output = NULL;
    g_autoptr(GError) error = NULL;
    const char *argv[] = {
        binary, "-machine", "virt", "-bios", "none", "-nodefaults",
        "-display", "none", "-accel", "tcg,thread=single", "-accel",
        "qtest", "-cpu", cpu, NULL,
    };
    int status;

    g_assert_nonnull(binary);
    g_assert_true(g_spawn_sync(NULL, (char **)argv, NULL,
                               G_SPAWN_STDOUT_TO_DEV_NULL,
                               NULL, NULL, NULL, &error_output, &status,
                               &error));
    g_assert_true(WIFEXITED(status));
    g_assert_cmpint(WEXITSTATUS(status), !=, 0);
    g_assert_nonnull(strstr(error_output, message));
}

static void assert_realize_fails(const char *options, const char *message)
{
    assert_cpu_realize_fails(ztt_cpu_model(), options, message);
}

static void test_ztt_realize_validation(void)
{
    if (!qtest_has_accel("tcg")) {
        g_test_skip("TCG acceleration is unavailable");
        return;
    }

    if (g_str_equal(qtest_get_arch(), "riscv64")) {
        QTestState *qts = qtest_initf(
            "-machine virt -bios none -nodefaults -S -accel tcg "
            "-cpu thead-c906");

        qtest_quit(qts);
        assert_cpu_realize_fails("thead-c906", "x-ztt=true",
                                 "CPU does not allow enabling extensions");
    }
    assert_realize_fails("x-ztt=true,amenlen=3",
                         "Ztt N must be a power of two");
    assert_realize_fails("x-ztt=true,ameudsz=128",
                         "Ztt U must be a power of two");
    assert_realize_fails("x-ztt=true,ame-num-mregs=24",
                         "Ztt NM must be 16 or 32");
    assert_realize_fails("x-ztt=true,ame-num-accregs=3",
                         "Ztt NA must be a power of two in [1, 16]");
    if (g_str_equal(qtest_get_arch(), "riscv64")) {
        assert_cpu_realize_fails("x-rv128", "x-ztt=true",
                                 "Ztt is not supported with RV128");
    }
}

static void test_ztt_kvm_rejected(void)
{
    const char *binary = g_getenv("QTEST_QEMU_BINARY");
    g_autofree char *cpu = g_strdup_printf("%s,x-ztt=true",
                                           ztt_cpu_model());
    g_autofree char *error_output = NULL;
    g_autoptr(GError) error = NULL;
    const char *argv[] = {
        binary, "-machine", "virt", "-bios", "none", "-nodefaults",
        "-display", "none", "-accel", "kvm", "-cpu", cpu, NULL,
    };
    int status;

    if (!qtest_has_accel("kvm")) {
        g_test_skip("RISC-V KVM acceleration is unavailable");
        return;
    }

    g_assert_true(g_spawn_sync(NULL, (char **)argv, NULL,
                               G_SPAWN_STDOUT_TO_DEV_NULL,
                               NULL, NULL, NULL, &error_output, &status,
                               &error));
    g_assert_true(WIFEXITED(status));
    g_assert_cmpint(WEXITSTATUS(status), !=, 0);
    g_assert_nonnull(strstr(error_output, "not available with KVM"));
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);
    g_test_add_func("/riscv/ztt/properties/default",
                    test_ztt_default_properties);
    g_test_add_func("/riscv/ztt/properties/small",
                    test_ztt_small_properties);
    g_test_add_func("/riscv/ztt/realize/validation",
                    test_ztt_realize_validation);
    g_test_add_func("/riscv/ztt/kvm/rejected", test_ztt_kvm_rejected);
    return g_test_run();
}
