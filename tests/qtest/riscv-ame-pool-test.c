/*
 * RISC-V AME backend pool machine tests
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "libqtest.h"
#include "qobject/qbool.h"
#include "qobject/qdict.h"
#include "qobject/qnum.h"

static QTestState *ame_pool_start(const char *machine_options, unsigned harts)
{
    return qtest_initf("-machine virt%s%s -bios none -nodefaults -S "
                       "-accel tcg -smp %u",
                       machine_options[0] == '\0' ? "" : ",",
                       machine_options, harts);
}

static QObject *qom_get(QTestState *qts, const char *property)
{
    QDict *response;
    QObject *value;

    response = qtest_qmp(qts,
                         "{ 'execute': 'qom-get', 'arguments': { "
                         "  'path': '/machine', 'property': %s } }",
                         property);
    g_assert_false(qdict_haskey(response, "error"));
    value = qdict_get(response, "return");
    qobject_ref(value);
    qobject_unref(response);
    return value;
}

static int64_t qom_get_int(QTestState *qts, const char *property)
{
    QObject *object = qom_get(qts, property);
    QNum *number = qobject_to(QNum, object);
    int64_t value = qnum_get_int(number);

    qobject_unref(object);
    return value;
}

static bool qom_get_bool(QTestState *qts, const char *property)
{
    QObject *object = qom_get(qts, property);
    QBool *boolean = qobject_to(QBool, object);
    bool value = qbool_get_bool(boolean);

    qobject_unref(object);
    return value;
}

static void test_ame_pool_default_properties(void)
{
    QTestState *qts = ame_pool_start("", 1);

    g_assert_cmpint(qom_get_int(qts, "x-ame-backends"), ==, 1);
    g_assert_true(qom_get_bool(qts, "x-ame-backend-enabled"));
    qtest_quit(qts);
}

static void test_ame_pool_properties(void)
{
    QTestState *qts = ame_pool_start(
        "x-ame-backends=2,x-ame-backend-enabled=off", 3);
    QDict *response;

    g_assert_cmpint(qom_get_int(qts, "x-ame-backends"), ==, 2);
    g_assert_false(qom_get_bool(qts, "x-ame-backend-enabled"));

    response = qtest_qmp(qts,
                         "{ 'execute': 'qom-set', 'arguments': { "
                         "  'path': '/machine', "
                         "  'property': 'x-ame-backends', "
                         "  'value': 1 } }");
    g_assert_true(qdict_haskey(response, "error"));
    qobject_unref(response);
    response = qtest_qmp(qts,
                         "{ 'execute': 'qom-set', 'arguments': { "
                         "  'path': '/machine', "
                         "  'property': 'x-ame-backend-enabled', "
                         "  'value': true } }");
    g_assert_true(qdict_haskey(response, "error"));
    qobject_unref(response);

    g_assert_cmpint(qom_get_int(qts, "x-ame-backends"), ==, 2);
    g_assert_false(qom_get_bool(qts, "x-ame-backend-enabled"));

    qtest_system_reset(qts);
    g_assert_cmpint(qom_get_int(qts, "x-ame-backends"), ==, 2);
    g_assert_false(qom_get_bool(qts, "x-ame-backend-enabled"));
    qtest_quit(qts);
}

static void assert_machine_realize_fails(const char *machine_options,
                                         unsigned harts,
                                         const char *message)
{
    const char *binary = g_getenv("QTEST_QEMU_BINARY");
    g_autofree char *machine = g_strdup_printf("virt,%s", machine_options);
    g_autofree char *smp = g_strdup_printf("%u", harts);
    g_autofree char *error_output = NULL;
    g_autoptr(GError) error = NULL;
    const char *argv[] = {
        binary, "-machine", machine, "-bios", "none", "-nodefaults",
        "-display", "none", "-accel", "tcg,thread=single",
        "-smp", smp, NULL,
    };
    int status;
    bool spawned;

    g_assert_nonnull(binary);
    spawned = g_spawn_sync(NULL, (char **)argv, NULL,
                           G_SPAWN_STDOUT_TO_DEV_NULL,
                           NULL, NULL, NULL, &error_output, &status,
                           &error);
    g_assert_no_error(error);
    g_assert_true(spawned);
    g_assert_true(WIFEXITED(status));
    g_assert_cmpint(WEXITSTATUS(status), !=, 0);
    g_assert_nonnull(strstr(error_output, message));
}

static void test_ame_pool_validation(void)
{
    assert_machine_realize_fails("x-ame-backends=0", 1,
                                 "x-ame-backends must be greater than zero");
    assert_machine_realize_fails("x-ame-backends=2", 1,
                                 "x-ame-backends cannot exceed the number "
                                 "of CPUs");
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);
    g_test_add_func("/riscv/ame-pool/properties/default",
                    test_ame_pool_default_properties);
    g_test_add_func("/riscv/ame-pool/properties/custom",
                    test_ame_pool_properties);
    g_test_add_func("/riscv/ame-pool/validation",
                    test_ame_pool_validation);
    return g_test_run();
}
