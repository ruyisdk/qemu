/*
 * RISC-V Attached Matrix Extension backend state tests
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qapi/error.h"
#include "qemu/main-loop.h"
#include "qemu/units.h"
#include "qom/object.h"
#include "target/riscv/ame/backend.h"

#define TYPE_AME_TEST_HART "riscv-ame-test-hart"

OBJECT_DECLARE_SIMPLE_TYPE(AmeTestHart, AME_TEST_HART)

struct AmeTestHart {
    Object parent_obj;
    unsigned kicks;
};

static unsigned ame_test_hart_finalize_count;

static void ame_test_hart_finalize(Object *obj)
{
    ame_test_hart_finalize_count++;
}

static const TypeInfo ame_test_hart_type_info = {
    .name = TYPE_AME_TEST_HART,
    .parent = TYPE_OBJECT,
    .instance_size = sizeof(AmeTestHart),
    .instance_finalize = ame_test_hart_finalize,
};

static void ame_test_wait_kick(Object *hart)
{
    qatomic_inc(&AME_TEST_HART(hart)->kicks);
}

static Object *new_test_hart(void)
{
    return object_new(TYPE_AME_TEST_HART);
}

static const AmeCapabilities *maximum(void)
{
    return riscv_ame_maximum_capabilities();
}

static void configure_small_pool(RISCVAMEBackendPoolCore *pool)
{
    g_assert_true(riscv_ame_pool_configure_cpu(
        pool, 4, 8, 16, 1, AME_CAP_POLICY_EXPLICIT, maximum(), &error_abort));
}

static void test_pool_configuration(void)
{
    RISCVAMEBackendPoolCore pool = { 0 };
    RISCVAMEBackendPoolCore unconfigured = { 0 };
    AmeCapabilities equivalent = { 0 };
    Error *err = NULL;

    g_assert_false(riscv_ame_pool_init(NULL, 1, true, &err));
    g_assert_nonnull(err);
    error_free(err);
    err = NULL;
    g_assert_false(riscv_ame_pool_init(&pool, 0, true, &err));
    g_assert_nonnull(err);
    error_free(err);
    err = NULL;
    g_assert_false(riscv_ame_pool_init(&pool, UINT32_MAX, true, &err));
    g_assert_nonnull(err);
    error_free(err);
    err = NULL;

    g_assert_true(riscv_ame_pool_init(&pool, 2, false, &error_abort));
    g_assert_false(pool.enabled);
    g_assert_false(riscv_ame_pool_init(&pool, 2, false, &err));
    g_assert_nonnull(err);
    error_free(err);
    err = NULL;

    riscv_ame_capabilities_copy(&equivalent, maximum());
    g_assert_true(riscv_ame_pool_configure_cpu(
        &pool, 4, 8, 16, 1, AME_CAP_POLICY_EXPLICIT,
        &equivalent, &error_abort));
    g_assert_true(pool.configured);
    g_assert_true(riscv_ame_capabilities_equal(
        &pool.capabilities, maximum()));
    g_assert_true(pool.capabilities.entries != equivalent.entries);

    equivalent.dtypes.words[0] = 0;
    equivalent.entries[0].tuples.operand_types[0].words[0] = 0;
    g_assert_true(riscv_ame_capabilities_equal(
        &pool.capabilities, maximum()));
    g_assert_true(riscv_ame_pool_configure_cpu(
        &pool, 4, 8, 16, 1, AME_CAP_POLICY_EXPLICIT, maximum(), &error_abort));
    g_assert_false(riscv_ame_pool_configure_cpu(
        &pool, 8, 8, 16, 1, AME_CAP_POLICY_EXPLICIT, maximum(), &err));
    g_assert_nonnull(err);
    error_free(err);

    riscv_ame_capabilities_clear(&equivalent);
    g_assert_true(riscv_ame_pool_realize(&pool, &error_abort));
    err = NULL;
    g_assert_false(riscv_ame_pool_realize(&pool, &err));
    g_assert_nonnull(err);
    error_free(err);
    riscv_ame_pool_destroy(&pool);

    g_assert_true(riscv_ame_pool_init(
        &unconfigured, 1, true, &error_abort));
    g_assert_true(riscv_ame_pool_realize(&unconfigured, &error_abort));
    err = NULL;
    g_assert_false(riscv_ame_pool_configure_cpu(
        &unconfigured, 4, 8, 16, 1, AME_CAP_POLICY_EXPLICIT, maximum(), &err));
    g_assert_nonnull(err);
    error_free(err);
    riscv_ame_pool_destroy(&unconfigured);
}

static void test_pool_failure_rollback(void)
{
    RISCVAMEBackendPoolCore pool = { 0 };
    RISCVAMEState state = { 0 };
    Error *err = NULL;

    g_assert_true(riscv_ame_pool_init(&pool, 2, true, &error_abort));
    g_assert_false(riscv_ame_pool_configure_cpu(
        &pool, 3, 8, 16, 1, AME_CAP_POLICY_EXPLICIT, maximum(), &err));
    g_assert_nonnull(err);
    g_assert_false(pool.configured);
    g_assert_null(pool.capabilities.entries);
    error_free(err);
    err = NULL;

    g_assert_false(riscv_ame_state_attach(&state, &pool, &err));
    g_assert_nonnull(err);
    g_assert_null(state.pool);
    g_assert_null(state.memory.chunks);
    error_free(err);

    configure_small_pool(&pool);
    g_assert_true(riscv_ame_pool_realize(&pool, &error_abort));
    g_assert_nonnull(pool.slots);
    riscv_ame_pool_destroy(&pool);

    err = NULL;
    g_assert_false(riscv_ame_state_init(
        &state, 3, 8, 16, 1, AME_CAP_POLICY_EXPLICIT, maximum(), &err));
    g_assert_nonnull(err);
    g_assert_null(state.pool);
    g_assert_null(state.private_pool);
    g_assert_null(state.memory.chunks);
    error_free(err);
    riscv_ame_state_destroy(&state);
}

static void test_pool_capability_authority(void)
{
    static const uint32_t encodings[] = { 0x40000008 };
    AmeDTypeSet dtypes = { 0 };
    AmeCapabilities small = { 0 };
    RISCVAMEBackendPoolCore explicit = { 0 }, automatic = { 0 };
    RISCVAMEBackendPoolCore invalid = { 0 };
    Error *err = NULL;

    g_assert_true(riscv_ame_dtype_set_from_encodings(
        &dtypes, encodings, 1, &error_abort));
    g_assert_true(riscv_ame_capabilities_init(
        &small, encodings[0], &dtypes, NULL, 0, &error_abort));
    g_assert_true(riscv_ame_pool_init(&explicit, 2, true, &error_abort));
    g_assert_true(riscv_ame_pool_configure(
        &explicit, 4, 8, 16, 1, &small, &error_abort));
    g_assert_cmpint(explicit.config_source, ==, RISCV_AME_POOL_EXPLICIT);
    g_assert_cmpuint(explicit.layout.acc_bytes, ==, 16);
    g_assert_true(riscv_ame_pool_configure_cpu(
        &explicit, 4, 8, 16, 1, AME_CAP_POLICY_FOLLOW_MAX, NULL, &error_abort));
    g_assert_true(riscv_ame_capabilities_equal(&explicit.capabilities, &small));
    /* Generic input must not replace the smaller pool's Acc geometry. */
    g_assert_cmpuint(explicit.layout.acc_bytes, ==, 16);
    g_assert_true(riscv_ame_pool_configure_cpu(
        &explicit, 4, 8, 16, 1, AME_CAP_POLICY_EXPLICIT, &small, &error_abort));
    g_assert_false(riscv_ame_pool_configure_cpu(
        &explicit, 4, 8, 16, 1, AME_CAP_POLICY_EXPLICIT,
        maximum(), &err));
    g_assert_nonnull(err);
    error_free(err);
    err = NULL;
    g_assert_false(riscv_ame_pool_configure_cpu(
        &explicit, 8, 8, 16, 1, AME_CAP_POLICY_FOLLOW_MAX, NULL, &err));
    g_assert_nonnull(err);
    error_free(err);
    err = NULL;
    /* A second explicit configuration is forbidden, even when identical. */
    g_assert_false(riscv_ame_pool_configure(
        &explicit, 4, 8, 16, 1, &small, &err));
    g_assert_nonnull(err);
    error_free(err);
    err = NULL;
    g_assert_cmpuint(explicit.num_attached_harts, ==, 0);
    g_assert_true(riscv_ame_pool_realize(&explicit, &error_abort));
    g_assert_true(riscv_ame_pool_configure_cpu(
        &explicit, 4, 8, 16, 1, AME_CAP_POLICY_FOLLOW_MAX, NULL, &error_abort));

    g_assert_true(riscv_ame_pool_init(&automatic, 1, true, &error_abort));
    g_assert_true(riscv_ame_pool_configure_cpu(
        &automatic, 4, 8, 16, 1, AME_CAP_POLICY_EXPLICIT,
        &small, &error_abort));
    /* Fixed CPU policy and explicit Machine pool configuration are distinct. */
    g_assert_cmpint(automatic.config_source, ==, RISCV_AME_POOL_AUTO);
    g_assert_true(riscv_ame_pool_configure_cpu(
        &automatic, 4, 8, 16, 1, AME_CAP_POLICY_FOLLOW_MAX,
        NULL, &error_abort));
    g_assert_true(riscv_ame_capabilities_equal(
        &automatic.capabilities, &small));

    g_assert_true(riscv_ame_pool_init(&invalid, 1, true, &error_abort));
    g_assert_false(riscv_ame_pool_configure_cpu(
        &invalid, 4, 8, 16, 1, AME_CAP_POLICY_UNSET, NULL, &err));
    g_assert_nonnull(err);
    error_free(err);
    g_assert_false(invalid.configured);
    g_assert_null(invalid.capabilities.entries);
    g_assert_cmpuint(invalid.capabilities.max_int_dtype, ==, 0);
    riscv_ame_pool_destroy(&invalid);
    riscv_ame_capabilities_clear(&small);
    g_assert_cmphex(explicit.capabilities.max_int_dtype, ==, encodings[0]);
    riscv_ame_pool_destroy(&explicit);
    riscv_ame_pool_destroy(&automatic);
}

static void test_pool_slots_and_reset(void)
{
    RISCVAMEBackendPoolCore pool = { 0 };
    RISCVAMEBackendSlot *first;
    RISCVAMEBackendSlot *second;
    size_t i;

    g_assert_true(riscv_ame_pool_init(&pool, 2, true, &error_abort));
    configure_small_pool(&pool);
    g_assert_true(riscv_ame_pool_realize(&pool, &error_abort));
    first = &pool.slots[0];
    second = &pool.slots[1];

    g_assert_true(first->mregs == pool.mregs);
    g_assert_true(second->mregs == pool.mregs + pool.layout.mregs_bytes);
    g_assert_true(first->accs == pool.accs);
    g_assert_true(second->accs == pool.accs + pool.layout.accs_bytes);
    first->mregs[0] = 0xa5;
    second->mregs[0] = 0x5a;
    first->accs[0] = 0x33;
    second->accs[0] = 0xcc;
    first->md[3] = UINT32_MAX;
    second->ad[0] = UINT32_MAX;
    first->md_dtype_id[3] = 1;
    second->ad_dtype_id[0] = 2;
    first->scalar_dtype = UINT32_MAX;
    first->scalar_dtype_id = 3;
    first->status = UINT32_MAX;

    g_assert_cmphex(first->mregs[0], ==, 0xa5);
    g_assert_cmphex(second->mregs[0], ==, 0x5a);
    riscv_ame_pool_reset(&pool);
    g_assert_cmphex(first->mregs[0], ==, 0);
    g_assert_cmphex(second->mregs[0], ==, 0);
    g_assert_cmphex(first->accs[0], ==, 0);
    g_assert_cmphex(second->accs[0], ==, 0);
    for (i = 0; i < ARRAY_SIZE(first->md_dtype_id); i++) {
        g_assert_cmpuint(first->md[i], ==, 0);
        g_assert_cmpuint(second->md[i], ==, 0);
        g_assert_cmpuint(first->md_dtype_id[i], ==, AME_DTYPE_ID_INVALID);
        g_assert_cmpuint(second->md_dtype_id[i], ==, AME_DTYPE_ID_INVALID);
    }
    for (i = 0; i < ARRAY_SIZE(first->ad_dtype_id); i++) {
        g_assert_cmpuint(first->ad[i], ==, 0);
        g_assert_cmpuint(second->ad[i], ==, 0);
        g_assert_cmpuint(first->ad_dtype_id[i], ==, AME_DTYPE_ID_INVALID);
        g_assert_cmpuint(second->ad_dtype_id[i], ==, AME_DTYPE_ID_INVALID);
    }
    g_assert_cmpuint(first->scalar_dtype, ==, 0);
    g_assert_cmpuint(first->scalar_dtype_id, ==, AME_DTYPE_ID_INVALID);
    g_assert_cmpuint(first->status, ==, 0);

    riscv_ame_pool_destroy(&pool);
    g_assert_false(pool.lock_initialized);
    g_assert_null(pool.slots);
    riscv_ame_pool_destroy(&pool);
}

static void test_fp_only_backend(void)
{
    static const uint32_t encodings[] = { 0x14300110, 0x20300120 };
    AmeDTypeSet dtypes = { 0 };
    AmeCapabilities capabilities = { 0 };
    RISCVAMEBackendPoolCore pool = { 0 };
    RISCVAMEState shared = { 0 }, private = { 0 };
    Object *hart = new_test_hart();
    bool new_grant;

    g_assert_true(riscv_ame_dtype_set_from_encodings(
        &dtypes, encodings, G_N_ELEMENTS(encodings), &error_abort));
    g_assert_true(riscv_ame_capabilities_init(
        &capabilities, 0, &dtypes, NULL, 0, &error_abort));
    g_assert_true(riscv_ame_pool_init(&pool, 1, true, &error_abort));
    g_assert_true(riscv_ame_pool_configure(
        &pool, 4, 8, 16, 1, &capabilities, &error_abort));
    /* A generic CPU adopts the pool without adding any integer types. */
    g_assert_true(riscv_ame_pool_configure_cpu(
        &pool, 4, 8, 16, 1, AME_CAP_POLICY_FOLLOW_MAX, NULL, &error_abort));
    g_assert_true(riscv_ame_pool_configure_cpu(
        &pool, 4, 8, 16, 1, AME_CAP_POLICY_EXPLICIT,
        &capabilities, &error_abort));
    g_assert_true(riscv_ame_capabilities_equal(
        &pool.capabilities, &capabilities));
    g_assert_true(riscv_ame_state_attach(&shared, &pool, &error_abort));
    g_assert_true(riscv_ame_pool_realize(&pool, &error_abort));
    g_assert_cmpint(riscv_ame_acquire(
        &shared, hart, false, 0, 0, &new_grant), ==,
        RISCV_AME_ACQUIRE_GRANTED);
    g_assert_true(new_grant);
    g_assert_cmpuint(riscv_ame_state_layout(&shared)->acc_bytes, ==, 64);
    g_assert_true(riscv_ame_release(&shared, hart));
    riscv_ame_state_destroy(&shared);
    riscv_ame_pool_destroy(&pool);

    g_assert_true(riscv_ame_state_init(
        &private, 4, 8, 16, 1, AME_CAP_POLICY_EXPLICIT,
        &capabilities, &error_abort));
    g_assert_nonnull(private.private_pool);
    g_assert_true(riscv_ame_capabilities_equal(
        &private.pool->capabilities, &capabilities));
    g_assert_cmpuint(private.pool->capabilities.max_int_dtype, ==, 0);
    g_assert_cmpuint(riscv_ame_state_layout(&private)->acc_bytes, ==, 64);
    riscv_ame_state_destroy(&private);
    object_unref(hart);
    riscv_ame_capabilities_clear(&capabilities);
}

static void test_private_state_lifecycle(void)
{
    RISCVAMEState state = { 0 };
    RISCVAMEBackendSlot *slot;
    const RISCVAMELayout *layout;
    Error *err = NULL;
    size_t i;

    g_assert_true(riscv_ame_state_init(
        &state, 4, 8, 16, 1, AME_CAP_POLICY_EXPLICIT, maximum(), &error_abort));
    g_assert_true(state.pool == state.private_pool);
    g_assert_cmpuint(state.pool->num_slots, ==, 1);
    g_assert_cmpuint(state.pool->num_attached_harts, ==, 1);
    g_assert_cmpint(state.active_slot, ==, -1);
    layout = riscv_ame_state_layout(&state);
    g_assert_nonnull(layout);
    slot = &state.pool->slots[0];
    g_assert_cmpuint(slot->scratch_bytes, ==,
                     MAX(layout->mregs_bytes, layout->accs_bytes));

    memset(slot->mregs, 0xa5, layout->mregs_bytes);
    memset(slot->accs, 0x5a, layout->accs_bytes);
    memset(slot->scratch, 0xff, slot->scratch_bytes);
    slot->md[0] = UINT32_MAX;
    slot->md_dtype_id[0] = 0;
    riscv_ame_state_reset(&state);
    for (i = 0; i < layout->mregs_bytes; i++) {
        g_assert_cmphex(slot->mregs[i], ==, 0);
    }
    for (i = 0; i < layout->accs_bytes; i++) {
        g_assert_cmphex(slot->accs[i], ==, 0);
    }
    for (i = 0; i < slot->scratch_bytes; i++) {
        g_assert_cmphex(slot->scratch[i], ==, 0);
    }
    g_assert_cmpuint(slot->md[0], ==, 0);
    g_assert_cmpuint(slot->md_dtype_id[0], ==, AME_DTYPE_ID_INVALID);

    g_assert_false(riscv_ame_state_init(
        &state, 4, 8, 16, 1, AME_CAP_POLICY_EXPLICIT, maximum(), &err));
    g_assert_nonnull(err);
    error_free(err);
    riscv_ame_state_destroy(&state);
    g_assert_null(state.pool);
    g_assert_null(state.private_pool);
    g_assert_null(state.memory.chunks);
    g_assert_null(riscv_ame_state_layout(&state));

    riscv_ame_state_destroy(&state);
    g_assert_true(riscv_ame_state_init(
        &state, 4, 8, 16, 1, AME_CAP_POLICY_EXPLICIT, maximum(), &error_abort));
    riscv_ame_state_destroy(&state);
}

static void test_shared_state_lifecycle(void)
{
    RISCVAMEBackendPoolCore pool = { 0 };
    RISCVAMEState first = { 0 };
    RISCVAMEState second = { 0 };
    Error *err = NULL;

    g_assert_true(riscv_ame_pool_init(&pool, 2, true, &error_abort));
    configure_small_pool(&pool);
    g_assert_true(riscv_ame_state_attach(&first, &pool, &error_abort));
    g_assert_true(riscv_ame_state_attach(&second, &pool, &error_abort));
    g_assert_cmpuint(pool.num_attached_harts, ==, 2);
    g_assert_null(first.private_pool);
    g_assert_null(second.private_pool);
    /* Capability geometry is available without owning a slot. */
    g_assert_cmpint(first.active_slot, ==, -1);
    g_assert_cmpint(second.active_slot, ==, -1);
    g_assert_true(riscv_ame_state_layout(&first) == &pool.layout);
    g_assert_true(riscv_ame_state_layout(&second) == &pool.layout);
    g_assert_cmpuint(riscv_ame_state_layout(&first)->n, ==, 4);
    g_assert_cmpuint(riscv_ame_state_layout(&second)->u, ==, 8);
    first.memory.fault_address = 0x100;
    first.memory.fault_active = true;
    second.memory.fault_address = 0x200;
    second.memory.fault_active = true;
    riscv_ame_state_reset(&first);
    g_assert_cmphex(first.memory.fault_address, ==, 0);
    g_assert_false(first.memory.fault_active);
    g_assert_cmphex(second.memory.fault_address, ==, 0x200);
    g_assert_true(second.memory.fault_active);

    g_assert_false(riscv_ame_state_attach(&first, &pool, &err));
    g_assert_nonnull(err);
    error_free(err);
    g_assert_true(riscv_ame_pool_realize(&pool, &error_abort));
    pool.slots[0].mregs[0] = 0x7e;
    riscv_ame_state_reset(&first);
    g_assert_cmphex(pool.slots[0].mregs[0], ==, 0x7e);

    riscv_ame_state_destroy(&first);
    g_assert_cmpuint(pool.num_attached_harts, ==, 1);
    g_assert_null(riscv_ame_state_layout(&first));
    g_assert_true(riscv_ame_state_layout(&second) == &pool.layout);
    riscv_ame_state_destroy(&second);
    g_assert_cmpuint(pool.num_attached_harts, ==, 0);
    riscv_ame_pool_destroy(&pool);
}

static void test_slot_scratch_lifecycle(void)
{
    RISCVAMEBackendPoolCore pool = { 0 };
    RISCVAMEState states[3] = { 0 };
    Object *harts[3];
    uint8_t *buffers[2];
    bool new_grant;

    g_assert_true(riscv_ame_pool_init(&pool, 2, true, &error_abort));
    configure_small_pool(&pool);
    for (size_t i = 0; i < ARRAY_SIZE(states); i++) {
        harts[i] = new_test_hart();
        g_assert_true(riscv_ame_state_attach(&states[i], &pool, &error_abort));
    }
    g_assert_null(pool.scratch);
    g_assert_true(riscv_ame_pool_realize(&pool, &error_abort));
    for (size_t i = 0; i < 2; i++) {
        RISCVAMEBackendSlot *slot = &pool.slots[i];

        g_assert_cmpuint(slot->scratch_bytes, ==,
                         MAX(pool.layout.mregs_bytes, pool.layout.accs_bytes));
        g_assert_nonnull(slot->scratch);
        buffers[i] = slot->scratch;
        g_assert_cmpint(riscv_ame_acquire(
            &states[i], harts[i], false, 0, 0, &new_grant), ==,
            RISCV_AME_ACQUIRE_GRANTED);
        g_assert_true(new_grant);
        memset(slot->scratch, i + 0xa5, slot->scratch_bytes);
    }
    g_assert_true(buffers[0] != buffers[1]);

    /* Repeated acquire retains the same ownership and buffer. */
    g_assert_cmpint(riscv_ame_acquire(
        &states[0], harts[0], false, 0, 0, &new_grant), ==,
        RISCV_AME_ACQUIRE_GRANTED);
    g_assert_false(new_grant);
    g_assert_cmphex(buffers[0][0], ==, 0xa5);

    /* A new owner must not inherit a previous instruction's partial result. */
    g_assert_true(riscv_ame_release(&states[0], harts[0]));
    g_assert_cmpint(riscv_ame_acquire(
        &states[2], harts[2], false, 0, 0, &new_grant), ==,
        RISCV_AME_ACQUIRE_GRANTED);
    g_assert_cmpint(states[2].active_slot, ==, 0);
    g_assert_true(pool.slots[0].scratch == buffers[0]);
    for (size_t i = 0; i < pool.slots[0].scratch_bytes; i++) {
        g_assert_cmphex(buffers[0][i], ==, 0);
    }
    memset(buffers[0], 0xc7, pool.slots[0].scratch_bytes);
    riscv_ame_state_reset(&states[0]);
    g_assert_cmphex(buffers[0][0], ==, 0xc7);
    g_assert_cmphex(buffers[1][0], ==, 0xa6);

    /* The caller quiesces all harts before a pool-wide reset. */
    riscv_ame_pool_reset(&pool);
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < pool.slots[i].scratch_bytes; j++) {
            g_assert_cmphex(buffers[i][j], ==, 0);
        }
        g_assert_true(pool.slots[i].scratch == buffers[i]);
    }
    for (size_t i = 0; i < ARRAY_SIZE(states); i++) {
        g_assert_cmpint(states[i].active_slot, ==, -1);
        riscv_ame_state_destroy(&states[i]);
        object_unref(harts[i]);
    }
    riscv_ame_pool_destroy(&pool);
    g_assert_null(pool.scratch);
}

static void test_acquire_descriptor(void)
{
    static const int64_t bounded_ns[] = {
        1 * SCALE_MS,
        10 * SCALE_MS,
        100 * SCALE_MS,
        1000 * SCALE_MS,
    };
    bool bounded;
    int64_t timeout_ns;
    unsigned timeout_class;

    for (timeout_class = 0; timeout_class < 16; timeout_class++) {
        g_assert_true(riscv_ame_acquire_descriptor_decode(
            timeout_class << 2, &bounded, &timeout_ns));
        g_assert_false(bounded);
        g_assert_cmpint(timeout_ns, ==, 0);
    }
    for (timeout_class = 0; timeout_class < ARRAY_SIZE(bounded_ns);
         timeout_class++) {
        g_assert_true(riscv_ame_acquire_descriptor_decode(
            (timeout_class << 2) | 1, &bounded, &timeout_ns));
        g_assert_true(bounded);
        g_assert_cmpint(timeout_ns, ==, bounded_ns[timeout_class]);
    }
    g_assert_true(riscv_ame_acquire_descriptor_decode(
        (UINT64_C(0xf) << 2) | 1, &bounded, &timeout_ns));
    g_assert_true(bounded);
    g_assert_cmpint(timeout_ns, ==, 1000 * SCALE_MS);

    for (timeout_class = 4; timeout_class < 15; timeout_class++) {
        g_assert_false(riscv_ame_acquire_descriptor_decode(
            (timeout_class << 2) | 1, NULL, NULL));
    }
    g_assert_false(riscv_ame_acquire_descriptor_decode(2, NULL, NULL));
    g_assert_false(riscv_ame_acquire_descriptor_decode(3, NULL, NULL));
    g_assert_false(riscv_ame_acquire_descriptor_decode(
        UINT64_C(1) << 6, NULL, NULL));
}

static void test_try_acquire_release(void)
{
    RISCVAMEBackendPoolCore pool = { 0 };
    RISCVAMEState states[3] = { 0 };
    Object *harts[3];
    bool new_grant;
    size_t i;

    g_assert_true(riscv_ame_pool_init(&pool, 2, true, &error_abort));
    configure_small_pool(&pool);
    for (i = 0; i < ARRAY_SIZE(states); i++) {
        harts[i] = new_test_hart();
        g_assert_true(riscv_ame_state_attach(&states[i], &pool,
                                             &error_abort));
    }
    g_assert_cmpint(riscv_ame_acquire(
                        &states[0], harts[0], false, 0, 0, &new_grant), ==,
                    RISCV_AME_ACQUIRE_UNSUPPORTED);
    g_assert_false(new_grant);
    g_assert_true(riscv_ame_pool_realize(&pool, &error_abort));

    pool.enabled = false;
    g_assert_cmpint(riscv_ame_acquire(
                        &states[0], harts[0], false, 0, 0, &new_grant), ==,
                    RISCV_AME_ACQUIRE_DISABLED);
    pool.enabled = true;
    g_assert_cmpint(riscv_ame_acquire(
                        &states[0], harts[0], false, 0, 0, &new_grant), ==,
                    RISCV_AME_ACQUIRE_GRANTED);
    g_assert_true(new_grant);
    g_assert_cmpint(states[0].active_slot, ==, 0);
    pool.slots[0].scalar_dtype = UINT32_C(0x40000008);

    g_assert_cmpint(riscv_ame_acquire(
                        &states[1], harts[1], false, 0, 0, &new_grant), ==,
                    RISCV_AME_ACQUIRE_GRANTED);
    g_assert_true(new_grant);
    g_assert_cmpint(states[1].active_slot, ==, 1);
    g_assert_cmpint(riscv_ame_acquire(
                        &states[0], harts[0], false, 0, 0, &new_grant), ==,
                    RISCV_AME_ACQUIRE_GRANTED);
    g_assert_false(new_grant);
    g_assert_cmphex(pool.slots[0].scalar_dtype, ==,
                    UINT32_C(0x40000008));
    g_assert_cmpint(riscv_ame_acquire(
                        &states[2], harts[2], false, 0, 0, &new_grant), ==,
                    RISCV_AME_ACQUIRE_BUSY);
    g_assert_false(new_grant);
    g_assert_false(riscv_ame_release(&states[0], harts[1]));
    g_assert_true(riscv_ame_state_owned(&states[0], harts[0]));

    g_assert_true(riscv_ame_release(&states[0], harts[0]));
    g_assert_cmpint(riscv_ame_acquire(
                        &states[2], harts[2], false, 0, 0, &new_grant), ==,
                    RISCV_AME_ACQUIRE_GRANTED);
    g_assert_true(new_grant);
    g_assert_cmpint(states[2].active_slot, ==, 0);
    g_assert_cmphex(pool.slots[0].scalar_dtype, ==, 0);
    g_assert_nonnull(riscv_ame_state_backend(&states[2], harts[2]));
    g_assert_null(riscv_ame_state_backend(&states[2], harts[0]));

    riscv_ame_state_reset(&states[1]);
    g_assert_false(riscv_ame_state_owned(&states[1], harts[1]));
    g_assert_true(riscv_ame_state_owned(&states[2], harts[2]));
    riscv_ame_pool_reset(&pool);
    for (i = 0; i < ARRAY_SIZE(states); i++) {
        g_assert_cmpint(states[i].active_slot, ==, -1);
    }
    for (i = 0; i < pool.num_slots; i++) {
        g_assert_cmpint(pool.slots[i].state, ==, RISCV_AME_BACKEND_FREE);
        g_assert_null(pool.slots[i].owner);
        g_assert_null(pool.slots[i].bound_state);
    }

    for (i = 0; i < ARRAY_SIZE(states); i++) {
        riscv_ame_state_destroy(&states[i]);
        object_unref(harts[i]);
    }
    riscv_ame_pool_destroy(&pool);
}

static void force_wait_timer(RISCVAMEState *state)
{
    int64_t now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);

    state->wait_deadline_ns = now - 1;
    timer_mod_ns(state->wait_timer, now - 1);
    g_assert_true(qemu_clock_run_timers(QEMU_CLOCK_VIRTUAL));
}

static void test_bounded_wait(void)
{
    RISCVAMEBackendPoolCore pool = { 0 };
    RISCVAMEState owner_state = { 0 };
    RISCVAMEState waiter_state = { 0 };
    Object *owner = new_test_hart();
    Object *waiter = new_test_hart();
    bool new_grant;

    g_assert_true(riscv_ame_pool_init(&pool, 1, true, &error_abort));
    configure_small_pool(&pool);
    g_assert_true(riscv_ame_pool_realize(&pool, &error_abort));
    g_assert_true(riscv_ame_state_attach(&owner_state, &pool, &error_abort));
    g_assert_true(riscv_ame_state_attach(&waiter_state, &pool, &error_abort));
    riscv_ame_state_set_wait_kick(&waiter_state, ame_test_wait_kick);
    g_assert_cmpint(riscv_ame_acquire(
                        &owner_state, owner, false, 0, 0, &new_grant), ==,
                    RISCV_AME_ACQUIRE_GRANTED);

    g_assert_cmpint(riscv_ame_acquire(
                        &waiter_state, waiter, true, 1, SCALE_MS,
                        &new_grant), ==, RISCV_AME_ACQUIRE_WAIT);
    g_assert_true(waiter_state.wait_active);
    g_assert_true(riscv_ame_wait_interrupt(&waiter_state, waiter));
    g_assert_false(riscv_ame_wait_interrupt(&waiter_state, owner));
    g_assert_cmpint(riscv_ame_acquire(
                        &waiter_state, waiter, true, 1, SCALE_MS,
                        &new_grant), ==, RISCV_AME_ACQUIRE_INTERRUPTED);
    g_assert_false(waiter_state.wait_active);

    AME_TEST_HART(waiter)->kicks = 0;
    g_assert_cmpint(riscv_ame_acquire(
                        &waiter_state, waiter, true, 1, SCALE_MS,
                        &new_grant), ==, RISCV_AME_ACQUIRE_WAIT);
    force_wait_timer(&waiter_state);
    g_assert_cmpuint(qatomic_read(&AME_TEST_HART(waiter)->kicks), ==, 1);
    g_assert_true(riscv_ame_wait_has_wake(&waiter_state));
    g_assert_cmpint(waiter_state.wait_terminal, ==, RISCV_AME_WAIT_TIMEOUT);
    g_assert_cmpint(riscv_ame_acquire(
                        &waiter_state, waiter, true, 1, SCALE_MS,
                        &new_grant), ==, RISCV_AME_ACQUIRE_TIMEOUT);

    /* A release that commits first invalidates a delayed timeout callback. */
    AME_TEST_HART(waiter)->kicks = 0;
    g_assert_cmpint(riscv_ame_acquire(
                        &waiter_state, waiter, true, 1, SCALE_MS,
                        &new_grant), ==, RISCV_AME_ACQUIRE_WAIT);
    g_assert_true(riscv_ame_release(&owner_state, owner));
    g_assert_cmpuint(qatomic_read(&AME_TEST_HART(waiter)->kicks), ==, 1);
    force_wait_timer(&waiter_state);
    g_assert_cmpint(waiter_state.wait_terminal, ==, RISCV_AME_WAIT_NONE);
    g_assert_cmpint(riscv_ame_acquire(
                        &waiter_state, waiter, true, 1, SCALE_MS,
                        &new_grant), ==, RISCV_AME_ACQUIRE_GRANTED);
    g_assert_true(new_grant);

    /* A terminal result that commits first is not replaced by release. */
    g_assert_true(riscv_ame_release(&waiter_state, waiter));
    g_assert_cmpint(riscv_ame_acquire(
                        &owner_state, owner, false, 0, 0, &new_grant), ==,
                    RISCV_AME_ACQUIRE_GRANTED);
    AME_TEST_HART(waiter)->kicks = 0;
    g_assert_cmpint(riscv_ame_acquire(
                        &waiter_state, waiter, true, 1, SCALE_MS,
                        &new_grant), ==, RISCV_AME_ACQUIRE_WAIT);
    g_assert_true(riscv_ame_wait_interrupt(&waiter_state, waiter));
    g_assert_true(riscv_ame_release(&owner_state, owner));
    g_assert_cmpuint(qatomic_read(&AME_TEST_HART(waiter)->kicks), ==, 0);
    g_assert_cmpint(riscv_ame_acquire(
                        &waiter_state, waiter, true, 1, SCALE_MS,
                        &new_grant), ==, RISCV_AME_ACQUIRE_INTERRUPTED);

    g_assert_cmpint(riscv_ame_acquire(
                        &owner_state, owner, false, 0, 0, &new_grant), ==,
                    RISCV_AME_ACQUIRE_GRANTED);
    g_assert_cmpint(riscv_ame_acquire(
                        &waiter_state, waiter, true, 1, SCALE_MS,
                        &new_grant), ==, RISCV_AME_ACQUIRE_WAIT);
    riscv_ame_wait_cancel(&waiter_state);
    g_assert_false(waiter_state.wait_active);
    AME_TEST_HART(waiter)->kicks = 0;
    g_assert_true(riscv_ame_release(&owner_state, owner));
    g_assert_cmpuint(qatomic_read(&AME_TEST_HART(waiter)->kicks), ==, 0);

    /* Pool reset cancels pending waits and clears every ownership binding. */
    g_assert_cmpint(riscv_ame_acquire(
                        &owner_state, owner, false, 0, 0, &new_grant), ==,
                    RISCV_AME_ACQUIRE_GRANTED);
    g_assert_cmpint(riscv_ame_acquire(
                        &waiter_state, waiter, true, 1, SCALE_MS,
                        &new_grant), ==, RISCV_AME_ACQUIRE_WAIT);
    riscv_ame_pool_reset(&pool);
    g_assert_false(waiter_state.wait_active);
    g_assert_false(riscv_ame_wait_has_wake(&waiter_state));
    g_assert_cmpint(owner_state.active_slot, ==, -1);
    g_assert_true(QTAILQ_EMPTY(&pool.waiters));
    g_assert_cmpint(pool.slots[0].state, ==, RISCV_AME_BACKEND_FREE);
    g_assert_null(pool.slots[0].owner);

    /* Destroying an owner releases its slot and wakes a pending hart. */
    g_assert_cmpint(riscv_ame_acquire(
                        &owner_state, owner, false, 0, 0, &new_grant), ==,
                    RISCV_AME_ACQUIRE_GRANTED);
    pool.slots[0].mregs[0] = 0xa5;
    g_assert_cmpint(riscv_ame_acquire(
                        &waiter_state, waiter, true, 1, SCALE_MS,
                        &new_grant), ==, RISCV_AME_ACQUIRE_WAIT);
    AME_TEST_HART(waiter)->kicks = 0;
    riscv_ame_state_destroy(&owner_state);
    g_assert_cmpuint(qatomic_read(&AME_TEST_HART(waiter)->kicks), ==, 1);
    g_assert_cmpint(riscv_ame_acquire(
                        &waiter_state, waiter, true, 1, SCALE_MS,
                        &new_grant), ==, RISCV_AME_ACQUIRE_GRANTED);
    g_assert_true(new_grant);
    g_assert_cmphex(pool.slots[0].mregs[0], ==, 0);
    g_assert_true(riscv_ame_release(&waiter_state, waiter));

    riscv_ame_state_destroy(&waiter_state);
    riscv_ame_state_destroy(&owner_state);
    riscv_ame_pool_destroy(&pool);
    object_unref(waiter);
    object_unref(owner);
}

static void test_wait_restore(void)
{
    RISCVAMEBackendPoolCore pool = { 0 };
    RISCVAMEState owner_state = { 0 };
    RISCVAMEState waiter_state = { 0 };
    Object *owner = new_test_hart();
    Object *waiter = new_test_hart();
    Error *err = NULL;
    bool new_grant;

    g_assert_true(riscv_ame_pool_init(&pool, 1, true, &error_abort));
    configure_small_pool(&pool);
    g_assert_true(riscv_ame_pool_realize(&pool, &error_abort));
    g_assert_true(riscv_ame_state_attach(&owner_state, &pool, &error_abort));
    g_assert_true(riscv_ame_state_attach(&waiter_state, &pool, &error_abort));

    g_assert_cmpint(riscv_ame_acquire(
                        &owner_state, owner, false, 0, 0, &new_grant), ==,
                    RISCV_AME_ACQUIRE_GRANTED);
    g_assert_true(riscv_ame_wait_restore(
        &waiter_state, waiter, 1, RISCV_AME_WAIT_NONE,
        SCALE_MS / 2, &error_abort));
    g_assert_true(waiter_state.wait_active);
    g_assert_true(riscv_ame_wait_has_wake(&waiter_state));
    g_assert_true(waiter_state.wait_hart == waiter);
    g_assert_false(QTAILQ_EMPTY(&pool.waiters));

    g_assert_false(riscv_ame_wait_restore(
        &waiter_state, waiter, 1, RISCV_AME_WAIT_NONE,
        SCALE_MS / 2, &err));
    g_assert_nonnull(err);
    error_free(err);
    err = NULL;
    riscv_ame_wait_cancel(&waiter_state);

    g_assert_true(riscv_ame_wait_restore(
        &waiter_state, waiter, 1, RISCV_AME_WAIT_TIMEOUT, 0,
        &error_abort));
    g_assert_cmpint(riscv_ame_acquire(
                        &waiter_state, waiter, true, 1, SCALE_MS,
                        &new_grant), ==, RISCV_AME_ACQUIRE_TIMEOUT);
    g_assert_false(waiter_state.wait_active);

    g_assert_false(riscv_ame_wait_restore(
        &waiter_state, waiter, 0, RISCV_AME_WAIT_NONE, 0, &err));
    g_assert_nonnull(err);
    error_free(err);
    err = NULL;
    g_assert_false(riscv_ame_wait_restore(
        &waiter_state, waiter, 1, RISCV_AME_WAIT_INTERRUPTED, 1, &err));
    g_assert_nonnull(err);
    error_free(err);

    g_assert_true(riscv_ame_release(&owner_state, owner));
    riscv_ame_state_destroy(&waiter_state);
    riscv_ame_state_destroy(&owner_state);
    riscv_ame_pool_destroy(&pool);
    object_unref(waiter);
    object_unref(owner);
}

static void check_wait_restore_boundary(uint64_t generation,
                                        RISCVAMEWaitTerminal terminal,
                                        bool busy, bool remaining,
                                        bool timer_first)
{
    const uint64_t descriptor = (UINT64_C(3) << 2) | 1;
    const int64_t timeout_ns = 1000 * SCALE_MS;
    RISCVAMEBackendPoolCore pool = { 0 };
    RISCVAMEState owner = { 0 }, waiter = { 0 };
    Object *owner_hart = new_test_hart();
    Object *waiter_hart = new_test_hart();
    Error *err = NULL;
    RISCVAMEAcquireResult expected, actual;
    bool new_grant;
    bool valid = terminal == RISCV_AME_WAIT_NONE || !remaining;

    g_assert_true(riscv_ame_pool_init(&pool, 1, true, &error_abort));
    configure_small_pool(&pool);
    g_assert_true(riscv_ame_pool_realize(&pool, &error_abort));
    g_assert_true(riscv_ame_state_attach(&owner, &pool, &error_abort));
    g_assert_true(riscv_ame_state_attach(&waiter, &pool, &error_abort));
    if (busy) {
        g_assert_cmpint(riscv_ame_acquire(
            &owner, owner_hart, false, 0, 0, &new_grant), ==,
            RISCV_AME_ACQUIRE_GRANTED);
    }
    pool.generation = generation;

    g_assert_cmpint(riscv_ame_wait_restore(
        &waiter, waiter_hart, descriptor, terminal,
        remaining ? timeout_ns : 0, &err), ==, valid);
    if (!valid) {
        g_assert_nonnull(err);
        error_free(err);
        g_assert_false(waiter.wait_active);
        g_assert_cmpint(waiter.active_slot, ==, -1);
        g_assert_true(QTAILQ_EMPTY(&pool.waiters));
        g_assert_false(timer_pending(waiter.wait_timer));
    } else {
        g_assert_null(err);
        g_assert_cmpuint(waiter.wait_generation, ==, generation - 1);
        if (generation == 0) {
            g_assert_cmpuint(waiter.wait_generation, ==, UINT64_MAX);
        }
        g_assert_true(riscv_ame_wait_has_wake(&waiter));
        g_assert_cmpint(timer_pending(waiter.wait_timer), ==,
                        terminal == RISCV_AME_WAIT_NONE);
        if (timer_first) {
            force_wait_timer(&waiter);
            g_assert_cmpint(waiter.wait_terminal, ==, RISCV_AME_WAIT_NONE);
        }

        /* Terminal results win; otherwise rescan availability before expiry. */
        if (terminal == RISCV_AME_WAIT_TIMEOUT) {
            expected = RISCV_AME_ACQUIRE_TIMEOUT;
        } else if (terminal == RISCV_AME_WAIT_INTERRUPTED) {
            expected = RISCV_AME_ACQUIRE_INTERRUPTED;
        } else if (!busy) {
            expected = RISCV_AME_ACQUIRE_GRANTED;
        } else {
            expected = remaining ? RISCV_AME_ACQUIRE_WAIT :
                                   RISCV_AME_ACQUIRE_TIMEOUT;
        }
        actual = riscv_ame_acquire(&waiter, waiter_hart, true, descriptor,
                                    timeout_ns, &new_grant);
        g_assert_cmpint(actual, ==, expected);
        g_assert_cmpint(new_grant, ==, expected == RISCV_AME_ACQUIRE_GRANTED);
        if (actual == RISCV_AME_ACQUIRE_WAIT) {
            /* The bypass is one-shot: an unsuccessful rescan can expire. */
            g_assert_cmpuint(waiter.wait_generation, ==, pool.generation);
            force_wait_timer(&waiter);
            g_assert_cmpint(waiter.wait_terminal, ==, RISCV_AME_WAIT_TIMEOUT);
            g_assert_cmpint(riscv_ame_acquire(
                &waiter, waiter_hart, true, descriptor, timeout_ns,
                &new_grant), ==, RISCV_AME_ACQUIRE_TIMEOUT);
            g_assert_false(new_grant);
        }
        g_assert_false(waiter.wait_active);
        g_assert_true(QTAILQ_EMPTY(&pool.waiters));
        g_assert_false(timer_pending(waiter.wait_timer));
        g_assert_cmpint(waiter.wait_terminal, ==, RISCV_AME_WAIT_NONE);
        g_assert_cmpint(riscv_ame_state_owned(&waiter, waiter_hart), ==,
                        expected == RISCV_AME_ACQUIRE_GRANTED);
    }

    riscv_ame_state_destroy(&waiter);
    riscv_ame_state_destroy(&owner);
    riscv_ame_pool_destroy(&pool);
    object_unref(waiter_hart);
    object_unref(owner_hart);
}

static void test_wait_restore_boundaries(void)
{
    static const uint64_t generations[] = { 0, 7 };
    static const RISCVAMEWaitTerminal terminals[] = {
        RISCV_AME_WAIT_NONE, RISCV_AME_WAIT_TIMEOUT, RISCV_AME_WAIT_INTERRUPTED,
    };
    unsigned checked = 0;

    for (size_t g = 0; g < ARRAY_SIZE(generations); g++) {
        for (size_t t = 0; t < ARRAY_SIZE(terminals); t++) {
            for (unsigned busy = 0; busy < 2; busy++) {
                for (unsigned remaining = 0; remaining < 2; remaining++) {
                    check_wait_restore_boundary(generations[g], terminals[t],
                                                 busy, remaining, false);
                    checked++;
                    if (terminals[t] == RISCV_AME_WAIT_NONE && !remaining) {
                        check_wait_restore_boundary(
                            generations[g], terminals[t], busy, remaining,
                            true);
                        checked++;
                    }
                }
            }
        }
    }
    g_assert_cmpuint(checked, ==, 28);
    g_test_message("%u restore cases: 20 valid and 8 rejected", checked);
}

typedef struct AmeConcurrentAcquire {
    RISCVAMEState *state;
    Object *hart;
    QemuEvent *start;
    QemuEvent *release;
    QemuSemaphore *acquired;
    RISCVAMEAcquireResult result;
    bool new_grant;
    uint8_t pattern;
} AmeConcurrentAcquire;

static void *concurrent_acquire_thread(void *opaque)
{
    AmeConcurrentAcquire *context = opaque;

    qemu_event_wait(context->start);
    context->result = riscv_ame_acquire(
        context->state, context->hart, false, 0, 0, &context->new_grant);
    if (context->result == RISCV_AME_ACQUIRE_GRANTED) {
        RISCVAMEBackendSlot *slot = riscv_ame_state_backend(context->state,
                                                           context->hart);

        memset(slot->scratch, context->pattern, slot->scratch_bytes);
    }
    qemu_sem_post(context->acquired);
    qemu_event_wait(context->release);
    if (context->result == RISCV_AME_ACQUIRE_GRANTED) {
        g_assert_true(riscv_ame_release(context->state, context->hart));
    }
    return NULL;
}

static void test_concurrent_try(void)
{
    enum { NUM_SLOTS = 4, NUM_HARTS = 16 };
    RISCVAMEBackendPoolCore pool = { 0 };
    RISCVAMEState states[NUM_HARTS] = { 0 };
    Object *harts[NUM_HARTS];
    AmeConcurrentAcquire contexts[NUM_HARTS];
    QemuThread threads[NUM_HARTS];
    QemuEvent start;
    QemuEvent release;
    QemuSemaphore acquired;
    uint64_t occupied = 0;
    unsigned granted = 0;
    size_t i;

    g_assert_true(riscv_ame_pool_init(
        &pool, NUM_SLOTS, true, &error_abort));
    configure_small_pool(&pool);
    g_assert_true(riscv_ame_pool_realize(&pool, &error_abort));
    qemu_event_init(&start, false);
    qemu_event_init(&release, false);
    qemu_sem_init(&acquired, 0);

    for (i = 0; i < NUM_HARTS; i++) {
        harts[i] = new_test_hart();
        g_assert_true(riscv_ame_state_attach(&states[i], &pool,
                                             &error_abort));
        contexts[i] = (AmeConcurrentAcquire) {
            .state = &states[i],
            .hart = harts[i],
            .start = &start,
            .release = &release,
            .acquired = &acquired,
            .pattern = i + 1,
        };
        qemu_thread_create(&threads[i], "ame-acquire",
                           concurrent_acquire_thread, &contexts[i],
                           QEMU_THREAD_JOINABLE);
    }
    qemu_event_set(&start);
    for (i = 0; i < NUM_HARTS; i++) {
        qemu_sem_wait(&acquired);
    }

    for (i = 0; i < NUM_HARTS; i++) {
        if (contexts[i].result == RISCV_AME_ACQUIRE_GRANTED) {
            unsigned slot = states[i].active_slot;

            g_assert_true(contexts[i].new_grant);
            g_assert_cmpuint(slot, <, NUM_SLOTS);
            g_assert_false(occupied & BIT_ULL(slot));
            occupied |= BIT_ULL(slot);
            g_assert_true(riscv_ame_state_owned(&states[i], harts[i]));
            for (size_t j = 0; j < pool.slots[slot].scratch_bytes; j++) {
                g_assert_cmphex(pool.slots[slot].scratch[j], ==, i + 1);
            }
            granted++;
        } else {
            g_assert_cmpint(contexts[i].result, ==,
                            RISCV_AME_ACQUIRE_BUSY);
            g_assert_false(contexts[i].new_grant);
        }
    }
    g_assert_cmpuint(granted, ==, NUM_SLOTS);
    g_assert_cmphex(occupied, ==, MAKE_64BIT_MASK(0, NUM_SLOTS));

    qemu_event_set(&release);
    for (i = 0; i < NUM_HARTS; i++) {
        qemu_thread_join(&threads[i]);
        riscv_ame_state_destroy(&states[i]);
        object_unref(harts[i]);
    }
    for (i = 0; i < NUM_SLOTS; i++) {
        g_assert_cmpint(pool.slots[i].state, ==, RISCV_AME_BACKEND_FREE);
        g_assert_null(pool.slots[i].owner);
    }

    qemu_sem_destroy(&acquired);
    qemu_event_destroy(&release);
    qemu_event_destroy(&start);
    riscv_ame_pool_destroy(&pool);
}

typedef struct AmeLockHolder {
    RISCVAMEBackendPoolCore *pool;
    QemuEvent *locked;
    QemuEvent *release;
} AmeLockHolder;

static void *hold_pool_lock_thread(void *opaque)
{
    AmeLockHolder *context = opaque;

    qemu_mutex_lock(&context->pool->lock);
    qemu_event_set(context->locked);
    qemu_event_wait(context->release);
    qemu_mutex_unlock(&context->pool->lock);
    return NULL;
}

static void test_owned_access_is_lockless(void)
{
    RISCVAMEState state = { 0 };
    Object *hart = new_test_hart();
    RISCVAMEBackendSlot *slot;
    AmeLockHolder context;
    QemuEvent locked;
    QemuEvent release;
    QemuThread thread;
    bool new_grant;

    g_assert_true(riscv_ame_state_init(
        &state, 4, 8, 16, 1, AME_CAP_POLICY_EXPLICIT, maximum(), &error_abort));
    g_assert_cmpint(riscv_ame_acquire(
                        &state, hart, false, 0, 0, &new_grant), ==,
                    RISCV_AME_ACQUIRE_GRANTED);
    qemu_event_init(&locked, false);
    qemu_event_init(&release, false);
    context = (AmeLockHolder) {
        .pool = state.pool,
        .locked = &locked,
        .release = &release,
    };
    qemu_thread_create(&thread, "ame-pool-lock", hold_pool_lock_thread,
                       &context, QEMU_THREAD_JOINABLE);
    qemu_event_wait(&locked);

    slot = riscv_ame_state_backend(&state, hart);
    g_assert_nonnull(slot);
    slot->mregs[0] = 0xa5;

    qemu_event_set(&release);
    qemu_thread_join(&thread);
    qemu_event_destroy(&release);
    qemu_event_destroy(&locked);
    riscv_ame_state_destroy(&state);
    object_unref(hart);
}

static void test_waiter_lifetime(void)
{
    RISCVAMEBackendPoolCore pool = { 0 };
    RISCVAMEState owner_state = { 0 };
    RISCVAMEState waiter_state = { 0 };
    Object *owner = new_test_hart();
    Object *waiter = new_test_hart();
    bool new_grant;

    ame_test_hart_finalize_count = 0;
    g_assert_true(riscv_ame_pool_init(&pool, 1, true, &error_abort));
    configure_small_pool(&pool);
    g_assert_true(riscv_ame_pool_realize(&pool, &error_abort));
    g_assert_true(riscv_ame_state_attach(&owner_state, &pool, &error_abort));
    g_assert_true(riscv_ame_state_attach(&waiter_state, &pool, &error_abort));
    g_assert_cmpint(riscv_ame_acquire(
                        &owner_state, owner, false, 0, 0, &new_grant), ==,
                    RISCV_AME_ACQUIRE_GRANTED);
    g_assert_cmpint(riscv_ame_acquire(
                        &waiter_state, waiter, true, 1, SCALE_MS,
                        &new_grant), ==, RISCV_AME_ACQUIRE_WAIT);

    object_unref(waiter);
    g_assert_cmpuint(ame_test_hart_finalize_count, ==, 0);
    riscv_ame_state_destroy(&waiter_state);
    g_assert_cmpuint(ame_test_hart_finalize_count, ==, 1);
    g_assert_true(QTAILQ_EMPTY(&pool.waiters));
    riscv_ame_state_destroy(&owner_state);
    object_unref(owner);
    g_assert_cmpuint(ame_test_hart_finalize_count, ==, 2);
    riscv_ame_pool_destroy(&pool);
}

int main(int argc, char **argv)
{
    qemu_init_main_loop(&error_abort);
    qemu_clock_enable(QEMU_CLOCK_VIRTUAL, true);
    g_test_init(&argc, &argv, NULL);
    type_register_static(&ame_test_hart_type_info);

    g_test_add_func("/riscv/ame/backend/configuration",
                    test_pool_configuration);
    g_test_add_func("/riscv/ame/backend/capability-authority",
                    test_pool_capability_authority);
    g_test_add_func("/riscv/ame/backend/fp-only", test_fp_only_backend);
    g_test_add_func("/riscv/ame/backend/failure-rollback",
                    test_pool_failure_rollback);
    g_test_add_func("/riscv/ame/backend/slots-reset",
                    test_pool_slots_and_reset);
    g_test_add_func("/riscv/ame/backend/private-state",
                    test_private_state_lifecycle);
    g_test_add_func("/riscv/ame/backend/shared-state",
                    test_shared_state_lifecycle);
    g_test_add_func("/riscv/ame/backend/slot-scratch",
                    test_slot_scratch_lifecycle);
    g_test_add_func("/riscv/ame/backend/acquire-descriptor",
                    test_acquire_descriptor);
    g_test_add_func("/riscv/ame/backend/try", test_try_acquire_release);
    g_test_add_func("/riscv/ame/backend/bounded-wait", test_bounded_wait);
    g_test_add_func("/riscv/ame/backend/wait-restore", test_wait_restore);
    g_test_add_func("/riscv/ame/backend/wait-restore-boundaries",
                    test_wait_restore_boundaries);
    g_test_add_func("/riscv/ame/backend/concurrent-try", test_concurrent_try);
    g_test_add_func("/riscv/ame/backend/lockless-access",
                    test_owned_access_is_lockless);
    g_test_add_func("/riscv/ame/backend/waiter-lifetime",
                    test_waiter_lifetime);

    return g_test_run();
}
