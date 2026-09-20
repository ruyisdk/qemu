/*
 * RISC-V Attached Matrix Extension GDB register access
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "exec/gdbstub.h"
#include "gdbstub/helpers.h"
#include "target/riscv/cpu.h"
#include "target/riscv/ame/gdbstub.h"

static uint32_t riscv_ame_gdb_ldl(const uint8_t *buf)
{
    return (target_big_endian() ? ldl_be_p : ldl_le_p)(buf);
}

bool riscv_ame_gdb_is_csr(int csrno)
{
    switch (csrno) {
    case CSR_AMESTATUS:
    case CSR_AMESTYPE:
    case CSR_AMEFFLAGS:
    case CSR_AMEXSAT:
    case CSR_AMENLEN:
    case CSR_AMEUDSZ:
    case CSR_AMEOWN:
        return true;
    default:
        return false;
    }
}

enum {
    RISCV_AME_GDB_CSR_STATUS,
    RISCV_AME_GDB_CSR_AMESTYPE,
    RISCV_AME_GDB_CSR_FFLAGS,
    RISCV_AME_GDB_CSR_XSAT,
    RISCV_AME_GDB_CSR_NLEN,
    RISCV_AME_GDB_CSR_UDSZ,
    RISCV_AME_GDB_CSR_OWN,
    RISCV_AME_GDB_CFG_MAX_INT_DTYPE,
};

#define RISCV_AME_GDB_CHUNK_BYTES (64 * 1024)

static unsigned riscv_ame_gdb_chunks(size_t bytes)
{
    return DIV_ROUND_UP(bytes, RISCV_AME_GDB_CHUNK_BYTES);
}

static size_t riscv_ame_gdb_chunk_bytes(size_t bytes)
{
    g_assert(bytes > 0);
    g_assert(bytes <= RISCV_AME_GDB_CHUNK_BYTES ||
             bytes % RISCV_AME_GDB_CHUNK_BYTES == 0);
    return MIN(bytes, RISCV_AME_GDB_CHUNK_BYTES);
}

static unsigned riscv_ame_gdb_md_base(const RISCVAMELayout *layout)
{
    return layout->num_mregs * riscv_ame_gdb_chunks(layout->mreg_bytes);
}

static unsigned riscv_ame_gdb_acc_base(const RISCVAMELayout *layout)
{
    return riscv_ame_gdb_md_base(layout) + layout->num_mregs;
}

static unsigned riscv_ame_gdb_ad_base(const RISCVAMELayout *layout)
{
    return riscv_ame_gdb_acc_base(layout) + layout->num_accregs *
           riscv_ame_gdb_chunks(layout->acc_bytes);
}

static unsigned riscv_ame_gdb_state_base(const RISCVAMELayout *layout)
{
    return riscv_ame_gdb_ad_base(layout) + layout->num_accregs;
}

static int riscv_gdb_get_ame(CPUState *cs, GByteArray *buf, int n)
{
    RISCVCPU *cpu = RISCV_CPU(cs);
    RISCVAMEState *ame = &cpu->ame;
    const RISCVAMELayout *layout = riscv_ame_state_layout(ame);
    const RISCVAMEBackendSlot *slot = riscv_ame_state_backend_const(
        ame, OBJECT(cpu));
    unsigned index, chunk, chunks;
    size_t chunk_bytes;

    g_assert(layout != NULL);
    if (n < riscv_ame_gdb_md_base(layout)) {
        chunks = riscv_ame_gdb_chunks(layout->mreg_bytes);
        chunk_bytes = riscv_ame_gdb_chunk_bytes(layout->mreg_bytes);
        index = n / chunks;
        chunk = n % chunks;
        if (slot == NULL) {
            return gdb_get_zeroes(buf, chunk_bytes);
        }
        g_byte_array_append(
            buf, slot->mregs + index * layout->mreg_bytes +
                 chunk * chunk_bytes, chunk_bytes);
        return chunk_bytes;
    }
    if (n < riscv_ame_gdb_acc_base(layout)) {
        index = n - riscv_ame_gdb_md_base(layout);
        return gdb_get_reg32(buf, slot != NULL ? slot->md[index] : 0);
    }
    if (n < riscv_ame_gdb_ad_base(layout)) {
        n -= riscv_ame_gdb_acc_base(layout);
        chunks = riscv_ame_gdb_chunks(layout->acc_bytes);
        chunk_bytes = riscv_ame_gdb_chunk_bytes(layout->acc_bytes);
        index = n / chunks;
        chunk = n % chunks;
        if (slot == NULL) {
            return gdb_get_zeroes(buf, chunk_bytes);
        }
        g_byte_array_append(
            buf, slot->accs + index * layout->acc_bytes +
                 chunk * chunk_bytes, chunk_bytes);
        return chunk_bytes;
    }
    if (n < riscv_ame_gdb_state_base(layout)) {
        index = n - riscv_ame_gdb_ad_base(layout);
        return gdb_get_reg32(buf, slot != NULL ? slot->ad[index] : 0);
    }

    switch (n - riscv_ame_gdb_state_base(layout)) {
    case RISCV_AME_GDB_CSR_STATUS:
        return gdb_get_reg32(buf, slot != NULL ?
                             slot->status & AME_STATUS_UN : 0);
    case RISCV_AME_GDB_CSR_AMESTYPE:
        return gdb_get_reg32(buf, slot != NULL ? slot->scalar_dtype : 0);
    case RISCV_AME_GDB_CSR_FFLAGS:
        return gdb_get_reg32(buf, cpu->env.amefflags & AME_FFLAGS_MASK);
    case RISCV_AME_GDB_CSR_XSAT:
        return gdb_get_reg32(buf, cpu->env.amexsat);
    case RISCV_AME_GDB_CSR_NLEN:
        return gdb_get_reg32(buf, layout->n);
    case RISCV_AME_GDB_CSR_UDSZ:
        return gdb_get_reg32(buf, layout->u);
    case RISCV_AME_GDB_CSR_OWN:
        return gdb_get_reg32(buf, slot != NULL);
    case RISCV_AME_GDB_CFG_MAX_INT_DTYPE:
        return gdb_get_reg32(buf, ame->pool->capabilities.max_int_dtype);
    default:
        return 0;
    }
}

static bool riscv_ame_gdb_set_dtype(RISCVCPU *cpu, uint32_t raw,
                                    unsigned index, bool accumulator)
{
    RISCVAMEState *ame = &cpu->ame;
    const RISCVAMELayout *layout = riscv_ame_state_layout(ame);
    RISCVAMEBackendSlot *slot = riscv_ame_state_backend(ame, OBJECT(cpu));
    const AmeDTypeDesc *dtype;
    uint16_t id = AME_DTYPE_ID_INVALID;
    uint32_t group_size;

    if (slot == NULL) {
        return false;
    }
    if (raw != 0) {
        dtype = riscv_ame_dtype_by_encoding(raw);
        if (dtype == NULL ||
            !riscv_ame_dtype_set_has(&ame->pool->capabilities.dtypes,
                                     dtype->id)) {
            return false;
        }
        id = dtype->id;
        if (!accumulator) {
            group_size = DIV_ROUND_UP(dtype->bits, layout->u);
            if (index % group_size != 0 ||
                group_size > layout->num_mregs - index) {
                return false;
            }
        }
    }

    if (accumulator) {
        slot->ad[index] = raw;
        slot->ad_dtype_id[index] = id;
    } else {
        slot->md[index] = raw;
        slot->md_dtype_id[index] = id;
    }
    riscv_ame_mark_dirty(&cpu->env);
    return true;
}

static int riscv_gdb_set_ame(CPUState *cs, uint8_t *mem_buf, int n)
{
    RISCVCPU *cpu = RISCV_CPU(cs);
    CPURISCVState *env = &cpu->env;
    RISCVAMEState *ame = &cpu->ame;
    const RISCVAMELayout *layout = riscv_ame_state_layout(ame);
    RISCVAMEBackendSlot *slot = riscv_ame_state_backend(ame, OBJECT(cpu));
    unsigned index, chunk, chunks;
    size_t chunk_bytes;
    uint32_t value;

    g_assert(layout != NULL);
    if (n < riscv_ame_gdb_md_base(layout)) {
        if (slot == NULL) {
            return 0;
        }
        chunks = riscv_ame_gdb_chunks(layout->mreg_bytes);
        chunk_bytes = riscv_ame_gdb_chunk_bytes(layout->mreg_bytes);
        index = n / chunks;
        chunk = n % chunks;
        memcpy(slot->mregs + index * layout->mreg_bytes +
               chunk * chunk_bytes, mem_buf, chunk_bytes);
        riscv_ame_mark_dirty(env);
        return chunk_bytes;
    }
    if (n < riscv_ame_gdb_acc_base(layout)) {
        index = n - riscv_ame_gdb_md_base(layout);
        value = riscv_ame_gdb_ldl(mem_buf);
        return riscv_ame_gdb_set_dtype(cpu, value, index, false) ?
               sizeof(value) : 0;
    }
    if (n < riscv_ame_gdb_ad_base(layout)) {
        if (slot == NULL) {
            return 0;
        }
        n -= riscv_ame_gdb_acc_base(layout);
        chunks = riscv_ame_gdb_chunks(layout->acc_bytes);
        chunk_bytes = riscv_ame_gdb_chunk_bytes(layout->acc_bytes);
        index = n / chunks;
        chunk = n % chunks;
        memcpy(slot->accs + index * layout->acc_bytes +
               chunk * chunk_bytes, mem_buf, chunk_bytes);
        riscv_ame_mark_dirty(env);
        return chunk_bytes;
    }
    if (n < riscv_ame_gdb_state_base(layout)) {
        index = n - riscv_ame_gdb_ad_base(layout);
        value = riscv_ame_gdb_ldl(mem_buf);
        return riscv_ame_gdb_set_dtype(cpu, value, index, true) ?
               sizeof(value) : 0;
    }

    value = riscv_ame_gdb_ldl(mem_buf);
    switch (n - riscv_ame_gdb_state_base(layout)) {
    case RISCV_AME_GDB_CSR_STATUS:
        if (slot == NULL) {
            return 0;
        }
        slot->status &= value & AME_STATUS_UN;
        break;
    case RISCV_AME_GDB_CSR_AMESTYPE: {
        const AmeDTypeDesc *dtype = riscv_ame_dtype_by_encoding(value);

        if (slot == NULL) {
            return 0;
        }
        slot->scalar_dtype = value;
        slot->scalar_dtype_id =
            dtype != NULL && riscv_ame_dtype_set_has(
                                 &ame->pool->capabilities.dtypes, dtype->id) ?
            dtype->id : AME_DTYPE_ID_INVALID;
        break;
    }
    case RISCV_AME_GDB_CSR_FFLAGS:
        env->amefflags = value & AME_FFLAGS_MASK;
        break;
    case RISCV_AME_GDB_CSR_XSAT:
        env->amexsat = (value & AME_XSAT_SAT) != 0;
        break;
    case RISCV_AME_GDB_CSR_NLEN:
    case RISCV_AME_GDB_CSR_UDSZ:
    case RISCV_AME_GDB_CSR_OWN:
    case RISCV_AME_GDB_CFG_MAX_INT_DTYPE:
        return sizeof(value);
    default:
        return 0;
    }
    riscv_ame_mark_dirty(env);
    return sizeof(value);
}

static GDBFeature *riscv_gen_dynamic_ame_feature(CPUState *cs, int base_reg)
{
    RISCVCPU *cpu = RISCV_CPU(cs);
    RISCVAMEState *ame = &cpu->ame;
    const RISCVAMELayout *layout = riscv_ame_state_layout(ame);
    GDBFeatureBuilder builder;
    unsigned i, j, m_chunks, acc_chunks;
    size_t m_chunk_bytes, acc_chunk_bytes;

    gdb_feature_builder_init(&builder, &cpu->dyn_ame_feature,
                             "org.gnu.gdb.riscv.ame", "riscv-ame.xml",
                             base_reg);
    g_assert(layout != NULL);
    m_chunks = riscv_ame_gdb_chunks(layout->mreg_bytes);
    acc_chunks = riscv_ame_gdb_chunks(layout->acc_bytes);
    m_chunk_bytes = riscv_ame_gdb_chunk_bytes(layout->mreg_bytes);
    acc_chunk_bytes = riscv_ame_gdb_chunk_bytes(layout->acc_bytes);
    gdb_feature_builder_append_tag(
        &builder,
        "<vector id=\"ame_m_bytes\" type=\"uint8\" count=\"%zu\"/>",
        m_chunk_bytes);
    gdb_feature_builder_append_tag(
        &builder,
        "<union id=\"ame_m\"><field name=\"b\" type=\"ame_m_bytes\"/>"
        "</union>");
    gdb_feature_builder_append_tag(
        &builder,
        "<vector id=\"ame_acc_bytes\" type=\"uint8\" count=\"%zu\"/>",
        acc_chunk_bytes);
    gdb_feature_builder_append_tag(
        &builder,
        "<union id=\"ame_acc\"><field name=\"b\" type=\"ame_acc_bytes\"/>"
        "</union>");

    for (i = 0; i < layout->num_mregs; i++) {
        for (j = 0; j < m_chunks; j++) {
            char *name = m_chunks == 1 ? g_strdup_printf("m%d", i) :
                                         g_strdup_printf("m%d_%d", i, j);

            gdb_feature_builder_append_reg(
                &builder, name, m_chunk_bytes * 8, i * m_chunks + j,
                "ame_m", "matrix");
        }
    }
    for (i = 0; i < layout->num_mregs; i++) {
        gdb_feature_builder_append_reg(
            &builder, g_strdup_printf("md%d", i), 32,
            riscv_ame_gdb_md_base(layout) + i, "uint32", "matrix");
    }
    for (i = 0; i < layout->num_accregs; i++) {
        for (j = 0; j < acc_chunks; j++) {
            char *name = acc_chunks == 1 ? g_strdup_printf("acc%d", i) :
                                           g_strdup_printf("acc%d_%d", i, j);

            gdb_feature_builder_append_reg(
                &builder, name, acc_chunk_bytes * 8,
                riscv_ame_gdb_acc_base(layout) + i * acc_chunks + j,
                "ame_acc", "matrix");
        }
    }
    for (i = 0; i < layout->num_accregs; i++) {
        gdb_feature_builder_append_reg(
            &builder, g_strdup_printf("ad%d", i), 32,
            riscv_ame_gdb_ad_base(layout) + i, "uint32", "matrix");
    }

#define AME_GDB_STATE_REG(_name, _id) \
    gdb_feature_builder_append_reg( \
        &builder, g_strdup(_name), 32, \
        riscv_ame_gdb_state_base(layout) + (_id), \
        "uint32", "system")
    AME_GDB_STATE_REG("amestatus", RISCV_AME_GDB_CSR_STATUS);
    AME_GDB_STATE_REG("amestype", RISCV_AME_GDB_CSR_AMESTYPE);
    AME_GDB_STATE_REG("amefflags", RISCV_AME_GDB_CSR_FFLAGS);
    AME_GDB_STATE_REG("amexsat", RISCV_AME_GDB_CSR_XSAT);
    AME_GDB_STATE_REG("amenlen", RISCV_AME_GDB_CSR_NLEN);
    AME_GDB_STATE_REG("ameudsz", RISCV_AME_GDB_CSR_UDSZ);
    AME_GDB_STATE_REG("ameown", RISCV_AME_GDB_CSR_OWN);
    AME_GDB_STATE_REG("ame_max_int_dtype",
                      RISCV_AME_GDB_CFG_MAX_INT_DTYPE);
#undef AME_GDB_STATE_REG

    gdb_feature_builder_end(&builder);
    return &cpu->dyn_ame_feature;
}

void riscv_ame_register_gdb_regs(CPUState *cs)
{
    gdb_register_coprocessor(cs, riscv_gdb_get_ame, riscv_gdb_set_ame,
                             riscv_gen_dynamic_ame_feature(
                                 cs, cs->gdb_num_regs));
}

void riscv_ame_gdb_finalize(CPUState *cs)
{
    GDBFeature *feature = &RISCV_CPU(cs)->dyn_ame_feature;

    for (unsigned i = 0; i < feature->num_regs; i++) {
        g_free((void *)feature->regs[i]);
    }
    g_free((void *)feature->regs);
    g_free((void *)feature->xml);
    memset(feature, 0, sizeof(*feature));
}
