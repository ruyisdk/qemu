/*
 * RISC-V Attached Matrix Extension (Ztt) provisional CSR definitions
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef TESTS_TCG_RISCV_ZTT_CSR_H
#define TESTS_TCG_RISCV_ZTT_CSR_H

/* Provisional CSR addresses shared by user-mode and system-mode tests. */
#define CSR_AMESTATUS          0x800
#define CSR_AMESTYPE           0x810
#define CSR_AMEFFLAGS          0x811
#define CSR_AMEXSAT            0x812
#define CSR_AMENLEN            0xcc0
#define CSR_AMEUDSZ            0xcc1
#define CSR_AMEOWN             0xcc3
#define CSR_FFLAGS             0x001

/* FS enables scalar floating point; MS controls AME instruction access. */
#define MSTATUS_FS_INITIAL     (1 << 13)
#define MSTATUS_AME_MS_SHIFT   25
#define MSTATUS_AME_MS_MASK    (3 << MSTATUS_AME_MS_SHIFT)
#define MSTATUS_AME_MS_INITIAL (1 << MSTATUS_AME_MS_SHIFT)
#define MSTATUS_AME_MS_CLEAN   (2 << MSTATUS_AME_MS_SHIFT)
#define MSTATUS_AME_MS_DIRTY   (3 << MSTATUS_AME_MS_SHIFT)

#define AME_FFLAGS_MASK        0x1f
#define AME_FFLAG_NX           0x01
#define AME_FFLAG_UF           0x02
#define AME_FFLAG_OF           0x04
#define AME_FFLAG_DZ           0x08
#define AME_FFLAG_NV           0x10
#define AME_XSAT_SAT           1
#define AME_STATUS_UN          1

/* ame.acquire result: ownership in bit 0, failure reason starting at bit 1. */
#define AME_ACQUIRE_GRANTED    1
#define AME_ACQUIRE_BUSY       (1 << 1)
#define AME_ACQUIRE_UNSUPPORTED (2 << 1)
#define AME_ACQUIRE_DISABLED   (3 << 1)
#define AME_ACQUIRE_BAD_DESC   (4 << 1)
#define AME_ACQUIRE_TIMEOUT    (5 << 1)
#define AME_ACQUIRE_INTERRUPTED (6 << 1)

/* Integer-register numbers used in instruction words, not register values. */
#define ZTT_R_T0               5
#define ZTT_R_T1               6
#define ZTT_R_T2               7
#define ZTT_R_A0               10
#define ZTT_R_A1               11
#define ZTT_R_S3               19
#define ZTT_R_S4               20
#define ZTT_R_S5               21

/*
 * Insert register numbers into an instruction word whose operand fields are
 * zero. Arguments name encoding fields: an AME instruction may use those
 * fields for M/Acc registers rather than integer registers. ZIP has no rd.
 * Tests emit these words with .word so no AME-aware assembler is required.
 */
#define ZTT_ENCODE_R1(base, rd) \
    ((base) | ((rd) << 7))
#define ZTT_ENCODE_R2(base, rd, rs1) \
    (ZTT_ENCODE_R1(base, rd) | ((rs1) << 15))
#define ZTT_ENCODE_R3(base, rd, rs1, rs2) \
    (ZTT_ENCODE_R2(base, rd, rs1) | ((rs2) << 20))
#define ZTT_ENCODE_ZIP(base, rs1, rs2) \
    ((base) | ((rs1) << 15) | ((rs2) << 20))

#define ZTT_AME_ACQUIRE(rd, rs) ZTT_ENCODE_R2(0xa490002b, rd, rs)
#define ZTT_AME_RELEASE         0xa4a0002b
#define ZTT_AGETTYP(rd, ad)     ZTT_ENCODE_R2(0xa200002b, rd, ad)
#define ZTT_ASETTYP(ad, rs)     ZTT_ENCODE_R2(0xa210002b, ad, rs)
#define ZTT_MGETTYP(rd, ms)     ZTT_ENCODE_R2(0xa220002b, rd, ms)
#define ZTT_MSETTYP(md, rs)     ZTT_ENCODE_R2(0xa230002b, md, rs)
#define ZTT_MMOV_M_A(md, acc)   ZTT_ENCODE_R2(0xa3c0002b, md, acc)
#define ZTT_MMOV_A_M(acc, ms)   ZTT_ENCODE_R2(0xa3d0002b, acc, ms)
#define ZTT_MMOV_M_M(md, ms)    ZTT_ENCODE_R2(0xa3e0002b, md, ms)
#define ZTT_MCONV_EW(md, ms1)    ZTT_ENCODE_R2(0xa2d0002b, md, ms1)
#define ZTT_MLS_1R(md, rs1)     ZTT_ENCODE_R2(0xa360002b, md, rs1)
#define ZTT_MLS_CM(md, rs1)     ZTT_ENCODE_R2(0xa370002b, md, rs1)
#define ZTT_MLS_RM(md, rs1)     ZTT_ENCODE_R2(0xa380002b, md, rs1)
#define ZTT_MLS_ST(md, rs1, rs2) \
    ZTT_ENCODE_R3(0xa800002b, md, rs1, rs2)
#define ZTT_MLS_TST(md, rs1, rs2) \
    ZTT_ENCODE_R3(0xaa00002b, md, rs1, rs2)
#define ZTT_MSS_1R(ms, rs1)     ZTT_ENCODE_R2(0xa390002b, ms, rs1)
#define ZTT_MSS_CM(ms, rs1)     ZTT_ENCODE_R2(0xa3a0002b, ms, rs1)
#define ZTT_MSS_RM(ms, rs1)     ZTT_ENCODE_R2(0xa3b0002b, ms, rs1)
#define ZTT_MSS_ST(ms, rs1, rs2) \
    ZTT_ENCODE_R3(0xac00002b, ms, rs1, rs2)
#define ZTT_MSS_TST(ms, rs1, rs2) \
    ZTT_ENCODE_R3(0xae00002b, ms, rs1, rs2)
#define ZTT_MMOVE8_M_X(md, rs2, rs1) \
    ZTT_ENCODE_R3(0xb200002b, md, rs2, rs1)
#define ZTT_MMOVE16_M_X(md, rs2, rs1) \
    ZTT_ENCODE_R3(0xb400002b, md, rs2, rs1)
#define ZTT_MMOVE32_M_X(md, rs2, rs1) \
    ZTT_ENCODE_R3(0xb600002b, md, rs2, rs1)
#define ZTT_MMOVE64_M_X(md, rs2, rs1) \
    ZTT_ENCODE_R3(0xb800002b, md, rs2, rs1)
#define ZTT_MMOVE8_X_M(rd, ms2, rs1) \
    ZTT_ENCODE_R3(0xba00002b, rd, ms2, rs1)
#define ZTT_MMOVE16_X_M(rd, ms2, rs1) \
    ZTT_ENCODE_R3(0xbc00002b, rd, ms2, rs1)
#define ZTT_MMOVE32_X_M(rd, ms2, rs1) \
    ZTT_ENCODE_R3(0xbe00002b, rd, ms2, rs1)
#define ZTT_MMOVE64_X_M(rd, ms2, rs1) \
    ZTT_ENCODE_R3(0xc000002b, rd, ms2, rs1)
#define ZTT_MABS_EW(md, ms1)    ZTT_ENCODE_R2(0xa240002b, md, ms1)
#define ZTT_MFRINTM_EW(md, ms1) ZTT_ENCODE_R2(0xa250002b, md, ms1)
#define ZTT_MFRINTN_EW(md, ms1) ZTT_ENCODE_R2(0xa260002b, md, ms1)
#define ZTT_MFRINTP_EW(md, ms1) ZTT_ENCODE_R2(0xa270002b, md, ms1)
#define ZTT_MFRINTZ_EW(md, ms1) ZTT_ENCODE_R2(0xa280002b, md, ms1)
#define ZTT_MREC_EW(md, ms1)    ZTT_ENCODE_R2(0xa310002b, md, ms1)
#define ZTT_MSQRT_EW(md, ms1)   ZTT_ENCODE_R2(0xa340002b, md, ms1)
#define ZTT_MADD_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x0400002b, md, ms1, ms2)
#define ZTT_MADD_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x0600002b, md, rs1, ms2)
#define ZTT_MABSDIFF_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x0000002b, md, ms1, ms2)
#define ZTT_MABSDIFF_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x0200002b, md, rs1, ms2)
#define ZTT_MAND_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x3400002b, md, ms1, ms2)
#define ZTT_MAND_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x3600002b, md, rs1, ms2)
#define ZTT_MANDNOT_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x3800002b, md, ms1, ms2)
#define ZTT_MANDNOT_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x3a00002b, md, rs1, ms2)
#define ZTT_MCMOVGE_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x5400002b, md, ms1, ms2)
#define ZTT_MCMOVLT_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x5600002b, md, ms1, ms2)
#define ZTT_MCMPGE_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x5800002b, md, ms1, ms2)
#define ZTT_MCMPGE_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x5a00002b, md, rs1, ms2)
#define ZTT_MCMPLT_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x5c00002b, md, ms1, ms2)
#define ZTT_MCMPLT_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x5e00002b, md, rs1, ms2)
#define ZTT_MHDIFF_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x0800002b, md, ms1, ms2)
#define ZTT_MHDIFF_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x0a00002b, md, rs1, ms2)
#define ZTT_MLDEXP_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x7e00002b, md, ms1, ms2)
#define ZTT_MLDEXP_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x8000002b, md, rs1, ms2)
#define ZTT_MLDEXPACC_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x8200002b, md, ms1, ms2)
#define ZTT_MLDEXPACC_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x8400002b, md, rs1, ms2)
#define ZTT_MRDEXP_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x8a00002b, md, ms1, ms2)
#define ZTT_MRDEXPACC_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x8c00002b, md, ms1, ms2)
#define ZTT_MMAX_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x0c00002b, md, ms1, ms2)
#define ZTT_MMAX_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x0e00002b, md, rs1, ms2)
#define ZTT_MMEAN_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x1000002b, md, ms1, ms2)
#define ZTT_MMEAN_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x1200002b, md, rs1, ms2)
#define ZTT_MMIN_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x1400002b, md, ms1, ms2)
#define ZTT_MMIN_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x1600002b, md, rs1, ms2)
#define ZTT_MMUL_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x1800002b, md, ms1, ms2)
#define ZTT_MMUL_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x1a00002b, md, rs1, ms2)
#define ZTT_MMULACC_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x1c00002b, md, ms1, ms2)
#define ZTT_MMULACC_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x1e00002b, md, rs1, ms2)
#define ZTT_MMULACCNEG_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x2000002b, md, ms1, ms2)
#define ZTT_MMULACCNEG_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x2200002b, md, rs1, ms2)
#define ZTT_MMULADD_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x2400002b, md, ms1, ms2)
#define ZTT_MMULADD_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x2600002b, md, rs1, ms2)
#define ZTT_MMULNEG_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x2800002b, md, ms1, ms2)
#define ZTT_MMULNEG_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x2a00002b, md, rs1, ms2)
#define ZTT_MMULSUB_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x2c00002b, md, ms1, ms2)
#define ZTT_MMULSUB_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x2e00002b, md, rs1, ms2)
#define ZTT_MMULACC_2D(acc, ms1, ms2) \
    ZTT_ENCODE_R3(0x9400002b, acc, ms1, ms2)
#define ZTT_MMULACCNEG_2D(acc, ms1, ms2) \
    ZTT_ENCODE_R3(0x9600002b, acc, ms1, ms2)
#define ZTT_MMULATACC_2D(acc, ms1, ms2) \
    ZTT_ENCODE_R3(0x9a00002b, acc, ms1, ms2)
#define ZTT_MMULATACCNEG_2D(acc, ms1, ms2) \
    ZTT_ENCODE_R3(0x9c00002b, acc, ms1, ms2)
#define ZTT_MMULBTACC_2D(acc, ms1, ms2) \
    ZTT_ENCODE_R3(0x9e00002b, acc, ms1, ms2)
#define ZTT_MMULBTACCNEG_2D(acc, ms1, ms2) \
    ZTT_ENCODE_R3(0xa000002b, acc, ms1, ms2)
#define ZTT_MOR_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x3c00002b, md, ms1, ms2)
#define ZTT_MOR_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x3e00002b, md, rs1, ms2)
#define ZTT_MORNOT_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x4000002b, md, ms1, ms2)
#define ZTT_MORNOT_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x4200002b, md, rs1, ms2)
#define ZTT_MSELGE_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x6000002b, md, ms1, ms2)
#define ZTT_MSELLT_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x6200002b, md, ms1, ms2)
#define ZTT_MSLL_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x4400002b, md, ms1, ms2)
#define ZTT_MSLL_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x4600002b, md, rs1, ms2)
#define ZTT_MSRA_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x4800002b, md, ms1, ms2)
#define ZTT_MSRA_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x4a00002b, md, rs1, ms2)
#define ZTT_MSRL_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x4c00002b, md, ms1, ms2)
#define ZTT_MSRL_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x4e00002b, md, rs1, ms2)
#define ZTT_MSUB_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x3000002b, md, ms1, ms2)
#define ZTT_MSUB_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x3200002b, md, rs1, ms2)
#define ZTT_MXOR_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x5000002b, md, ms1, ms2)
#define ZTT_MXOR_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x5200002b, md, rs1, ms2)
#define ZTT_MCOS_EW(md, ms1)    ZTT_ENCODE_R2(0xa2e0002b, md, ms1)
#define ZTT_MEXP2_EW(md, ms1)   ZTT_ENCODE_R2(0xa2f0002b, md, ms1)
#define ZTT_MLOG2_EW(md, ms1)   ZTT_ENCODE_R2(0xa300002b, md, ms1)
#define ZTT_MRSQRT_EW(md, ms1)  ZTT_ENCODE_R2(0xa320002b, md, ms1)
#define ZTT_MSIN_EW(md, ms1)    ZTT_ENCODE_R2(0xa330002b, md, ms1)
#define ZTT_MTANH_EW(md, ms1)   ZTT_ENCODE_R2(0xa350002b, md, ms1)
#define ZTT_MLOG2SUB_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x8600002b, md, ms1, ms2)
#define ZTT_MLOG2SUB_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x8800002b, md, rs1, ms2)
#define ZTT_MSUBLOG2_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x8e00002b, md, ms1, ms2)
#define ZTT_MSUBLOG2_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x9000002b, md, rs1, ms2)
#define ZTT_MCOLBCAST_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x6400002b, md, rs1, ms2)
#define ZTT_MCOLGATHER_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x6600002b, md, ms1, ms2)
#define ZTT_MCOLID_EW(md)       ZTT_ENCODE_R1(0xa601002b, md)
#define ZTT_MCOLSHIFT_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x6800002b, md, rs1, ms2)
#define ZTT_MCOLUNZIP_EW(ms1, ms2) \
    ZTT_ENCODE_ZIP(0xc200002b, ms1, ms2)
#define ZTT_MCOLZIP_EW(ms1, ms2) \
    ZTT_ENCODE_ZIP(0xc400002b, ms1, ms2)
#define ZTT_MBCAST_M_X(md, rs1, rs2) \
    ZTT_ENCODE_R3(0xb000002b, md, rs1, rs2)
#define ZTT_MPACK_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x7a00002b, md, rs1, ms2)
#define ZTT_MROWBCAST_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x6a00002b, md, rs1, ms2)
#define ZTT_MROWGATHER_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x6c00002b, md, ms1, ms2)
#define ZTT_MROWID_EW(md)       ZTT_ENCODE_R1(0xa601802b, md)
#define ZTT_MROWSHIFT_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x6e00002b, md, rs1, ms2)
#define ZTT_MROWUNZIP_EW(ms1, ms2) \
    ZTT_ENCODE_ZIP(0xc600002b, ms1, ms2)
#define ZTT_MROWZIP_EW(ms1, ms2) \
    ZTT_ENCODE_ZIP(0x7000002b, ms1, ms2)
#define ZTT_MUNPACK_EW_X(md, rs1, ms2) \
    ZTT_ENCODE_R3(0x7c00002b, md, rs1, ms2)
#define ZTT_MPREFIXADD_COL(md, ms1) ZTT_ENCODE_R2(0xa3f0002b, md, ms1)
#define ZTT_MPREFIXADD_ROW(md, ms1) ZTT_ENCODE_R2(0xa400002b, md, ms1)
#define ZTT_MPREFIXMAX_COL(md, ms1) ZTT_ENCODE_R2(0xa410002b, md, ms1)
#define ZTT_MPREFIXMAX_ROW(md, ms1) ZTT_ENCODE_R2(0xa420002b, md, ms1)
#define ZTT_MREDUCEADD_COL(md, ms1) ZTT_ENCODE_R2(0xa430002b, md, ms1)
#define ZTT_MREDUCEADD_ROW(md, ms1) ZTT_ENCODE_R2(0xa440002b, md, ms1)
#define ZTT_MREDUCEMAX_COL(md, ms1) ZTT_ENCODE_R2(0xa450002b, md, ms1)
#define ZTT_MREDUCEMAX_ROW(md, ms1) ZTT_ENCODE_R2(0xa460002b, md, ms1)
#define ZTT_MREDUCEMIN_COL(md, ms1) ZTT_ENCODE_R2(0xa470002b, md, ms1)
#define ZTT_MREDUCEMIN_ROW(md, ms1) ZTT_ENCODE_R2(0xa480002b, md, ms1)
#define ZTT_MROWSCATADD_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x7200002b, md, ms1, ms2)
#define ZTT_MCOLSCATADD_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x7400002b, md, ms1, ms2)
#define ZTT_MROWSCATMAX_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x7600002b, md, ms1, ms2)
#define ZTT_MCOLSCATMAX_EW(md, ms1, ms2) \
    ZTT_ENCODE_R3(0x7800002b, md, ms1, ms2)
#define ZTT_MZERO_2D_ACC(ad)    ZTT_ENCODE_R1(0xa600002b, ad)
#define ZTT_MZERO_ACC(ad)       ZTT_MZERO_2D_ACC(ad)
#define ZTT_MZERO_M(md)         ZTT_ENCODE_R1(0xa600802b, md)
#endif /* TESTS_TCG_RISCV_ZTT_CSR_H */
