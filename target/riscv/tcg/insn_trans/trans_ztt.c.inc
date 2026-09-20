/*
 * RISC-V Attached Matrix Extension (Ztt) translation routines
 *
 * Copyright (c) 2026 Alibaba Group
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

static bool trans_ame_acquire(DisasContext *ctx, arg_ame_acquire *a)
{
    TCGv result = tcg_temp_new();

    gen_helper_ame_acquire(result, tcg_env, get_gpr(ctx, a->rs1, EXT_NONE));
    gen_set_gpr(ctx, a->rd, result);
    /* A bounded wait may halt and restart this instruction. */
    ctx->base.is_jmp = DISAS_TOO_MANY;
    return true;
}

static bool trans_ame_release(DisasContext *ctx, arg_ame_release *a)
{
    gen_helper_ame_release(tcg_env);
    return true;
}

#define TRANS_AME_GETTYP(name) \
static bool trans_##name(DisasContext *ctx, arg_ztt_r2 *a) \
{ \
    TCGv result = tcg_temp_new(); \
    gen_helper_ame_##name(result, tcg_env, tcg_constant_i32(a->rs1)); \
    gen_set_gpr(ctx, a->rd, result); \
    return true; \
}
TRANS_AME_GETTYP(mgettyp)
TRANS_AME_GETTYP(agettyp)
#undef TRANS_AME_GETTYP

#define TRANS_AME_SETTYP(name) \
static bool trans_##name(DisasContext *ctx, arg_ztt_r2 *a) \
{ \
    gen_helper_ame_##name(tcg_env, tcg_constant_i32(a->rd), \
                          get_gpr(ctx, a->rs1, EXT_NONE)); \
    return true; \
}
TRANS_AME_SETTYP(msettyp)
TRANS_AME_SETTYP(asettyp)
#undef TRANS_AME_SETTYP

#define TRANS_AME_MOVE(name, helper) \
static bool trans_##name(DisasContext *ctx, arg_ztt_r2 *a) \
{ \
    gen_helper_ame_##helper(tcg_env, tcg_constant_i32(a->rd), \
                            tcg_constant_i32(a->rs1)); \
    return true; \
}
TRANS_AME_MOVE(mmov_m_m, mov_m_m)
TRANS_AME_MOVE(mmov_m_a, mov_m_a)
TRANS_AME_MOVE(mmov_a_m, mov_a_m)
TRANS_AME_MOVE(mconv_ew, convert)
#undef TRANS_AME_MOVE

#define TRANS_AME_ZERO(name, helper) \
static bool trans_##name(DisasContext *ctx, arg_ztt_r1 *a) \
{ \
    gen_helper_ame_##helper(tcg_env, tcg_constant_i32(a->rd)); \
    return true; \
}
TRANS_AME_ZERO(mzero_2d_m, zero_m)
TRANS_AME_ZERO(mzero_2d_acc, zero_acc)
#undef TRANS_AME_ZERO

/* Operation and encoded indices are constants; datatype remains dynamic. */
#define TRANS_AME_R2(name, helper, op) \
static bool trans_##name(DisasContext *ctx, arg_ztt_r2 *a) \
{ \
    gen_helper_ame_##helper(tcg_env, tcg_constant_i32(AME_OP_##op), \
                            tcg_constant_i32(a->rd), \
                            tcg_constant_i32(a->rs1)); \
    return true; \
}
TRANS_AME_R2(mabs_ew, unary, ABS)
TRANS_AME_R2(mfrintm_ew, unary, FRINTM)
TRANS_AME_R2(mfrintn_ew, unary, FRINTN)
TRANS_AME_R2(mfrintp_ew, unary, FRINTP)
TRANS_AME_R2(mfrintz_ew, unary, FRINTZ)
TRANS_AME_R2(mcos_ew, unary, COS)
TRANS_AME_R2(mexp2_ew, unary, EXP2)
TRANS_AME_R2(mlog2_ew, unary, LOG2)
TRANS_AME_R2(mrec_ew, unary, REC)
TRANS_AME_R2(mrsqrt_ew, unary, RSQRT)
TRANS_AME_R2(msin_ew, unary, SIN)
TRANS_AME_R2(msqrt_ew, unary, SQRT)
TRANS_AME_R2(mtanh_ew, unary, TANH)
TRANS_AME_R2(mprefixadd_col, scan, PREFIXADDCOL)
TRANS_AME_R2(mprefixadd_row, scan, PREFIXADD)
TRANS_AME_R2(mprefixmax_col, scan, PREFIXMAXCOL)
TRANS_AME_R2(mprefixmax_row, scan, PREFIXMAX)
TRANS_AME_R2(mreduceadd_col, scan, REDUCEADDCOL)
TRANS_AME_R2(mreduceadd_row, scan, REDUCEADD)
TRANS_AME_R2(mreducemax_col, scan, REDUCEMAXCOL)
TRANS_AME_R2(mreducemax_row, scan, REDUCEMAX)
TRANS_AME_R2(mreducemin_col, scan, REDUCEMINCOL)
TRANS_AME_R2(mreducemin_row, scan, REDUCEMIN)
#undef TRANS_AME_R2

#define TRANS_AME_R3(name, helper, op) \
static bool trans_##name(DisasContext *ctx, arg_ztt_r3 *a) \
{ \
    gen_helper_ame_##helper(tcg_env, tcg_constant_i32(AME_OP_##op), \
                            tcg_constant_i32(a->rd), \
                            tcg_constant_i32(a->rs1), \
                            tcg_constant_i32(a->rs2)); \
    return true; \
}
TRANS_AME_R3(mabsdiff_ew, binary, ABSDIFF)
TRANS_AME_R3(madd_ew, binary, ADD)
TRANS_AME_R3(mhdiff_ew, binary, HDIFF)
TRANS_AME_R3(mmax_ew, binary, MAX)
TRANS_AME_R3(mmean_ew, binary, MEAN)
TRANS_AME_R3(mmin_ew, binary, MIN)
TRANS_AME_R3(mmul_ew, binary, MUL)
TRANS_AME_R3(mmulneg_ew, binary, NMUL)
TRANS_AME_R3(msub_ew, binary, SUB)
TRANS_AME_R3(mand_ew, binary, AND)
TRANS_AME_R3(mandnot_ew, binary, ANDNOT)
TRANS_AME_R3(mor_ew, binary, OR)
TRANS_AME_R3(mornot_ew, binary, ORNOT)
TRANS_AME_R3(mxor_ew, binary, XOR)
TRANS_AME_R3(msll_ew, binary, SLL)
TRANS_AME_R3(msra_ew, binary, SRA)
TRANS_AME_R3(msrl_ew, binary, SRL)
TRANS_AME_R3(mcmpge_ew, binary, CMPGE)
TRANS_AME_R3(mcmplt_ew, binary, CMPLT)
TRANS_AME_R3(mselge_ew, binary, SELGE)
TRANS_AME_R3(msellt_ew, binary, SEL)
TRANS_AME_R3(mldexp_ew, binary, LDEXP)
TRANS_AME_R3(mrdexp_ew, binary, RDEXP)
TRANS_AME_R3(mlog2sub_ew, binary, LOG2SUB)
TRANS_AME_R3(msublog2_ew, binary, SUBLOG2)
TRANS_AME_R3(mcmovge_ew, ternary, MUXGE)
TRANS_AME_R3(mcmovlt_ew, ternary, MUXLT)
TRANS_AME_R3(mmulacc_ew, ternary, MULACC)
TRANS_AME_R3(mmulaccneg_ew, ternary, MULNACC)
TRANS_AME_R3(mmuladd_ew, ternary, MULADD)
TRANS_AME_R3(mmulsub_ew, ternary, MULSUB)
TRANS_AME_R3(mldexpacc_ew, ternary, LDEXPACC)
TRANS_AME_R3(mrdexpacc_ew, ternary, RDEXPACC)
TRANS_AME_R3(mmulacc_2d, matmul, MMACC)
TRANS_AME_R3(mmulaccneg_2d, matmul, MMACCNEG)
TRANS_AME_R3(mmulatacc_2d, matmul, MMATACC)
TRANS_AME_R3(mmulataccneg_2d, matmul, MMATACCNEG)
TRANS_AME_R3(mmulbtacc_2d, matmul, MMBTACC)
TRANS_AME_R3(mmulbtaccneg_2d, matmul, MMBTACCNEG)
TRANS_AME_R3(mcolgather_ew, gather, GATHER)
TRANS_AME_R3(mrowgather_ew, gather, GATHERROW)
TRANS_AME_R3(mcolscatadd_ew, scatter, COLSCATADD)
TRANS_AME_R3(mcolscatmax_ew, scatter, COLSCATMAX)
TRANS_AME_R3(mrowscatadd_ew, scatter, ROWSCATADD)
TRANS_AME_R3(mrowscatmax_ew, scatter, ROWSCATMAX)
#undef TRANS_AME_R3

#define TRANS_AME_SCALAR(name, helper, op, kind) \
static bool trans_##name(DisasContext *ctx, arg_ztt_r3 *a) \
{ \
    gen_helper_ame_##helper(tcg_env, tcg_constant_i32(AME_OP_##op), \
                            tcg_constant_i32(AME_EW_##kind), \
                            tcg_constant_i32(a->rd), \
                            get_gpr(ctx, a->rs1, EXT_NONE), \
                            tcg_constant_i32(a->rs2)); \
    return true; \
}
TRANS_AME_SCALAR(mabsdiff_ew_x, scalar, ABSDIFF, DATA_SCALAR_FIRST)
TRANS_AME_SCALAR(madd_ew_x, scalar, ADD, DATA_SCALAR_FIRST)
TRANS_AME_SCALAR(mhdiff_ew_x, scalar, HDIFF, DATA_SCALAR_FIRST)
TRANS_AME_SCALAR(mmax_ew_x, scalar, MAX, DATA_SCALAR_FIRST)
TRANS_AME_SCALAR(mmean_ew_x, scalar, MEAN, DATA_SCALAR_FIRST)
TRANS_AME_SCALAR(mmin_ew_x, scalar, MIN, DATA_SCALAR_FIRST)
TRANS_AME_SCALAR(mmul_ew_x, scalar, MUL, DATA_SCALAR_FIRST)
TRANS_AME_SCALAR(mmulneg_ew_x, scalar, NMUL, DATA_SCALAR_FIRST)
TRANS_AME_SCALAR(msub_ew_x, scalar, SUB, DATA_SCALAR_FIRST)
TRANS_AME_SCALAR(mand_ew_x, scalar, AND, DATA_SCALAR_FIRST)
TRANS_AME_SCALAR(mandnot_ew_x, scalar, ANDNOT, DATA_SCALAR_FIRST)
TRANS_AME_SCALAR(mor_ew_x, scalar, OR, DATA_SCALAR_FIRST)
TRANS_AME_SCALAR(mornot_ew_x, scalar, ORNOT, DATA_SCALAR_FIRST)
TRANS_AME_SCALAR(mxor_ew_x, scalar, XOR, DATA_SCALAR_FIRST)
TRANS_AME_SCALAR(mcmpge_ew_x, scalar, CMPGE, DATA_SCALAR_FIRST)
TRANS_AME_SCALAR(mcmplt_ew_x, scalar, CMPLT, DATA_SCALAR_FIRST)
TRANS_AME_SCALAR(mlog2sub_ew_x, scalar, LOG2SUB, DATA_SCALAR_SECOND)
TRANS_AME_SCALAR(msublog2_ew_x, scalar, SUBLOG2, DATA_SCALAR_SECOND)
TRANS_AME_SCALAR(msll_ew_x, scalar, SLL, CONTROL_SCALAR)
TRANS_AME_SCALAR(msra_ew_x, scalar, SRA, CONTROL_SCALAR)
TRANS_AME_SCALAR(msrl_ew_x, scalar, SRL, CONTROL_SCALAR)
TRANS_AME_SCALAR(mldexp_ew_x, scalar, LDEXP, EXPONENT_SCALAR)
TRANS_AME_SCALAR(mmulacc_ew_x, ternary_scalar, MULACC, DATA_SCALAR_FIRST)
TRANS_AME_SCALAR(mmulaccneg_ew_x, ternary_scalar, MULNACC, DATA_SCALAR_FIRST)
TRANS_AME_SCALAR(mmuladd_ew_x, ternary_scalar, MULADD, DATA_SCALAR_FIRST)
TRANS_AME_SCALAR(mmulsub_ew_x, ternary_scalar, MULSUB, DATA_SCALAR_FIRST)
TRANS_AME_SCALAR(mldexpacc_ew_x, ternary_scalar, LDEXPACC, EXPONENT_SCALAR)
#undef TRANS_AME_SCALAR

#define TRANS_AME_PERMUTE(name, op) \
static bool trans_##name(DisasContext *ctx, arg_ztt_r3 *a) \
{ \
    gen_helper_ame_permute(tcg_env, tcg_constant_i32(AME_OP_##op), \
                           tcg_constant_i32(a->rd), \
                           get_gpr(ctx, a->rs1, EXT_NONE), \
                           tcg_constant_i32(a->rs2)); \
    return true; \
}
TRANS_AME_PERMUTE(mcolbcast_ew_x, COLBCAST)
TRANS_AME_PERMUTE(mrowbcast_ew_x, ROWBCAST)
TRANS_AME_PERMUTE(mcolshift_ew_x, SHIFT)
TRANS_AME_PERMUTE(mrowshift_ew_x, ROWSHIFT)
#undef TRANS_AME_PERMUTE

#define TRANS_AME_INDEX(name, op) \
static bool trans_##name(DisasContext *ctx, arg_ztt_r1 *a) \
{ \
    gen_helper_ame_index(tcg_env, tcg_constant_i32(AME_OP_##op), \
                         tcg_constant_i32(a->rd)); \
    return true; \
}
TRANS_AME_INDEX(mcolid_ew, COLID)
TRANS_AME_INDEX(mrowid_ew, ROWID)
#undef TRANS_AME_INDEX

#define TRANS_AME_ZIP(name, op) \
static bool trans_##name(DisasContext *ctx, arg_ztt_zip *a) \
{ \
    gen_helper_ame_zip(tcg_env, tcg_constant_i32(AME_OP_##op), \
                       tcg_constant_i32(a->rs1), tcg_constant_i32(a->rs2)); \
    return true; \
}
TRANS_AME_ZIP(mcolzip_ew, COLZIP)
TRANS_AME_ZIP(mcolunzip_ew, COLUNZIP)
TRANS_AME_ZIP(mrowzip_ew, ROWZIP)
TRANS_AME_ZIP(mrowunzip_ew, ROWUNZIP)
#undef TRANS_AME_ZIP

static bool trans_mbcast_m_x(DisasContext *ctx, arg_ztt_r3 *a)
{
    gen_helper_ame_bcast(tcg_env, tcg_constant_i32(a->rd),
                         get_gpr(ctx, a->rs1, EXT_NONE),
                         get_gpr(ctx, a->rs2, EXT_NONE));
    return true;
}

#define TRANS_AME_PACK(name, unpack) \
static bool trans_##name(DisasContext *ctx, arg_ztt_r3 *a) \
{ \
    gen_helper_ame_pack(tcg_env, tcg_constant_i32(unpack), \
                        tcg_constant_i32(a->rd), \
                        get_gpr(ctx, a->rs1, EXT_NONE), \
                        tcg_constant_i32(a->rs2)); \
    return true; \
}
TRANS_AME_PACK(mpack_ew_x, false)
TRANS_AME_PACK(munpack_ew_x, true)
#undef TRANS_AME_PACK

#define TRANS_AME_MEM(name, helper, format, stride) \
static bool trans_##name(DisasContext *ctx, arg_##name *a) \
{ \
    gen_helper_ame_##helper(tcg_env, tcg_constant_i32(a->rd), \
                            get_gpr(ctx, a->rs1, EXT_NONE), stride, \
                            tcg_constant_i32(AME_MEM_##format)); \
    return true; \
}
TRANS_AME_MEM(mls_1r, load, WHOLE_REGISTER, ctx->zero)
TRANS_AME_MEM(mls_cm, load, COLUMN_MAJOR, ctx->zero)
TRANS_AME_MEM(mls_rm, load, ROW_MAJOR, ctx->zero)
TRANS_AME_MEM(mls_st, load, STRIDED, get_gpr(ctx, a->rs2, EXT_NONE))
TRANS_AME_MEM(mls_tst, load, TRANSPOSED_STRIDED, get_gpr(ctx, a->rs2, EXT_NONE))
TRANS_AME_MEM(mss_1r, store, WHOLE_REGISTER, ctx->zero)
TRANS_AME_MEM(mss_cm, store, COLUMN_MAJOR, ctx->zero)
TRANS_AME_MEM(mss_rm, store, ROW_MAJOR, ctx->zero)
TRANS_AME_MEM(mss_st, store, STRIDED, get_gpr(ctx, a->rs2, EXT_NONE))
TRANS_AME_MEM(mss_tst, store, TRANSPOSED_STRIDED,
              get_gpr(ctx, a->rs2, EXT_NONE))
#undef TRANS_AME_MEM

#define TRANS_AME_DEBUG(bits) \
static bool trans_mmove##bits##_m_x(DisasContext *ctx, \
                                  arg_ztt_r3 *a) \
{ \
    gen_helper_ame_debug_write(tcg_env, tcg_constant_i32(a->rd), \
                               get_gpr(ctx, a->rs1, EXT_NONE), \
                               get_gpr(ctx, a->rs2, EXT_NONE), \
                               tcg_constant_i32(bits)); \
    return true; \
} \
static bool trans_mmove##bits##_x_m(DisasContext *ctx, \
                                  arg_ztt_r3 *a) \
{ \
    TCGv result = tcg_temp_new(); \
    gen_helper_ame_debug_read(result, tcg_env, tcg_constant_i32(a->rs1), \
                              get_gpr(ctx, a->rs2, EXT_NONE), \
                              tcg_constant_i32(bits)); \
    gen_set_gpr(ctx, a->rd, result); \
    return true; \
}
TRANS_AME_DEBUG(8)
TRANS_AME_DEBUG(16)
TRANS_AME_DEBUG(32)
TRANS_AME_DEBUG(64)
#undef TRANS_AME_DEBUG
