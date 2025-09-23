import triton
import triton.language as tl


@triton.jit
def flashattention_2_fwd(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    Q_i = tl.load(Q_block_ptr, boundary_check=(1, 0), padding_option="zero")
    L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    O_i = tl.load(O_block_ptr, boundary_check=(1, 0), padding_option="zero")

    O_i_j = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    m_i_j = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l_i_j = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    if is_causal:
        offs_q = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(1, 0), padding_option="zero")

        S_i_j = tl.dot(Q_i, K_j) * scale
        if is_causal:
            offs_k = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            tri_mask = offs_q[:, None] >= offs_k[None, :]
            S_i_j = tl.where(tri_mask, S_i_j, float("-inf"))

        m_i_jm1 = m_i_j
        m_i_j = tl.maximum(m_i_j, tl.max(S_i_j, axis=-1))

        P_i_j = tl.exp(S_i_j - m_i_j[:, None]).to(dtype=V_j.dtype)

        l_i_jm1 = l_i_j
        l_i_j = tl.exp(m_i_jm1 - m_i_j) * l_i_jm1 + tl.sum(P_i_j, axis=-1)

        O_i_jm1 = O_i_j
        O_i_j = tl.exp(m_i_jm1 - m_i_j)[:, None] * O_i_jm1 + tl.dot(P_i_j, V_j)

        K_block_ptr = K_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    O_i = (1 / l_i_j)[:, None] * O_i_j
    L_i = m_i_j + tl.log(l_i_j)
    
    tl.store(O_block_ptr, O_i, boundary_check=(0,1))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))


@triton.jit
def flashattention_2_bwd_dkv(
    dQ_ptr, dK_ptr, dV_ptr,
    dO_ptr,
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    D_ptr,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    stride_dob, stride_doq, stride_dod,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS,
    scale,
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):

    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, d),
        strides=(stride_dqq, stride_dqd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, d),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, d),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, d),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, d),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, d),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, d),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, d),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    K_j = tl.load(K_block_ptr, boundary_check=(1, 0), padding_option="zero")
    V_j = tl.load(V_block_ptr, boundary_check=(1, 0), padding_option="zero")

    dK_j = tl.zeros((K_TILE_SIZE, d), dtype=tl.float32)
    dV_j = tl.zeros((K_TILE_SIZE, d), dtype=tl.float32)

    if is_causal:
        offs_k = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Q_i= tl.load(Q_block_ptr, boundary_check=(1, 0), padding_option="zero")
        O_i = tl.load(O_block_ptr, boundary_check=(1, 0), padding_option="zero")
        L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
        dQ_i = tl.load(dQ_block_ptr, boundary_check=(1, 0), padding_option="zero")
        dO_i = tl.load(dO_block_ptr, boundary_check=(1, 0), padding_option="zero")

        S_i_j = tl.dot(Q_i, tl.trans(K_j)) * scale
        if is_causal:
            offs_q = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            tri_mask = offs_q[:, None] >= offs_k[None, :]
            S_i_j = tl.where(tri_mask, S_i_j, float("-inf"))

        P_i_j = tl.exp(S_i_j - L_i[:, None])
        
        dV_j += tl.dot(tl.trans(P_i_j), dO_i)

        dP_i_j = tl.dot(dO_i, tl.trans(V_j))

        dS_i_j = P_i_j * (dP_i_j - D_i[:, None]) * scale
        
        # dQ_i = tl.dot(dS_i_j, tl.trans(K_j))
        # tl.atomic_add(dQ_block_ptr, dQ_i)
        
        dK_j += tl.dot(tl.trans(dS_i_j), Q_i)

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        O_block_ptr = O_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))
        dQ_block_ptr = dQ_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
    
    tl.store(dK_block_ptr, dK_j, boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV_j, boundary_check=(1, 0))


@triton.jit
def flashattention_2_bwd_dq(
    dQ_ptr,
    dO_ptr,
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    D_ptr,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dob, stride_doq, stride_dod,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS,
    scale,
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):

    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, d),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, d),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, d),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, d),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, d),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, d),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q_i= tl.load(Q_block_ptr, boundary_check=(1, 0), padding_option="zero")
    L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
    dQ_i = tl.load(dQ_block_ptr, boundary_check=(1, 0), padding_option="zero")
    dO_i = tl.load(dO_block_ptr, boundary_check=(1, 0), padding_option="zero")

    if is_causal:
        offs_q = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(1, 0), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(1, 0), padding_option="zero")

        S_i_j = tl.dot(Q_i, tl.trans(K_j)) * scale
        if is_causal:
            offs_k = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            tri_mask = offs_q[:, None] >= offs_k[None, :]
            S_i_j = tl.where(tri_mask, S_i_j, float("-inf"))

        P_i_j = tl.exp(S_i_j - L_i[:, None])

        dP_i_j = tl.dot(dO_i, tl.trans(V_j))

        dS_i_j = P_i_j * (dP_i_j - D_i[:, None]) * scale
        
        dQ_i += tl.dot(dS_i_j, K_j)
        
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    tl.store(dQ_block_ptr, dQ_i, boundary_check=(1, 0))
