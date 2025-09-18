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

    for _ in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(1, 0), padding_option="zero")

        S_i_j = tl.dot(Q_i, K_j) * scale

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
