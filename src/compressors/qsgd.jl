using LinearAlgebra

# counter = 0

function int_to_elias(v)
    enc = BitArray([0])
    while v > 0
        v_bits = digits(v, base=2)    # potentially inefficient
        append!(enc, v_bits)
        v = length(v_bits) - 1
    end
    return reverse(enc)
end

function elias_to_int(idx, bits)
    m = 0
    while bits[idx] != 0
        m_bits = bits[idx:idx+m]
        idx += m+1
        # digits() uses little-endian. reverse() turns
        # that into big-endian.
        next_m = 0 
        for i in 1:(m+1)
            next_m += m_bits[i] << (m+1 - i)
        end
        m = next_m
    end
    idx += 1
    return idx, m
end

function encode_qsgd(array, s, type, seed, cid, client_name)
    if seed != nothing
        Random.seed!(seed)
    end
    result = Vector{UInt8}()
    if type == "LFL"
        arr_norm = maximum(array) - minimum(array)
    else
        arr_norm = norm(array)
    end
    signs = array .< 0
    in_betweens = (abs.(array) .* s) ./ arr_norm
    ls = floor.(Int32, in_betweens)
    p_highs = in_betweens .- ls
    rands = rand(Float32, length(array))
    # global counter
    # println("cid = $(cid) | counter = $(counter) | client_name = $(client_name) | rands[1:2] = $(rands[1:2])")
    # counter += 1
    choices = UInt8.(rands .<= p_highs)
    ms = ls .+ choices
    zero_lre_ctr = 0
    bits = BitArray([])
    for idx in 1:length(ms)
        m = ms[idx]
        sign = signs[idx]
        if m == 0 && idx < length(ms)
            zero_lre_ctr += 1
        else
            append!(bits, int_to_elias(zero_lre_ctr))
            zero_lre_ctr = 0
            append!(bits, int_to_elias(m))
            push!(bits, sign)
        end
    end
    length_bits = reinterpret(UInt8, [UInt32(length(bits))])
    bytes = reinterpret(UInt8, bits.chunks)
    append!(result, length_bits)
    append!(result, reinterpret(UInt8, [Float32(arr_norm)]))
    append!(result, bytes)
    return result
end

function decode_qsgd(array, s, type, use_lo_quant)
    use_lo_quant &= s > 1
    array = Vector{UInt8}(join(array))
    result = Vector{Float32}()
    bits_length = reinterpret(UInt32, array[1:4])[1]
    norm = reinterpret(Float32, array[5:8])[1]
    # @show norm
    bits = BitArray([])
    bits.chunks = reinterpret(UInt64, array[9:end])
    bits.len = (length(array) - 8) * 8
    idx = 1
    while idx <= bits_length
        idx, n_zeros = elias_to_int(idx, bits)
        append!(result, zeros(Float32, n_zeros))
        idx, m = elias_to_int(idx, bits)
        sign = bits[idx]
        idx += 1
        if use_lo_quant
            fl = (mod(m,2)*rand([0,1]) + div(m,2)) * (1 - sign * 2) * (norm / div(s,2))
        else
            fl = m * (1 - sign * 2) * (norm / s)
        end
        append!(result, fl)
    end
    return result
end
