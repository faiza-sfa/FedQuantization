function encode_fp8(array, vals, fp8s, seed)
    if seed != nothing
        Random.seed!(seed)
    end
    fp8s = Vector{UInt8}(join(fp8s))
    idxs_high = [searchsortedfirst(vals, f) for f in array]
    highs = [vals[idx] for idx in idxs_high]
    lows = [vals[idx-1] for idx in idxs_high]
    probs = (array .- lows) ./ (highs .- lows)
    rands = rand(Float32, length(array))
    choices = UInt8.(rands .> probs)
    idxs = idxs_high - choices 
    result = [fp8s[idx] for idx in idxs]
    return result
end

function decode_fp8(array)
    array = Vector{UInt8}(join(array))
    result = Array{Float32}(undef, length(array))
    for idx in 1:length(array)
        num = array[idx]
        sign8 = num>>7 & 0b1
        exponent8 = (num>>2) & 0b11111
        significand8 = num & 0b11
        # Handle special-case 0
        if exponent8 == 0 && significand8 == 0
            result[idx] = 0
        # Handle +-Inf
        elseif ~exponent8 == 0 && significand8 == 0
            result[idx] = sign8 == 0 ? Inf : -Inf
        # Handle subnormals
        elseif exponent8 == 0 && significand8 != 0
            emax8 = (2^(5-1)) - 1
            emin8 = 1 - emax8
            res = Float32(2)^(emin8-2) * significand8
            result[idx] = sign8 == 0 ? res : -res
        # Handle normals
        else
            res = UInt32(0)
            bias8 = (2^(5-1)) - 1
            bias32 = (2^(8-1)) - 1
            exponent32 = UInt32(exponent8) + bias32 - bias8
            significand32 = UInt32(significand8)
            sign32 = UInt32(sign8)
            res += significand32 << (23 - 2)
            res += exponent32 << 23
            res += sign32 << 31
            result[idx] = reinterpret(Float32, UInt32(res))
        end
    end
    return result
end