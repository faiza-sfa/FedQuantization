using LinearAlgebra
using Random
using Statistics

include("../compressors/qsgd.jl")

function lemma1_variance(f, s)
    b = rem(f, s)
    u = s - b
    return u * b
end

array = Float32[0.6, 0.4, 1.3, 0.01]

# array ./= norm(array)

l = 2
s = norm(array) / l

@show array

reps = 1000000

nums = zeros(Float32, reps, length(array))

for rep in 1:reps
    # function encode_qsgd(array, s, type, seed, cid, client_name)
    enc = encode_qsgd(array, l, "QSGD", nothing, 0, "0")
    # function decode_qsgd(array, s, type, use_lo_quant)
    dec = decode_qsgd(String(enc), l, "QSGD", false)
    nums[rep, :] = dec
    # @show dec
end


# The mathematical variance should roughly equal the empirical variance.

@show lemma1_variance.(array, s)

@show var(nums[:,1]), var(nums[:,2]), var(nums[:,3]), var(nums[:,4])
