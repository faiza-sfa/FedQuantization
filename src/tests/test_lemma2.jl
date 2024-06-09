using LinearAlgebra
using Random
using Statistics

include("../compressors/qsgd.jl")

function lemma1_variance(f, s)
    b = rem(f, s)
    u = s - b
    return u * b
end

function lemma2_variance(weights, ls, t)
    return (t^2/6) * sum(weights.^2 ./ ls.^2)
end

t = 5

weights = [0.4, 0.6]

ls = [5, 4]

reps = 100000

expected_variance = 0

for rep in 1:reps

    fs = rand(Float32, 2) .* t
    ss = t ./ ls
    vars = lemma1_variance.(fs, ss)
    global expected_variance
    expected_variance += sum(weights .^2 .* vars)
end

expected_variance /= reps


# The mathematical variance should roughly equal the empirical variance.

@show lemma2_variance(weights, ls, t)

@show expected_variance
