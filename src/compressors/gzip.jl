using GZip
using LinearAlgebra

function encode_gzip(array, s, seed)
    if seed != nothing
        Random.seed!(seed)
    end
    arr_norm = norm(array)
    signs = array .> 0
    in_betweens = (abs.(array) .* s) ./ arr_norm
    ls = floor.(Int32, in_betweens)
    p_highs = in_betweens .- ls
    rands = rand(Float32, length(array))
    choices = UInt8.(rands .<= p_highs)
    ms = ls .+ choices
    ms .*= (2 .* signs) .- 1
    if s >= 128
        ms = Int16.(ms)
    else
        ms = Int8.(ms)
    end
    filename = tempname()
    fh = GZip.open(filename, "w")
    write(fh, ms)
    write(fh, Float32(arr_norm))
    close(fh)
    fh = open(filename, "r")
    result = read(fh)
    close(fh)
    return result
end

function decode_gzip(array, s)
    array = Vector{UInt8}(join(array))
    filename = tempname()
    fh = open(filename, "w")
    write(fh, array)
    close(fh)
    fh = GZip.open(filename, "r")
    array = read(fh)
    close(fh)
    result = Vector{Float32}()
    norm = reinterpret(Float32, array[end-3:end])[1]
    if s >= 128
        result = reinterpret(Int16, array[1:end-4]) .* (norm / s)
    else
        result = reinterpret(Int8, array[1:end-4]) .* (norm / s)
    end
    return result
end
