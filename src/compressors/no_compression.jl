function encode_no_compression(array)
    return collect(reinterpret(UInt8, array))
end

function decode_no_compression(array)
    array = Vector{UInt8}(join(array))
    return collect(reinterpret(Float32, array))
end