
import BitIntegers
BitIntegers.@define_integers 24

"""
LinQuantArray{T,N}
Represents a quantised array using UInts, with fields for the minimum and maximum of the original range.
"""
struct LinQuantArray{T,N} <: AbstractArray{Unsigned,N}
    A::Array{T,N}       # Array of UInts
    min::Float64        # Minimum value
    max::Float64        # Maximum value
end

# Basic array operations for LinQuantArray
Base.size(QA::LinQuantArray) = size(QA.A)
Base.getindex(QA::LinQuantArray, i...) = getindex(QA.A, i...)
Base.eltype(Q::LinQuantArray{T,N}) where {T,N} = T

"""
LinQuantization(T, A)
Quantises an array A into a LinQuantArray of type T.
"""
function LinQuantization(::Type{T}, A::AbstractArray) where {T<:Unsigned}
    all(isfinite.(A)) || throw(DomainError("Linear quantization only in (-∞,∞)"))
    Amin, Amax = Float64(minimum(A)), Float64(maximum(A))
    Δ = (Amin == Amax) ? 0.0 : (2^(sizeof(T)*8)-1) / (Amax - Amin)
    Q = similar(A, T)
    @inbounds for i in eachindex(Q)
        Q[i] = round((A[i] - Amin) * Δ)
    end
    return LinQuantArray{T,ndims(Q)}(Q, Amin, Amax)
end

# Predefined quantizations for different bit sizes
LinQuant8Array(A::AbstractArray{T,N}) where {T,N} = LinQuantization(UInt8, A)
LinQuant16Array(A::AbstractArray{T,N}) where {T,N} = LinQuantization(UInt16, A)
LinQuant24Array(A::AbstractArray{T,N}) where {T,N} = LinQuantization(UInt24, A)
LinQuant32Array(A::AbstractArray{T,N}) where {T,N} = LinQuantization(UInt32, A)

"""
Base.Array{T}(n, Q)
De-quantises a LinQuantArray Q into an array of floats of type T.
"""
function Base.Array{T}(n::Integer, Q::LinQuantArray) where {T<:AbstractFloat}
    Qmin, Qmax, Δ = Q.min, Q.max, (Q.max - Q.min) / (2^n - 1)
    A = similar(Q, T)
    @inbounds for i in eachindex(A)
        A[i] = Qmin + Q[i] * Δ
    end
    return A
end

# Default conversions for LinQuantArray with standard bit sizes
Base.Array{T}(Q::LinQuantArray{UInt8,N}) where {T,N} = Array{T}(8, Q)
Base.Array{T}(Q::LinQuantArray{UInt16,N}) where {T,N} = Array{T}(16, Q)
Base.Array{T}(Q::LinQuantArray{UInt24,N}) where {T,N} = Array{T}(24, Q)
Base.Array{T}(Q::LinQuantArray{UInt32,N}) where {T,N} = Array{T}(32, Q)
Base.Array(Q::LinQuantArray{UInt8,N}) where N = Array{Float32}(8, Q)
Base.Array(Q::LinQuantArray{UInt16,N}) where N = Array{Float32}(16, Q)
Base.Array(Q::LinQuantArray{UInt24,N}) where N = Array{Float32}(24, Q)
Base.Array(Q::LinQuantArray{UInt32,N}) where N = Array{Float64}(32, Q)

"""
LinQuantArray(TUInt, A, dim)
Performs linear quantization independently for every element along dimension dim in array A.
Returns a Vector of LinQuantArrays.
"""
function LinQuantArray(::Type{TUInt}, A::AbstractArray{T,N}, dim::Int) where {TUInt,T,N}
    @assert dim <= N "Can't quantize a $N-dimensional array in dim=$dim"
    L = [LinQuantization(TUInt, A[axes(A, dim) .== i]) for i in axes(A, dim)]
    return L
end

# Dimensional quantizations for standard bit sizes
LinQuant8Array(A::AbstractArray{T,N}, dim::Int) where {T,N} = LinQuantArray(UInt8, A, dim)
LinQuant16Array(A::AbstractArray{T,N}, dim::Int) where {T,N} = LinQuantArray(UInt16, A, dim)
LinQuant24Array(A::AbstractArray{T,N}, dim::Int) where {T,N} = LinQuantArray(UInt24, A, dim)
LinQuant32Array(A::AbstractArray{T,N}, dim::Int) where {T,N} = LinQuantArray(UInt32, A, dim)

"""
Base.Array{T}(L)
Reconstructs the original array from a Vector of LinQuantArrays L, permuted such that the quantized dimension is last.
"""
function Base.Array{T}(L::Vector{LinQuantArray}) where T
    N = ndims(L[1])
    n = length(L)
    s = size(L[1])
    t = axes(L[1])
    A = Array{T,N+1}(undef,s...,length(L))
    for i in 1:n
        A[t...,i] = Array{T}(L[i])
    end
    return A
end
