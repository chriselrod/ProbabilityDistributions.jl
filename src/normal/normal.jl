
using StructuredMatrices: nlogdet
using ReverseDiffExpressionsBase: tadd

"""
σ⁻²   : multiply with δ² / -2 to get kernel-component of density
ld    : multiply with -logdet_coef to get logdet component of density
∂k∂i  : multiply with δ²σ⁻² to get ∂kernel-component/∂input
∂ld∂i : multiply with -logdet_coef to get ∂logdet-component/∂input
"""
struct Precision{T,P,D}
    σ⁻²::T
    ld::D
    ∂k∂i::P
end
struct PrecisionArray{S,T,N,X,L,V<:AbstractFixedSizeArray{S,T,N,X,L},P,D<:AbstractFixedSizeArray{S,T,N,X,L}} <: AbstractFixedSizeArray{S,T,N,X,L}
    σ⁻²::V
    ld::D
    ∂k∂i::P
end
@inline Base.getindex(A::PrecisionArray, i...) = A.σ⁻²[i...]
const AbstractPrecision{T,P,D} = Union{Precision{T,P,D},PrecisionArray{<:Any,<:Any,<:Any,<:Any,<:Any,T,P,D}}

using ReverseDiffExpressionsBase: One, Zero

@inline logdet_coef(Y, args...) = Float64(size(Y, 1))
@inline function canonicalize_Σ(σ::Real)
    σ⁻¹ = Base.FastMath.inv_fast( VectorizationBase.extract_data(σ) )
    σ⁻² = Base.FastMath.abs2_fast( σ⁻¹ )
    Precision( σ⁻², log(σ), nothing )
end
@inline function ∂canonicalize_Σ(σ::Real)
    σ⁻¹ = Base.FastMath.inv_fast(VectorizationBase.extract_data(σ))
    σ⁻² = Base.FastMath.abs2_fast( σ⁻¹ )
    # nσ⁻¹ = Base.FastMath.sub_fast( σ⁻¹ )
    Precision( σ⁻², log(σ), σ⁻¹ )
end
@inline precision(v) = SIMDPirates.vabs2(SIMDPirates.vinv(v))
@inline function canonicalize_Σ(σ::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    σ⁻² = LazyMap( precision, σ )
    logσ = LazyMap( SLEEFPirates.log, σ )
    PrecisionArray{S,T,N,X,L,typeof(σ⁻²),Nothing,typeof(logσ)}( σ⁻², logσ, nothing )
end
@inline function ∂canonicalize_Σ(σ::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    # Trust compiler to elliminate redundant
    σ⁻¹ = LazyMap( vinv, σ ) 
    σ⁻² = LazyMap( precision, σ )
    logσ = LazyMap( SLEEFPirates.log, σ )
    PrecisionArray{S,T,N,X,L,typeof(σ⁻²),typeof(σ⁻¹),typeof(logσ)}( σ⁻², logσ, σ⁻¹ )
end
@inline function canonicalize_Σ(σ::S) where {S <: Integer}
    T = promote_type(S, Float64)
    canonicalize_Σ(convert(T, σ))
end
@inline canonicalize_Σ(σ²::UniformScaling{Bool}) = One()
@inline function canonicalize_Σ(σ²I::UniformScaling{T}) where {T <: Real}
    σ² = σ²I.λ
    σ⁻² = Base.FastMath.inv_fast(VectorizationBase.extract_data(σ²))
    Precision(
        σ⁻², Base.FastMat.div_fast(Base.log(σ⁻²),2), nothing
    )
end
@inline canonicalize_Σ(σ::AbstractPrecision) = σ
@inline ∂canonicalize_Σ(σ::AbstractPrecision) = σ
@inline Base.inv(σ::AbstractPrecision) = σ.σb⁻¹
@inline SIMDPirates.vinv(σ::AbstractPrecision) = σ.σ⁻¹
@inline loginvroot(σ::AbstractPrecision) = σ.logσ
@inline function Base.materialize(σ::Precision)
    Precision(
        Base.materialize(σ.σ⁻²),
        Base.materialize(σ.ld),
        Base.materialize(σ.∂k∂i)
    )
end
@inline function Base.materialize(σ::PrecisionArray{S,T,N,X,L,V,P,D}) where {S,T,N,X,L,V,P,D}
    PrecisionArray(
        Base.materialize(σ.σ⁻²),
        Base.materialize(σ.ld),
        Base.materialize(σ.∂k∂i)
    )
end
# @inline Base.inv(σ::Union{Precision,PrecisionArray}) = σ.σ⁻¹
# @inline SIMDPirates.vinv(σ::Union{Precision,PrecisionArray}) = σ.σ⁻¹
# @inline loginvroot(σ::Union{Precision,PrecisionArray}) = σ.logσ

@inline precision(λ::AbstractPrecision) = λ.σ⁻²
@inline precision(::One) = One()
@inline LinearAlgebra.logdet(λ::AbstractPrecision) = λ.ld
@inline StructuredMatrices.nlogdet(λ::AbstractPrecision) = Base.FastMath.sub_fast(λ.ld)
@inline LinearAlgebra.logdet(::One) = Zero()
@inline StructuredMatrices.nlogdet(::One) = Zero()
@inline ∂k∂i(λ::AbstractPrecision) = λ.∂k∂i
@inline ∂k∂i(λ::AbstractPrecision{T,Nothing}) where {T} = Base.FastMath.mul_fast(T(0.5), λ.σ⁻²) # Variance
@inline ∂ld∂i(λ::AbstractPrecision{T,Nothing}) where {T} = Base.FastMath.mul_fast(T(0.5), λ.σ⁻²) # Variance
@inline ∂ld∂i(λ::AbstractPrecision{T,P}) where {T,P<:Real} = λ.∂k∂i # St.Dev

"""
The normal distribution can be split into
Normal(args...) = Normal_kernel(args...) + logdet_coef(args...)*logdet(last(args))
"""
Normal(y) = Normal_kernel(y)
Normal(::Val{track}, y) where {track} = (first(track) == true ? Normal_kernel(y) : Zero())
∂Normal!(∂y, y) = ∂Normal_kernel!(∂y, y)
function Normal(::Val{track}, args...) where {track}
    fargs = Base.front(args)
    L = canonicalize_Σ(last(args))
    if last(track) == true
        tadd(vmul(logdet_coef(fargs..., L), nlogdet(L)), Normal_kernel(fargs...,L))
    else
        Normal_kernel(fargs...,L)
    end
end
@generated function ∂Normal!(args::Vararg{<:Any,N}) where {N}
    Nargs = N >>> 1
    # @assert N & 1 == 0 "Number of arguments is not odd." # separate ::StackPointer function
    if args[Nargs + (N & 1)] === Nothing
        quote
            @inbounds L = canonicalize_Σ(args[$N])
            @inbounds $(Expr(:call, :∂Normal_kernel!, [:(args[$n]) for n ∈ 1:N-1]..., L)) # if uninitialized, this initializes
        end
    else
        if isodd(N) # then there is a StackPointer
            quote
                @inbounds (sp,(∂Ldiv∂Σ, L)) = ∂canonicalize_Σ(first(args), args[$N])
                @inbounds (sp, kern) = $(Expr(:call, :∂Normal_kernel!, [:(args[$n]) for n ∈ 1:N-1]..., L)) # if uninitialized, this initializes
                @inbounds ∂TARGETdiv∂L = initialized(args[$(Nargs+(N&1))]) # so here we adjust type
                nlogdetL = ∂nlogdet!(∂TARGETdiv∂L, L, logdet_coef(args...))
                ∂TARGETdiv∂Σ = ∂TARGETdiv∂L # alias
                ReverseDiffExpressionsBase.RESERVED_INCREMENT_SEED_RESERVED!(∂TARGETdiv∂Σ, ∂Ldiv∂Σ, ∂TARGETdiv∂L) # adjust based on canonicalization
                (sp, tadd(nlogdetL, kern))
            end
        else # there is no StackPointer
            quote
                @inbounds (∂Ldiv∂Σ, L) = ∂canonicalize_Σ(args[$N])
                @inbounds kern = $(Expr(:call, :∂Normal_kernel!, [:(args[$n]) for n ∈ 1:N-1]..., L)) # if uninitialized, this initializes
                @inbounds ∂TARGETdiv∂L = initialized(args[$Nargs]) # so here we adjust type
                nlogdetL = ∂nlogdet!(∂TARGETdiv∂L, L, logdet_coef(args...))
                ∂TARGETdiv∂Σ = ∂TARGETdiv∂L # alias
                ReverseDiffExpressionsBase.RESERVED_INCREMENT_SEED_RESERVED!(∂TARGETdiv∂Σ, ∂Ldiv∂Σ, ∂TARGETdiv∂L) # adjust based on canonicalization
                tadd(nlogdetL, kern)
            end
        end
    end
end


push!(DISTRIBUTION_DIFF_RULES, :Normal)
