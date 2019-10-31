

struct StandardDeviation{T,P,L}
    σ::T
    σ⁻¹::T
    ∂σ∂i::P
    logσ::L
end
struct StandardDeviationArray{S,I,L}
    σ::S
    σ⁻¹::I
    logσ::L
end

@inline logdet_coef(Y, args...) = -Float64(size(Y, 1))
@inline canonicalize_Σ(σ::T) where {T<:Real} = StandardDeviation{T,One}( σ, Base.FastMath.inv_fast(σ), One(), log(σ) )
@inline function canonicalize_Σ(σ::S) where {S <: Integer}
    T = promote_type(S, Float64)
    canonicalize_Σ(convert(T, σ))
end
@inline canonicalize_Σ(σ²::UniformScaling{Bool}) = One()
@inline function canonicalize_Σ(σ²I::UniformScaling{T}) where {T <: Real}
    σ² = σ²I.λ
    σ = Base.FastMath.sqrt_fast(σ²)
    σ⁻¹ = Base.FastMath.inv_fast(σ)
    ∂σ∂σ² = Base.FastMath.mul_fast(0.5, σ⁻¹)
    StandardDeviation(
        σ, σ⁻¹, ∂σ∂σ², log(σ)
    )
end
@inline function canonicalize_Σ(σ::RealFloat)
    StandardDeviation( σ.r, Base.FastMath.inv_fast(σ.r), One(), log(σ) )
end
@inline function canonicalize_Σ(σ::AbstractFixedSizeArray)
    StandardDeviation( PtrArray(σ), LazyMap(vinv, σ), LazyMap(SLEEFPirates.log, σ) )
end

"""
The normal distribution can be split into
Normal(args...) = Normal_kernel(args...) + logdet_coef(args...)*logdet(last(args))
"""

function Normal(args...)
    fargs = Base.front(args)
    L = canonicalize_Σ(last(args))
    tadd(vmul(logdet_coef(fargs..., L), logdet(L)), Normal_kernel(fargs...,L))
end
@generated function ∂Normal(args::Vararg{<:Any,N}) where {N}
    Nargs = N >>> 1
    @assert N & 1 == 0 "Number of arguments is not odd." # separate ::StackPointer function
    quote
        @inbounds (∂Ldiv∂Σ, L) = ∂canonicalize_Σ(args[$N])
        @inbounds kern = $(Expr(:call, :∂Normal_kernel!, [:(args[$n]) for n ∈ 1:N-1]..., L)) # if uninitialized, this initializes
        @inbounds ∂TARGETdiv∂L = initialized(args[$Nargs]) # so here we adjust type
        logdetL = ∂logdet!(∂TARGETdiv∂L, L)
        ∂TARGETdiv∂Σ = ∂TARGETdiv∂L # alias
        ReverseDiffExpressionsBase.RESERVED_INCREMENT_SEED_RESERVED!(∂TARGETdiv∂Σ, ∂Ldiv∂Σ, ∂TARGETdiv∂L) # adjust based on canonicalization
        tadd(vmul(logdet_coef(args...), logdetL), kern)
    end    
end


push!(DISTRIBUTION_DIFF_RULES, :Normal)
