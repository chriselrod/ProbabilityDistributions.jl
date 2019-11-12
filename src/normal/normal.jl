
using StructuredMatrices: nlogdet
using ReverseDiffExpressionsBase: tadd

"""
σ⁻²   : multiply with δ² / -2 to get kernel-component of density
ld    : multiply with -logdet_coef to get logdet component of density
∂k∂i  : multiply with δ²σ⁻² to get ∂kernel-component/∂input
∂ld∂i : multiply with -logdet_coef to get ∂logdet-component/∂input
"""
struct Precision{T,P}
    σ⁻²::T
    ld::T
    ∂k∂i::P
end
# struct PrecisionArray{I,P}
    # σ⁻¹::I
    # σ⁻²::P
    # logσ::L
# end
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
@inline function canonicalize_Σ(σ::AbstractFixedSizeArray)
    σ⁻² = LazyMap( precision, σ )
    logσ = LazyMap( SLEEFPirates.log, σ )
    Precision( σ⁻², logσ, nothing )
end
@inline function ∂canonicalize_Σ(σ::AbstractFixedSizeArray)
    # Trust compiler to elliminate redundant
    σ⁻¹ = LazyMap( vinv, σ ) 
    σ⁻² = LazyMap( precision, σ )
    logσ = LazyMap( SLEEFPirates.log, σ )
    Precision( σ⁻², logσ, σ⁻¹ )
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
@inline canonicalize_Σ(σ::Precision) = σ
@inline Base.inv(σ::Precision) = σ.σ⁻¹
@inline SIMDPirates.vinv(σ::Precision) = σ.σ⁻¹
@inline loginvroot(σ::Precision) = σ.logσ
# @inline Base.inv(σ::Union{Precision,PrecisionArray}) = σ.σ⁻¹
# @inline SIMDPirates.vinv(σ::Union{Precision,PrecisionArray}) = σ.σ⁻¹
# @inline loginvroot(σ::Union{Precision,PrecisionArray}) = σ.logσ

@inline precision(λ::Precision) = λ.σ⁻²
@inline precision(::One) = One()
@inline LinearAlgebra.logdet(λ::Precision) = λ.ld
@inline StructuredMatrices.nlogdet(λ::Precision) = Base.FastMath.sub_fast(λ.ld)
@inline LinearAlgebra.logdet(::One) = Zero()
@inline StructuredMatrices.nlogdet(::One) = Zero()
@inline ∂k∂i(λ::Precision) = λ.∂k∂i
@inline ∂k∂i(λ::Precision{T,Nothing}) where {T} = Base.FastMath.mul_fast(T(0.5), λ.σ⁻²) # Variance
@inline ∂ld∂i(λ::Precision{T,Nothing}) where {T} = Base.FastMath.mul_fast(T(0.5), λ.σ⁻²) # Variance
@inline ∂ld∂i(λ::Precision{T,P}) where {T,P<:Real} = λ.∂k∂i # St.Dev

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
