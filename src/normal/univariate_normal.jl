
@inline logdet_coef(y::AbstractFixedSizeVector, ::Any, σ::AbstractFixedSizeVector) = One()

function univariate_normal_kernel_quote(
    M::Int, @nospecialize(T), yisvec::Bool,
    @nospecialize(μisvec::Union{Bool,Nothing}), @nospecialize(σisvec::Union{Bool,Nothing}),
    (track_y, track_μ, track_σ)::NTuple{3,Bool},
    (inity, initμ, initσ)::NTuple{3,Bool},
    partial::Bool, calclogdet::Bool = false, @nospecialize(S) = Tuple{M}
)
    if M > 1
        pre_quote = quote
            qf = SIMDPirates.vbroadcast($(VectorizationBase.pick_vector(M,T)), zero($T))
        end
    else
        pre_quote = quote
            qf = zero($T)
        end
    end
    return_expr = quote end
    if yisvec # not allowed to be nothing, must be bool
        yexpr = :(y[i])
    else
        yexpr = :y
    end
    if μisvec === nothing
        δexpr = yexpr
        #if not, then it is a bool
    elseif μisvec # == true
        δexpr = :($yexpr - μ[i])
    else # μisvec == false
        δexpr = :($yexpr - μ)
    end
    # add target
    if σisvec === nothing
        # push!(pre_quote.args, :(σ⁻¹ = One() ))
        loop_expr = quote
            δ = $δexpr
            δσ⁻² = δ
            qf = vfmadd(δ, δ, qf)
        end
    elseif σisvec# == true
        push!(pre_quote.args, :(σ⁻² = precision(σ)))
        loop_expr = quote
            δ = $δexpr
            σ⁻²ᵢ = σ⁻²[i]
            δσ⁻² = δ * σ⁻²ᵢ
            qf = vfmadd(δσ⁻², δ, qf)
        end
        if calclogdet && track_σ
            push!(pre_quote.args, :(nld = vbroadcast(Vec{$W,$T},zero($T))))
            push!(pre_quote.args, :(ldv = nlogdet(σ)))
            push!(loop_expr, :(nld = vadd(nld, ldv[i])))
        end
    elseif partial && (track_y || track_μ) #σisvec == false, and we're calculating ∂y or ∂μ
        push!(pre_quote.args, :(σ⁻² = precision(σ)))
        if track_σ
            loop_expr = quote
                δ = $δexpr
                δσ⁻² = δ * σ⁻²
                δ²σ⁻² = δσ⁻² * δ
                qf = vadd(δ²σ⁻², qf)
            end
        else
            loop_expr = quote
                δ = $δexpr
                δσ⁻² = δ * σ⁻2
                qf = vfmadd(δσ⁻², δ, qf)
            end
        end
    else #σisvec == false
        # we do not need to keep track of δσ⁻², so we multiply at the end of the reduction.
        loop_expr = quote
            δ = $δexpr
            qf = vfmadd(δ, δ, qf)
        end
        push!(return_expr.args, :(qf = vmul(qf, precision(σ))))
    end
    if !σisvec && calclogdet && track_σ
        if M == 1
            push!(retun_expr.args, :(nld = nlogdet(σ)))
        else
            push!(retun_expr.args, :(nld = vmul($(T(M)), nlogdet(σ))))
        end
    end
    if partial
        if track_y
            if yisvec
                if inity
                    push!(loop_expr.args, :(∂y[i]  = -δσ⁻²))
                else
                    push!(loop_expr.args, :(∂y[i] -=  δσ⁻²))
                end
            else
                push!(pre_quote.args, :(∂ys  = zero($T)))
                push!(loop_expr.args, :(∂ys -= δσ⁻²))
                push!(return_expr.args, inity ? :(∂y[] = ∂ys) : :(∂y[] += ∂ys))
            end
        end
        if track_μ
            if μisvec# == true
                if initμ
                    push!(loop_expr.args, :(∂μ[i]  = δσ⁻²))
                else
                    push!(loop_expr.args, :(∂μ[i] += δσ⁻²))
                end
            elseif μisvec# == false
                if yisvec
                    push!(pre_quote.args, :(∂μs = zero($T)))
                    push!(loop_expr.args, :(∂μs += δσ⁻²))
                else # both yisvec && μisvec
                    push!(return_expr.args, :(∂μs = Base.FastMath.sub_fast(∂y)))
                end
                push!(return_expr.args, initμ ? :(∂μ[] = ∂μs) : :(∂μ[] += ∂μs))
            end
        end
        if track_σ
            eq = initσ ? :(=) : :(+=)
            if σisvec# == true
                push!(pre_quote.args, :(∂k∂σ  = ∂k∂i(σ)))
                if calclogdet
                    push!(pre_quote.args, :( ∂ld∂σ = ∂ld∂i(σ) ) )
                    set∂σ = :(∂k∂σ[i] * δσ⁻² - ∂ld∂σ[i])
                else
                    set∂σ = :(∂k∂σ[i] * δσ⁻²)
                end
                push!(loop_expr.args, Expr(eq, :(∂σ[i]), set∂σ))
            elseif σisvec# == false
                if calclogdet
                    if yisvec
                        push!(return_expr.args, Expr(eq, :(∂σ[]), :(∂k∂i(σ) * qf + $(T(-M)) * ∂ld∂i(σ))))
                    else
                        push!(return_expr.args, Expr(eq, :(∂σ[]), :(∂k∂i(σ) * qf - ∂ld∂i(σ))))
                    end
                else
                    push!(return_expr.args, Expr(eq, :(∂σ[]), :(∂k∂i(σ) * qf)))
                end
            end
        end
    end
    if calclogdet
        if yisvec && !σisvec
            retex = :(tadd(vmul($(T(-0.5)), qf), nld))
        else # either both isvec, or neither
            retex = :(vfmadd($(T(-0.5)), qf, nld))
        end
    else
        retex = :(vmul($(T(-0.5)), qf))
    end
    q = if yisvec
        loop_expr = quote
            LoopVectorization.@vvectorize $T 4 for i ∈ 1:$M
                $loop_expr
            end
        end
        quote
            $(Expr(:meta,:inline))
            $pre_quote
            $(macroexpand(ProbabilityDistributions, loop_expr))
            $return_expr
            $retex
        end
    else
        quote
            $(Expr(:meta,:inline))
            @fastmath begin
                $pre_quote
                $loop_expr
                $return_expr
            end
            $retex
        end
    end
    simplify_expr(q)
end
function alloc_univariate_normal_quote(M, S, @nospecialize(T), (track_y, track_μ, track_σ), sp = true)
    N = length(S.parameters)
    X = Vector{Int}(undef, N)
    X[1] = 1
    for n in 2:N
        X[n] = X[n-1] * (S.parameters[n-1])::Int
    end
    P = Tuple{X...}
    q = quote end
    if track_y
        if yisvec
            if sp
                push!(q.args, :((sp,∂y) = PtrArray{$S,$T,$N,$P}(sp)))
            else
                push!(q.args, :(∂y = FixedSizeArray{$S,$T,$N,$P}(undef) ))
            end
        else
            push!(q.args, :(∂y = Ref{$T}()))
        end
    end
    if track_μ
        if μisvec == true
            if sp
                push!(q.args, :((sp,∂μ) = PtrArray{$S,$T,$N,$P}(sp)))
            else
                push!(q.args, :(∂μ = FixedSizeArray{$S,$T,$N,$P}(undef) ))
            end
        elseif μisvec == false
            push!(q.args, :(∂μ = Ref{$T}()))
        end
    end
    if track_σ
        if σisvec == true
            if sp
                push!(q.args, :((sp,∂σ) = PtrArray{$S,$T,$N,$P}(sp)))
            else
                push!(q.args, :(∂σ = FixedSizeArray{$S,$T,$N,$P}(undef) ))
            end
        elseif σisvec == false
            push!(q.args, :(∂σ = Ref{$T}()))
        end
    end
    q
end

# @eval loop to specify normals
for (y,μ,σ) ∈ []
    

end


@inline Normal(y::T) where {T <: Real} = Base.FastMath.mul_fast(T(-0.5), Base.FastMath.abs2_fast(y))
@inline Normal(::Val{true}, y::Real) = Normal(y)
@inline Normal(::Val{false}, y::T) where {T <: Real} = zero(T)

# univariate_normal_kernel_quote(
    # M::Int, @nospecialize(T), yisvec::Bool,
    # @nospecialize(μisvec::Union{Bool,Nothing}), @nospecialize(σisvec::Union{Bool,Nothing}),
    # (track_y, track_μ, track_σ)::NTuple{3,Bool},
    # (inity, initμ, initσ)::NTuple{3,Bool},
    # partial::Bool, calclogdet::Bool = false, @nospecialize(S) = Tuple{M}
# )
@generated function Normal(::Val{track}, y::T, σ::Union{T,Int}) where {T <: Real, track}
    univariate_normal_quote(
        1, T, false, nothing, false,
        (track[1], false, track[2]),
        (false, false, false),
        false, true
    )
end
@generated function Normal_kernel(::Val{track}, y::T, σ::Union{T,Int}) where {T <: Real, track}
    univariate_normal_quote(
        1, T, false, nothing, false,
        (track[1], false, track[2]),
        (false, false, false),
        false, false
    )
end
@generated function Normal_kernel(::Val{track}, y::T, μ::T, σ::Union{T,Int}) where {T <: Real, track}
    univariate_normal_quote(
        1, T, false, false, false,
        (track[1], track[2], track[3]),
        (false, false, false),
        false, false
    )
end
@noinline function univariate_normal_length(SV::Core.SimpleVector, N, RV::Core.SimpleVector, L)
    fsv = first(SV)::Int
    first(RV)::Int == 1 || throw("Non-unit stride not yet supported.")
    if N == 1
        M = fsv
    else
        fsv == (RV[2])::Int || throw("Arrays with more than 1 dimension cannot be padded.")
        M = L
    end
    M
end
@generated function Normal_kerel(::Val{track}, y::AbstractFixedSizeArray{S,T,N,R,L}) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R.parameters, L)
    univariate_normal_quote( M, T, true, nothing, nothing, (track[1], false, false), (false,false,false), false, false )
end
@generated function Normal_kernel(
    y::AbstractFixedSizeArray{S,T,N,R,L}, σ::Union{T,Int,<:AbstractFixedSizeArray{S,T,N,R,L}}, ::Val{track}
) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R.parameters, L)
    univariate_normal_quote(
        M, T, true, nothing, σ <: AbstractFixedSizeArray, (track[1], false, track[2]), (false,false,false), false, false
    )
end
@generated function Normal_kernel(
    ::Val{track},
    y::AbstractFixedSizeArray{S,T,N,R,L},
    μ::Union{T,Int,<:AbstractFixedSizeArray{S,T,N,R,L}},
    σ::Union{T,Int,<:AbstractFixedSizeArray{S,T,N,R,L}}
) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R.parameters, L)
    univariate_normal_quote(
        M, T, true, μ <: AbstractFixedSizeArray, σ <: AbstractFixedSizeArray,
        track, (false,false,false), false, false
    )
end
@generated function Normal(
    ::Val{track},
    y::AbstractFixedSizeArray{S,T,N,R,L},
    μ::Union{T,Int,<:AbstractFixedSizeArray{S,T,N,R,L}},
    σ::AbstractFixedSizeArray{S,T,N,R,L}
) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R.parameters, L)
    univariate_normal_quote( M, T, true,
        μ <: AbstractFixedSizeArray, true, track, (false,false,false), false, true
    )
end
@generated function Normal_kernel(
    ::Val{track},
    y::AbstractFixedSizeArray{S,T,N,R,L},
    μ::Union{T,Int,<:AbstractFixedSizeArray{S,T,N,R,L}},
    σ::AbstractFixedSizeArray{S,T,N,R,L}
) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R.parameters, L)
    univariate_normal_quote( M, T, true,
        μ <: AbstractFixedSizeArray, true, track, (false,false,false), false, false
    )
end



@inline function ∂Normal!(∂y::U, y::T) where {T <: Real}
    t = Base.FastMath.mul_fast(T(-0.5), Base.FastMath.abs2_fast(y))
    if isinitialized(U)
        ∂y[] = FastMat.sub_fast(∂y[], y)
    else
        ∂y[] = FastMat.sub_fast(y)
    end
    t
end
@inline ∂Normal!(::Nothing, y::T) where {T <: Real, U} = zero(T)

# univariate_normal_kernel_quote(
    # M::Int, @nospecialize(T), yisvec::Bool,
    # @nospecialize(μisvec::Union{Bool,Nothing}), @nospecialize(σisvec::Union{Bool,Nothing}),
    # (track_y, track_μ, track_σ)::NTuple{3,Bool},
    # (inity, initμ, initσ)::NTuple{3,Bool},
    # partial::Bool, calclogdet::Bool = false, @nospecialize(S) = Tuple{M}
# )
@generated function ∂Normal(∂y::∂YN, ∂σ::∂ΣN, y::T, σ::Union{T,Int}) where {T <: Real, ∂YN, ∂ΣN}
    univariate_normal_quote(
        1, T, false, nothing, false,
        (∂YN !== Nothing, false, ∂YN !== Nothing),
        (!isinitialized(∂YN), false, !isinitialized(∂ΣN)),
        true, true
    )
end
@generated function ∂Normal_kernel!(∂y::∂YN, ∂σ::∂ΣN, y::T, σ::Union{T,Int}) where {T <: Real, ∂YN, ∂ΣN}
    univariate_normal_quote(
        1, T, false, nothing, false,
        (∂YN !== Nothing, false, ∂YN !== Nothing),
        (!isinitialized(∂YN), false, !isinitialized(∂ΣN)),
        true, false
    )
end
@generated function ∂Normal_kernel!(∂y::∂YN, ∂μ::∂MN, ∂σ::∂ΣN, y::T, ∂μ, σ::Union{T,Int}) where {T <: Real, ∂YN, ∂MN, ∂ΣN}
    univariate_normal_quote(
        1, T, false, false, false,
        (∂YN !== Nothing, ∂MN !== Nothing, ∂YN !== Nothing),
        (!isinitialized(∂YN), !isinitialized(∂MN), !isinitialized(∂ΣN)),
        true, false
    )
end
@noinline function univariate_normal_length(SV::Core.SimpleVector, N, RV::Core.SimpleVector, L)
    fsv = first(SV)::Int
    first(RV)::Int == 1 || throw("Non-unit stride not yet supported.")
    if N == 1
        M = fsv
    else
        fsv == (RV[2])::Int || throw("Arrays with more than 1 dimension cannot be padded.")
        M = L
    end
    M
end
@generated function ∂Normal_kerel!(∂y::∂YN, y::AbstractFixedSizeArray{S,T,N,R,L}) where {S,T,N,R,L, ∂YN}
    M = univariate_normal_length(S.parameters, N, R.parameters, L)
    univariate_normal_quote(
        M, T, true, nothing, nothing,
        (∂YN !== Nothing, false, false)
        (!isinitialized(∂YN), false, false),
        true, false
    )
end
@generated function ∂Normal_kernel!(∂y::∂YN, ∂σ::∂ΣN, y::AbstractFixedSizeArray{S,T,N,R,L}, σ) where {S,T,N,R,L,∂YN,∂ΣN}
    M = univariate_normal_length(S.parameters, N, R.parameters, L)
    univariate_normal_quote(
        M, T, true, nothing, σ <: AbstractFixedSizeArray,
        (∂YN !== Nothing, false, ∂ΣN !== Nothing)
        (!isinitialized(∂YN), false, !isinitialized(∂ΣN)),
        true, false
    )
end
@generated function Normal_kernel(
    ::Val{track},
    y::AbstractFixedSizeArray{S,T,N,R,L},
    μ::Union{T,Int,<:AbstractFixedSizeArray{S,T,N,R,L}},
    σ::Union{T,Int}
) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R.parameters, L)
    univariate_normal_quote( M, T, true,
        μ <: AbstractFixedSizeArray, false, track, (false,false,false), false, false
    )
end
@generated function Normal(
    ::Val{track},
    y::AbstractFixedSizeArray{S,T,N,R,L},
    μ::Union{T,Int,<:AbstractFixedSizeArray{S,T,N,R,L}},
    σ::AbstractFixedSizeArray{S,T,N,R,L}
) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R.parameters, L)
    univariate_normal_quote( M, T, true,
        μ <: AbstractFixedSizeMatrix, true, track, (false,false,false), false, true
    )
end
@generated function Normal_kernel(
    ::Val{track},
    y::AbstractFixedSizeArray{S,T,N,R,L},
    μ::Union{T,Int,<:AbstractFixedSizeArray{S,T,N,R,L}},
    σ::AbstractFixedSizeArray{S,T,N,R,L}
) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R.parameters, L)
    univariate_normal_quote( M, T, true,
        μ <: AbstractFixedSizeMatrix, true, track, (false,false,false), false, false
    )
end





