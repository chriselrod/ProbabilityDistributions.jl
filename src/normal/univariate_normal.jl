

@inline logdet_coef(y::AbstractFixedSizeVector, ::Any, σ::AbstractFixedSizeVector) = One()

function univariate_normal_quote(
    M::Int, @nospecialize(T), yisvec::Bool,
    @nospecialize(μisvec::Union{Bool,Nothing}), @nospecialize(σisvec::Union{Bool,Nothing}),
    (track_y, track_μ, track_σ)::NTuple{3,Bool},
    (inity, initμ, initσ)::NTuple{3,Bool},
    partial::Bool, calclogdet::Bool = false
)
    if M > 1
        pre_quote = quote
            qf = SIMDPirates.vbroadcast($(VectorizationBase.pick_vector(M,T)), zero($T))
        end
    else
        pre_quote = quote qf = zero($T) end
    end
    if partial && track_σ
        push!(pre_quote.args, :(σ = ∂canonicalize_Σ(σin)))
    else
        push!(pre_quote.args, :(σ = canonicalize_Σ(σin)))
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
    # @show M, T, yisvec, μisvec, σisvec
    # @show track_y, track_μ, track_σ
    # @show inity, initμ, initσ
    # @show partial, calclogdet
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
                δσ⁻² = δ * σ⁻²
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
            push!(return_expr.args, :(nld = nlogdet(σ)))
        else
            push!(return_expr.args, :(nld = vmul($(T(M)), nlogdet(σ))))
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
            else#if μisvec == false
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
            else#if σisvec == false
                if calclogdet
                    if yisvec
                        push!(return_expr.args, Expr(eq, :(∂σ[]), :(∂k∂i(σ) * vsum(qf) - $(T(M)) * ∂ld∂i(σ))))
                    else
                        push!(return_expr.args, Expr(eq, :(∂σ[]), :(∂k∂i(σ) * vsum(qf) - ∂ld∂i(σ))))
                    end
                else
                    push!(return_expr.args, Expr(eq, :(∂σ[]), :(∂k∂i(σ) * vsum(qf))))
                end
            end
        end
    end
    if calclogdet && track_σ
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
            # $(Expr(:meta,:inline))
            $pre_quote
            $(macroexpand(ProbabilityDistributions, loop_expr))
            $return_expr
            $retex
        end
    else
        quote
            # $(Expr(:meta,:inline))
            @fastmath begin
                $pre_quote
                $loop_expr
                $return_expr
            end
            $retex
        end
    end
    (yisvec && !σisvec) && pushfirst!(q.args, Expr(:meta, :inline)) # inline Target...
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

@noinline function univariate_normal_length(SV::Core.SimpleVector, RV::Core.SimpleVector)
    PaddedMatrices.isdense(SV, RV) || throw("Non-unit stride not yet supported.")
    # because isdense passed, last(RV) must equal prod(SV[1:N-1])
    last(RV)::Int * last(SV)::Int
end

@inline Normal(y::T) where {T <: Real} = Base.FastMath.mul_fast(T(-0.5), Base.FastMath.abs2_fast(y))
@inline Normal(::Val{(true,)}, y::Real) = Normal(y)
@inline Normal(::Val{(false,)}, y::T) where {T <: Real} = zero(T)

@generated function Normal(::Val{(true,)}, y::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    nrows = first(S.parameters)::Int
    if PaddedMatrices.isdense(S.parameters, X.parameters)
        M = N == 1 ? nrows : L
        W = VectorizationBase.pick_vector_width(M, T)
        return quote
            out = vbroadcast(Vec{$W,$T}, zero($T))
            @vvectorize $T 4 for m ∈ 1:$M
                yₘ = y[m]
                out = vmuladd(yₘ, yₘ, out)
            end
            vmul($(T(-0.5)), out)
        end
    else
        P = (X.parameters[2])::Int
        W = VectorizationBase.pick_vector_width(nrows, T)
        return quote
            out = vbroadcast(Vec{$W,$T}, zero($T))
            ind = 0
            Base.Cartesian.@nloops $(N-1) i j -> 1:size(y,j+1) begin
                @vvectorize $T 4 for i_0 ∈ 1:$nrows
                    yᵢ = y[ind + i_0]
                    out = vmuladd(yᵢ, yᵢ, out)
                end
                ind += $P
            end
            vmul($(T(-0.5)), out)
        end
    end
end
@generated function ∂Normal!(∂y::AbstractFixedSizeArray{S,T,N,X,L}, y::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    nrows = first(S.parameters)::Int
    if PaddedMatrices.isdense(S.parameters, X.parameters)
        M = N == 1 ? nrows : L
        W = VectorizationBase.pick_vector_width(M, T)
        return quote
            out = vbroadcast(Vec{$W,$T}, zero($T))
            @vvectorize $T 4 for m ∈ 1:$M
                yₘ = y[m]
                ∂y[m] = -yₘ
                out = vmuladd(yₘ, yₘ, out)
            end
            vmul($(T(-0.5)), out)
        end
    else
        P = (X.parameters[2])::Int
        W = VectorizationBase.pick_vector_width(nrows, T)
        return quote
            out = vbroadcast(Vec{$W,$T}, zero($T))
            ind = 0
            Base.Cartesian.@nloops $(N-1) i j -> 1:size(y,j+1) begin
                @vvectorize $T 4 for i_0 ∈ 1:$nrows
                    yᵢ = y[ind + i_0]
                    ∂y[ind + i_0] = -yᵢ
                    out = vmuladd(yᵢ, yᵢ, out)
                end
                ind += $P
            end
            vmul($(T(-0.5)), out)
        end
    end
end


@inline function ∂Normal!(∂y::U, y::T) where {T <: Real, U}
    t = Base.FastMath.mul_fast(T(-0.5), Base.FastMath.abs2_fast(y))
    if isinitialized(U)
        ∂y[] = Base.FastMath.sub_fast(∂y[], y)
    else
        ∂y[] = Base.FastMath.sub_fast(y)
    end
    t
end
@inline ∂Normal!(::Nothing, y::T) where {T <: Real} = zero(T)
@generated function Normal_kernel(x::AbstractFixedSizeArray{S,T,N,X,L}) where {S,T,N,X,L}
    P = univariate_normal_length(S.parameters, X.parameters)
    V = VectorizationBase.pick_vector(P,T)
    quote
        # $(Expr(:meta,:inline))
        target = vbroadcast($V, zero($T))
        @vvectorize $T 4 for p ∈ 1:$P
            xₚ = x[p]
            target = vmuladd(xₚ, xₚ, target)
        end
        vmul(target, vbroadcast($V, $(T(-0.5))))
    end
end
@inline function Normal_kernel(::Val{track}, x::AbstractFixedSizeArray{S,T}) where {S,T,track}
    first(track) ? Normal_kernel(x) : Zero()
end

# @eval loop to specify normals
for yisvec ∈ (true,false)
    if yisvec
        args1 = [:(y::AbstractFixedSizeArray{S,T,N,R,L})]
        M = :(univariate_normal_length(S.parameters, R.parameters))
        whereparams = [:S,:N,:R,:L]
        σisvec = :(σin <: AbstractFixedSizeArray)
    else
        args1 = [:(y::Union{T,RealFloat{<:Any,T}})]
        M = 1
        whereparams = Symbol[]
        σisvec = false
    end
    ∂args1 = [:(∂y::∂YN)]; ∂whereparams1 = [:∂YN, :∂ΣN]
    for μ ∈ (false,true)
        if μ
            args2 = push!(copy(args1), yisvec ? :(μ::Union{T,Int,<:AbstractFixedSizeArray{S,T,N,R,L}}) : :(μ::Union{T,Int}))
            track_μ = :(track[2])
            initμ = :(!isinitialized(∂MN))
            ∂args2 = push!(copy(∂args1), :(∂μ::∂MN));
            ∂whereparams2 = push!(copy(∂whereparams1), :∂MN)
            μisvec = :(μ <: AbstractFixedSizeArray)
            ∂track_μ = :(∂MN !== Nothing)
        else
            args2 = args1
            ∂args2 = ∂args1
            ∂whereparams2 = ∂whereparams1
            track_μ = false
            μisvec = nothing
            initμ = false
            ∂track_μ = false
        end
        ∂args3 = push!(copy(∂args2), :(∂σ::∂ΣN))
        args3 = push!(copy(args2), yisvec ? :(σin::Union{T,Int,RealFloat{<:Any,T},Precision,<:AbstractFixedSizeArray{S,T,N,R,L}}) : :(σin::Union{T,Int,RealFloat{<:Any,T},Precision{T}}))
        # @show μ, args3, μisvec
        for calclogdet ∈ (true,false)
            n = calclogdet ? :Normal : :Normal_kernel
            ∂n = calclogdet ? :∂Normal! : :∂Normal_kernel!
            @eval @generated function $n(::Val{track}, $(args3...)) where {track, $(whereparams...), T <: Union{Float32,Float64}}
                univariate_normal_quote(
                    $M, T, $yisvec, $μisvec, $σisvec,
                    (first(track), $track_μ, last(track)),
                    (false, false, false),
                    false, $calclogdet
                )
            end
            @eval @generated function $n($(args3...)) where {$(whereparams...), T <: Union{Float32,Float64}}
                univariate_normal_quote(
                    $M, T, $yisvec, $μisvec, $σisvec,
                    ((true, $μ, true)),
                    (false, false, false),
                    false, $calclogdet
                )
            end
            
            @eval @generated function $∂n($(∂args3...), $(args3...)) where {$(∂whereparams2...), $(whereparams...), T <: Union{Float32,Float64}}
                univariate_normal_quote(
                    $M, T, $yisvec, $μisvec, $σisvec,
                    (∂YN !== Nothing, $∂track_μ, ∂ΣN !== Nothing),
                    (!isinitialized(∂y), $initμ, !isinitialized(∂σ)),
                    true, $calclogdet
                )
            end
        end
    end
end


