

const DISTRIBUTION_DIFF_RULES = Set{Symbol}()

using ReverseDiffExpressionsBase: adj, InitializedVarTracker

"""
Do we inline the expression, or try and take advantage of multiple dispatch?
The latter option sounds more appealing flexible, and easier to test.
Constants will be dropped based on their type.

The normal distribution has 3 arguments, dispatching to either:
y ~ normal(μ = 0, σ = 1) # default arguments
or
y ~ normal(μ, inv(chol(Σ)))

This may be handled under the hood via how we represent Σ.

"""
@noinline function distribution_diff_rule!(ivt::InitializedVarTracker, first_pass::Vector{Any}, tracked_vars, mod, out, A, f, verbose = false)
    track_out = false
    # verbose = true
    function_call = Expr(:call, :($mod.ProbabilityDistributions.$(Symbol(:∂, f, :!))))
    for a ∈ A
        if a ∉ tracked_vars
            push!(function_call.args, nothing)
            continue
        end
        track_out = true
        # push!(function_call.args, adj(out, a))
        push!(function_call.args, uninitialize_update!(first_pass, initialized_seeds, adj(a), mod, aliases)) # ∂target/∂a
    end
    for a ∈ A
        if a ∈ tracked_vars && ReverseDiffExpressionsBase.initialize!(ivt, first_pass, a, a, mod) # must_initialize
            push!(function_call.args, :($mod.uninitialized($a)))
        else
            push!(function_call.args, a)
        end
    end
    if track_out
        if verbose
            printstring = "distribution $f (ret: $out): "
            push!(first_pass, :(println($printstring)))
        end
        push!(first_pass, Expr(:(=), out, function_call))
        if verbose
            push!(first_pass, :(@show $out))
            for a ∈ A
                a ∈ tracked_vars && push!(first_pass, :(@show $(adj(a))))
            end
        end
    end
    nothing
end
# """
# Arguments are: y, logitθ
# """
# function ∂Bernoulli_logit_quote(y_is_param, logitθ_is_param)
#     @assert y_is_param == false
#
#     out = zero(eltype(p))
#     @inbounds @simd for i ∈ eachindex(y,p)
#         OmP = 1 - p[i]
#         out += y[i] ? p[i] : OmP
#     end
#     out
# end
const FMADD_DISTRIBUTIONS = Set{Symbol}()
push!(FMADD_DISTRIBUTIONS, :Bernoulli_logit)

# """
# Arguments are: y, logitθ
# """
# function ∂Bernoulli_logit_quote(y_is_param, β_is_param, X_is_param, α_is_param)
#     @assert y_is_param == false
#     quote
#         T = promote_type(eltype(α, X, β))
#         target = zero(T)
#         @fastmath @inbounds @simd ivdep for i ∈ eachindex(y)
#             OmP = one(T) / (one(T) + SLEEFPirates.exp( α + β[i] * x[i]  ))
#             P = one(T) - P
#             target += y[i] ? P : OmP
#         end
#         target
#     end
#
# end

function Bernoulli_logit_quote(T)
    W = VectorizationBase.pick_vector_width(T)
    q = quote
        $(Expr(:meta, :inline))
        target = vbroadcast(Vec{$W,$T}, zero($T))
        @vvectorize $T for i ∈ eachindex(y)
            αᵢ = α[i]
            invOmP = one($T) + SLEEFPirates.exp( αᵢ )
            nlogOmP = log(invOmP)
            nlogP = nlogOmP - αᵢ
            target = vsub(target, y[i] ? nlogP : nlogOmP)
        end
        target
    end
    simplify_expr(q)
end

@generated function Bernoulli_logit(
    ::Val{track},
    y::BitVector, α::AbstractVector{T}
) where {T, track}
    y_is_param, α_is_param = track
    @assert y_is_param == false
    Bernoulli_logit_quote(T)
end
function Bernoulli_logit_constant(y::BitVector, α::AbstractVector{T}, ::Val{track} = Val{(false,true)}()) where {T,track}
    zero(T)
end

function ∂Bernoulli_logit_quote(T, initialized::Bool = false)
    W = VectorizationBase.pick_vector_width(T)
    ∂αop = initialized ? :(+=) : :(=)
    q = quote
        $(Expr(:meta, :inline))
        target = vbroadcast(Vec{$W,$T}, zero($T))
        @vvectorize $T for i ∈ eachindex(y)
            αᵢ = α[i]
            invOmP = (one($T) + SLEEFPirates.exp( αᵢ ))
            ∂logP = one($T) / invOmP
            nlogOmP = SLEEFPirates.log(invOmP)
            nlogP = nlogOmP - αᵢ
            target = vsub(target, y[i] ? nlogP : nlogOmP)
            $(Expr(∂αop, :(∂α[i]), :(y[i] ? ∂logP : ∂logP - one($T))))
        end
        target
    end
    simplify_expr(q)
end

@generated function ∂Bernoulli_logit!(
    ::Nothing, ∂α::AbstractVector{T}, y::BitVector, α::AbstractVector{T}
) where {T}
    ∂Bernoulli_logit_quote(T, isinitialized(∂α))
end

function ∂Bernoulli_logit(y::BitVector, α::AbstractVector{T}, ::Val{track} = Val{(false,true)}()) where {T,track}
    y_is_param, α_is_param = track
    @assert y_is_param == false
    if α_is_param
        ∂α = similar(α)
        return ∂Bernoulli_logit!(∂α, y, α), ∂α
    else
        return Bernoulli_logit(y, α)
    end
end

function ∂Bernoulli_logit(
    sptr::StackPointer, ::Val{track},# = Val{(false,true)}(),
    y::BitVector, α::AbstractVector{T}
) where {T,track}
    y_is_param, α_is_param = track
    @assert y_is_param == false
    if α_is_param
        sptr, ∂α = similar(sptr, α)
        return sptr, (∂Bernoulli_logit!(∂α, y, α), ∂α)
    else
        return sptr, Bernoulli_logit(y, α)
    end
end
function ∂Bernoulli_logit(
        sptr::StackPointer, y::BitVector, α::AbstractVector{T}
) where {T}
    ∂Bernoulli_logit(sptr, Val{(false,true)}(), y, α)
end
push!(DISTRIBUTION_DIFF_RULES, :Bernoulli_logit)

function Binomial_logit_quote(T, yconst::Bool = false)
    W = VectorizationBase.pick_vector_width(T)
    q = quote
        $(Expr(:meta, :inline))
        target = vbroadcast(Vec{$W,$T}, zero($T))
        @vvectorize $T for i ∈ eachindex(s)
            αᵢ = α[i]
            invOmP = one($T) + SLEEFPirates.exp( αᵢ )
            nlogOmP = log(invOmP)
            target = vsub(target, $(yconst ? :N : :(N[i])) * nlogOmP - s[i] * αᵢ)
        end
        target
    end
    simplify_expr(q)
end

@generated function Binomial_logit(
    ::Val{track},
    s::AbstractVector{T}, α::AbstractVector{T}, N::AbstractVector{T}    
) where {track, T}
    s_is_param, α_is_param, N_is_param = track
    @assert !s_is_param && !N_is_param
    Binomial_logit_quote(T)
end

@generated function Binomial_logit(
    ::Val{track},
    s::AbstractVector{T}, α::AbstractVector{T}, N::T
) where {track, T}
    s_is_param, α_is_param, N_is_param = track
    @assert !s_is_param && !N_is_param
    Binomial_logit_quote(T, true)
end
@generated function Binomial_logit(
    s::AbstractVector{T}, α::AbstractVector{T}, N::Union{T,<: AbstractVector{T}}
) where {track, T}
    Binomial_logit(Val{(false,true,false)}(), s, α, N)
end

# faster, works for Floats that can't be convered to Int
# lbinom(n, p) = lgamma(n+1) - lgamma(n-p+1) - lgamma(p+1) 
# more accurate, but doesn't allow n or p to have decimals.
lbinom(n, p) = first(logabsbinomial(trunc(Int,n), trunc(Int,p))) 
function Binomial_logit_constant(
    ::Val{track}, s::AbstractVector{T}, α::AbstractVector{T}, N::AbstractVector{T}
) where {track,T}
    c = zero(T)
    @inbounds for i ∈ eachindex(s, N)
        c += lbinom(N[i],s[i])
    end
    c
end
function Binomial_logit_constant(
    ::Val{track}, s::AbstractVector{T}, α::AbstractVector{T}, N::T
) where {track,T}
    c = zero(T)
    @inbounds for i ∈ eachindex(s)
        c += lbinom(N,s[i])
    end
    c
end
function Binomial_logit_constant(
    s::AbstractVector{T}, α::AbstractVector{T}, N::Union{T,<:AbstractVector{T}}    
) where {track,T}
    Binomial_logit_constant(Val{(false,true,false)}(), s, α, N)
end

function ∂Binomial_logit_quote(T, nconst::Bool = false, initialized::Bool = false)
    W = VectorizationBase.pick_vector_width(T)
    ∂αop = initialized ? :(+=) : :(=)
    q = quote
        $(Expr(:meta, :inline))
        target = vbroadcast(Vec{$W,$T}, zero($T))
        @vvectorize $T for i ∈ eachindex(s)
            αᵢ = α[i]
            expαᵢ = SLEEFPirates.exp( αᵢ )
            invOmP = ( one($T) + expαᵢ )
            ∂logP = one($T) / invOmP
            nlogOmP = SLEEFPirates.log(invOmP)
            $(nconst ? :(sᵢ = s[i]) : :(sᵢ = s[i]; Nᵢ = N[i]))
            target = vsub(target, Nᵢ * nlogOmP - sᵢ * αᵢ )
            $(Expr(∂αop, :(∂α[i]), :(sᵢ - Nᵢ * ∂logP * expαᵢ)))
        end
        target
    end
    simplify_expr(q)
end

@generated function ∂Binomial_logit!(
    ::Nothing, ∂α::∂Α, ::Nothing,
    s::AbstractVector{T}, α::AbstractVector{T}, N::AbstractVector{T}
) where {T, ∂Α <: AbstractVector{T}}
    ∂Binomial_logit_quote(T, false, isinitialized(∂Α))
end
@generated function ∂Binomial_logit!(
    ::Nothing, ∂α::∂Α, ::Nothing,
    s::AbstractVector{T}, α::AbstractVector{T}, Nᵢ::T
) where {T, ∂Α <: AbstractVector{T}}
    ∂Binomial_logit_quote(T, true, isinitialized(∂Α))
end

 #    ::Val{track} = Val{(false,true,false)}()
function ∂Binomial_logit(
    ::Val{track}, s::AbstractVector{T}, α::AbstractVector{T}, N::Union{T,AbstractVector{T}}
) where {T,track}
    s_is_param, α_is_param, N_is_param = track
    @assert !s_is_param && !N_is_param
    if α_is_param
        ∂α = similar(α)
        return ∂Binomial_logit!(∂α, s, uninitialized(α), N), ∂α
    else
        return Binomial_logit(s, α, N)
    end
end
#::Val{track} = Val{(false,true,false)}()
function ∂Binomial_logit(
    sptr::StackPointer, ::Val{track}, s::AbstractVector{T}, α::AbstractVector{T}, N::Union{T,AbstractVector{T}}
) where {T,track}
    s_is_param, α_is_param, N_is_param = track
    @assert !s_is_param && !N_is_param
    if α_is_param
        sptr, ∂α = similar(sptr, α)
        return sptr, (∂Binomial_logit!(∂α, s, uninitialized(α), N), ∂α')
    else
        return sptr, Binomial_logit(s, α, N)
    end
end

push!(DISTRIBUTION_DIFF_RULES, :Binomial_logit)

#::Val{track} = Val{(false,false,true,true)}()
@generated function Bernoulli_logit(
    ::Val{track}, y::BitVector, X::AbstractMatrix{T}, β::AbstractVector{T}, α::AbstractFloat
) where {T, track}
    y_is_param, β_is_param, X_is_param, α_is_param = track
    @assert y_is_param == false
    if PaddedMatrices.is_sized(β)
        N_β = PaddedMatrices.type_length(β)
        q = quote
            $(Expr(:meta, :inline))
            # T = promote_type(eltype(α),eltype(β),eltype(X))
#            target = zero($T)
            target = vbroadcast(Vec{$(VectorizationBase.pick_vector_width(T)),$T}, zero($T))
            @vectorize $T for i ∈ eachindex(y)
                # a = $(Expr(:call, :+, :α, [:(X[i,$n] * β[$n]) for n ∈ 1:N_β]...))
                # Break it up, so inference still works for N_β > 15
                a = vmuladd(X[i,1], β[1], α)
                $([:(a = vmuladd(X[i,$n], β[$n], a)) for n ∈ 2:N_β]...)
                OmP = one($T) / (one($T) + SLEEFPirates.exp( a ))
                # P = one($T) - OmP
                logOmP = log(OmP)
                logP = a + logOmP
                target = vadd(target, y[i] ? logP : logOmP)
            end
            target
        end
    else
        throw("""
            Dynamically sized coefficient vector β is not yet supported.
            Feel free to file an issue, so that adding support will be prioritized.
            As a workaround, use a β with type parameterized by size.
            Eg, a padded vector from PaddedMatrices.jl or SVector from StaticArrays.jl.
        """)
    end
    simplify_expr(q)
end
Bernoulli_logit_constant(y, X, β, α, ::Any) = zero(eltype(β))
@generated function ∂Bernoulli_logit!(
    ::Nothing, ::Nothing, ∂β::∂Β, ∂α::∂Α,
    y::BitVector, X::AbstractMatrix{T}, β::AbstractVector{T}, α::AbstractFloat,
) where {T, ∂Β, ∂Α}
    β_is_param = ∂Β !== Nothing
    α_is_param = ∂Α !== Nothing
    if PaddedMatrices.is_sized(β)
        N_β = PaddedMatrices.type_length(β)
        init_q = quote
            # T = promote_type(eltype(α),eltype(β),eltype(X))
#            target = zero($T)
            target = vbroadcast(Vec{$(VectorizationBase.pick_vector_width(T)),$T}, zero($T))
        end
        out_expr = quote end
        partial_exprs = Expr[]
        ∂P_undefined = true
        if β_is_param
            L_β = PaddedMatrices.pick_L(N_β, T)
            ∂P_undefined = false
            # push!(partial_exprs, :(∂P = OmP * P))
            push!(partial_exprs, :(∂logP = OmP ))
            push!(partial_exprs, :(∂logOmP = - P))
            out∂β = if isinitialized(∂Β)
                quote $([:(@inbounds ∂β[$n] += $(Symbol(:∂βP_, n))) for n ∈ 1:N_β]...) end
            else
                quote $([:(@inbounds ∂β[$n] = $(Symbol(:∂βP_, n))) for n ∈ 1:N_β]...) end
            end
            push!(out_expr.args, out∂β)
            for n ∈ 1:N_β
                push!(init_q.args, :($(Symbol(:∂βP_, n)) = zero($T)) )
            end
            partial_exprs_q = quote end
            for n ∈ 1:N_β
                push!(partial_exprs_q.args, :($(Symbol(:∂βP_, n)) += y[i] ? ∂logP * X[i,$n] : ∂logOmP * X[i,$n] ) )
            end
            push!(partial_exprs, partial_exprs_q)
        end
        # if X_is_param
        #     ∂P_undefined && push!(partial_exprs, :(∂P = OmP * P))
        #     ∂P_undefined = false
        #     push!(out_expr.args, Expr(:tuple, [Symbol(:∂XP_, n) for n ∈ 1:N_β]...))
        #     for n ∈ 1:N_β
        #         push!(init_q.args, :($(Symbol(:∂XP_, n)) = zero($T)) )
        #     end
        #     push!(partial_exprs, quote
        #         $([:($(Symbol(:∂XP_, n)) += y[i] ? β[$n] * ∂P : - β[$n] * ∂P) for n ∈ 1:N_β]...)
        #     end)
        # end
        if α_is_param
            # ∂P_undefined && push!(partial_exprs, :(∂P = OmP * P))
            ∂P_undefined && push!(partial_exprs, :(∂logP = OmP ))
            ∂P_undefined && push!(partial_exprs, :(∂logOmP = - P))
            ∂P_undefined = false
            push!(out_expr.args, isinitialized(∂Α) ? :(∂α[] += ∂αP) : :(∂α[] = ∂αP))
            push!(init_q.args, :(∂αP = zero($T)))
            push!(partial_exprs, :(∂αP += y[i] ? ∂logP : ∂logOmP))
        end
        W, Wshift = VectorizationBase.pick_vector_width_shift(T)
        # unroll_factor = max(8 >>> Wshift, 1)
        q = quote
            $(Expr(:meta, :inline))
            $init_q
            @vvectorize $T for i ∈ eachindex(y)
                # a = $(Expr(:call, :+, :α, [:(X[i,$n] * β[$n]) for n ∈ 1:N_β]...))
                a = vmuladd(X[i,1], β[1], α)
                $([:(a = vmuladd(X[i,$n], β[$n], a)) for n ∈ 2:N_β]...)
                OmP = one($T) / (one($T) + SLEEFPirates.exp( a ))
                P = one($T) - OmP
                logOmP = SLEEFPirates.log(OmP)
                logP = a + logOmP
                target = vadd(target, y[i] ? logP : logOmP)
                $(partial_exprs...)
            end
            $out_expr
            target
        end
    else
        throw("""
            Dynamically sized coefficient vector β is not yet supported.
            Feel free to file an issue, so that adding support will be prioritized.
            As a workaround, use a β with type parameterized by size.
            Eg, a padded vector from PaddedMatrices.jl or SVector from StaticArrays.jl.
        """)
    end
    simplify_expr(q)
end

push!(DISTRIBUTION_DIFF_RULES, :Bernoulli_logit)

@generated function LKJ(::Val{track}, L::AbstractCorrCholesky{N,T}, η::T) where {N,T,track}
    W = VectorizationBase.pick_vector_width(N-1,T)
    # If the remainder is 1, we'll skip the first element (which equals 1, so the log is 0)
    # otherwise, we wont skip, because memory should be aligned with accessing the first element.
    rem1 = (N % W) == 1
    track_L, track_η = track
    if !track_L && !track_η
        return zero(T)
    end
    if track_L
        q = quote
            $(Expr(:meta,:inline))
            target = vbroadcast(SVec{$W,$T}, zero($T))
            logd = logdiag(L)
            @vvectorize $T for n ∈ 1:$(rem1 ? N-1 : N)
                target = vfmadd( ($(N - 2 - rem1) - n + 2η), logd[$(rem1 ? :(n+1) : :n)], target)
            end
            vtarget = extract_data(target)
        end
    end
    if track_η
        ηtarget = track_L ? :starget : :target
        ηoneadjust = if isodd(N)
            T(0.25*(N - 1)*log(π) - 0.25abs2(N-1) * log(2) - (N-1) * first(logabsgamma(0.5*(N+1))))
        else
            T(-0.25*(N)*log(π) + 0.25N * (3N - 4)*log(2) + N*first(logabsgamma(0.5N)) - (N-1)*first(logabsgamma(N)))
        end
        q_η = quote
            $ηtarget = zero($T)
            if η == one($T)
                for n ∈ 1:$((N-1) >>> 1)
                    $ηtarget -= first(logabsgamma($(T(2))*n))
                end
                # more accurate to sum this way?
                $ηtarget -= $ηoneadjust
            else
                for n ∈ 1:$(N-1)
                    $ηtarget -= first(logabsgamma(η + 0.5*($(N-1) - n)))
                end
                $ηtarget += $(N-1)*first(logabsgamma(η + $(0.5*(N-1))))
            end
        end
        push!(q.args, q_η)
        if track_L
            vstarget = :(Base.setindex(vbroadcast(Vec{$W,$T}, zero($T)), Core.VecElement($ηtarget), 1))
            push!(q.args, :(vadd(vtarget, $vstarget)))
        else
            push!(q.args, ηtarget)
        end
    end
    simplify_expr(q)
end
@generated function LKJ_constant(L::AbstractCorrCholesky{N,T}, η::T, ::Val{track}) where {N,T,track}
    :(LKJ(L,η,Val{$((!track[1],!track[2]))}()) - $(0.5*log(π)*StructuredMatrices.binomial2(N)))
end
@generated function ∂LKJ!(
    ∂L::PL, ∂η::Pη,
    L::AbstractCorrCholesky{N,T}, η::T
) where {N,T,PL,Pη}
    track_L = PL !== Nothing
    track_η = Pη !== Nothing
    if track_L
        L_uninit = !isinitialized(PL)
        ∂Lop = L_uninit ? :(=) : :(+=)
    end
    if track_η
        η_uninit = !isinitialized(Pη)
        ∂ηop = η_uninit ? :(=) : :(+=)
    end
    W = VectorizationBase.pick_vector_width(N-1,T)
    if track_η
        ∂η_q = quote
            starget = zero($T)
            for n ∈ 1:$(N-1)
                c = η + 0.5*($(N-1) - n) # ∂c/∂η = 1
                starget -= first(logabsgamma(c))
                ∂ηs -= digamma(c) # ∂target/∂η = ∂target/∂c * ∂c/∂η = -digamma(c) * 1
            end
            $(Expr(∂ηop, :(∂η[]), ∂ηs))
            vadd(extract_data(target), Base.setindex(vbroadcast(Vec{$W,$T},zero($T)), startget, 1))
        end
    end
    if track_L && track_η
        q = quote
            $(Expr(:meta,:inline))
            target = vbroadcast(SVec{$W,$T}, zero($T))
            logd = logdiag(L)
            ∂ηs = zero($T)
            @vvectorize $T for n ∈ 1:$(N-1)
                ∂ηₙ = logd[n+1]
                coef = ($(N - 3) - n + 2η)
                target = vfmadd( coef, ∂ηₙ, target )
                $(Expr(∂Lop, :(∂L[n+1]), :(coef / L[n+1])))
                ∂ηs += $(T(2))*∂ηₙ
            end
            $∂η_q
        end
        L_uninit && pushfirst!(q.args, :(@inbounds ∂L[1] = zero($T)))
    elseif track_L
        q = quote
            $(Expr(:meta,:inline))
            target = vbroadcast(SVec{$W,$T}, zero($T))
            logd = logdiag(L)
            @vvectorize $T for n ∈ 1:$(N-1)
                ∂ηₙ = logd[n+1]
                coef = ($(N - 3) - n + 2η)
                target = vfmadd(coef, ∂ηₙ, target)
                $(Expr(∂Lop, :(∂L[n+1]), :(coef / L[n+1])))
            end
            extract_data(target)
        end
        L_uninit && pushfirst!(q.args, :(@inbounds ∂L[1] = zero($T)))
    elseif track_η
        q = quote
            $(Expr(:meta,:inline))
            target = vbroadcast(SVec{$W,$T}, zero($T))
            logd = logdiag(L)
            ∂ηs = zero($T)
            @vvectorize $T for n ∈ 1:$(N-1)
                ∂ηₙ = logd[n+1]
                coef = ($(N - 3) - n + 2η)
                target = vfmadd( coef, ∂ηₙ, target )
                ∂ηs += $(T(2))*∂ηₙ
            end
            $∂η_q
        end
    else
        return zero(T)
    end
    simplify_expr(q)
end
@generated function ∂LKJ(sp::PaddedMatrices.StackPointer, ::Val{track}, L::AbstractCorrCholesky{N,T}, η::T) where {N,T,track}
    track_L, track_η = track
    q = quote end
    ret_expr = Expr(:tuple, :target)
    if track_L
        push!(q.args, :((sp, ∂L) = PtrVector{$N,$T}(sp)))
        push!(ret_expr.args, :(Diagonal(∂L)))
    else
        push!(q.args, :(∂L = nothing))
    end
    if track_η
        push!(q.args, :(∂η = Ref{$T}()))
        push!(ret_expr.args, :(∂η[]))
    else
        push!(q.args, :(∂η = nothing))
    end
    push!(q.args, :(∂LKJ!(uninitialized(∂L), uninitialized(∂η), L, η)))
    push!(q.args, :(sp, $ret_expr))
    q
end
push!(DISTRIBUTION_DIFF_RULES, :LKJ)

function gamma_quote(
    M::Int, T, (yisvec, αisvec, βisvec)::NTuple{3,Bool},
    (track_y, track_α, track_β)::NTuple{3,Bool}, partial::Bool,
    (inity, initα, initβ)::NTuple{3,Bool} = (true,true,true)
)
    q = quote end
    pre_quote = quote end
    return_expr = quote end
    loop = any((yisvec, αisvec, βisvec))
    # set initialized_target to loop; if we are looping, we'll start out at zero
    initialized_target = loop
    if yisvec
        yexpr = :(y[i])
        push!(pre_quote.args, :(logy = PaddedMatrices.LazyMap(log, y)))
        logyexpr = :(logy[i])
    else
        yexpr = :y
        push!(pre_quote.args, :(logy = Base.log(y)))
        logyexpr = :logy
    end
    if αisvec
        αexpr = :(α[i])
        lgammaαexpr = :(first(SpecialFunctions.logabsgamma(α[i])))
        αm1expr = :(α[i] - one(eltype(α)))
    else
        αexpr = :α
        lgammaαexpr = :(lgammaα)
        αm1expr = :(αm1)
        push!(pre_quote.args, :(lgammaα = first(SpecialFunctions.logabsgamma(α))))
        push!(pre_quote.args, :(αm1 = α - one(eltype(α))))
    end
    if βisvec
        βexpr = :(β[i])
        push!(pre_quote.args, :(logβ = PaddedMatrices.LazyMap(log, y)))
        logβexpr = :(logβ[i])
    else
        βexpr = :β
        logβexpr = :logβ
        push!(pre_quote.args, :(logβ = Base.log(β)))
    end
    if partial
        if track_y
            if yisvec
                ∂yassignment = :(=)
                ∂ystorage = :(∂yᵢ)
                # push!(return_expr.args, :(∂y'))
            else
                ∂ystorage = :(∂ys)
                if inity
                    push!(return_expr.args, :(∂y[] = ∂ys))
                else
                    push!(return_expr.args, :(∂y[] += ∂ys))
                end
                if loop
                    ∂yassignment = :(+=)
                    push!(pre_quote.args, :(∂ys = zero($T)))
                else
                    ∂yassignment = :(=)
                end
            end
        end
        if track_α
            if αisvec
                ∂αassignment = :(=)
                ∂αstorage = :(∂αᵢ)
            else
                ∂αstorage = :(∂αs)
                if initα
                    push!(return_expr.args, :(∂α[] = ∂αs))
                else
                    push!(return_expr.args, :(∂α[] += ∂αs))
                end
                if loop
                    ∂αassignment = :(+=)
                    push!(pre_quote.args, :(∂αs = zero($T)))
                else
                    ∂αassignment = :(=)
                end
            end
        end
        if track_β
            if βisvec
                ∂βassignment = :(=)
                ∂βstorage = :(∂βᵢ)
            else
                ∂βstorage = :(∂βs)
                if initβ
                    push!(return_expr.args, :(∂β[] = ∂βs))
                else
                    push!(return_expr.args, :(∂β[] += ∂βs))
                end
                if loop
                    ∂βassignment = :(+=)
                    push!(pre_quote.args, :(∂βs = zero($T)))
                else
                    ∂βassignment = :(=)
                end
            end
        end
    end
    if track_α || track_β
        push!(q.args, :( lβ = $logβexpr))
        if initialized_target
            push!(q.args, :( target = vmuladd($αexpr, lβ, target) ) )
        else
            push!(q.args, :( target = $αexpr * lβ) )
            initialized_target = true
        end
        if partial
            track_α && push!(q.args, Expr(∂αassignment, ∂αstorage, :lβ) )
            track_β && push!(q.args, Expr(∂βassignment, ∂βstorage, αexpr) )
        end
    end
    if track_α || track_y
        push!(q.args, :(ly = $logyexpr))
        if initialized_target
            push!(q.args, :( target = vmuladd($αm1expr, ly, target) ) )
        else
            push!(q.args, :( target = $αm1expr * ly ) )
            initialized_target = true
        end
        if partial
            if track_α
                if αisvec
                    push!(q.args, :(∂α₂ = $∂αstorage + ly))
                else
                    push!(q.args, Expr(:(+=), ∂αstorage, :ly) )
                end
            end
            track_y && push!(q.args, Expr(∂yassignment, ∂ystorage, :($αm1expr / $yexpr) ) )
        end
    end
    if track_β || track_y
        # initialized_target == true
        push!(q.args, :(target = vfnmadd($βexpr, $yexpr, target)))
        if partial
            if track_β
                if βisvec
                    push!(q.args, Expr(initβ ? :(=) : :(+=), :(∂β[i]), :($∂βstorage - $yexpr) ) )
                else
                    push!(q.args, Expr(:(-=), ∂βstorage, yexpr ) )
                end
            end
            if track_y
                if yisvec
                    push!(q.args, Expr(inity ? :(=) : :(+=), :(∂y[i]), :($∂ystorage - $βexpr) ) )
                else
                    push!(q.args, Expr(:(-=), ∂ystorage, βexpr ) )
                end
            end
        end
    end
    if track_α
        # initialized_target == true, because (if track_α || track_β) == true
        push!(q.args, :(target = vsub(target, $lgammaαexpr)))
        if partial
            if αisvec
                push!(q.args, Expr(initα ? :(=) : :(+=), :(∂α[i]), :(∂α₂ - SpecialFunctions.digamma($αexpr) ) ) )
            else
                push!(pre_quote.args, Expr(:(-=), ∂αstorage, :($M * SpecialFunctions.digamma(α)) ) )
            end
        end
    end
    q = if loop
        quote
            $(Expr(:meta,:inline))
            @fastmath begin
                $pre_quote
            end
            target = vbroadcast(Vec{$(VectorizationBase.pick_vector_width(M,T)),$T}, zero($T))
#            out = zero($T)
            @vvectorize $T for i ∈ 1:$M
                $q
            end
            @fastmath begin
                $return_expr
            end
            target
        end
    else
        quote
            $(Expr(:meta,:inline))
            @fastmath begin
                $pre_quote
                $q
                $return_expr
                target
            end
        end
    end
    simplify_expr(q)
end

function gamma_alloc_quote(
    M::Int, T, (yisvec, αisvec, βisvec)::NTuple{3,Bool},
    (track_y, track_α, track_β)::NTuple{3,Bool}, sp::Bool = true
)
    q = quote end
    return_expr = Expr(:tuple, :target)
    if track_y
        if yisvec
            if sp
                push!(q.args, :((sp,∂y) = PaddedMatrices.PtrVector{$M,$T}(sp)))
            else
                push!(q.args, :(∂y = PaddedMatrices.FixedSizeVector{$M,$T}(undef)))
            end
            push!(return_expr.args, :(∂y))
        else
            push!(q.args, :(∂y = Ref{$T}()))
            push!(return_expr.args, :(∂y[]))
        end
    else
        push!(q.args, :(∂y = nothing))
    end
    if track_α
        if αisvec
            if sp
                push!(q.args, :(∂α = PaddedMatrices.PtrVector{$M,$T}(sp)))
            else
                push!(q.args, :(∂α = PaddedMatrices.FixedSizeVector{$M,$T}(undef)))
            end
            push!(return_expr.args, :(∂α))
        else
            push!(q.args, :(∂α = Ref{$T}()))
            push!(return_expr.args, :(∂α[]))
        end
    else
        push!(q.args, :(∂α = nothing))
    end
    if track_β
        if βisvec
            if sp
                push!(q.args, :(∂β = PaddedMatrices.PtrVector{$M,$T}(sp)))
            else
                push!(q.args, :(∂β = PaddedMatrices.FixedSizeVector{$M,$T}(undef)))
            end
            push!(return_expr.args, :(∂β))
        else
            push!(q.args, :(∂β = Ref{$T}()))
            push!(return_expr.args, :(∂β[]))
        end
    else
        push!(q.args, :(∂β = nothing))
    end
    push!(q.args, :(target = ∂Gamma!(uninitialized(∂y), uninitialized(∂α), uninitialized(∂β), y, α, β)))
    push!(q.args, return_expression(return_expr, sp))
    q
end

# α * log(β) + (α-1) * log(y) - β*y - lgamma(α)
@generated function Gamma(
    ::Val{track},
    y::PaddedMatrices.AbstractFixedSizeVector{M,T},
    α::Union{T, <: PaddedMatrices.AbstractFixedSizeVector{M,T}},
    β::Union{T, <: PaddedMatrices.AbstractFixedSizeVector{M,T}}
) where {track,M,T}
#) where {track,T,M}
    αisvec = isa(α, PaddedMatrices.AbstractFixedSizeVector)
    βisvec = isa(β, PaddedMatrices.AbstractFixedSizeVector)
    gamma_quote(M, T, (true, αisvec, βisvec), track, false)
end

@generated function ∂Gamma!(
    ∂y::∂YN, ∂α::∂ΑN, ∂β::∂ΒN,
    y::PaddedMatrices.AbstractFixedSizeVector{M,T},
    α::Union{T,<:PaddedMatrices.AbstractFixedSizeVector{M,T}},
    β::Union{T,<:PaddedMatrices.AbstractFixedSizeVector{M,T}}
) where {∂YN, ∂ΑN, ∂ΒN, M, T}
    yisvec = true
    αisvec = isa(α, PaddedMatrices.AbstractFixedSizeVector)
    βisvec = isa(β, PaddedMatrices.AbstractFixedSizeVector)
    track_y = ∂YN !== Nothing
    track_α = ∂ΑN !== Nothing
    track_β = ∂ΒN !== Nothing
    inity = !isinitialized(∂YN)
    initα = !isinitialized(∂ΑN)
    initβ = !isinitialized(∂ΒN)
    gamma_quote(
        M, T, (yisvec, αisvec, βisvec),
        (track_y, track_α, track_β), true,
        (inity, initα, initβ)
    )
end

@generated function ∂Gamma(
    ::Val{track},
    y::PaddedMatrices.AbstractFixedSizeVector{M,T},
    α::Union{T,<:PaddedMatrices.AbstractFixedSizeVector{M,T}},
    β::Union{T,<:PaddedMatrices.AbstractFixedSizeVector{M,T}}
#) where {track,M,T}
) where {track,T,M}
    αisvec = isa(α, PaddedMatrices.AbstractFixedSizeVector)
    βisvec = isa(β, PaddedMatrices.AbstractFixedSizeVector)
    gamma_alloc_quote(M, T, (true, αisvec, βisvec), track, false)
end
@generated function ∂Gamma(
    sp::StackPointer,
    ::Val{track},
    y::PaddedMatrices.AbstractFixedSizeVector{M,T},
    α::Union{T,<:PaddedMatrices.AbstractFixedSizeVector{M,T}},
    β::Union{T,<:PaddedMatrices.AbstractFixedSizeVector{M,T}}
) where {track,M,T}
            # ::Val{track}) where {track,T,M}
    αisvec = isa(α, PaddedMatrices.AbstractFixedSizeVector)
    βisvec = isa(β, PaddedMatrices.AbstractFixedSizeVector)
    gamma_alloc_quote(M, T, (true, αisvec, βisvec), track, true)
end
@generated function Gamma(::Val{track}, y::T, α::T, β::T) where {track,T <: Real}
    gamma_quote(1, T, (false, false, false), track, false)
end
@generated function ∂Gamma(::Val{track}, y::T, α::T, β::T) where {track,T <: Real}
    gamma_alloc_quote(1, T, (false, false, false), track, false)
end
@generated function ∂Gamma(sp::StackPointer, ::Val{track}, y::T, α::T, β::T) where {track,T <: Real}
    gamma_alloc_quote(1, T, (false, false, false), track, true)
end

push!(DISTRIBUTION_DIFF_RULES, :Gamma)


function beta_quote(M, T, (yisvec, αisvec, βisvec), (track_y, track_α, track_β), (inity, initα, initβ), partial)
    q = quote end
    pre_quote = quote end
    return_expr = quote end
    loop = any((yisvec, αisvec, βisvec))
    sp &= loop
    # set initialized_target to loop; if we are looping, we'll start out at zero
    initialized_target = loop
    if yisvec
        yexpr = :(y[i])
        push!(pre_quote.args, :(logity = PaddedMatrices.LazyMap(SLEEFPirates.logit, y)))
        logyexpr = :(SLEEFPirates.log(y[i]))
        logomyexpr = :(logy - logity[i])
    else
        yexpr = :y
        logyexpr = :logy
        logomyexpr = :logomy
        push!(pre_quote.args, :(logy = Base.log(y)))
        push!(pre_quote.args, :(logomy = SLEEFPirates.logit(y) - logy))
    end
    if αisvec
        αexpr = :(α[i])
        αm1expr = :(α[i] - one($T))
    else
        αexpr = :α
        αm1expr = :(αm1)
        push!(pre_quote.args, :(αm1 = α - one(α)))
    end
    if βisvec
        βexpr = :(β[i])
        βm1expr = :(β[i] - one($T))
    else
        βexpr = :β
        βm1expr = :(βm1)
        push!(pre_quote.args, :(βm1 = β - one($T)))
    end
    if αisvec || βisvec
        lbetaβexpr = :(first(SpecialFunctions.logabsbeta($αexpr, $βexpr)))
    else # neither are vectors
        lbetaβexpr = :lbetaαβ
        push!(pre_quote.args, :(lbetaαβ = first(SpecialFunctions.logabsbeta(α, β))))
    end
    if partial
        if track_y
            if yisvec
                yassignment = :(=)
                ∂ystorage = :∂yᵢ
            else
                ∂ystorage = :∂ys
                if inity
                    push!(return_expr.args, :(∂y[] = ∂ys))
                else
                    push!(return_expr.args, :(∂y[] += ∂ys))
                end
                if loop
                    push!(pre_quote.args, :(∂ys = zero($T)))
                    yassignment = :(+=)
                else
                    yassignment = :(=)
                end
            end
        end
        if track_α
            if αisvec
                αassignment = :(=)
                ∂αstorage = :∂αᵢ
                dgαexpr = :(SpecialFunctions.digamma(α[i]))
            else
                ∂αstorage = :∂αs
                dgαexpr = :dgα
                push!(pre_quote.args, :(dgα = SpecialFunctions.digamma(α)))
                if initα
                    push!(return_expr.args, :(∂α[] = ∂αs))
                else
                    push!(return_expr.args, :(∂α[] += ∂αs))
                end
                if loop
                    push!(pre_quote.args, :(∂αs = zero($T)))
                    αassignment = :(+=)
                else
                    αassignment = :(=)
                end
            end
        end
        if track_β
            if βisvec
                βassignment = :(=)
                ∂βstorage = :(∂βᵢ)
                dgβexpr = :(SpecialFunctions.digamma(β[i]))
            else
                ∂βstorage = :∂βs
                dgβexpr = :dgβ
                push!(pre_quote.args, :(dgβ = SpecialFunctions.digamma(β)))
                if initβ
                    push!(return_expr.args, :(∂β[] = ∂βs))
                else
                    push!(return_expr.args, :(∂β[] += ∂βs))
                end
                if loop
                    push!(pre_quote.args, :(∂βs = zero($T)))
                    βassignment = :(+=)
                else
                    βassignment = :(=)
                end
            end
        end
        if track_α || track_β
            if αisvec || βisvec
                push!(q.args, :(dgαβ = SpecialFunctions.digamma($αexpr + $βexpr) ))
            else # both are scalars
                push!(pre_quote.args, :(dgαβ = SpecialFunctions.digamma(α + β) ))
            end
        end
    end
    if track_α || track_y
        push!(q.args, :( am1 = $αm1expr))
        push!(q.args, :( logy = $logyexpr))
        if initialized_target
            push!(q.args, :( target = vmuladd(am1, logy, target) ) )
        else
            push!(q.args, :( target = am1 * logy ) )
            initialized_target = true
        end
        if partial
            track_α && push!(q.args, Expr(αassignment, ∂αstorage, :logy) )
            track_y && push!(q.args, Expr(yassignment, ∂ystorage, :(am1 / $yexpr) ) )
        end
    end
    if track_β || track_y
        push!(q.args, :( bm1 = $βm1expr))
        push!(q.args, :( logomy = $logomyexpr))
        if initialized_target
            push!(q.args, :( target = vmuladd(bm1, logomy, target) ) )
        else
            push!(q.args, :( target = bm1 * logomy ) )
            initialized_target = true
        end
        if partial
            track_β && push!(q.args, Expr(βassignment, ∂βstorage, :logomy) )
            if track_y
                if yisvec
                    push!(q.args, Expr(inity ? :(=) : :(+=), :(∂y[i]), :($∂ystorage - bm1 / (one($T) - $yexpr) ) ))
                else
                    push!(q.args, Expr(:(=), ∂ystorage, :($∂ystorage - bm1 / (one($T) - $yexpr) ) ))
                end
            end
        end
    end
    if track_α || track_β
        push!(q.args, :(ly = $logyexpr))
        if initialized_target
            push!(q.args, :( target = vsub(target, $lbetaβexpr) ) )
        else
            push!(q.args, :( target = -$lbetaβexpr ) )
            initialized_target = true
        end
        if partial
            if track_α
                if αisvec
                    push!(q.args, Expr(initα ? :(=) : :(+=), :(∂α[i]), :($∂αstorage + dgαβ - $dgαexpr) ) )
                else
                    push!(q.args, Expr(:(=), ∂αstorage, :($∂αstorage + dgαβ - $dgαexpr) ) )
                end
            end
            if track_β
                if βisvec
                    push!(q.args, Expr(initβ ? :(=) : :(+=), :(∂β[i]), :($∂βstorage + dgαβ - $dgβexpr) ) )
                else
                    push!(q.args, Expr(:(=), ∂βstorage, :($∂βstorage + dgαβ - $dgβexpr) ) )
                end
            end
        end
    end
    q = if loop
        quote
            $(Expr(:meta,:inline))
            @fastmath begin
                $pre_quote
                target = vbroadcast(Vec{$(VectorizationBase.pick_vector_width(T)),$T}, zero($T))
#                out = zero($T)
            end
            @vectorize $T for i ∈ 1:$M
                $q
            end
            @fastmath begin
                $return_expr
            end
            target
        end
    else
        quote
            $(Expr(:meta,:inline))
            @fastmath begin
                $pre_quote
                $q
                $return_expr
                target
            end
        end
    end
    simplify_expr(q)
end
function beta_alloc_quote(M, T, (yisvec, αisvec, βisvec), (track_y, tack_α, track_β), sp::Bool = true)
    q = quote end
    if track_y
        if yisvec
            if sp
                push!(q.args, :((sp,∂y) = PaddedMatrices.PtrVector{$M,$T}(sp)))
            else
                push!(q.args, :(∂y = PaddedMatrices.FixedSizeVector{$M,$T}(undef)))
            end
            push!(return_expr.args, :(∂y))
        else
            push!(q.args, :(∂y = Ref{$T}()))
            push!(return_expr.args, :∂y[])
        end
    else
        push!(q.args, :(∂y = nothing))
    end
    if track_α
        if αisvec
            if sp
                push!(q.args, :((sp,∂α) = PaddedMatrices.PtrVector{$M,$T}(undef)))
            else
                push!(q.args, :(∂α = PaddedMatrices.FixedSizeVector{$M,$T}(undef)))
            end
            push!(return_expr.args, :(∂α))
        else
            push!(q.args, :(∂α = Ref{$T}()))
            push!(return_expr.args, :(∂α[]))
        end
    else
        push!(q.args, :(∂α = nothing))
    end
    if track_β
        if βisvec
            if sp
                push!(pre_quote.args, :((sp,∂β) = PaddedMatrices.PtrVector{$M,$T}(undef)))
            else
                push!(pre_quote.args, :(∂β = PaddedMatrices.FixedSizeVector{$M,$T}(undef)))
            end
            push!(return_expr.args, :(∂β))
        else
            push!(q.args, :(∂β = Ref{$T}()))
            push!(return_expr.args, :(∂β[]))
        end
    else
        push!(q.args, :(∂β = nothing))
    end
    push!(q.args, :(target = ∂Beta!(uninitialized(∂y), uninitialized(∂α), uninitialized(∂β), y, α, β)))
    push!(q.args, return_expression(return_expr, sp))
    q
end

# α * log(β) + (α-1) * log(y) - β*y - lgamma(α)
@generated function Beta(
    ::Val{track},
    y::PaddedMatrices.AbstractFixedSizeVector{M,T},
    α::Union{T,Int,<:PaddedMatrices.AbstractFixedSizeVector{M,T}},
    β::Union{T,Int,<:PaddedMatrices.AbstractFixedSizeVector{M,T}}
            # ::Val{track}) where {track,T,M}
) where {track,M,T}
    αisvec = isa(α, PaddedMatrices.AbstractFixedSizeVector)
    βisvec = isa(β, PaddedMatrices.AbstractFixedSizeVector)
    beta_quote(M, T, (true, αisvec, βisvec), track, (false,false,false), false)
end
@generated function ∂Beta!(
    ∂y::∂YN, ∂α::∂ΑN, ∂β::∂ΒN,
    y::PaddedMatrices.AbstractFixedSizeVector{M,T},
    α::Union{T,Int,<:PaddedMatrices.AbstractFixedSizeVector{M,T}},
    β::Union{T,Int,<:PaddedMatrices.AbstractFixedSizeVector{M,T}}    
) where {∂YN, ∂ΑN, ∂ΒN, M, T}
    yisvec = true
    αisvec = isa(α, PaddedMatrices.AbstractFixedSizeVector)
    βisvec = isa(β, PaddedMatrices.AbstractFixedSizeVector)
    track_y = ∂YN !== Nothing
    track_α = ∂YΑ !== Nothing
    track_β = ∂YΒ !== Nothing
    inity = !isinitialized(∂YN)
    initα = !isinitialized(∂YΑ)
    initβ = !isinitialized(∂YΒ)
    beta_quote(
        M, T, (yisvec, αisvec, βisvec), (track_y, track_α, track_β), (inity, initα, initβ), true
    )
end
@generated function ∂Beta(
    ::Val{track},
    y::PaddedMatrices.AbstractFixedSizeVector{M,T},
    α::Union{T,Int,<:PaddedMatrices.AbstractFixedSizeVector{M,T}},
    β::Union{T,Int,<:PaddedMatrices.AbstractFixedSizeVector{M,T}}
) where {track,M,T}
            # ::Val{track}) where {track,T,M}
    αisvec = isa(α, PaddedMatrices.AbstractFixedSizeVector)
    βisvec = isa(β, PaddedMatrices.AbstractFixedSizeVector)
    beta_alloc_quote(M, T, (true, αisvec, βisvec), track, false)
end
@generated function ∂Beta(
    sp::StackPointer,
    ::Val{track},
    y::PaddedMatrices.AbstractFixedSizeVector{M,T},
    α::Union{T,Int,<:PaddedMatrices.AbstractFixedSizeVector{M,T}},
    β::Union{T,Int,<:PaddedMatrices.AbstractFixedSizeVector{M,T}}
) where {track,M,T}
# ) where {track,T,M}
    αisvec = isa(α, PaddedMatrices.AbstractFixedSizeVector)
    βisvec = isa(β, PaddedMatrices.AbstractFixedSizeVector)
    beta_quote(M, T, (true, αisvec, βisvec), track, true)
end
@generated function Beta(::Val{track}, y::T, α::Union{T,Int}, β::Union{T,Int}) where {T <: Real,track}
    beta_quote(1, T, (false, false, false), track, false)
end
@generated function ∂Beta(::Val{track}, y::T, α::Union{T,Int}, β::Union{T,Int}) where {track,T <: Real}
    beta_quote(1, T, (false, false, false), track, true)
end
push!(DISTRIBUTION_DIFF_RULES, :Beta)


function lsgg_quote(
    M::Int, T,
    (yisvec, αisvec, ϕisvec, δisvec, σisvec)::NTuple{5,Bool},
    (track_y, track_α, track_ϕ, track_δ, track_σ)::NTuple{5,Bool},
    (inity, initα, initϕ, initδ, initσ)::NTuple{5,Bool},
    partial::Bool
)
    @assert !any((track_α, track_ϕ, track_δ, track_σ))
    q = quote end
    yexpr = yisvec ? :(y[m]) : :y
    αexpr = αisvec ? :(α[m]) : :α
    ϕexpr = ϕisvec ? :(ϕ[m]) : :ϕ
    δexpr = δisvec ? :(δ[m]) : :δ
    σexpr = σisvec ? :(σ[m]) : :σ
    push!(q.args, :(smy = $σexpr - $yexpr) )
    push!(q.args, :(asmy = $αexpr * smy))
    push!(q.args, :(smyd = smy - $ϕexpr) )
    push!(q.args, :(ptsmyd = smyd * $δexpr) )
    push!(q.args, :(eptsmyd = $(any((yisvec, ϕisvec, δisvec, σisvec)) ? :SLEEFPirates : :Base).exp(ptsmyd)) )
    W = VectorizationBase.pick_vector_width(M,T)
    if partial
        ∂yop = inity ? :(=) : :(+=)
        if yisvec
            return quote
                # Inlined because of Julia SIMD corruption bug (if sp)
                # inlined to avoid heap allocation of mvector (if !sp)
                $(Expr(:meta,:inline))
                target = SIMDPirates.vbroadcast(Vec{$W,$T}, zero($T))
                LoopVectorization.@vvectorize for m ∈ 1:$M
                    $q
                    target = LoopVectorization.SIMDPirates.vadd(SIMDPirates.vsub(asmy, eptsmyd), target)
                    $(Expr(∂yop, :(∂y[m]), :(SIMDPirates.vfmsub($δexpr, eptsmyd, $αexpr))))
                end
                target
            end
        else
            @assert !any((αisvec, ϕisvec, δisvec, σisvec))
            return quote
                @fastmath begin
                    $q
                    target = asmy - eptsmyd
                    $(Expr(∂yop, :(∂y[]), :($δexpr * eptsmyd - $αexpr)))
                end
                target
            end
        end
    else
        if yisvec
            return quote
                # Inlined because of Julia SIMD corruption bug
                $(Expr(:meta,:inline))
                target = SIMDPirates.vbroadcast(Vec{$W,$T}, zero($T))
                LoopVectorization.@vvectorize for m ∈ 1:$M
                    $q
                    target = LoopVectorization.SIMDPirates.vadd(SIMDPirates.vsub(asmy, eptsmyd), target)
                end
                target
            end
        else
            @assert !any((αisvec, ϕisvec, δisvec, σisvec))
            return quote
                @fastmath begin
                    $q
                    target = asmy - eptsmyd
                end
                target
            end
        end
    end
end
function lsgg_alloc_quote(
    M::Int, T,
    (yisvec, αisvec, ϕisvec, δisvec, σisvec)::NTuple{5,Bool},
    (track_y, track_α, track_ϕ, track_δ, track_σ)::NTuple{5,Bool},
    sp::Bool
)
    @assert !any((track_α, track_ϕ, track_δ, track_σ))
    if yisvec
        return quote
            # Inlined because of Julia SIMD corruption bug (if sp)
            # inlined to avoid heap allocation of mvector (if !sp)
            $(Expr(:meta,:inline))
            $(sp ? :((sp, ∂y) = PtrVector{$M,$T}(sp)) : :( ∂y = FixedSizeVector{$M,$T}(undef)))
            target = ∂lsgg!(uninitialized(∂y),nothing,nothing,nothing,nothing,y,α,ϕ,δ,σ)
            $(sp ? :(sp, (target, ∂y)) : :(target, ∂y))
        end
    else
        @assert !any((αisvec, ϕisvec, δisvec, σisvec))
        return quote
            ∂y = Ref{$T}()
            target = ∂lsgg!(uninitialized(∂y),nothing,nothing,nothing,nothing,y,α,ϕ,δ,σ)
            $(sp ? :(sp, (target, ∂y[])) : :(target, ∂y[]))
        end
    end
end

@generated function lsgg(::Val{track}, y::AbstractFixedSizeVector{M,T}, α, ϕ, δ, σ) where {M,T,track}
    lsgg_quote(
        M, T, (true,
        α <: AbstractArray,
        ϕ <: AbstractArray,
        δ <: AbstractArray,
        σ <: AbstractArray),
        track, ntuple(_ -> false, Val(5)), false
    )
end
@generated function ∂lsgg!(
    ∂y::∂YN, ∂α::Nothing, ∂ϕ::Nothing, ∂δ::Nothing, ∂σ::Nothing,
    y::AbstractFixedSizeVector{M,T}, α, ϕ, δ, σ
) where {∂YN, M, T}
    yisvec = true
    αisvec = isa(α, PaddedMatrices.AbstractFixedSizeVector)
    ϕisvec = isa(ϕ, PaddedMatrices.AbstractFixedSizeVector)
    δisvec = isa(δ, PaddedMatrices.AbstractFixedSizeVector)
    σisvec = isa(σ, PaddedMatrices.AbstractFixedSizeVector)
    track = (true, false, false, false, false)
    init = (!isinitialized(∂YN), false, false, false, false)
    lsgg_quote(
       M, T, (yisvec, αisvec, ϕisvec, δisvec, σisvec), track, init, true
    )
end
@generated function ∂lsgg(::Val{track}, y::AbstractFixedSizeVector{M,T}, α, ϕ, δ, σ) where {M,T,track}
    lsgg_alloc_quote(
        M, T, (true,
        α <: AbstractArray,
        ϕ <: AbstractArray,
        δ <: AbstractArray,
        σ <: AbstractArray),
        track, false
    )
end
@generated function ∂lsgg(sp::StackPointer, ::Val{track}, y::AbstractFixedSizeVector{M,T}, α, ϕ, δ, σ) where {M,T,track}
    lsgg_alloc_quote(
        M, T, (true,
        α <: AbstractArray,
        ϕ <: AbstractArray,
        δ <: AbstractArray,
        σ <: AbstractArray),
        track, true
    )
end
@generated function lsgg(::Val{track}, y::T, α, ϕ, δ, σ) where {T<:Real,track}
    @assert !any((
        α <: AbstractArray,
        ϕ <: AbstractArray,
        δ <: AbstractArray,
        σ <: AbstractArray,
    ))
    lsgg_quote(
        1, T, (false,false,false,false,false),
        track, (false,false,false,false,false), false
    )
end
@generated function ∂lsgg!(
    ∂y::∂YN, ∂α::Nothing, ∂ϕ::Nothing, ∂δ::Nothing, ∂σ::Nothing,
    y::T, α, ϕ, δ, σ
) where {T<:Real,∂YN}
    @assert !any((
        α <: AbstractArray,
        ϕ <: AbstractArray,
        δ <: AbstractArray,
        σ <: AbstractArray,
    ))
    lsgg_alloc_quote(
        1, T, (false,false,false,false,false),
        track, (!isinitialized(T),false,false,false,false), true
    )
end
@generated function ∂lsgg(::Val{track}, y::T, α, ϕ, δ, σ) where {T<:Real,track}
    @assert !any((
        α <: AbstractArray,
        ϕ <: AbstractArray,
        δ <: AbstractArray,
        σ <: AbstractArray,
    ))
    lsgg_alloc_quote(
        1, T, (false,false,false,false,false), track, false
    )
end
@generated function ∂lsgg(sp::StackPointer, ::Val{track}, y::T, α, ϕ, δ, σ) where {T,track}
    @assert !any((
        α <: AbstractArray,
        ϕ <: AbstractArray,
        δ <: AbstractArray,
        σ <: AbstractArray,
    ))
    lsgg_alloc_quote(
        1, T, (false,false,false,false,false), track, true
    )
end
push!(DISTRIBUTION_DIFF_RULES, :lsgg)


emax(a, em, ed50) = em * a / (a + ed50)
emaxi(a, em, ed50) = @fastmath a * em * ed50 / (1 + ed50 * a)
emaxn(a, em, ed50) = @fastmath a*em*ed50 / (a + ed50)
normal_lpdf(y, μ, nhτ, nlogrootτ) = nhτ * abs2(y-μ) - nlogrootτ

# Val{(true,true,true,true,true,false,false)}()
@generated function EₘₐₓNMA(
    ::Val{track}, α::AbstractVector{T}, σᵤ::T, Eₘₐₓ::AbstractVector{T}, ED₅₀::AbstractVector{T}, σ::T,
    Treatments::StructuredMatrices.FixedSizeRaggedMatrix{M,N,P,<:Integer,<:Integer}, dose::AbstractVector{T}
) where {T, track, M, N, P}
    track_α, track_σᵤ, track_Eₘₐₓ, track_ED₅₀, track_σ, track_treat, track_dose = track
    @assert track_treat == false && track_dose == false
    padM = PaddedMatrices.calc_padding(M-1, T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(M-1)

    quote
        target = zero($T)
        ταr = 1 / σᵤ
        τ = 1 / (σ*σ)
        j = 1
        col_lengths = Treatments.column_lengths
        vτ = SIMDPirates.vbroadcast(Vec{$W,$T}, τ)
        vnh = SIMDPirates.vbroadcast(Vec{$W,$T}, $(T(-0.5)))
        $([:($(Symbol(:vτma_,k)) = vmul(vτ, $(Expr(:tuple, [Core.VecElement{T}(2m/(m+1)) for m in k*W+1:(k+1)*W]...)))) for k in 0:((padM>>>Wshift)-1)]...)
        $([:($(Symbol(:vnhτ_,k)) = vmul(vnh, $(Symbol(:vτma_,k)))) for k in 0:((padM>>>Wshift)-1)]...)
        $([:($(Symbol(:vnlogrootτ_,k)) = vmul(vnh, SLEEFPirates.log($(Symbol(:vτma_,k))))) for k in 0:((padM>>>Wshift)-1)]...)
        @inbounds nhτ = $(Expr(:tuple, [:(($(Symbol(:vnhτ_,m>>>Wshift))[$(1+m&(W-1))]).value) for m in 0:padM-1]...))
        @inbounds nlogrootτ = $(Expr(:tuple, [:(($(Symbol(:vnlogrootτ_,m>>>Wshift))[$(1+m&(W-1))]).value) for m in 0:padM-1]...))
        @fastmath @inbounds for i in 1:$N
            t = Treatments[j]
            αᵢ = α[j]
            emi1 = αᵢ
            t == 0 || (emi1 -= emax(dose[j], Eₘₐₓ[t], ED₅₀[t]))
            αᵢ *= ταr
            target -= 0.5αᵢ*αᵢ
            s = zero($T)
            j += 1
            for k in 1:col_lengths[i]-1
                t = Treatments[j]
                emik = emax(dose[j], Eₘₐₓ[t], ED₅₀[t])
                δik = α[j] - emik - emi1 # - αi1
                # target += normal_lpdf(δik, s/(k-1), τ * (2*(k-1)) / k )
                # @show δik, s/k, nhτ[k], nlogrootτ[k]
                target += normal_lpdf(δik, s/k, nhτ[k], nlogrootτ[k])
                s += δik
                j += 1
            end
        end
        muladd($(T(N)), log(ταr), target)
    end
end
@inline function ∂emax(a, em, ed50)
    #    f(a,b,c) =  a*c   / (b + c)
    # ∂f∂a(a,b,c) =    c   / (b + c)
    # ∂f∂b(a,b,c) = -a*c   / (b + c)^2
    # ∂f∂c(a,b,c) =  a * b / (b + c)^2
    @fastmath begin
        invaped50 = 1 / ( a + ed50 )
        eminvaped50 = em * invaped50
        f = a * eminvaped50
        dfda = ed50 * eminvaped50 * invaped50
        dfdemax = a * invaped50
        dfded50 = - f * invaped50
    end
    f, dfda, dfdemax, dfded50
end
@inline function ∂emaxi(a, em, ed50)
    #    f(a,b,c) = a*b*c / (1 + a*c)
    # ∂f∂a(a,b,c) =   b*c / (1 + a*c)^2
    # ∂f∂b(a,b,c) = a * c / (1 + a*c)
    # ∂f∂c(a,b,c) = a*b   / (1 + a*c)^2
    @fastmath begin
        invdenom = 1 / muladd(ed50, a, one(ed50))
        doseed50 = a * ed50
        dfdemax = doseed50 * invdenom
        f = dfdemax * em
        emdivdenomdivdenom = em * invdenom * invdenom
        dfda = emdivdenomdivdenom * ed50
        dfded50 = emdivdenomdivdenom * a
    end
    f, dfda, dfdemax, dfded50
end
@inline function ∂emaxn(a, em, ed50)
    @fastmath begin
        invdenom = 1 / (ed50 + a)
        dfdem = ed50 * a * invdenom
        f = dfdem * em
        eminvdenominvdenom = em * invdenom * invdenom
        dfded50 = eminvdenominvdenom * a * a
        dfda = eminvdenominvdenom * ed50 * ed50
    end
    f, dfda, dfdem, dfded50
end
function ∂normal_lpdf(y, μ, nhτ, nlogrootτ, nhσ²)
    @fastmath begin
        z = y - μ
        nτ = nhτ + nhτ
        z² = z * z
        f = nhτ * z² - nlogrootτ
        ∂f∂y = nτ * z
        ∂f∂μ = -∂f∂y
        ∂f∂τ =  - 0.5 * z² - nhσ²
    end
    f, ∂f∂y, ∂f∂μ, ∂f∂τ
end

@generated function ∂EₘₐₓNMA!(
    dαv::∂ΑN, ∂σᵤ::∂ΣμN, demaxv::DEMAX, ded50v::DED50, ∂σ::∂ΣN,
    α::AbstractVector{T}, σᵤ::T, Eₘₐₓ::AbstractFixedSizeVector{C,T}, ED₅₀::AbstractFixedSizeVector{C,T}, σ::T,
    treatments::StructuredMatrices.FixedSizeRaggedMatrix{M,N,P,<:Integer,<:Integer}, dose::AbstractVector{T}
) where {T,C,M,N,P,∂ΑN, ∂ΣμN, DEMAX, DED50, ∂ΣN}
    padM = PaddedMatrices.calc_padding(M-1, T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(M-1)
    inits_q = quote target = zero($T) end
    if !isinitialized(DEMAX) || !isinitialized(DED50)
        loop_body = quote end
        isinitialized(DEMAX) || push!(loop_body.args, :(demaxv[i] = zero($T)))
        isinitialized(DED50) || push!(loop_body.args, :(ded50v[i] = zero($T)))
        loop_q = quote
            @inbounds for i ∈ eachindex(demaxv)
                $loop_body
            end
        end
        push!(inits_q.args, loop_q)
    end
    quote
        dsdemax = PtrMatrix{2,$M,$T}(SIMDPirates.alloca(Val{$(2M)}(), $T))
        $inits_q
        invσ = 1/σ
        # σ² =  σ * σ
        τ = abs2(invσ)
        dτdσ = -2τ*invσ
        # @show dτdσ
        # tau, dtaudsigma = 1/abs2(sigma), -2/sigma^3
        j = 1
        dτ = zero(σ)
        col_lengths = treatments.column_lengths
        ταr = 1 / σᵤ
        ταr² = ταr * ταr
        dταrdσᵤ = -ταr * ταr
        dταr = zero($T)
        vτ = SIMDPirates.vbroadcast(Vec{$W,$T}, τ)
        vnh = SIMDPirates.vbroadcast(Vec{$W,$T}, $(T(-0.5)))
        $([:($(Symbol(:vτma_,k)) = vmul(vτ, $(Expr(:tuple, [Core.VecElement{T}(2m/(m+1)) for m in k*W+1:(k+1)*W]...)))) for k in 0:((padM>>>Wshift)-1)]...)
        $([:($(Symbol(:vnhτ_,k)) = vmul(vnh, $(Symbol(:vτma_,k)))) for k in 0:((padM>>>Wshift)-1)]...)
        $([:($(Symbol(:vnhσ²_,k)) = SIMDPirates.vfdiv(vnh,$(Symbol(:vτma_,k)))) for k in 0:((padM>>>Wshift)-1)]...)
        $([:($(Symbol(:vnlogrootτ_,k)) = vmul(vnh, SLEEFPirates.log($(Symbol(:vτma_,k))))) for k in 0:((padM>>>Wshift)-1)]...)
        τscales = $(Expr(:tuple, [T(2k/(k+1)) for k in 1:padM]...))
        @inbounds nhτ = $(Expr(:tuple, [:(($(Symbol(:vnhτ_,m>>>Wshift))[$(1+m&(W-1))]).value) for m in 0:padM-1]...))
        @inbounds nlogrootτ = $(Expr(:tuple, [:(($(Symbol(:vnlogrootτ_,m>>>Wshift))[$(1+m&(W-1))]).value) for m in 0:padM-1]...))
        @inbounds nhσ² = $(Expr(:tuple, [:(($(Symbol(:vnhσ²_,m>>>Wshift))[$(1+m&(W-1))]).value) for m in 0:padM-1]...))
        @fastmath @inbounds for i in 1:$N
            ti = treatments[j]
            ji = j
            αi1 = α[ji]
            if ti == 0
                emi1, demi1ddose, demi1demaxv, demi1ded50v = (zero($T),zero($T),zero($T),zero($T))
            else
                emi1, demi1ddose, demi1demaxv, demi1ded50v = ∂emax(dose[j], Eₘₐₓ[ti], ED₅₀[ti])
            end
            αi1²ταr = αi1 * αi1 * ταr
            target -= 0.5 * αi1²ταr * ταr
            dταr -= αi1²ταr
            dtargetdαin = αi1*ταr²
            emi1minusαi1 = emi1 - αi1
            dtargetdαi = zero($T)
            dtargetdemaxv = zero($T)
            dtargetded50v = zero($T)
            s = zero($T)
            j += 1
            # fill!(dsdemax, 0.0)
            for k in 1:col_lengths[i]-1
                t = treatments[j]
                emik, demikddose, demikdemaxv, demikded50v = ∂emax(dose[j], Eₘₐₓ[t], ED₅₀[t])
                δik = α[j] - emik + emi1minusαi1
                sscale = 1 / k
                # f, dfdy, dfdμ, dfdτ = dnormal_lpdf( δik, s*sscale, τ * τscale, logτs[k-1] )
                f, dfdy, dfdμ, dfdτ = ∂normal_lpdf( δik, s*sscale, nhτ[k], nlogrootτ[k], nhσ²[k] )
                target += f
                # @show dfdτ, τscales[k]
                dτ += dfdτ * τscales[k]
                dtargetdαi -= dfdy * sscale
                if ti > 0
                    dtargetdemaxv += dfdy*demi1demaxv*sscale
                    dtargetded50v += dfdy*demi1ded50v*sscale
                end
                for p in 1:k-1
                    tp = treatments[j + p - k]
                    dαv[j + p - k] += dfdμ*sscale
                    demaxv[tp] -= dsdemax[1,p]*dfdμ*sscale
                    ded50v[tp] -= dsdemax[2,p]*dfdμ*sscale
                end
                $(Expr(isinitialized(∂ΑN) ? :(+=) : :(=), :(dαv[j]), :dfdy))
                demaxv[t] -= dfdy*demikdemaxv
                ded50v[t] -= dfdy*demikded50v
                dsdemax[1,k] = demikdemaxv
                dsdemax[2,k] = demikded50v
                s += δik
                j += 1
            end
            $(Expr(isinitialized(∂ΑN) ? :(+=) : :(=), :(dαv[ji]), :(dtargetdαi - dtargetdαin)))
            if ti > 0
                demaxv[ti] += dtargetdemaxv
                ded50v[ti] += dtargetded50v
            end
        end
        target = muladd($(T(N)), Base.log(ταr), target)
        dταr = muladd($(T(N)), σᵤ, dταr)
        $(Expr(isinitialized(∂ΣμN) ? :(+=) : :(=), :(∂σᵤ[]), :(dταrdσᵤ*dταr)))
        $(Expr(isinitialized(∂ΣN) ? :(+=) : :(=), :(∂σ[]), :(dτ*dτdσ)))
        target
    end
end
# Val{(true,true,true,true,false,false)}()
@generated function ∂EₘₐₓNMA(
    sp::StackPointer, ::Val{track}, α::AbstractVector{T}, σᵤ::T, Eₘₐₓ::AbstractFixedSizeVector{C,T}, ED₅₀::AbstractFixedSizeVector{C,T}, σ::T,
    treatments::StructuredMatrices.FixedSizeRaggedMatrix{M,N,P,<:Integer,<:Integer}, dose::AbstractVector{T}
) where {T,C,M,N,P,track}
    padM = PaddedMatrices.calc_padding(M-1, T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(M-1)
    quote
        sp, dαv = PtrVector{$P,$T}(sp)
        sp, demaxv = PtrVector{$C,$T}(sp)
        sp, ded50v = PtrVector{$C,$T}(sp)
        dsdemax = PtrMatrix{2,$M,$T}(pointer(sp,$T))
        target = zero($T)
        @inbounds for i in eachindex(demaxv)#, ded50v)
            ded50v[i] = zero($T)
            demaxv[i] = zero($T)
        end
        invσ = 1/σ
        # σ² =  σ * σ
        τ = abs2(invσ)
        dτdσ = -2τ*invσ
        # @show dτdσ
        # tau, dtaudsigma = 1/abs2(sigma), -2/sigma^3
        j = 1
        dτ = zero(σ)
        col_lengths = treatments.column_lengths
        ταr = 1 / σᵤ
        ταr² = ταr * ταr
        dταrdσᵤ = -ταr * ταr
        dταr = zero($T)
        vτ = SIMDPirates.vbroadcast(Vec{$W,$T}, τ)
        vnh = SIMDPirates.vbroadcast(Vec{$W,$T}, $(T(-0.5)))
        $([:($(Symbol(:vτma_,k)) = vmul(vτ, $(Expr(:tuple, [Core.VecElement{T}(2m/(m+1)) for m in k*W+1:(k+1)*W]...)))) for k in 0:((padM>>>Wshift)-1)]...)
        $([:($(Symbol(:vnhτ_,k)) = vmul(vnh, $(Symbol(:vτma_,k)))) for k in 0:((padM>>>Wshift)-1)]...)
        $([:($(Symbol(:vnhσ²_,k)) = SIMDPirates.vfdiv(vnh,$(Symbol(:vτma_,k)))) for k in 0:((padM>>>Wshift)-1)]...)
        $([:($(Symbol(:vnlogrootτ_,k)) = vmul(vnh, SLEEFPirates.log($(Symbol(:vτma_,k))))) for k in 0:((padM>>>Wshift)-1)]...)
        τscales = $(Expr(:tuple, [T(2k/(k+1)) for k in 1:padM]...))
        @inbounds nhτ = $(Expr(:tuple, [:(($(Symbol(:vnhτ_,m>>>Wshift))[$(1+m&(W-1))]).value) for m in 0:padM-1]...))
        @inbounds nlogrootτ = $(Expr(:tuple, [:(($(Symbol(:vnlogrootτ_,m>>>Wshift))[$(1+m&(W-1))]).value) for m in 0:padM-1]...))
        @inbounds nhσ² = $(Expr(:tuple, [:(($(Symbol(:vnhσ²_,m>>>Wshift))[$(1+m&(W-1))]).value) for m in 0:padM-1]...))
        @fastmath @inbounds for i in 1:$N
            ti = treatments[j]
            ji = j
            αi1 = α[ji]
            if ti == 0
                emi1, demi1ddose, demi1demaxv, demi1ded50v = (zero($T),zero($T),zero($T),zero($T))
            else
                emi1, demi1ddose, demi1demaxv, demi1ded50v = ∂emax(dose[j], Eₘₐₓ[ti], ED₅₀[ti])
            end
            αi1²ταr = αi1 * αi1 * ταr
            target -= 0.5 * αi1²ταr * ταr
            dταr -= αi1²ταr
            dtargetdαin = αi1*ταr²
            emi1minusαi1 = emi1 - αi1
            dtargetdαi = zero($T)
            dtargetdemaxv = zero($T)
            dtargetded50v = zero($T)
            s = zero($T)
            j += 1
            # fill!(dsdemax, 0.0)
            for k in 1:col_lengths[i]-1
                t = treatments[j]
                emik, demikddose, demikdemaxv, demikded50v = ∂emax(dose[j], Eₘₐₓ[t], ED₅₀[t])
                δik = α[j] - emik + emi1minusαi1
                sscale = 1 / k
                # f, dfdy, dfdμ, dfdτ = dnormal_lpdf( δik, s*sscale, τ * τscale, logτs[k-1] )
                f, dfdy, dfdμ, dfdτ = ∂normal_lpdf( δik, s*sscale, nhτ[k], nlogrootτ[k], nhσ²[k] )
                target += f
                # @show dfdτ, τscales[k]
                dτ += dfdτ * τscales[k]
                dtargetdαi -= dfdy * sscale
                if ti > 0
                    dtargetdemaxv += dfdy*demi1demaxv*sscale
                    dtargetded50v += dfdy*demi1ded50v*sscale
                end
                for p in 1:k-1
                    tp = treatments[j + p - k]
                    dαv[j + p - k] += dfdμ*sscale
                    demaxv[tp] -= dsdemax[1,p]*dfdμ*sscale
                    ded50v[tp] -= dsdemax[2,p]*dfdμ*sscale
                end
                dαv[j] = dfdy
                demaxv[t] -= dfdy*demikdemaxv
                ded50v[t] -= dfdy*demikded50v
                dsdemax[1,k] = demikdemaxv
                dsdemax[2,k] = demikded50v
                s += δik
                j += 1
            end
            dαv[ji] = dtargetdαi - dtargetdαin
            if ti > 0
                demaxv[ti] += dtargetdemaxv
                ded50v[ti] += dtargetded50v
            end
        end
        target = muladd($(T(N)), Base.log(ταr), target)
        dταr = muladd($(T(N)), σᵤ, dταr)
        sp, (target, dαv', dταrdσᵤ*dταr, demaxv', ded50v', dτ*dτdσ)
    end
end


push!(DISTRIBUTION_DIFF_RULES, :EₘₐₓNMA)

#= 

function multinomial_quote(M, isvararg::Bool, T)
    q = quote end
    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    Wm1 = W - 1
    niter = M >>> Wshift
    nrem = (M + Wm1) & Wm1
    if isvararg
        push!(q.args, :(target = SIMDPirates.vmul(y, SLEEFPirates.log()) ))
        for i ∈ 1:niter-1
            push!(q.args, :( ))
        end

    else
        for i ∈ 0:niter-1
            
        end
    end

end

 
=#
