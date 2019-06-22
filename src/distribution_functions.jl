
# using StaticArrays
#
# @inline PaddedMatrices.is_sized(::SVector) = true
# @inline PaddedMatrices.is_sized(::Type{<:SVector}) = true
# @inline PaddedMatrices.is_sized(::SMatrix) = true
# @inline PaddedMatrices.is_sized(::Type{<:SMatrix}) = true
# @inline PaddedMatrices.type_length(::SVector{N}) where {N} = N
# @inline PaddedMatrices.type_length(::Type{<:SVector{N}}) where {N} = N
# @inline PaddedMatrices.type_length(::SMatrix{M,N}) where {M,N} = M*N
# @inline PaddedMatrices.type_length(::Type{<:SMatrix{M,N}}) where {M,N} = M*N

const DISTRIBUTION_DIFF_RULES = Set{Symbol}()


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
function distribution_diff_rule!(mod, first_pass, second_pass, tracked_vars, out, A, f)
    track_out = false
    function_output = Expr(:tuple, out)
    track_tup = Expr(:tuple,)
    for a ∈ A
        if a ∈ tracked_vars
            push!(track_tup.args, true)
        else
            push!(track_tup.args, false)
            continue
        end
        track_out = true
        ∂ = Symbol("###adjoint###_##∂", out, "##∂", a, "##")
        push!(function_output.args, ∂)
        pushfirst!(second_pass.args, :( $(Symbol("###seed###", a)) = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED($(Symbol("###seed###", out)), $∂, $(Symbol("###seed###", a)))))
        # pushfirst!(second_pass.args, :( $(Symbol("###seed###", a)) = $(Symbol("###seed###", out)) * $∂ ))
    end
    if track_out
        push!(tracked_vars, out)
        push!(first_pass.args, :($function_output = $(mod).$(Symbol(:∂, f))($(A...), Val{$track_tup}())))
        # ret_string  = "function: $f: "
        # push!(first_pass.args, :(println($ret_string, $function_output)))
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
# function ∂Bernoulli_logit_fmadd_quote(y_is_param, β_is_param, X_is_param, α_is_param)
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


@generated function Bernoulli_logit_fmadd(y::BitVector, X::AbstractMatrix{T}, β::AbstractVector{T}, α::AbstractFloat,
                            ::Val{track} = Val{(false,false,true,true)}()) where {T, track}
    y_is_param, β_is_param, X_is_param, α_is_param = track
    @assert y_is_param == false
    if PaddedMatrices.is_sized(β)
        N_β = PaddedMatrices.type_length(β)
        q = quote
            # $(Expr(:meta, :inline))
            # T = promote_type(eltype(α),eltype(β),eltype(X))
            target = zero($T)
            @vectorize $T for i ∈ eachindex(y)
                # a = $(Expr(:call, :+, :α, [:(X[i,$n] * β[$n]) for n ∈ 1:N_β]...))
                # Break it up, so inference still works for N_β > 15
                a = SIMDPirates.vmuladd(X[i,1], β[1], α)
                $([:(a = SIMDPirates.vmuladd(X[i,$n], β[$n], a)) for n ∈ 2:N_β]...)
                OmP = one($T) / (one($T) + SLEEFPirates.exp( a ))
                # P = one($T) - OmP
                logOmP = log(OmP)
                logP = a + logOmP
                target += y[i] ? logP : logOmP
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
    q
end
@generated function ∂Bernoulli_logit_fmadd(y::BitVector, X::AbstractMatrix{T}, β::AbstractVector{T}, α::AbstractFloat,
                            ::Val{track}) where {T, track}
    y_is_param, X_is_param, β_is_param, α_is_param = track
    @assert y_is_param == false
    @assert X_is_param == false
    X_is_param && throw("X as parameter is not yet supported.")
    if PaddedMatrices.is_sized(β)
        N_β = PaddedMatrices.type_length(β)
        init_q = quote
            # T = promote_type(eltype(α),eltype(β),eltype(X))
            target = zero($T)
        end
        out_expr = Expr(:tuple, :target)
        partial_exprs = Expr[]
        ∂P_undefined = true
        if β_is_param
            L_β = PaddedMatrices.pick_L(N_β, T)
            ∂P_undefined = false
            # push!(partial_exprs, :(∂P = OmP * P))
            push!(partial_exprs, :(∂logP = OmP ))
            push!(partial_exprs, :(∂logOmP = - P))
            push!(out_expr.args, :(ConstantFixedSizePaddedVector{$N_β}($(Expr(:tuple, [Symbol(:∂βP_, n) for n ∈ 1:N_β]...,[zero(T) for n ∈ 1:L_β-N_β]...)))'))
            for n ∈ 1:N_β
                push!(init_q.args, :($(Symbol(:∂βP_, n)) = zero($T)) )
            end
            partial_exprs_q = quote end
            for n ∈ 1:N_β
                # push!(partial_exprs_q.args, :($(Symbol(:∂PxX_, n)) = ∂P * X[i,$n] ) )
                # push!(partial_exprs_q.args, :($(Symbol(:∂βP_, n)) += y[i] ? $(Symbol(:∂PxX_, n)) : - $(Symbol(:∂PxX_, n)) ) )
                # push!(partial_exprs_q.args, :($(Symbol(:∂PxX_, n)) = ∂P * X[i,$n] ) )
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
            push!(out_expr.args, :(∂αP))
            push!(init_q.args, :(∂αP = zero($T)))
            push!(partial_exprs, :(∂αP += y[i] ? ∂logP : ∂logOmP))
        end
        W, Wshift = VectorizationBase.pick_vector_width_shift(T)
        unroll_factor = max(8 >> Wshift, 1)
        q = quote
            # $(Expr(:meta, :inline))
            $init_q
            @vectorize $T $unroll_factor for i ∈ eachindex(y)
                # a = $(Expr(:call, :+, :α, [:(X[i,$n] * β[$n]) for n ∈ 1:N_β]...))
                a = SIMDPirates.vmuladd(X[i,1], β[1], α)
                $([:(a = SIMDPirates.vmuladd(X[i,$n], β[$n], a)) for n ∈ 2:N_β]...)
                OmP = one($T) / (one($T) + SLEEFPirates.exp( a ))
                P = one($T) - OmP
                logOmP = SLEEFPirates.log(OmP)
                logP = a + logOmP
                target += y[i] ? logP : logOmP
                $(partial_exprs...)
            end
            $out_expr
        end
    else
        throw("""
            Dynamically sized coefficient vector β is not yet supported.
            Feel free to file an issue, so that adding support will be prioritized.
            As a workaround, use a β with type parameterized by size.
            Eg, a padded vector from PaddedMatrices.jl or SVector from StaticArrays.jl.
        """)
    end
    q
end

push!(DISTRIBUTION_DIFF_RULES, :Bernoulli_logit_fmadd)

function ∂Bernoulli_logit_fnmadd_quote()

end
function ∂Bernoulli_logit_fmsub_quote()

end
function ∂Bernoulli_logit_fnmsub_quote()

end


@generated function LKJ(L::LKJCorrCholesky{N,T}, η::T, ::Val{track}) where {N,T,track}
    quote
        out = zero($T)
        # @fastmath @inbounds @simd ivdep for n ∈ 1:$(N-1)
        @vectorize $T for n ∈ 1:$(N-1)
            out += ($(N - 3) - n + 2η) * SLEEFPirates.log(L[n+1])
        end
        out
    end
end
@generated function ∂LKJ(L::LKJCorrCholesky{N,T}, η::T, ::Val{track}) where {N,T,track}
    track_L, track_η = track
    if track_L && track_η
        quote
            out = zero($T)
            ∂L = MutableFixedSizePaddedVector{$N,$T}(undef)
            @inbounds ∂L[1] = 0
            ∂η = zero($T)
            @vectorize $T for n ∈ 1:$(N-1)
            # @fastmath @inbounds @simd ivdep for n ∈ 1:$(N-1)
                ∂ηₙ = log(L[n+1])
                coef = ($(N - 3) - n + 2η)
                out += coef * ∂ηₙ
                ∂L[n+1] = coef / L[n+1]
                ∂η += 2∂ηₙ
            end
            out, Diagonal(ConstantFixedSizePaddedVector(∂L)), ∂η
        end
    elseif track_L
        quote
            out = zero($T)
            ∂L = MutableFixedSizePaddedVector{$N,$T}(undef)
            @inbounds ∂L[1] = 0
            ∂η = zero($T)
            @vectorize $T for n ∈ 1:$(N-1)
            # @fastmath @inbounds @simd ivdep for n ∈ 1:$(N-1)
                ∂ηₙ = log(L[n+1])
                coef = ($(N - 3) - n + 2η)
                out += coef * ∂ηₙ
                ∂L[n+1] = coef / L[n+1]
            end
            out, Diagonal(ConstantFixedSizePaddedVector(∂L))
        end
    elseif track_η
        quote
            out = zero($T)
            ∂η = zero($T)
            @vectorize $T for n ∈ 1:$(N-1)
            # @fastmath @inbounds @simd ivdep for n ∈ 1:$(N-1)
                ∂ηₙ = log(L[n+1])
                coef = ($(N - 3) - n + 2η)
                out += coef * ∂ηₙ
                ∂η += 2∂ηₙ
            end
            out, ∂η
        end
    else
        quote
            out = zero($T)
            @vectorize $T for n ∈ 1:$(N-1)
            # @fastmath @inbounds @simd ivdep for n ∈ 1:$(N-1)
                ∂ηₙ = log(L[n+1])
                coef = ($(N - 3) - n + 2η)
                out += coef * ∂ηₙ
            end
            out
        end
    end
end
@generated function ∂LKJ(sp::PaddedMatrices.StackPointer, L::LKJCorrCholesky{N,T}, η::T, ::Val{track}) where {N,T,track}
    track_L, track_η = track
    if track_L && track_η
        quote
            out = zero($T)
            (sp,∂L) = PtrVector{$N,$T}(sp)
            @inbounds ∂L[1] = 0
            ∂η = zero($T)
            @vectorize $T for n ∈ 1:$(N-1)
            # @fastmath @inbounds @simd ivdep for n ∈ 1:$(N-1)
                ∂ηₙ = log(L[n+1])
                coef = ($(N - 3) - n + 2η)
                out += coef * ∂ηₙ
                ∂L[n+1] = coef / L[n+1]
                ∂η += 2∂ηₙ
            end
            sp, (out, Diagonal(∂L), ∂η)
        end
    elseif track_L
        quote
            out = zero($T)
            (sp, ∂L) = PtrVector{$N,$T}(sp)
            @inbounds ∂L[1] = 0
            ∂η = zero($T)
            @vectorize $T for n ∈ 1:$(N-1)
            # @fastmath @inbounds @simd ivdep for n ∈ 1:$(N-1)
                ∂ηₙ = log(L[n+1])
                coef = ($(N - 3) - n + 2η)
                out += coef * ∂ηₙ
                ∂L[n+1] = coef / L[n+1]
            end
            sp, (out, Diagonal(∂L))
        end
    elseif track_η
        quote
            out = zero($T)
            ∂η = zero($T)
            @vectorize $T for n ∈ 1:$(N-1)
            # @fastmath @inbounds @simd ivdep for n ∈ 1:$(N-1)
                ∂ηₙ = log(L[n+1])
                coef = ($(N - 3) - n + 2η)
                out += coef * ∂ηₙ
                ∂η += 2∂ηₙ
            end
            out, ∂η
        end
    else
        quote
            out = zero($T)
            @vectorize $T for n ∈ 1:$(N-1)
            # @fastmath @inbounds @simd ivdep for n ∈ 1:$(N-1)
                ∂ηₙ = log(L[n+1])
                coef = ($(N - 3) - n + 2η)
                out += coef * ∂ηₙ
            end
            out
        end
    end
end
push!(DISTRIBUTION_DIFF_RULES, :LKJ)
PaddedMatrices.@support_stack_pointer ∂LKJ

function gamma_quote(M, T, yisvec, αisvec, βisvec, (track_y, track_α, track_β), partial)
    q = quote end
    pre_quote = quote end
    return_expr = Expr(:tuple, :out)
    loop = any((yisvec, αisvec, βisvec))
    # set initialized to loop; if we are looping, we'll start out at zero
    initialized = loop
    if yisvec
        yexpr = :(y[i])
        logyexpr = :(SLEEFPirates.log(y[i]))
    else
        yexpr = :y
        push!(pre_quote.args, :(logy = Base.log(y)))
        logyexpr = :logy
    end
    if αisvec
        αexpr = :(α[i])
        lgammaαexpr = :(SpecialFunctions.lgamma(α[i]))
        αm1expr = :(α[i] - one(eltype(α)))
    else
        αexpr = :α
        lgammaαexpr = :(lgammaα)
        αm1expr = :(αm1)
        push!(pre_quote.args, :(lgammaα = SpecialFunctions.lgamma(α)))
        push!(pre_quote.args, :(αm1 = α - one(eltype(α))))
    end
    if βisvec
        βexpr = :(β[i])
        logβexpr = :(SLEEFPirates.log(β[i]))
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
                push!(pre_quote.args, :(∂y = PaddedMatrices.MutableFixedSizePaddedVector{$M,$T}(undef)))
                push!(return_expr.args, :(∂y'))
                # push!(return_expr.args, :(PaddedMatrices.ConstantFixedSizePaddedVector(∂y)'))
            else
                ∂ystorage = :∂y
                push!(return_expr.args, :(∂y))
                if loop
                    ∂yassignment = :(+=)
                    push!(pre_quote.args, :(∂y = zero($T)))
                else
                    ∂yassignment = :(=)
                end
            end
        end
        if track_α
            if αisvec
                ∂αassignment = :(=)
                ∂αstorage = :(∂αᵢ)
                push!(pre_quote.args, :(∂α = PaddedMatrices.MutableFixedSizePaddedVector{$M,$T}(undef)))
                push!(return_expr.args, :(PaddedMatrices.ConstantFixedSizePaddedVector(∂α)'))
            else
                ∂αstorage = :(∂α)
                push!(return_expr.args, :(∂α))
                if loop
                    ∂αassignment = :(+=)
                    push!(pre_quote.args, :(∂α = zero($T)))
                else
                    ∂αassignment = :(=)
                end
            end
        end
        if track_β
            if βisvec
                ∂βassignment = :(=)
                ∂βstorage = :(∂βᵢ)
                push!(pre_quote.args, :(∂β = PaddedMatrices.MutableFixedSizePaddedVector{$M,$T}(undef)))
                push!(return_expr.args, :(PaddedMatrices.ConstantFixedSizePaddedVector(∂β)'))
            else
                ∂βstorage = :(∂β)
                push!(return_expr.args, :(∂β))
                if loop
                    ∂βassignment = :(+=)
                    push!(pre_quote.args, :(∂β = zero($T)))
                else
                    ∂βassignment = :(=)
                end
            end
        end
    end
    if track_α || track_β
        push!(q.args, :( lβ = $logβexpr))
        if initialized
            push!(q.args, :( out += $αexpr * lβ ) )
        else
            push!(q.args, :( out = $αexpr * lβ) )
            initialized = true
        end
        if partial
            track_α && push!(q.args, Expr(∂αassignment, ∂αstorage, :lβ) )
            track_β && push!(q.args, Expr(∂βassignment, ∂βstorage, αexpr) )
        end
    end
    if track_α || track_y
        push!(q.args, :(ly = $logyexpr))
        if initialized
            push!(q.args, :( out += $αm1expr * ly ) )
        else
            push!(q.args, :( out = $αm1expr * ly ) )
            initialized = true
        end
        if partial
            if track_α
                if αisvec
                    push!(q.args, Expr(:(=), :∂α₂, :($αstorage + ly)) )
                else
                    push!(q.args, Expr(:(+=), ∂αstorage, :ly) )
                end
            end
            track_y && push!(q.args, Expr(∂yassignment, ∂ystorage, :($αm1expr / $yexpr) ) )
        end
    end
    if track_β || track_y
        # initialized == true
        push!(q.args, :(out -= $βexpr*$yexpr))
        if partial
            if track_β
                if βisvec
                    push!(q.args, Expr(:(=), :(∂β[i]), :($∂βstorage - $yexpr) ) )
                else
                    push!(q.args, Expr(:(-=), ∂βstorage, yexpr ) )
                end
            end
            if track_y
                if yisvec
                    push!(q.args, Expr(:(=), :(∂y[i]), :($∂ystorage - $βexpr) ) )
                else
                    push!(q.args, Expr(:(-=), ∂ystorage, βexpr ) )
                end
            end
        end
    end
    if track_α
        # initialized == true, because (if track_α || track_β) == true
        push!(q.args, :(out -= $lgammaαexpr))

        if partial
            if αisvec
                push!(q.args, Expr(:(=), :(∂α[i]), :(∂α₂ - SpecialFunctions.digamma($αexpr) ) ) )
            else
                push!(pre_quote.args, Expr(:(-=), ∂αstorage, :($M * SpecialFunctions.digamma(α)) ) )
            end
        end
    end
    if loop
        quote
            $(Expr(:meta,:inline))
            @fastmath begin
                $pre_quote
            end
            out = zero($T)
            @vectorize $T for i ∈ 1:$M
                $q
            end
            @fastmath begin
                $(return_expression(return_expr))
            end
        end
    else
        quote
            $(Expr(:meta,:inline))
            @fastmath begin
                $pre_quote
                $q
                $(return_expression(return_expr))
            end
        end
    end
end

# α * log(β) + (α-1) * log(y) - β*y - lgamma(α)
@generated function Gamma(
            y::PaddedMatrices.AbstractFixedSizePaddedVector{M,T},
            α::Union{T, <: PaddedMatrices.AbstractFixedSizePaddedVector{M,T}},
            β::Union{T, <: PaddedMatrices.AbstractFixedSizePaddedVector{M,T}},
            ::Val{track}) where {track,T,M}
    αisvec = isa(α, PaddedMatrices.AbstractFixedSizePaddedVector)
    βisvec = isa(β, PaddedMatrices.AbstractFixedSizePaddedVector)
    gamma_quote(M, T, true, αisvec, βisvec, track, false)
end
@generated function ∂Gamma(
            y::PaddedMatrices.AbstractFixedSizePaddedVector{M,T},
            α::Union{T,<:PaddedMatrices.AbstractFixedSizePaddedVector{M,T}},
            β::Union{T,<:PaddedMatrices.AbstractFixedSizePaddedVector{M,T}},
            ::Val{track}) where {track,M,T}
            # ::Val{track}) where {track,T,M}
    αisvec = isa(α, PaddedMatrices.AbstractFixedSizePaddedVector)
    βisvec = isa(β, PaddedMatrices.AbstractFixedSizePaddedVector)
    gamma_quote(M, T, true, αisvec, βisvec, track, true)
end
@generated function Gamma(y::T, α::T, β::T, ::Val{track}) where {track,T <: Real}
    gamma_quote(1, T, false, false, false, track, false)
end
@generated function ∂Gamma(y::T, α::T, β::T, ::Val{track}) where {track,T <: Real}
    gamma_quote(1, T, false, false, false, track, true)
end

push!(DISTRIBUTION_DIFF_RULES, :Gamma)







function beta_quote(M, T, yisvec, αisvec, βisvec, (track_y, track_α, track_β), partial)
    q = quote end
    pre_quote = quote end
    return_expr = Expr(:tuple, :out)
    loop = any((yisvec, αisvec, βisvec))
    # set initialized to loop; if we are looping, we'll start out at zero
    initialized = loop
    if yisvec
        yexpr = :(y[i])
        logyexpr = :(SLEEFPirates.log(y[i]))
        logomyexpr = :(SLEEFPirates.log(one($T) - y[i]))
    else
        yexpr = :y
        logyexpr = :logy
        logomyexpr = :logomy
        push!(pre_quote.args, :(logy = Base.log(y)))
        push!(pre_quote.args, :(logomy = Base.log(one($T) - y)))
    end
    if αisvec
        αexpr = :(α[i])
        # lgammaαexpr = :(lgamma(α[i]))
        αm1expr = :(α[i] - one($T))
    else
        αexpr = :α
        # digammaαexpr = :(lgammaα)
        αm1expr = :(αm1)
        # push!(pre_quote.args, :(lgammaα = lgamma(α)))
        push!(pre_quote.args, :(αm1 = α - one(α)))
    end
    if βisvec
        βexpr = :(β[i])
        # lgammaαexpr = :(lgamma(α[i]))
        βm1expr = :(β[i] - one($T))
    else
        βexpr = :β
        # digammaαexpr = :(lgammaα)
        βm1expr = :(βm1)
        # push!(pre_quote.args, :(lgammaα = lgamma(α)))
        push!(pre_quote.args, :(βm1 = β - one($T)))
    end
    if αisvec || βisvec
        lbetaβexpr = :(SpecialFunctions.lbeta($αexpr, $βexpr))
    else # neither are vectors
        lbetaβexpr = :lbetaαβ
        push!(pre_quote.args, :(lbetaαβ = SpecialFunctions.lbeta(α, β)))
    end

    if partial
        if track_y
            if yisvec
                yassignment = :(=)
                ∂ystorage = :∂yᵢ
                push!(pre_quote.args, :(∂y = PaddedMatrices.MutableFixedSizePaddedVector{$M,$T}(undef)))
                push!(return_expr.args, :(PaddedMatrices.ConstantFixedSizePaddedVector(∂y)'))
            else
                ∂ystorage = :∂y
                push!(return_expr.args, :∂y)
                if loop
                    push!(pre_quote.args, :(∂y = zero($T)))
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
                push!(pre_quote.args, :(∂α = PaddedMatrices.MutableFixedSizePaddedVector{$M,$T}(undef)))
                push!(return_expr.args, :(PaddedMatrices.ConstantFixedSizePaddedVector(∂α)'))
            else
                ∂αstorage = :∂α
                dgαexpr = :dgα
                push!(pre_quote.args, :(dgα = SpecialFunctions.digamma(α)))
                push!(return_expr.args, :(∂α))
                if loop
                    push!(pre_quote.args, :(∂α = zero($T)))
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
                push!(pre_quote.args, :(∂β = PaddedMatrices.MutableFixedSizePaddedVector{$M,$T}(undef)))
                push!(return_expr.args, :(PaddedMatrices.ConstantFixedSizePaddedVector(∂β)'))
            else
                ∂βstorage = :(∂β)
                dgβexpr = :dgβ
                push!(pre_quote.args, :(dgβ = SpecialFunctions.digamma(β)))
                push!(return_expr.args, :(∂β))
                if loop
                    push!(pre_quote.args, :(∂β = zero($T)))
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
        if initialized
            push!(q.args, :( out += am1 * logy ) )
        else
            push!(q.args, :( out = am1 * logy ) )
            initialized = true
        end
        if partial
            track_α && push!(q.args, Expr(αassignment, ∂αstorage, :logy) )
            track_y && push!(q.args, Expr(yassignment, ∂ystorage, :(am1 / $yexpr) ) )
        end
    end
    if track_β || track_y
        push!(q.args, :( bm1 = $βm1expr))
        push!(q.args, :( logomy = $logomyexpr))
        if initialized
            push!(q.args, :( out += bm1 * logomy ) )
        else
            push!(q.args, :( out = bm1 * logomy ) )
            initialized = true
        end
        if partial
            track_β && push!(q.args, Expr(βassignment, ∂βstorage, :logomy) )
            if track_y
                if yisvec
                    push!(q.args, Expr(:(=), :(∂y[i]), :($∂ystorage - bm1 / (one($T) - $yexpr) ) ))
                else
                    push!(q.args, Expr(:(=), ∂ystorage, :($∂ystorage - bm1 / (one($T) - $yexpr) ) ))
                end
            end
        end
    end
    if track_α || track_β
        push!(q.args, :(ly = $logyexpr))
        if initialized
            push!(q.args, :( out -= $lbetaβexpr ) )
        else
            push!(q.args, :( out = -$lbetaβexpr ) )
            initialized = true
        end
        if partial
            if track_α
                if αisvec
                    push!(q.args, Expr(:(=), :(∂α[i]), :($∂αstorage + dgαβ - $dgαexpr) ) )
                else
                    push!(q.args, Expr(:(=), ∂αstorage, :($∂αstorage + dgαβ - $dgαexpr) ) )
                end
            end
            if track_β
                if βisvec
                    push!(q.args, Expr(:(=), :(∂β[i]), :($∂βstorage + dgαβ - $dgβexpr) ) )
                else
                    push!(q.args, Expr(:(=), ∂βstorage, :($∂βstorage + dgαβ - $dgβexpr) ) )
                end
            end
        end
    end


    if loop
        quote
            $(Expr(:meta,:inline))
            @fastmath begin
                $pre_quote
                out = zero($T)
            end
            @vectorize $T for i ∈ 1:$M
                $q
            end
            @fastmath begin
                $(return_expression(return_expr))
            end
        end
    else
        quote
            $(Expr(:meta,:inline))
            @fastmath begin
                $pre_quote
                $q
                $(return_expression(return_expr))
            end
        end
    end
end

# α * log(β) + (α-1) * log(y) - β*y - lgamma(α)
@generated function Beta(
            y::PaddedMatrices.AbstractFixedSizePaddedVector{M,T},
            α::Union{T,Int,<:PaddedMatrices.AbstractFixedSizePaddedVector{M,T}},
            β::Union{T,Int,<:PaddedMatrices.AbstractFixedSizePaddedVector{M,T}},
            ::Val{track}) where {track,T,M}
    αisvec = isa(α, PaddedMatrices.AbstractFixedSizePaddedVector)
    βisvec = isa(β, PaddedMatrices.AbstractFixedSizePaddedVector)
    beta_quote(M, T, true, αisvec, βisvec, track, false)
end
@generated function ∂Beta(
            y::PaddedMatrices.AbstractFixedSizePaddedVector{M,T},
            α::Union{T,Int,<:PaddedMatrices.AbstractFixedSizePaddedVector{M,T}},
            β::Union{T,Int,<:PaddedMatrices.AbstractFixedSizePaddedVector{M,T}},
            # ::Val{track}) where {track,M,T}
            ::Val{track}) where {track,T,M}
    αisvec = isa(α, PaddedMatrices.AbstractFixedSizePaddedVector)
    βisvec = isa(β, PaddedMatrices.AbstractFixedSizePaddedVector)
    beta_quote(M, T, true, αisvec, βisvec, track, true)
end
@generated function Beta(y::T, α::Union{T,Int}, β::Union{T,Int}, ::Val{track}) where {T <: Real,track}
    beta_quote(1, T, false, false, false, track, false)
end
@generated function ∂Beta(y::T, α::Union{T,Int}, β::Union{T,Int}, ::Val{track}) where {track,T <: Real}
    beta_quote(1, T, false, false, false, track, true)
end
push!(DISTRIBUTION_DIFF_RULES, :Beta)
