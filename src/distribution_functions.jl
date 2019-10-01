

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
function distribution_diff_rule!(mod, first_pass, second_pass, tracked_vars, out, A, f, verbose = false)
    track_out = false
#    verbose = true
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
        if verbose
            printstring = "distribution $f (ret: $out): "
            push!(first_pass.args, :(println($printstring)))
        end
        push!(first_pass.args, :($function_output = $(mod).$(Symbol(:∂, f))($(A...), Val{$track_tup}())))
        if verbose
            push!(first_pass.args, :(($out isa AbstractArray) ? ((length($out) < 100) && (@show $out)) : :(@show $out)))
            for a ∈ A
                a ∈ tracked_vars && push!(first_pass.args, :(@show $(Symbol("###adjoint###_##∂", out, "##∂", a, "##"))))
            end
        end
##        ret_string  = "function: $f: (ret"
##        push!(first_pass.args, :(println($ret_string, $function_output)))
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

function Bernoulli_logit_quote(T)
    W = VectorizationBase.pick_vector_width(T)
    q = quote
        # $(Expr(:meta, :inline))
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
    y::BitVector, α::AbstractVector{T},
    ::Val{track} = Val{(false,true)}()
) where {T, track}
    y_is_param, α_is_param = track
    @assert y_is_param == false
    Bernoulli_logit_quote(T)
end
function ∂Bernoulli_logit_quote(T)
    W = VectorizationBase.pick_vector_width(T)
    q = quote
        # $(Expr(:meta, :inline))
        target = vbroadcast(Vec{$W,$T}, zero($T))
        @vvectorize $T for i ∈ eachindex(y)
            αᵢ = α[i]
            invOmP = (one($T) + SLEEFPirates.exp( αᵢ ))
            ∂logP = one($T) / invOmP
            nlogOmP = SLEEFPirates.log(invOmP)
            nlogP = nlogOmP - αᵢ
            target = vsub(target, y[i] ? nlogP : nlogOmP)
            ∂α[i] = y[i] ? ∂logP : ∂logP - one($T)
        end
        target
    end
    simplify_expr(q)
end

@generated function ∂Bernoulli_logit!(
    ∂α::AbstractVector{T}, y::BitVector, α::AbstractVector{T}
) where {T}
    # y_is_param, α_is_param = track
    # @assert y_is_param == false
    # α_is_param ? ∂Bernoulli_logit_quote(T) : Bernoulli_logit_quote(T)
    ∂Bernoulli_logit_quote(T)
end

function ∂Bernoulli_logit(y::BitVector, α::AbstractVector{T}, ::Val{track} = Val{(false,true)}()) where {T,track}
    y_is_param, α_is_param = track
    @assert y_is_param == false
    if α_is_param
        ∂α = similar(α)
        return Bernoulli_logit!(∂α, y, α), ∂α
    else
        return Bernoulli_logit(y, α)
    end
end
push!(DISTRIBUTION_DIFF_RULES, :Bernoulli_logit)


@generated function Bernoulli_logit_fmadd(y::BitVector, X::AbstractMatrix{T}, β::AbstractVector{T}, α::AbstractFloat,
                            ::Val{track} = Val{(false,false,true,true)}()) where {T, track}
    y_is_param, β_is_param, X_is_param, α_is_param = track
    @assert y_is_param == false
    if PaddedMatrices.is_sized(β)
        N_β = PaddedMatrices.type_length(β)
        q = quote
            # $(Expr(:meta, :inline))
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
#            target = zero($T)
            target = vbroadcast(Vec{$(VectorizationBase.pick_vector_width(T)),$T}, zero($T))
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
            push!(out_expr.args, :(ConstantFixedSizeVector{$N_β}($(Expr(:tuple, [Symbol(:∂βP_, n) for n ∈ 1:N_β]...,[zero(T) for n ∈ 1:L_β-N_β]...)))'))
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
        unroll_factor = max(8 >>> Wshift, 1)
        q = quote
            # $(Expr(:meta, :inline))
            $init_q
            @vectorize $T $unroll_factor for i ∈ eachindex(y)
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

push!(DISTRIBUTION_DIFF_RULES, :Bernoulli_logit_fmadd)

# function ∂Bernoulli_logit_fnmadd_quote()

# end
# function ∂Bernoulli_logit_fmsub_quote()

# end
# function ∂Bernoulli_logit_fnmsub_quote()

# end


@generated function LKJ(L::AbstractLKJCorrCholesky{N,T}, η::T, ::Val{track}) where {N,T,track}
    quote
        #out = zero($T)
        target = vbroadcast(SVec{$(VectorizationBase.pick_vector_width(N-1,T)),$T}, zero($T))

        # @fastmath @inbounds @simd ivdep for n ∈ 1:$(N-1)
        @vectorize $T for n ∈ 1:$(N-1)
            target = vmuladd( ($(N - 3) - n + 2η), SLEEFPirates.log(L[n+1]), target)
        end
        extract_data(target)
    end |> simplify_expr
end
@generated function ∂LKJ(L::AbstractLKJCorrCholesky{N,T}, η::T, ::Val{track}) where {N,T,track}
    track_L, track_η = track
    q = if track_L && track_η
        quote
#            out = zero($T)
            target = vbroadcast(Vec{$(VectorizationBase.pick_vector_width(N-1,T)),$T}, zero($T))
            ∂L = MutableFixedSizeVector{$N,$T}(undef)
            @inbounds ∂L[1] = 0
            ∂η = zero($T)
            @vvectorize $T for n ∈ 1:$(N-1)
            # @fastmath @inbounds @simd ivdep for n ∈ 1:$(N-1)
                ∂ηₙ = log(L[n+1])
                coef = ($(N - 3) - n + 2η)
                target = vmuladd( coef, ∂ηₙ, target )
                ∂L[n+1] = coef / L[n+1]
                ∂η += 2∂ηₙ
            end
            target, Diagonal(ConstantFixedSizeVector(∂L)), ∂η
        end
    elseif track_L
        quote
            target = vbroadcast(SVec{$(VectorizationBase.pick_vector_width(N-1,T)),$T}, zero($T))
#            out = zero($T)
            ∂L = MutableFixedSizeVector{$N,$T}(undef)
            @inbounds ∂L[1] = 0
            ∂η = zero($T)
            @vectorize $T for n ∈ 1:$(N-1)
            # @fastmath @inbounds @simd ivdep for n ∈ 1:$(N-1)
                ∂ηₙ = log(L[n+1])
                coef = ($(N - 3) - n + 2η)
                target = vmuladd(coef, ∂ηₙ, target)
                ∂L[n+1] = coef / L[n+1]
            end
            extract_data(target), Diagonal(ConstantFixedSizeVector(∂L))
        end
    elseif track_η
        quote
#            out = zero($T)
            target = vbroadcast(Vec{$(VectorizationBase.pick_vector_width(N-1,T)),$T}, zero($T))
            ∂η = zero($T)
            @vvectorize $T for n ∈ 1:$(N-1)
            # @fastmath @inbounds @simd ivdep for n ∈ 1:$(N-1)
                ∂ηₙ = log(L[n+1])
                coef = ($(N - 3) - n + 2η)
                target = vmuladd( coef, ∂ηₙ, target )
                ∂η += 2∂ηₙ
            end
            target, ∂η
        end
    else
        quote
            target = vbroadcast(Vec{$(VectorizationBase.pick_vector_width(N-1,T)),$T}, zero($T))
#            out = zero($T)
            @vvectorize $T for n ∈ 1:$(N-1)
            # @fastmath @inbounds @simd ivdep for n ∈ 1:$(N-1)
                ∂ηₙ = log(L[n+1])
                coef = ($(N - 3) - n + 2η)
                target = vmuladd( coef, ∂ηₙ, target)
            end
            target
        end
    end
    simplify_expr(q)
end
@generated function ∂LKJ(sp::PaddedMatrices.StackPointer, L::AbstractLKJCorrCholesky{N,T}, η::T, ::Val{track}) where {N,T,track}
    track_L, track_η = track
    if track_L && track_η
        quote
            # Inlined because of:
            # https://github.com/JuliaLang/julia/issues/32414
            # Stop forcing inlining when the issue is fixed.
            $(Expr(:meta,:inline))
            target = vbroadcast(Vec{$(VectorizationBase.pick_vector_width(N-1,T)),$T}, zero($T))
#            out = zero($T)
            (sp,∂L) = PtrVector{$N,$T}(sp)
            @inbounds ∂L[1] = 0
            ∂η = zero($T)
            @vvectorize $T for n ∈ 1:$(N-1)
            # @fastmath @inbounds @simd ivdep for n ∈ 1:$(N-1)
                ∂ηₙ = log(L[n+1])
                coef = ($(N - 3) - n + 2η)
                target = vmuladd( coef, ∂ηₙ, target )
                ∂L[n+1] = coef / L[n+1]
                ∂η += 2∂ηₙ
            end
            sp, (target, Diagonal(∂L), ∂η)
        end
    elseif track_L
        quote
            # Inlined because of:
            # https://github.com/JuliaLang/julia/issues/32414
            # Stop forcing inlining when the issue is fixed.
            $(Expr(:meta,:inline))
            target = vbroadcast(Vec{$(VectorizationBase.pick_vector_width(N-1,T)),$T}, zero($T))
            #            out = zero($T)
#            i_init = reinterpret(Int, pointer(sp))
            (sp, ∂L) = PtrVector{$N,$T}(sp)
#            i_final = reinterpret(Int, pointer(sp))
#            @show i_final - i_init, $N, typeof(∂L)
            @inbounds ∂L[1] = 0
            ∂η = zero($T)
            @vvectorize $T for n ∈ 1:$(N-1)
            # @fastmath @inbounds @simd ivdep for n ∈ 1:$(N-1)
                ∂ηₙ = log(L[n+1])
                coef = ($(N - 3) - n + 2η)
                target = vmuladd(coef, ∂ηₙ, target)
                ∂L[n+1] = coef / L[n+1]
            end
#            println("\nReturning stack pointer:")
#            @show pointer(sp)
            sp, (target, Diagonal(∂L))
        end
    elseif track_η
        quote
            # Inlined because of:
            # https://github.com/JuliaLang/julia/issues/32414
            # Stop forcing inlining when the issue is fixed.
            $(Expr(:meta,:inline))
            target = vbroadcast(Vec{$(VectorizationBase.pick_vector_width(N-1,T)),$T}, zero($T))
#            out = zero($T)
            ∂η = zero($T)
            @vvectorize $T for n ∈ 1:$(N-1)
            # @fastmath @inbounds @simd ivdep for n ∈ 1:$(N-1)
                ∂ηₙ = log(L[n+1])
                coef = ($(N - 3) - n + 2η)
                target = vmuladd(coef, ∂ηₙ, target)
                ∂η += 2∂ηₙ
            end
            sp, (target, ∂η)
        end
    else
        quote
            # Inlined because of:
            # https://github.com/JuliaLang/julia/issues/32414
            # Stop forcing inlining when the issue is fixed.
            $(Expr(:meta,:inline))
            target = vbroadcast(Vec{$(VectorizationBase.pick_vector_width(N-1,T)),$T}, zero($T))
#            out = zero($T)
            @vvectorize $T for n ∈ 1:$(N-1)
            # @fastmath @inbounds @simd ivdep for n ∈ 1:$(N-1)
                ∂ηₙ = log(L[n+1])
                coef = ($(N - 3) - n + 2η)
                target = vmuladd(coef, ∂ηₙ, target)
            end
            sp, target
        end
    end |> simplify_expr
end
push!(DISTRIBUTION_DIFF_RULES, :LKJ)

function gamma_quote(M, T, yisvec, αisvec, βisvec, (track_y, track_α, track_β), partial, sp = false)
    q = quote end
    pre_quote = quote end
    return_expr = Expr(:tuple, :(extract_data(target)))
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
                if sp
                    push!(pre_quote.args, :((sp,∂y) = PaddedMatrices.PtrVector{$M,$T}(sp)))
                else
                    push!(pre_quote.args, :(∂y = PaddedMatrices.MutableFixedSizeVector{$M,$T}(undef)))
                end
                push!(return_expr.args, :(∂y'))
                # push!(return_expr.args, :(PaddedMatrices.ConstantFixedSizeVector(∂y)'))
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
                if sp
                    push!(pre_quote.args, :(∂α = PaddedMatrices.PtrVector{$M,$T}(sp)))
                else
                    push!(pre_quote.args, :(∂α = PaddedMatrices.MutableFixedSizeVector{$M,$T}(undef)))
                end
                push!(return_expr.args, :(∂α'))
#                push!(return_expr.args, :(PaddedMatrices.ConstantFixedSizeVector(∂α)'))
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
                if sp
                    push!(pre_quote.args, :(∂β = PaddedMatrices.PtrVector{$M,$T}(sp)))
                else
                    push!(pre_quote.args, :(∂β = PaddedMatrices.MutableFixedSizeVector{$M,$T}(undef)))
                end
                push!(return_expr.args, :(∂β'))
#                push!(return_expr.args, :(PaddedMatrices.ConstantFixedSizeVector(∂β)'))
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
            push!(q.args, :( target = vmuladd($αexpr, lβ, target) ) )
        else
            push!(q.args, :( target = $αexpr * lβ) )
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
            push!(q.args, :( target = vmuladd($αm1expr, ly, target) ) )
        else
            push!(q.args, :( target = $αm1expr * ly ) )
            initialized = true
        end
        if partial
            if track_α
                if αisvec
                    push!(q.args, Expr(:(=), :∂α₂, :($∂αstorage + ly)) )
                else
                    push!(q.args, Expr(:(+=), ∂αstorage, :ly) )
                end
            end
            track_y && push!(q.args, Expr(∂yassignment, ∂ystorage, :($αm1expr / $yexpr) ) )
        end
    end
    if track_β || track_y
        # initialized == true
        push!(q.args, :(target = vfnmadd($βexpr, $yexpr, target)))
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
        push!(q.args, :(target = vsub(target, $lgammaαexpr)))

        if partial
            if αisvec
                push!(q.args, Expr(:(=), :(∂α[i]), :(∂α₂ - SpecialFunctions.digamma($αexpr) ) ) )
            else
                push!(pre_quote.args, Expr(:(-=), ∂αstorage, :($M * SpecialFunctions.digamma(α)) ) )
            end
        end
    end
#    println("\n\n\n\n\n\n\n\n\n\n")
#    println(q)
#    println("\n\n\n\n\n\n\n\n\n\n")
    q = if loop
        quote
            $(Expr(:meta,:inline))
            @fastmath begin
                $pre_quote
            end
            target = vbroadcast(SVec{$(VectorizationBase.pick_vector_width(M,T)),$T}, zero($T))
#            out = zero($T)
            @vectorize $T for i ∈ 1:$M
                $q
            end
            @fastmath begin
                $(return_expression(return_expr, sp))
            end
        end
    else
        quote
            $(Expr(:meta,:inline))
            @fastmath begin
                $pre_quote
                $q
                $(return_expression(return_expr, sp))
            end
        end
    end
    simplify_expr(q)
end

# α * log(β) + (α-1) * log(y) - β*y - lgamma(α)
@generated function Gamma(
    y::PaddedMatrices.AbstractFixedSizeVector{M,T},
    α::Union{T, <: PaddedMatrices.AbstractFixedSizeVector{M,T}},
    β::Union{T, <: PaddedMatrices.AbstractFixedSizeVector{M,T}},
    ::Val{track}
) where {track,M,T}
#) where {track,T,M}
    αisvec = isa(α, PaddedMatrices.AbstractFixedSizeVector)
    βisvec = isa(β, PaddedMatrices.AbstractFixedSizeVector)
    gamma_quote(M, T, true, αisvec, βisvec, track, false, false)
end
@generated function ∂Gamma(
    y::PaddedMatrices.AbstractFixedSizeVector{M,T},
    α::Union{T,<:PaddedMatrices.AbstractFixedSizeVector{M,T}},
    β::Union{T,<:PaddedMatrices.AbstractFixedSizeVector{M,T}},
    ::Val{track}
#) where {track,M,T}
) where {track,T,M}
    αisvec = isa(α, PaddedMatrices.AbstractFixedSizeVector)
    βisvec = isa(β, PaddedMatrices.AbstractFixedSizeVector)
    gamma_quote(M, T, true, αisvec, βisvec, track, true, false)
end
@generated function ∂Gamma(
    sp::StackPointer,
    y::PaddedMatrices.AbstractFixedSizeVector{M,T},
    α::Union{T,<:PaddedMatrices.AbstractFixedSizeVector{M,T}},
    β::Union{T,<:PaddedMatrices.AbstractFixedSizeVector{M,T}},
    ::Val{track}
) where {track,M,T}
            # ::Val{track}) where {track,T,M}
    αisvec = isa(α, PaddedMatrices.AbstractFixedSizeVector)
    βisvec = isa(β, PaddedMatrices.AbstractFixedSizeVector)
    gamma_quote(M, T, true, αisvec, βisvec, track, true, true)
end
@generated function Gamma(y::T, α::T, β::T, ::Val{track}) where {track,T <: Real}
    gamma_quote(1, T, false, false, false, track, false, false)
end
@generated function ∂Gamma(y::T, α::T, β::T, ::Val{track}) where {track,T <: Real}
    gamma_quote(1, T, false, false, false, track, true, false)
end
@generated function ∂Gamma(sp::StackPointer, y::T, α::T, β::T, ::Val{track}) where {track,T <: Real}
    gamma_quote(1, T, false, false, false, track, true, true)
end

push!(DISTRIBUTION_DIFF_RULES, :Gamma)







function beta_quote(M, T, yisvec, αisvec, βisvec, (track_y, track_α, track_β), partial, sp::Bool = false)
    q = quote end
    pre_quote = quote end
    return_expr = Expr(:tuple, :(extract_data(target)))
    loop = any((yisvec, αisvec, βisvec))
    sp &= loop
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
                if sp
                    push!(pre_quote.args, :((sp,∂y) = PaddedMatrices.PtrVector{$M,$T}(sp)))
                    push!(return_expr.args, :(∂y'))
                else
                    push!(pre_quote.args, :(∂y = PaddedMatrices.MutableFixedSizeVector{$M,$T}(undef)))
                    push!(return_expr.args, :(PaddedMatrices.ConstantFixedSizeVector(∂y)'))
                end
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
                if sp
                    push!(pre_quote.args, :((sp,∂α) = PaddedMatrices.PtrVector{$M,$T}(undef)))
                    push!(return_expr.args, :(∂α'))
                else
                    push!(pre_quote.args, :(∂α = PaddedMatrices.MutableFixedSizeVector{$M,$T}(undef)))
                    push!(return_expr.args, :(PaddedMatrices.ConstantFixedSizeVector(∂α)'))
                end
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
                if sp
                    push!(pre_quote.args, :((sp,∂β) = PaddedMatrices.PtrVector{$M,$T}(undef)))
                    push!(return_expr.args, :(∂β'))
                else
                    push!(pre_quote.args, :(∂β = PaddedMatrices.MutableFixedSizeVector{$M,$T}(undef)))
                    push!(return_expr.args, :(PaddedMatrices.ConstantFixedSizeVector(∂β)'))
                end
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
            push!(q.args, :( target = vmuladd(am1, logy, target) ) )
        else
            push!(q.args, :( target = am1 * logy ) )
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
            push!(q.args, :( target = vmuladd(bm1, logomy, target) ) )
        else
            push!(q.args, :( target = bm1 * logomy ) )
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
            push!(q.args, :( target = vsub(target, $lbetaβexpr) ) )
        else
            push!(q.args, :( target = -$lbetaβexpr ) )
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
                $(return_expression(return_expr, sp))
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
    simplify_expr(q)
end

# α * log(β) + (α-1) * log(y) - β*y - lgamma(α)
@generated function Beta(
            y::PaddedMatrices.AbstractFixedSizeVector{M,T},
            α::Union{T,Int,<:PaddedMatrices.AbstractFixedSizeVector{M,T}},
            β::Union{T,Int,<:PaddedMatrices.AbstractFixedSizeVector{M,T}},
            # ::Val{track}) where {track,T,M}
            ::Val{track}) where {track,M,T}
    αisvec = isa(α, PaddedMatrices.AbstractFixedSizeVector)
    βisvec = isa(β, PaddedMatrices.AbstractFixedSizeVector)
    beta_quote(M, T, true, αisvec, βisvec, track, false)
end
@generated function ∂Beta(
            y::PaddedMatrices.AbstractFixedSizeVector{M,T},
            α::Union{T,Int,<:PaddedMatrices.AbstractFixedSizeVector{M,T}},
            β::Union{T,Int,<:PaddedMatrices.AbstractFixedSizeVector{M,T}},
            ::Val{track}) where {track,M,T}
            # ::Val{track}) where {track,T,M}
    αisvec = isa(α, PaddedMatrices.AbstractFixedSizeVector)
    βisvec = isa(β, PaddedMatrices.AbstractFixedSizeVector)
    beta_quote(M, T, true, αisvec, βisvec, track, true)
end
@generated function ∂Beta(
    sp::StackPointer,
    y::PaddedMatrices.AbstractFixedSizeVector{M,T},
    α::Union{T,Int,<:PaddedMatrices.AbstractFixedSizeVector{M,T}},
    β::Union{T,Int,<:PaddedMatrices.AbstractFixedSizeVector{M,T}},
    ::Val{track}
) where {track,M,T}
# ) where {track,T,M}
    αisvec = isa(α, PaddedMatrices.AbstractFixedSizeVector)
    βisvec = isa(β, PaddedMatrices.AbstractFixedSizeVector)
    beta_quote(M, T, true, αisvec, βisvec, track, true, true)
end
@generated function Beta(y::T, α::Union{T,Int}, β::Union{T,Int}, ::Val{track}) where {T <: Real,track}
    beta_quote(1, T, false, false, false, track, false)
end
@generated function ∂Beta(y::T, α::Union{T,Int}, β::Union{T,Int}, ::Val{track}) where {track,T <: Real}
    beta_quote(1, T, false, false, false, track, true)
end
push!(DISTRIBUTION_DIFF_RULES, :Beta)


function lsgg_quotebeta_quote(
    M::Int, T,
    yisvec::Bool, αisvec::Bool, ϕisvec::Bool, δisvec::Bool, σisvec::Bool,
    (track_y, track_α, track_ϕ, track_δ, track_σ)::NTuple{5,Bool},
    partial::Bool, sp::Bool
)
#    s = T(log(log(100)))
    # smy = s - y
    # α * smy - exp(ϕ * (smy - δ));
    @assert !any((track_α, track_ϕ, track_δ, track_σ))

    # ϕ = log(β)
    
    q = quote end
#    pre_quote = quote end
#    return_expr = Expr(:tuple, :(extract_data(target)))
#    loop = any((yisvec, αisvec, ϕisvec, δisvec, σisvec))
    # set initialized to loop; if we are looping, we'll start out at zero
#    initialized = loop
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
        if yisvec
            return quote
                # Inlined because of Julia SIMD corruption bug (if sp)
                # inlined to avoid heap allocation of mvector (if !sp)
                $(Expr(:meta,:inline))
                $(sp ? :((sp, ∂y) = PtrVector{$M,$T}(sp)) : :( ∂y = MutableFixedSizeVector{$M,$T}(undef)))
                target = SIMDPirates.vbroadcast(Vec{$W,$T}, zero($T))
                LoopVectorization.@vvectorize for m ∈ 1:$M
                    $q
                    target = LoopVectorization.SIMDPirates.vadd(SIMDPirates.vsub(asmy, eptsmyd), target)
                    ∂y[m] = SIMDPirates.vfmsub($δexpr, eptsmyd, $αexpr)
                end
                $(sp ? :(sp, (target, ∂y')) : :(target, ∂y'))
            end
        else
            @assert !any((αisvec, ϕisvec, δisvec, σisvec))
            return quote
                @fastmath begin
                    $q
                    target = asmy - eptsmyd
                    ∂y = $δexpr * eptsmyd - $αexpr
                end
                $(sp ? :(sp, (target, ∂y)) : :(target, ∂y))
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
                $(sp ? :(sp, target) : :target)
            end
        else
            @assert !any((αisvec, ϕisvec, δisvec, σisvec))
            return quote
                @fastmath begin
                    $q
                    target = asmy - eptsmyd
                end
                $(sp ? :(sp, target) : :target)
            end
        end
    end
end

@generated function lsgg(y::AbstractFixedSizeVector{M,T}, α, ϕ, δ, σ, ::Val{track}) where {M,T,track}
    lsgg_quotebeta_quote(
        M, T, true,
        α <: AbstractArray,
        ϕ <: AbstractArray,
        δ <: AbstractArray,
        σ <: AbstractArray,
        track, false, false
    )
end
@generated function ∂lsgg(y::AbstractFixedSizeVector{M,T}, α, ϕ, δ, σ, ::Val{track}) where {M,T,track}
    lsgg_quotebeta_quote(
        M, T, true,
        α <: AbstractArray,
        ϕ <: AbstractArray,
        δ <: AbstractArray,
        σ <: AbstractArray,
        track, true, false
    )
end
@generated function ∂lsgg(sp::StackPointer, y::AbstractFixedSizeVector{M,T}, α, ϕ, δ, σ, ::Val{track}) where {M,T,track}
    lsgg_quotebeta_quote(
        M, T, true,
        α <: AbstractArray,
        ϕ <: AbstractArray,
        δ <: AbstractArray,
        σ <: AbstractArray,
        track, true, true
    )
end
@generated function lsgg(y::T, α, ϕ, δ, σ, ::Val{track}) where {T<:Real,track}
    @assert !any((
        α <: AbstractArray,
        ϕ <: AbstractArray,
        δ <: AbstractArray,
        σ <: AbstractArray,
    ))
    lsgg_quotebeta_quote(
        1, T, false,
        false,false,false,false,
        track, false, false
    )
end
@generated function ∂lsgg(y::T, α, ϕ, δ, σ, ::Val{track}) where {T<:Real,track}
    @assert !any((
        α <: AbstractArray,
        ϕ <: AbstractArray,
        δ <: AbstractArray,
        σ <: AbstractArray,
    ))
    lsgg_quotebeta_quote(
        1, T, false,
        false,false,false,false,
        track, true, false
    )
end
@generated function ∂lsgg(sp::StackPointer, y::T, α, ϕ, δ, σ, ::Val{track}) where {T<:Real,track}
    @assert !any((
        α <: AbstractArray,
        ϕ <: AbstractArray,
        δ <: AbstractArray,
        σ <: AbstractArray,
    ))
    lsgg_quotebeta_quote(
        1, T, false,
        false,false,false,false,
        track, true, true
    )
end
push!(DISTRIBUTION_DIFF_RULES, :lsgg)




normal_lpdf(y, μ, τ, logrootτ) = -0.5τ * abs2(y-μ) + logrootτ
@generated function EₘₐₓNMA(
    α::AbstractVector{T}, Eₘₐₓ::AbstractVector{T}, ED₅₀::AbstractVector{T}, σ::T,
    Treatments::StructuredMatrices.RaggedMatrix{Int,Int,Vector{Int},Vector{Int}}, Doses::AbstractVector{T},
    ::Val{track}() = Val{(true,true,true,true,false,false)}
) where {T, track}
    track_α, track_Eₘₐₓ, track_ED₅₀, track_σ, track_treat, track_dose = track
    @assert track_treat == false && track_dose == false

    target = zero(T)
    τ = 1 / abs2(σ)
    j = 1
    col_lengths = treatment.column_lengths
    for i in eachindex(col_lengths)
        t = treatment[j]
        emi1 = emax(dose[j], emaxv[t], ed50v[t]) - αv[j]
        # αi1 = αv[j]
        s = zero(T)
        j += 1
        for k in 2:col_lengths[i]
            t = treatment[j]
            emik = emax(dose[j], emaxv[t], ed50v[t])
            δik = αv[j] - emik + emi1 # - αi1
            target += normal_lpdf(δik, s/(k-1), τ * (2*(k-1)) / k )
            s += δik
            j += 1
        end
    end
    target

end
function dnormal_lpdf(y, μ, τ, logrootτ, σ²)
    @fastmath begin
        z = y - μ
        f = -0.5τ * abs2(z) + logrootτ
        ∂f∂y = -τ * z
        ∂f∂μ = τ * z
        ∂f∂τ = -0.5abs2(z) + 0.5σ²
    end
    f, ∂f∂y, ∂f∂μ, ∂f∂τ
end

@generated function ∂EₘₐₓNMA()

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
