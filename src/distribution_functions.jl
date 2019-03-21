
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
        pushfirst!(second_pass.args, :( $(Symbol("###seed###", a)) += $(Symbol("###seed###", out)) * $∂ ))
    end
    push!(first_pass.args, :($function_output = $(mod).$(Symbol(:∂, f))($(A...), Val{$track_tup}())))
    track_out && push!(tracked_vars, out)
    nothing
end
# """
# Arguments are: y, logitθ
# """
# function ∂Bernoulli_logit_logeval_dropconst_quote(y_is_param, logitθ_is_param)
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
# function ∂Bernoulli_logit_fmadd_logeval_dropconst_quote(y_is_param, β_is_param, X_is_param, α_is_param)
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


@generated function Bernoulli_logit_fmadd_logeval_dropconst(y::BitVector, X::AbstractMatrix{T}, β::AbstractVector{T}, α::AbstractFloat,
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
@generated function ∂Bernoulli_logit_fmadd_logeval_dropconst(y::BitVector, X::AbstractMatrix{T}, β::AbstractVector{T}, α::AbstractFloat,
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
            push!(out_expr.args, :(ConstantFixedSizePaddedVector{$N_β}($(Expr(:tuple, [Symbol(:∂βP_, n) for n ∈ 1:N_β]...,[zero(T) for n ∈ 1:L_β-N_β]...)))))
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
        q = quote
            # $(Expr(:meta, :inline))
            $init_q
            @vectorize $T for i ∈ eachindex(y)
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

push!(DISTRIBUTION_DIFF_RULES, :Bernoulli_logit_fmadd_logeval_dropconst)

function ∂Bernoulli_logit_fnmadd_logeval_dropconst_quote()

end
function ∂Bernoulli_logit_fmsub_logeval_dropconst_quote()

end
function ∂Bernoulli_logit_fnmsub_logeval_dropconst_quote()

end
