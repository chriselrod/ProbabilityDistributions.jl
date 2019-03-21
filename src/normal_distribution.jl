

push!(DISTRIBUTION_DIFF_RULES, :Normal_logeval_dropconst)

function Normal_logeval_dropconst_quote(y_type, μ_type, Σ_type, y_is_param, μ_is_param, Σ_is_param)
    if Σ_is_param
        if (Σ_type <: AbstractFloat) && ((y_type <: AbstractVector) || (μ_type <: AbstractVector))
            if y_type <: AbstractVector
                logdet_expr = :($(PaddedMatrices.type_length(y_type)) * logdet(R))
            else # μ_type <: AbstractVector
                logdet_expr = :($(PaddedMatrices.type_length(μ_type)) * logdet(R))
            end
        else
            logdet_expr = :(logdet(R))
        end
        out_expr = :($logdet_expr - eltype(Rδ)(0.5) * (Rδ' * Rδ))
    else
        out_expr = :(eltype(Rδ)(-0.5) * (Rδ' * Rδ))
    end
    quote
        $(Expr(:meta, :inline))
        δ = y - μ
        R = PaddedMatrices.invchol(Σ)
        Rδ = R*δ
        $out_expr
    end
end
"""
Note that the partial derivatives this emits are currently with respect to
the inverse of the cholesky factor of Σ.
This is because this is how the covariance matrix data types will store the underlying data.
"""
function ∂Normal_logeval_dropconst_quote(y_type, μ_type, Σ_type, y_is_param, μ_is_param, Σ_is_param)
    q = quote
        $(Expr(:meta, :inline))
        δ = y - μ
        R = PaddedMatrices.invchol(Σ)
        Rδ = R*δ
    end
    if Σ_is_param
        if (Σ_type <: AbstractFloat) && ((y_type <: AbstractVector) || (μ_type <: AbstractVector))
            if y_type <: AbstractVector
                logdet_expr = :(dR = det(R); $(PaddedMatrices.type_length(y_type)) * log(dR))
            else # μ_type <: AbstractVector
                logdet_expr = :(dR = det(R); $(PaddedMatrices.type_length(μ_type)) * log(dR))
            end
        else
            logdet_expr = :(dR = det(R); log(dR))
        end
        out_expr = Expr(:tuple, :($logdet_expr - eltype(Rδ)(0.5) * (Rδ' * Rδ)))
    else
        out_expr = Expr(:tuple, :(eltype(Rδ)(-0.5) * (Rδ' * Rδ)))
    end
    if y_is_param | μ_is_param
        push!(q.args, :(∂Rδ_∂δ = R' * Rδ))
        push!(q.args, :(∂Rδ_∂δ = ∂Rδ_∂δ + ∂Rδ_∂δ))
        y_is_param && push!(out_expr.args, :(-0.5∂Rδ_∂δ))
        μ_is_param && push!(out_expr.args, :(0.5∂Rδ_∂δ))
    end
    if Σ_is_param
        push!(q.args, :(∂Rδ_∂R = Rδ * δ'))
        push!(q.args, :(∂Rδ_∂R = ∂Rδ_∂R + ∂Rδ_∂R))
        push!(out_expr.args, :(1/dR - 0.5∂Rδ_∂R))
    end
    push!(q.args, out_expr)
    q
end
@generated function Normal_logeval_dropconst(y, μ, Σ, ::Val{track}) where {track}
    # Normal_logeval_dropconst_quote(y, μ, Σ, track[1], track[2], track[3])
    Normal_logeval_dropconst_quote(y, μ, Σ, track...)
end
@generated function Normal_logeval_dropconst(y, μ, Σ)
    Normal_logeval_dropconst_quote(y, μ, Σ, isparameter(y), isparameter(μ), isparameter(Σ))
end
@generated function ∂Normal_logeval_dropconst(y, μ, Σ, ::Val{track}) where {track}
    ∂Normal_logeval_dropconst_quote(y, μ, Σ, track...)
end
@generated function ∂Normal_logeval_dropconst(y, μ, Σ)
    ∂Normal_logeval_dropconst_quote(y, μ, Σ, isparameter(y), isparameter(μ), isparameter(Σ))
end

function matrix_normal_ar_lkj_partial_quote(M, N, T, V, (track_y, track_μ, track_Λ, track_L), partial)
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    @assert track_y == false
    q = quote
        $(track_Υ ? :( (U, ∂U) = ∂inv(L) ) : :(U = inv(L)) )
        μᵥ = vbroadcast(μ)
        Uᵥ = vbroadcast(U)
        Λᵥ = vbroadcast(Λ)
        Yᵥ = vectorizable(Y)

        @vectorize $T for i ∈ 1:length(y)
            δ = Yᵥ[i] - μᵥ
            ΛδΥ = Λ * (δ * Uᵥ)
            out += dot_self(ΛδΥ)
        end
    end
end

@generated function Normal_logeval_dropconst(
            Y::AbstractScatteredArray{T,2,<: Union{SMatrix{M,N,T},AbstractFixedSizePaddedMatrix{M,N,T}},1,2},
            μ::Union{SMatrix{M,N,T},AbstractFixedSizePaddedMatrix{M,N,T}}
            Λ::AbstractAutoregressiveMatrix{T,V},
            L::LKJ_Correlation_Cholesky{N,T}, ::Val{track}
        ) where {M,N,T,V,track}
    matrix_normal_ar_lkj_quote(M,N,T,V,track)
end
@generated function ∂Normal_logeval_dropconst(
            Y::AbstractScatteredArray{T,2,<: Union{SMatrix{M,N,T},AbstractFixedSizePaddedMatrix{M,N,T}},1,2},
            μ::Union{SMatrix{M,N,T},AbstractFixedSizePaddedMatrix{M,N,T}}
            Λ::AbstractAutoregressiveMatrix{T,V},
            L::LKJ_Correlation_Cholesky{N,T}, ::Val{track}
        ) where {M,N,T,V,track}
    matrix_normal_ar_lkj_partial_quote(M,N,T,V,track)
end
