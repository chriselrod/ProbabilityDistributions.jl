

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

function matrix_normal_ar_lkj_quote(M, N, T, (track_y, track_μ, track_Λ, track_L), partial)
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    @assert track_y == false
    initialize_block = quote
        Λᵥ = StructuredMatrices.cache(Λ)
        Ny = length(y)
        qf = vbroadcast(Vec{$W,$T}, zero($T))
        Yᵥ = vectorizable(Y)
        i = 1
    end
    loop_block = quote
        StructuredMatrices.diff!(δ, μ, Yᵥ[i])
        mul!(δU, δ, U)
    end
    closing_block = quote end
    return_expr = Expr(:tuple, :qfscalar)
    if partial
        # First, we look at initializations
        if track_μ
            push!(initialize_block.args, :(∂qf∂δ = zero(PaddedMatrices.MutableFixedSizePaddedMatrix{$M,$N,Vec{$W,$T}}) ))
            push!(return_expr.args, :( SIMDPirates.vsum(∂qf∂δ) ) )
        end
        if ( track_μ || track_L )
            push!(initialize_block.args, :(Λᵥ′ΛᵥδU = PaddedMatrices.MutableFixedSizePaddedMatrix{$M,$N,Vec{$W,$T}}(undef) ))
        end
        if track_Λ
            push!(initialize_block.args, :(∂qf∂Λ = vbroadcast(Vec{$W,$T}, zero($T))))
            push!(closing_block.args, :(logdetΛ, ∂logdetΛ = ∂logdet(Λ)))
            push!(return_expr.args, :(SIMDPirates.vsum(∂qf∂Λ)))
        end
        if track_L
            push!(initialize_block.args, :((U, ∂U∂L) = StructuredMatrices.∂inv(L)))
            push!(closing_block.args, :(logdetL, ∂logdetL = StructuredMatrices.∂logdet(L)))
            push!(initialize_block.args, :(∂qf∂U = zero(StructuredMatrices.MutableLowerTriangularMatrix{N,Vec{$W,$T},$(StructuredMatrices.binomial2(N+1))})))
            push!(return_expr.args, :∂qf∂L)
        else
            push!(initialize_block.args, :(U = inv(L)) )
        end

        # Now, at the loop block
        if (track_L || track_μ)
            if track_Λ
                push!(loop_block.args, quote
                    qfᵢ, ∂qf∂Λᵢ = StructuredMatrices.∂selfcrossmul_and_quadform!(Λᵥ′ΛᵥδU, Λᵥ, δU)
                    qf = SIMDPirates.vmuladd($(T(-0.5)), qfᵢ, qf)
                    ∂qf∂Λ = SIMDPirates.vadd(∂qf∂Λᵢ, ∂qf∂Λ)
                end)
            else
                push!(loop_block.args, quote
                    qf = SIMDPirates.vmuladd($(T(-0.5)), StructuredMatrices.selfcrossmul_and_quadform!(Λᵥ′ΛᵥδU, Λᵥ, δU), qf)
                end)
            end
        elseif track_Λ
            push!(loop_block.args, quote
                qfᵢ, ∂qf∂Λᵢ = StructuredMatrices.∂quadform(Λᵥ, δU)
                qf = SIMDPirates.vmuladd($(T(-0.5)), qfᵢ, qf)
                ∂qf∂Λ = SIMDPirates.vadd(∂qf∂Λᵢ, ∂qf∂Λ)
            end)
        else
            push!(loop_block.args, :(qf = SIMDPirates.vmuladd($(T(-0.5)), StructuredMatrices.quadform(Λᵥ, δU), qf)))
        end
        track_L && push!(loop_block.args, :(StructuredMatrices.submul!(∂qf∂U, δ', Λᵥ′ΛᵥδU)))
        track_μ && push!(loop_block.args, :(StructuredMatrices.submul!(∂qf∂δ, Λᵥ′ΛᵥδU, U')))
    else # We are not taking partials
        track_L && push!(closing_block.args, :(logdetL = logdet(L)))
        track_Λ && push!(closing_block.args, :(logdetΛ = logdet(Λ)))

        push!(loop_block.args, :(qf = SIMDPirates.vmuladd($(T(-0.5)), StructuredMatrices.quadform(Λᵥ, δU), qf)))
    end
    push!(loop_block.args, :(i += $W))
    # Here we handle the log determinants
    if track_L
        if track_Λ # track_L and track_Λ
            push!(closing_block.args, :(@fastmath qfscalar = -Ny * ( $M * logdetL + $N * logdetΛ) - 0.5 * SIMDPirates.vsum(qf) ))
        else # track_L but not Λ
            push!(closing_block.args, :(@fastmath qfscalar = -Ny * $M * logdetL - 0.5 * SIMDPirates.vsum(qf) ))
        end
        if partial
            push!(closing_block.args, quote
                ∂qf∂L_part = StructuredMatrices.vsumvec(∂qf∂U)' * ∂U∂L
                ∂qf∂L = SIMDPirates.vfma($T(-Ny * $M), ∂logdetL', ∂qf∂L_part)
            end)
        end
    elseif track_Λ
        push!(closing_block.args, :(@fastmath qfscalar = -Ny * $N * logdetΛ - 0.5 * SIMDPirates.vsum(qf) ))
    else
        push!(closing_block.args, :(@fastmath qfscalar = - 0.5 * SIMDPirates.vsum(qf) ))
    end
    quote
        $initialize_block
        for ifrac ∈ 1:size(Y.data, 3)
            $loop_block
        end
        $closing_block
        $return_expr
    end
end

@generated function Normal_logeval_dropconst(
            Y::AbstractScatteredArray{T,2,<: Union{SMatrix{M,N,T},AbstractFixedSizePaddedMatrix{M,N,T}},1,2},
            μ::Union{<:SMatrix{M,N,T},<:AbstractFixedSizePaddedMatrix{M,N,T}},
            Λ::AbstractAutoregressiveMatrix{T,V},
            L::LKJ_Correlation_Cholesky{N,T}, ::Val{track}
        ) where {M,N,T,V,track}
    matrix_normal_ar_lkj_quote(M,N,T,track,false)
end
@generated function ∂Normal_logeval_dropconst(
            Y::AbstractScatteredArray{T,2,<: Union{SMatrix{M,N,T},AbstractFixedSizePaddedMatrix{M,N,T}},1,2},
            μ::Union{<:SMatrix{M,N,T},<:AbstractFixedSizePaddedMatrix{M,N,T}},
            Λ::AbstractAutoregressiveMatrix{T,V},
            L::LKJ_Correlation_Cholesky{N,T}, ::Val{track}
        ) where {M,N,T,V,track}
    matrix_normal_ar_lkj_quote(M,N,T,track,true)
end
