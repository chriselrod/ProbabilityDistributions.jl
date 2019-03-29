

push!(DISTRIBUTION_DIFF_RULES, :Normal_logeval_dropconst)

function univariate_normal_quote(
                M::Int, T::DataType, yisvec::Bool,
                μisvec::Union{Bool,Nothing}, σisvec::Union{Bool,Nothing},
                (track_y, track_μ, track_σ)::NTuple{3,Bool}, partial::Bool, stddev::Bool)

    # q = quote end
    pre_quote = quote
        qf = zero($T)
    end
    if !stddev
        # For now, we will require σ to be a scalar; ie, th eonly way to reach this is through UniformScaling
        @assert σisvec != true
        # Then the argument is named σ², and it is a UniformScaling multiple.
        push!(pre_quote.args, :(σ = sqrt(σ².λ)))
        if partial && track_σ
            push!(pre_quote.args, :(∂σ∂σ² = $(T(0.5))/σ ))
        end
    end
    return_expr = Expr(:tuple,)
    loop = false

    if yisvec # not allowed to be nothing, must be bool
        yexpr = :(y[i])
        loop = true
    else
        yexpr = :y
    end
    if μisvec == nothing
        # μ == 0
        δexpr = yexpr
        #if not, then it is a bool
    elseif μisvec # == true
        # μexpr = :(μ[i])
        δexpr = :($yexpr - μ[i])
        loop = true
    else # μisvec == false
        δexpr = :($yexpr - μ)
    end
    if σisvec == nothing
        # σ == 1
        loop_expr = quote
            δ = $δexpr
            δσ = δ
            qf += δ * δ
        end
        push!(return_expr.args, :( $T(-0.5)*qf ))
    elseif σisvec# == true
        push!(pre_quote.args, :(logdetσ = zero($T)))
        if stddev
            loop_expr = quote
                δ = $δexpr
                σ⁻¹ = 1 / σ[i]
                δσ = δ * σ⁻¹
                δσ² = δσ * δσ
                qf += δσ²
            end
            push!(pre_quote.args, :(logdetσ = zero($T)))
            if track_σ
                push!(loop_expr.args, :(logdetσ -= SLEEFPirates.log(σ[i])))
                push!(return_expr.args, :( $(T(-0.5))*qf + logdetσ ))
            else
                push!(return_expr.args, :( $(T(-0.5))*qf ))
            end
        # else # variance parameter
        #     loop_expr = quote
        #         δ = $δexpr
        #         qf = δ * δ / σ[i] + qf
        #     end
        #     if track_σ
        #         push!(loop_expr.args, :(logdetσ -= SLEEFPirates.log(σ[i])))
        #         push!(return_expr.args, :( $(T(-0.5))*qf + $(T(0.5)) * logdetσ ))
        #     else
        #         push!(return_expr.args, :( $(T(-0.5))*qf ))
        #     end
        end
        loop = true
    elseif partial && (track_y || track_μ || track_σ) #σisvec == false
        push!(pre_quote.args, :(σ⁻¹ = 1 / σ))
        if track_σ
            loop_expr = quote
                δ = $δexpr
                δσ = δ * σ⁻¹
                δσ² = δσ * δσ
                qf += δσ²
            end
            push!(return_expr.args, :( $T(-0.5)*qf - $M * Base.log(σ) ))
        else
            loop_expr = quote
                δ = $δexpr
                δσ = δ * σ⁻¹
                qf += δσ * δσ
            end
            push!(return_expr.args, :( $T(-0.5)*qf ))
        end
    else #σisvec == false
        # we do not need to keep track of δ / σ
        loop_expr = quote
            δ = $δexpr
            qf += δ * δ
        end
        if track_σ
            if stddev
                push!(return_expr.args, :( $T(-0.5)*qf/(σ*σ) - $M * Base.log(σ) ))
            else # variance parameter
                push!(return_expr.args, :( $T(-0.5)*qf/σ + $(T(-0.5M)) * Base.log(σ) ))
            end
        else # σ not tracked, so we drop the constant term
            if stddev
                push!(return_expr.args, :( $T(-0.5)*qf/(σ*σ) ))
            else # variance parameter
                push!(return_expr.args, :( $T(-0.5)*qf/σ ))
            end
        end
    end

    if partial
        if track_y
            if yisvec
                push!(pre_quote.args, :(∂y = MutableFixedSizePaddedVector{$M,$T}(undef) ))
                push!(loop_expr.args, :(∂y[i] = - δσ))
                push!(return_expr.args, :(ConstantFixedSizePaddedVector(∂y)))
            else
                push!(pre_quote.args, :(∂y = zero($T)))
                push!(loop_expr.args, :(∂y -= δσ))
                push!(return_expr.args, :∂y)
            end
        end
        if track_μ
            if μisvec == true
                push!(pre_quote.args, :(∂μ = MutableFixedSizePaddedVector{$M,$T}(undef) ))
                push!(loop_expr.args, :(∂μ[i] = δσ))
                push!(return_expr.args, :(ConstantFixedSizePaddedVector(∂μ)))
            elseif μisvec == false
                push!(pre_quote.args, :(∂μ = zero($T)))
                push!(loop_expr.args, :(∂μ += δσ))
                push!(return_expr.args, :∂μ)
            end
        end
        if track_σ
            if σisvec == true
                push!(pre_quote.args, :(∂σ = MutableFixedSizePaddedVector{$M,$T}(undef) ))
                push!(loop_expr.args, :(∂σ[i] = δσ² * σ⁻¹ - σ⁻¹ ))
                push!(return_expr.args, :(ConstantFixedSizePaddedVector(∂σ)))
            elseif σisvec == false
                push!(pre_quote.args, :(∂σ = zero($T)))
                push!(loop_expr.args, :(∂σ += δσ² * σ⁻¹ - σ⁻¹ ))
                if stddev
                    push!(return_expr.args, :∂σ)
                else
                    push!(return_expr.args, :(∂σ * ∂σ∂σ²))
                end
            end
        end
    end
    if loop
        quote
            $(Expr(:meta,:inline))
            @fastmath begin
                $pre_quote
            end
            @vectorize $T for i ∈ 1:$M
                $loop_expr
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
                $loop_expr
                $(return_expression(return_expr))
            end
        end
    end
end
function multivariate_normal_lkj_quote(M, L, T, (track_y, track_μ, track_Σ), partial, lkjinv)
    if partial
        q = quote
            $(Expr(:meta, :inline))
            δ = y - μ
        end
        if !lkjinv
            if partial && track_Σ
                push!(q.args, :(R, ∂R∂L = StructuredMatrices.∂inv′(L)) )
            else
                push!(q.args, :(R = StructuredMatrices.inv′(L)) )
            end
        end
        push!(q.args, :(Rδ = R*δ))
        if track_Σ
            if (Σ_type <: AbstractFloat) && ((y_type <: AbstractVector) || (μ_type <: AbstractVector))
                if y_type <: AbstractVector
                    logdet_expr = :($(M) * LinearAlgebra.logdet(R))
                else # μ_type <: AbstractVector
                    logdet_expr = :($(M) * LinearAlgebra.logdet(R))
                end
            else
                logdet_expr = :(dR = det(R); log(dR))
            end
            out_expr = Expr(:tuple, :($logdet_expr - $(T(0.5)) * (Rδ' * Rδ)))
        else
            out_expr = Expr(:tuple, :($(T(-0.5)) * (Rδ' * Rδ)))
        end
        if track_y | track_μ
            push!(q.args, :(∂Rδ_∂δ = R' * Rδ))
            push!(q.args, :(∂Rδ_∂δ = ∂Rδ_∂δ + ∂Rδ_∂δ))
            track_y && push!(out_expr.args, :(-0.5∂Rδ_∂δ))
            track_μ && push!(out_expr.args, :(0.5∂Rδ_∂δ))
        end
        if track_Σ
            push!(q.args, :(∂Rδ_∂R = Rδ * δ'))
            push!(q.args, :(∂Rδ_∂R = ∂Rδ_∂R + ∂Rδ_∂R))
            push!(q.args, quote
                ∂out∂R = MutableFixedSizePaddedVector{$L,$T}(undef)
                @vectorize $T for m ∈ 1:$M
                    ∂out∂R[m] = 1/dR[m] - 0.5 * ∂Rδ_∂R[m]
                end
                @vectorize $T for l ∈ 1:$(L-M)
                    ∂out∂R[l+$M] = - 0.5 * ∂Rδ_∂R[l + $M]
                end
            end)
            if lkjinv
                push!(out_expr.args, :(∂out∂R))
            else
                push!(out_expr.args, :(∂out∂R*∂R∂L))
            end
        end
        push!(q.args, out_expr)
        return q
    else
        if track_Σ
            if (Σ_type <: AbstractFloat) && ((y_type <: AbstractVector) || (μ_type <: AbstractVector))
                if y_type <: AbstractVector
                    logdet_expr = :($(M) * LinearAlgebra.logdet(R))
                else # μ_type <: AbstractVector
                    logdet_expr = :($(M) * LinearAlgebra.logdet(R))
                end
            else
                logdet_expr = :(logdet(R))
            end
            out_expr = :($logdet_expr - $(T(0.5)) * (Rδ' * Rδ))
        else
            out_expr = :($(T(-0.5)) * (Rδ' * Rδ))
        end
        quote
            $(Expr(:meta, :inline))
            δ = y - μ
            R = PaddedMatrices.invchol(Σ)
            Rδ = R*δ
            $out_expr
        end
    end
end
# @generated function Normal_logeval_dropconst(y::AbstractFixedSizePaddedVector{M,T}, ::Val{track}) where {M,track,T}
# @generated
function Normal_logeval_dropconst(y::AbstractFixedSizePaddedVector{M,T}, ::Val{track}) where {M,T,track}
    univariate_normal_quote( M, T, true, nothing, nothing, (track[1], false, false), false, true)
end
@generated function Normal_logeval_dropconst(y::T, ::Val{track}) where {track,T <: Real}
    univariate_normal_quote(1, T, false, nothing, nothing, (track[1], false, false), false, true)
end
@generated function Normal_logeval_dropconst(y::AbstractFixedSizePaddedVector{M,T}, σ::Union{T,Int}, ::Val{track}) where {M, T <: Real, track}
    univariate_normal_quote(
                    M, T, true, nothing, false,
                    (track[1], false, track[2]), false, true)
end
@generated function Normal_logeval_dropconst(y::T, σ::Union{T,Int}, ::Val{track}) where {T <: Real, track}
    univariate_normal_quote(
                    1, T, false, nothing, false,
                    (track[1], false, track[2]), false, true)
end
@generated function Normal_logeval_dropconst(
                y::AbstractFixedSizePaddedVector{M,T},
                σ²::Union{LinearAlgebra.UniformScaling{T},LinearAlgebra.UniformScaling{Int}}, ::Val{track})::T where {M, T <: Real, track}
    univariate_normal_quote(
                    M, T, true, nothing, false,
                    (track[1], false, track[2]), false, false)
end
@generated function Normal_logeval_dropconst(y::T, σ²::LinearAlgebra.UniformScaling{T}, ::Val{track}) where {T <: Real, track}
    univariate_normal_quote(
                    1, T, false, nothing, false,
                    (track[1], false, track[2]), false, false)
end

@generated function Normal_logeval_dropconst(y::T, μ::T, σ::T, ::Val{track}) where {T <: Real, track}
    univariate_normal_quote( 1, T, false, false, false, track, false, true)
end
@generated function Normal_logeval_dropconst(
                    y::AbstractFixedSizePaddedVector{M,T},
                    μ::Union{T,Int,<:AbstractFixedSizePaddedVector{M,T}},
                    σ::Union{T,Int,<:AbstractFixedSizePaddedVector{M,T}},
                    ::Val{track}) where {M, T <: Real, track}
    univariate_normal_quote( M, T, true,
        μ <: AbstractFixedSizePaddedVector, σ <: AbstractFixedSizePaddedVector, track, false, true)
end
@generated function Normal_logeval_dropconst(
                    y::T,
                    μ::AbstractFixedSizePaddedVector{M,T},
                    σ::Union{T,Int,<:AbstractFixedSizePaddedVector{M,T}},
                    ::Val{track}) where {M, T <: Real, track}
    univariate_normal_quote( M, T, false,
        true, σ <: AbstractFixedSizePaddedVector, track, false, true)
end
@generated function Normal_logeval_dropconst(
                    y::T,
                    μ::Union{T,Int},
                    σ::AbstractFixedSizePaddedVector{M,T},
                    ::Val{track}) where {M, T <: Real, track}
    univariate_normal_quote( M, T, false, false, true, track, false, true)
end

@generated function Normal_logeval_dropconst(y::AbstractFixedSizePaddedVector{M,T},
                                    μ, L::StructuredMatrices.AbstractLowerTriangularMatrix{M,T,LL},
                                    ::Val{track}) where {M, T <: Real, LL, track}
    multivariate_normal_lkj_quote(M, LL, T, track, false, false)
end
@generated function Normal_logeval_dropconst(y::AbstractFixedSizePaddedVector{M,T},
                                    μ, R::StructuredMatrices.AbstractUpperTriangularMatrix{M,T,LL},
                                    ::Val{track}) where {M, T <: Real, LL, track}

    multivariate_normal_lkj_quote(M, LL, T, track, false, true)
end



@generated function ∂Normal_logeval_dropconst(y::AbstractFixedSizePaddedVector{M,T}, ::Val{track}) where {track,M,T}
    univariate_normal_quote( M, T, true, nothing, nothing, (track[1], false, false), true, true)
end
@generated function ∂Normal_logeval_dropconst(y::T, ::Val{track}) where {track,T <: Real}
    univariate_normal_quote(1, T, false, nothing, nothing, (track[1], false, false), true, true)
end
@generated function ∂Normal_logeval_dropconst(y::AbstractFixedSizePaddedVector{M,T}, σ::T, ::Val{track}) where {M, T <: Real, track}
    univariate_normal_quote(
                    M, T, true, nothing, false,
                    (track[1], false, track[2]), true, true)
end
@generated function ∂Normal_logeval_dropconst(y::T, σ::T, ::Val{track}) where {T <: Real, track}
    univariate_normal_quote(
                    1, T, false, nothing, false,
                    (track[1], false, track[2]), true, true)
end
@generated function ∂Normal_logeval_dropconst(
                y::AbstractFixedSizePaddedVector{M,T},
                σ²::LinearAlgebra.UniformScaling{T}, ::Val{track}) where {M, T <: Real, track}
    univariate_normal_quote(
                    M, T, true, nothing, false,
                    (track[1], false, track[2]), true, false)
end
@generated function ∂Normal_logeval_dropconst(y::T, σ²::LinearAlgebra.UniformScaling{T}, ::Val{track}) where {T <: Real, track}
    univariate_normal_quote(
                    1, T, false, nothing, false,
                    (track[1], false, track[2]), true, false)
end

@generated function ∂Normal_logeval_dropconst(y::T, μ::T, σ::T, ::Val{track}) where {T <: Real, track}
    univariate_normal_quote( 1, T, false, false, false, track, true, true)
end
@generated function ∂Normal_logeval_dropconst(
                    y::AbstractFixedSizePaddedVector{M,T},
                    μ::Union{T,<:AbstractFixedSizePaddedVector{M,T}},
                    σ::Union{T,<:AbstractFixedSizePaddedVector{M,T}},
                    ::Val{track}) where {M, T <: Real, track}
    univariate_normal_quote( M, T, true,
        μ <: AbstractFixedSizePaddedVector, σ <: AbstractFixedSizePaddedVector, track, true, true)
end
@generated function ∂Normal_logeval_dropconst(
                    y::T,
                    μ::AbstractFixedSizePaddedVector{M,T},
                    σ::Union{T,<:AbstractFixedSizePaddedVector{M,T}},
                    ::Val{track}) where {M, T <: Real, track}
    univariate_normal_quote( M, T, false,
        true, σ <: AbstractFixedSizePaddedVector, track, true, true)
end
@generated function ∂Normal_logeval_dropconst(
                    y::T,
                    μ::T,
                    σ::AbstractFixedSizePaddedVector{M,T},
                    ::Val{track}) where {M, T <: Real, track}
    univariate_normal_quote( M, T, false, false, true, track, true, true)
end

@generated function ∂Normal_logeval_dropconst(y::AbstractFixedSizePaddedVector{M,T},
                                    μ, L::StructuredMatrices.AbstractLowerTriangularMatrix{M,T,LL},
                                    ::Val{track}) where {M, T <: Real, LL, track}
    multivariate_normal_lkj_quote(M, LL, T, track, true, false)
end
@generated function ∂Normal_logeval_dropconst(y::AbstractFixedSizePaddedVector{M,T},
                                    μ, R::StructuredMatrices.AbstractUpperTriangularMatrix{M,T,LL},
                                    ::Val{track}) where {M, T <: Real, LL, track}

    multivariate_normal_lkj_quote(M, LL, T, track, true, true)
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
            push!(initialize_block.args, :(∂qf∂U = zero(StructuredMatrices.MutableUpperTriangularMatrix{N,Vec{$W,$T},$(StructuredMatrices.binomial2(N+1))})))
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
                ∂qf∂L = SIMDPirates.vmuladd($T(-Ny * $M), ∂logdetL', ∂qf∂L_part)
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
function matrix_normal_ar_lkjinv_quote(M, N, T, (track_y, track_μ, track_Λ, track_U), partial)
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
        if ( track_μ || track_U )
            push!(initialize_block.args, :(Λᵥ′ΛᵥδU = PaddedMatrices.MutableFixedSizePaddedMatrix{$M,$N,Vec{$W,$T}}(undef) ))
        end
        if track_Λ
            push!(initialize_block.args, :(∂qf∂Λ = vbroadcast(Vec{$W,$T}, zero($T))))
            push!(closing_block.args, :(logdetΛ, ∂logdetΛ = ∂logdet(Λ)))
            push!(return_expr.args, :(SIMDPirates.vsum(∂qf∂Λ)))
        end
        if track_U
            push!(closing_block.args, :(logdetU, ∂logdetU = StructuredMatrices.∂logdet(U)))
            push!(initialize_block.args, :(∂qf∂U = zero(StructuredMatrices.MutableUpperTriangularMatrix{N,Vec{$W,$T},$(StructuredMatrices.binomial2(N+1))})))
            push!(return_expr.args, :∂out∂U)
        end

        # Now, at the loop block
        if (track_U || track_μ)
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
        track_L && push!(closing_block.args, :(logdetU = logdet(U)))
        track_Λ && push!(closing_block.args, :(logdetΛ = logdet(Λ)))

        push!(loop_block.args, :(qf = SIMDPirates.vmuladd($(T(-0.5)), StructuredMatrices.quadform(Λᵥ, δU), qf)))
    end
    push!(loop_block.args, :(i += $W))
    # Here we handle the log determinants
    if track_U
        if track_Λ # track_L and track_Λ
            push!(closing_block.args, :(@fastmath qfscalar = Ny * ( $M * logdetU - $N * logdetΛ) - 0.5 * SIMDPirates.vsum(qf) ))
        else # track_L but not Λ
            push!(closing_block.args, :(@fastmath qfscalar = Ny * $M * logdetU - 0.5 * SIMDPirates.vsum(qf) ))
        end
        if partial
            push!(closing_block.args, quote
                ∂out∂U = SIMDPirates.vmuladd($T(Ny * $M), ∂logdetU, StructuredMatrices.vsumvec(∂qf∂U))'
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
            U::StructuredMatrices.AbstractUpperTriangularMatrix{N,T}, ::Val{track}
        ) where {M,N,T,V,track}
    matrix_normal_ar_lkjinv_quote(M,N,T,track,false)
end
@generated function ∂Normal_logeval_dropconst(
            Y::AbstractScatteredArray{T,2,<: Union{SMatrix{M,N,T},AbstractFixedSizePaddedMatrix{M,N,T}},1,2},
            μ::Union{<:SMatrix{M,N,T},<:AbstractFixedSizePaddedMatrix{M,N,T}},
            Λ::AbstractAutoregressiveMatrix{T,V},
            U::StructuredMatrices.AbstractUpperTriangularMatrix{N,T}, ::Val{track}
        ) where {M,N,T,V,track}
    matrix_normal_ar_lkjinv_quote(M,N,T,track,true)
end
