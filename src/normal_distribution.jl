

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
@generated function Normal(y::AbstractFixedSizePaddedVector{M,T},
                                    μ, L::StructuredMatrices.AbstractLowerTriangularMatrix{M,T,LL},
                                    ::Val{track}) where {M, T <: Real, LL, track}
    multivariate_normal_lkj_quote(M, LL, T, track, false, false)
end
@generated function Normal(y::AbstractFixedSizePaddedVector{M,T},
                                    μ, R::StructuredMatrices.AbstractUpperTriangularMatrix{M,T,LL},
                                    ::Val{track}) where {M, T <: Real, LL, track}

    multivariate_normal_lkj_quote(M, LL, T, track, false, true)
end

@inline function Normalc(
    sp::StackPointer,
    Y, μ,
    Σ::AbstractFixedSizeCovarianceMatrix{R,T,R},
    ::Val{track}
) where {R, K, T, track}
    Σcopy = DistributionParameters.MutableFixedSizeCovarianceMatrix{R,T,R}(undef)
    copyto!(Σcopy, Σ)
    Normal(sp, Y, μ, Σcopy, Val{track}())
end
@inline function Normalc(
    sp::StackPointer,
    Y, μ,
    Σ::AbstractFixedSizeCovarianceMatrix{R,T,R},
    Σcopy::AbstractFixedSizeCovarianceMatrix{R,T,R},
    ::Val{track}
) where {R, K, T, track}
    copyto!(Σcopy, Σ)
    Normal(sp, Y, μ, Σcopy, Val{track}())
end
@inline function ∂Normalc(
    sp::StackPointer,
    Y, μ,
    Σ::AbstractFixedSizeCovarianceMatrix{R,T,R},
    ::Val{track}
) where {R, K, T, track}
    Σcopy = DistributionParameters.MutableFixedSizeCovarianceMatrix{R,T,R}(undef)
    copyto!(Σcopy, Σ)
    ∂Normal(sp, Y, μ, Σcopy, Val{track}())
end
@inline function ∂Normalc(
    sp::StackPointer,
    Y, μ,
    Σ::AbstractFixedSizeCovarianceMatrix{R,T,R},
    Σcopy::AbstractFixedSizeCovarianceMatrix{R,T,R},
    ::Val{track}
) where {R, K, T, track}
    copyto!(Σcopy, Σ)
    ∂Normal(sp, Y, μ, Σcopy, Val{track}())
end


@generated function ∂Normal(y::AbstractFixedSizePaddedVector{M,T},
                                    μ, L::StructuredMatrices.AbstractLowerTriangularMatrix{M,T,LL},
                                    ::Val{track}) where {M, T <: Real, LL, track}
    multivariate_normal_lkj_quote(M, LL, T, track, true, false)
end
@generated function ∂Normal(y::AbstractFixedSizePaddedVector{M,T},
                                    μ, R::StructuredMatrices.AbstractUpperTriangularMatrix{M,T,LL},
                                    ::Val{track}) where {M, T <: Real, LL, track}

    multivariate_normal_lkj_quote(M, LL, T, track, true, true)
end


function matrix_normal_ar_lkj_quote(M, N, T, (track_y, track_μ, track_Λ, track_L), partial)
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    V = Vec{W,T}
    @assert track_y == false
    initialize_block = quote
        Λᵥ = StructuredMatrices.cache(Λ)
        Ny = length(Y)
        qf = vbroadcast(Vec{$W,$T}, zero($T))
        δ = MutableFixedSizePaddedMatrix{$M,$N,$V}(undef)
        δU = MutableFixedSizePaddedMatrix{$M,$N,$V}(undef)
        Yᵥ = vectorizable(Y)
        i = 0
    end
    loop_block = quote
        Yblock = vload($V, Yᵥ + i)
        PaddedMatrices.diff!(δ, μ, Yblock)
        mul!(δU, δ, U)
    end
    closing_block = quote end
    return_expr = Expr(:tuple, :qfscalar)
    if partial
        # First, we look at initializations
        if track_μ
            push!(initialize_block.args, :(∂qf∂δ = zero(PaddedMatrices.MutableFixedSizePaddedMatrix{$M,$N,Vec{$W,$T}}) ))
            push!(return_expr.args, :( vsum(∂qf∂δ) ) )
        end
        if ( track_μ || track_L )
            push!(initialize_block.args, :(Λᵥ′ΛᵥδU = PaddedMatrices.MutableFixedSizePaddedMatrix{$M,$N,Vec{$W,$T}}(undef) ))
        end
        if track_Λ
            push!(initialize_block.args, :(∂qf∂Λ = vbroadcast(Vec{$W,$T}, zero($T))))
            push!(closing_block.args, :((logdetΛ, ∂logdetΛ) = StructuredMatrices.∂logdet(Λ)))
            push!(return_expr.args, :(vsum(∂qf∂Λ) + $N * Ny * ∂logdetΛ))
        end
        if track_L
            push!(initialize_block.args, :((U, ∂U∂L) = StructuredMatrices.∂inv(L)))
            push!(closing_block.args, :((logdetL, ∂logdetL) = StructuredMatrices.∂logdet(L)))
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
                    qf = vadd(qfᵢ, qf)
                    ∂qf∂Λ = vadd(∂qf∂Λᵢ, ∂qf∂Λ)
                end)
            else
                push!(loop_block.args, quote
                    qf = vadd(StructuredMatrices.selfcrossmul_and_quadform!(Λᵥ′ΛᵥδU, Λᵥ, δU), qf)
                end)
            end
        elseif track_Λ
            push!(loop_block.args, quote
                qfᵢ, ∂qf∂Λᵢ = StructuredMatrices.∂quadform(Λᵥ, δU)
                qf = vadd(qfᵢ, qf)
                ∂qf∂Λ = vadd(∂qf∂Λᵢ, ∂qf∂Λ)
            end)
        else
            push!(loop_block.args, :(qf = vadd(StructuredMatrices.quadform(Λᵥ, δU), qf)))
        end
        track_L && push!(loop_block.args, :(StructuredMatrices.submul!(∂qf∂U, δ', Λᵥ′ΛᵥδU)))
        track_μ && push!(loop_block.args, :(StructuredMatrices.submul!(∂qf∂δ, Λᵥ′ΛᵥδU, U')))
    else # We are not taking partials
        track_L && push!(closing_block.args, :(logdetL = logdet(L)))
        track_Λ && push!(closing_block.args, :(logdetΛ = logdet(Λ)))

        push!(loop_block.args, :(qf = vadd(StructuredMatrices.quadform(Λᵥ, δU), qf)))
    end
    push!(loop_block.args, :(i += $W))
    # Here we handle the log determinants
    if track_L
        if track_Λ # track_L and track_Λ
            push!(closing_block.args, :(@fastmath qfscalar = Ny * ( $N * logdetΛ - $M * logdetL) - 0.5 * vsum(qf) ))
        else # track_L but not Λ
            push!(closing_block.args, :(@fastmath qfscalar = -Ny * $M * logdetL - 0.5 * vsum(qf) ))
        end
        if partial
            push!(closing_block.args, quote
                ∂qf∂L_part = StructuredMatrices.vsumvec(∂qf∂U)' * ∂U∂L
                ∂qf∂L = vmuladd($T(-Ny * $M), ∂logdetL', ∂qf∂L_part)
            end)
        end
    elseif track_Λ
        push!(closing_block.args, :(@fastmath qfscalar = Ny * $N * logdetΛ - 0.5 * vsum(qf) ))
    else
        push!(closing_block.args, :(@fastmath qfscalar = - 0.5 * vsum(qf) ))
    end
    quote
        $initialize_block
        Ysize = size(Y.data, 3)
        remmask = Y.mask
        @inbounds for ifrac ∈ 1:Ysize
            Yblock = vload($V, Yᵥ + i)
            PaddedMatrices.diff!(δ, μ, Yblock)
            if ifrac == Ysize
                PaddedMatrices.mask!(δ, remmask)
            end
        end
        $closing_block
        $(return_expression(return_expr))
    end
end

@generated function Normal(
            Y::AbstractScatteredArray{T,2,<: Union{SMatrix{M,N,T},AbstractFixedSizePaddedMatrix{M,N,T}},1,2},
            μ::Union{<:SMatrix{M,N,T},<:AbstractFixedSizePaddedMatrix{M,N,T}},
            Λ::AbstractAutoregressiveMatrix{T,V},
            L::LKJCorrCholesky{N,T}, ::Val{track}
        ) where {M,N,T,V,track}
    matrix_normal_ar_lkj_quote(M,N,T,track,false)
end
@generated function ∂Normal(
            Y::AbstractScatteredArray{T,2,<: Union{SMatrix{M,N,T},AbstractFixedSizePaddedMatrix{M,N,T}},1,2},
            μ::Union{<:SMatrix{M,N,T},<:AbstractFixedSizePaddedMatrix{M,N,T}},
            Λ::AbstractAutoregressiveMatrix{T,V},
            L::LKJCorrCholesky{N,T}, ::Val{track}
        ) where {M,N,T,V,track}
    matrix_normal_ar_lkj_quote(M,N,T,track,true)
end
function matrix_normal_ar_lkjinv_quote(M, N, T, (track_y, track_μ, track_Λ, track_U), partial)
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    V = Vec{W,T}
    WT = VectorizationBase.REGISTER_SIZE
    @assert track_y == false

    δ = MutableFixedSizePaddedMatrix{M,N,V,M,(M*N)}(undef)
    δU = MutableFixedSizePaddedMatrix{M,N,V,M,(M*N)}(undef)
    # buffer_size = 2M*N*WT
    initialize_block = quote
        Λᵥ = StructuredMatrices.cache(Λ)
        Ny = length(Y)
        qf = vbroadcast(Vec{$W,$T}, zero($T))
        Yᵥ = vectorizable(Y)
        # δ  = MutableFixedSizePaddedMatrix{$M,$N,$V,$M,$(M*N)}(undef)
        δ = $δ
        # δU = MutableFixedSizePaddedMatrix{$M,$N,$V,$M,$(M*N)}(undef)
        δU = $δU
        i = 0
    end
    # partial ||
    # push!(initialize_block.args, quote
    #     println("Mean:")
    #     println(μ)
    #     println("AR matrix:")
    #     println(Λ.spacing.rinvOmρ²ᵗ)
    #     println(Λ.spacing.ρᵗ .* Λ.spacing.rinvOmρ²ᵗ)
    #     println("U:")
    #     println(U)
    # end)
    loop_block = quote
        mul!(δU, δ, U)
    end
    closing_block = quote end
    return_expr = Expr(:tuple, :qfscalar)
    if partial
        # First, we look at initializations
        if track_μ
            # push!(initialize_block.args, :(∂qf∂δ = zero(PaddedMatrices.MutableFixedSizePaddedMatrix{$M,$N,Vec{$W,$T}}) ))
            ∂qf∂δ = PaddedMatrices.MutableFixedSizePaddedMatrix{M,N,Vec{W,T}}(undef)
            push!(initialize_block.args, quote
                ∂qf∂δ = $∂qf∂δ
                PaddedMatrices.zero!(∂qf∂δ)
            end)
            push!(return_expr.args, :( vsum(∂qf∂δ) ) )
        end
        if ( track_μ || track_U )
            # push!(initialize_block.args, :(Λᵥ′ΛᵥδU = PaddedMatrices.MutableFixedSizePaddedMatrix{$M,$N,Vec{$W,$T}}(undef) ))
            Λᵥ′ΛᵥδU = PaddedMatrices.MutableFixedSizePaddedMatrix{M,N,Vec{W,T}}(undef)
            push!(initialize_block.args, :(Λᵥ′ΛᵥδU = $Λᵥ′ΛᵥδU ))
        end
        if track_Λ
            push!(initialize_block.args, :(∂qf∂Λ = vbroadcast(Vec{$W,$T}, zero($T))))
            push!(closing_block.args, :((logdetΛ, ∂logdetΛ) = StructuredMatrices.∂logdet(Λ)))
            push!(return_expr.args, :(vsum(∂qf∂Λ) + $N * Ny * ∂logdetΛ))
        end
        if track_U
            push!(closing_block.args, :((logdetU, ∂logdetU) = StructuredMatrices.∂logdet(U)))
            # push!(initialize_block.args, :(∂qf∂U = zero(StructuredMatrices.MutableUpperTriangularMatrix{$N,Vec{$W,$T},$(StructuredMatrices.binomial2(N+1))})))
            ∂qf∂U = StructuredMatrices.MutableUpperTriangularMatrix{N,Vec{W,T},StructuredMatrices.binomial2(N+1)}(undef)
            push!(initialize_block.args, quote
                ∂qf∂U = $∂qf∂U
                PaddedMatrices.zero!(∂qf∂U)
            end)
            push!(return_expr.args, :∂out∂U)
        end

        # Now, at the loop block
        if (track_U || track_μ)
            if track_Λ
                push!(loop_block.args, quote
                    qfᵢ, ∂qf∂Λᵢ = StructuredMatrices.∂selfcrossmul_and_quadform!(Λᵥ′ΛᵥδU, Λᵥ, δU)
                    qf = vadd(qfᵢ, qf)
                    ∂qf∂Λ = vadd(∂qf∂Λᵢ, ∂qf∂Λ)
                end)
            else
                push!(loop_block.args, quote
                    qf = vadd(StructuredMatrices.selfcrossmul_and_quadform!(Λᵥ′ΛᵥδU, Λᵥ, δU), qf)
                end)
            end
        elseif track_Λ
            push!(loop_block.args, quote
                qfᵢ, ∂qf∂Λᵢ = StructuredMatrices.∂quadform(Λᵥ, δU)
                qf = vadd(qfᵢ, qf)
                ∂qf∂Λ = vadd(∂qf∂Λᵢ, ∂qf∂Λ)
            end)
        else
            push!(loop_block.args, :(qf = vadd(StructuredMatrices.quadform(Λᵥ, δU), qf)))
        end
        track_U && push!(loop_block.args, :(StructuredMatrices.submul!(∂qf∂U, δ', Λᵥ′ΛᵥδU)))
        # if track_U
        #     push!(loop_block.args, quote
        #         StructuredMatrices.submul!(∂qf∂U, δ', Λᵥ′ΛᵥδU)
        #         if !StructuredMatrices.all_finite(∂qf∂U)
        #             println("Not all elements of ∂qf∂U were finite!\n")
        #             println(∂qf∂U)
        #             println("\nArguments, δ':\n")
        #             println(δ')
        #             println("\nArguments, Λᵥ′ΛᵥδU:\n")
        #             println(Λᵥ′ΛᵥδU)
        #             throw("\nAll elements of ∂qf∂U should be finite!")
        #         end
        #     end)
        # end
        track_μ && push!(loop_block.args, :(StructuredMatrices.submul!(∂qf∂δ, Λᵥ′ΛᵥδU, U')))
    else # We are not taking partials
        track_U && push!(closing_block.args, :(logdetU = logdet(U)))
        track_Λ && push!(closing_block.args, :(logdetΛ = logdet(Λ)))

        push!(loop_block.args, :(qf = vadd(StructuredMatrices.quadform(Λᵥ, δU), qf)))
    end
    push!(loop_block.args, :(i += $W))
    # Here we handle the log determinants
    if track_U
        if track_Λ # track_U and track_Λ
            # if !partial
            #     push!(closing_block.args, :(@show Ny, $M, logdetU, $N, logdetΛ))
            #     push!(closing_block.args, :(@show -0.5qfvsum))
            #     push!(closing_block.args, :(@show qf))
            #     push!(closing_block.args, :(@show  Ny * ( $M * logdetU + $N * logdetΛ)))
            # end
            push!(closing_block.args, :(@fastmath qfscalar = Ny * ( $M * logdetU + $N * logdetΛ) - 0.5qfvsum))
        else # track_U but not Λ
            push!(closing_block.args, :(@fastmath qfscalar = Ny * $M * logdetU - 0.5qfvsum ))
        end
        if partial
            push!(closing_block.args, quote
                ∂qf∂Usummed = StructuredMatrices.vsum(∂qf∂U)
                coef = $T(Ny * $M)
                @vectorize $T for n ∈ 1:$N
                    ∂qf∂Usummed[n] = coef * ∂logdetU[n] + ∂qf∂Usummed[n]
                end
                ∂out∂U = UpperTriangularMatrix(∂qf∂Usummed)
            end)
        end
    elseif track_Λ
        push!(closing_block.args, :(@fastmath qfscalar = Ny * $N * logdetΛ - 0.5qfvsum ))
    else
        push!(closing_block.args, :(qfscalar = - 0.5qfvsum ))
    end
    quote
        # @show $((track_y, track_μ, track_Λ, track_U))
        $initialize_block
        Ysize = size(Y.data, 3)
        remmask = Y.mask
        @inbounds for ifrac ∈ 1:Ysize
            # Yblock = vload($V, Yᵥ + i)
            # PaddedMatrices.diff!(δ, μ, Yblock)
            # PaddedMatrices.vload!(δ, Yᵥ + i)
            Yᵢ = PaddedMatrices.vload($V, Yᵥ + i)
            PaddedMatrices.diff!(δ, μ, Yᵢ)
            ifrac == Ysize && PaddedMatrices.mask!(δ, remmask)
            $loop_block
        end
        qfvsum = vsum(qf)
        $closing_block
        $(return_expression(return_expr))
    end
end
@generated function Normal(
            Y::AbstractScatteredArray{T,2,<: Union{SMatrix{M,N,T},AbstractFixedSizePaddedMatrix{M,N,T}},1},
            μ::Union{<:SMatrix{M,N,T},<:AbstractFixedSizePaddedMatrix{M,N,T}},
            Λ::AbstractAutoregressiveMatrix{T,V},
            U::StructuredMatrices.AbstractUpperTriangularMatrix{N,T}, ::Val{track}
        # ) where {M,N,V,T,track}
        ) where {M,N,T,V,track}
    matrix_normal_ar_lkjinv_quote(M,N,T,track,false)
end
@generated function ∂Normal(
            Y::AbstractScatteredArray{T,2,<: Union{SMatrix{M,N,T},AbstractFixedSizePaddedMatrix{M,N,T}},1},
            μ::Union{<:SMatrix{M,N,T},<:AbstractFixedSizePaddedMatrix{M,N,T}},
            Λ::AbstractAutoregressiveMatrix{T,V},
            U::StructuredMatrices.AbstractUpperTriangularMatrix{N,T}, ::Val{track}
        # ) where {M,N,V,T,track}
        ) where {M,N,T,V,track}
    matrix_normal_ar_lkjinv_quote(M,N,T,track,true)
end



