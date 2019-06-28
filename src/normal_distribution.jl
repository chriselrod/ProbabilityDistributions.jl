

push!(DISTRIBUTION_DIFF_RULES, :Normal)

function univariate_normal_quote(
                M::Int, T::DataType, yisvec::Bool,
                μisvec::Union{Bool,Nothing}, σisvec::Union{Bool,Nothing},
                (track_y, track_μ, track_σ)::NTuple{3,Bool}, partial::Bool, stddev::Bool
            )

    # q = quote end
    pre_quote = quote
        qf = SIMDPirates.vbroadcast(SVec{$(VectorizationBase.pick_vector_width(M,T)),$T}, zero($T))
#        qf = zero($T)
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

    # add target
    if σisvec == nothing
        push!(pre_quote.args, :(σ⁻¹ = 1 ))
        # σ == 1
        loop_expr = quote
            δ = $δexpr
            δσ⁻¹ = δ
            qf = SIMDPirates.vmuladd(δ, δ, qf)
        end
        push!(return_expr.args, :( extract_data(vmul($(T(-0.5)),qf) )))
    elseif σisvec# == true
        push!(pre_quote.args, :(logdetσ = zero($T)))
        if stddev
            loop_expr = quote
                δ = $δexpr
                σ⁻¹ = 1 / σ[i]
                δσ⁻¹ = δ * σ⁻¹
                δσ⁻² = δσ⁻¹ * δσ⁻¹
                qf = SIMDPirates.vadd(qf, δσ⁻²)
            end
            if track_σ
                push!(pre_quote.args, :(logdetσ = vbroadcast(SVec{$(VectorizationBase.pick_vector_width(M,T)),$T}, zero($T)) ))
                push!(loop_expr.args, :(logdetσ = vsub( logdetσ, SLEEFPirates.log(σ[i]))))
                push!(return_expr.args, :( extract_data(vmuladd($(T(-0.5)), qf, logdetσ) )))
            else
                push!(return_expr.args, :( extract_data(vmul($(T(-0.5)), qf ))))
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
                δσ⁻¹ = δ * σ⁻¹
                δσ⁻² = δσ⁻¹ * δσ⁻¹
                qf = SIMDPirates.vadd(δσ⁻², qf)
            end
            push!(return_expr.args, :( DistributionParameters.Target( extract_data( vmul($(T(-0.5), qf) )), - $M * Base.log(σ) )))
        else
            loop_expr = quote
                δ = $δexpr
                δσ⁻¹ = δ * σ⁻¹
                qf = SIMDPirates.vmuladd(δσ⁻¹, δσ⁻¹, qf)
            end
            push!(return_expr.args, :( extract_data( vmul($(T(-0.5)), qf ))) )
        end
    else #σisvec == false
        # we do not need to keep track of δ / σ
        loop_expr = quote
            δ = $δexpr
            qf = SIMDPirates.vmuladd(δ, δ, qf)
        end
        if track_σ
            if stddev
                push!(return_expr.args, :( DistributionParameters.Target( extract_data( vmul($(T(-0.5))/(σ*σ),qf)),  $M * Base.log(σ) )))
            else # variance parameter
                push!(return_expr.args, :( DistributionParameters.Target( extract_data( vmul($(T(-0.5))/σ,qf) ), $(T(-0.5M)) * Base.log(σ) )))
            end
        else # σ not tracked, so we drop the constant term
            if stddev
                push!(return_expr.args, :( extract_data( vmul($(T(-0.5))/(σ*σ), qf) )) )
            else # variance parameter
                push!(return_expr.args, :( extract_data( vmul($(T(-0.5))/σ, qf) )) )
            end
        end
    end

    if partial
        if track_y
            if yisvec
                push!(pre_quote.args, :(∂y = MutableFixedSizePaddedVector{$M,$T}(undef) ))
                push!(loop_expr.args, :(∂y[i] = - δσ⁻¹ * σ⁻¹))
                push!(return_expr.args, :(ConstantFixedSizePaddedVector(∂y)'))
            else
                push!(pre_quote.args, :(∂y = zero($T)))
                push!(loop_expr.args, :(∂y -= δσ⁻¹ * σ⁻¹))
                push!(return_expr.args, :∂y)
            end
        end
        if track_μ
            if μisvec == true
                push!(pre_quote.args, :(∂μ = MutableFixedSizePaddedVector{$M,$T}(undef) ))
                push!(loop_expr.args, :(∂μ[i] = δσ⁻¹ * σ⁻¹))
                push!(return_expr.args, :(ConstantFixedSizePaddedVector(∂μ)'))
            elseif μisvec == false
                push!(pre_quote.args, :(∂μ = zero($T)))
                push!(loop_expr.args, :(∂μ += δσ⁻¹ * σ⁻¹))
                push!(return_expr.args, :∂μ)
            end
        end
        if track_σ
            if σisvec == true
                push!(pre_quote.args, :(∂σ = MutableFixedSizePaddedVector{$M,$T}(undef) ))
                push!(loop_expr.args, :(∂σ[i] = δσ⁻² * σ⁻¹ - σ⁻¹ ))
                push!(return_expr.args, :(ConstantFixedSizePaddedVector(∂σ)'))
            elseif σisvec == false
                push!(pre_quote.args, :(∂σ = zero($T)))
                push!(loop_expr.args, :(∂σ += δσ⁻² * σ⁻¹ - σ⁻¹ ))
                if stddev
                    push!(return_expr.args, :∂σ⁻¹)
                else
                    push!(return_expr.args, :(∂σ⁻¹ * ∂σ∂σ²))
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
# @generated function Normal(y::AbstractFixedSizePaddedVector{M,T}, ::Val{track}) where {M,track,T}
@generated function Normal(y::AbstractFixedSizePaddedVector{M,T}, ::Val{track}) where {M,T,track}
    univariate_normal_quote( M, T, true, nothing, nothing, (track[1], false, false), false, true)
end
@generated function Normal(y::T, ::Val{track}) where {track,T <: Real}
    univariate_normal_quote(1, T, false, nothing, nothing, (track[1], false, false), false, true)
end
@generated function Normal(y::AbstractFixedSizePaddedVector{M,T}, σ::Union{T,Int}, ::Val{track}) where {T <: Real, M, track}
    univariate_normal_quote(
                    M, T, true, nothing, false,
                    (track[1], false, track[2]), false, true)
end
@generated function Normal(y::T, σ::Union{T,Int}, ::Val{track}) where {T <: Real, track}
    univariate_normal_quote(
                    1, T, false, nothing, false,
                    (track[1], false, track[2]), false, true)
end
@generated function Normal(
                y::AbstractFixedSizePaddedVector{M,T},
                σ²::Union{LinearAlgebra.UniformScaling{T},LinearAlgebra.UniformScaling{Int}}, ::Val{track})::T where {T <: Real, M, track}
    univariate_normal_quote(
                    M, T, true, nothing, false,
                    (track[1], false, track[2]), false, false)
end
@generated function Normal(y::T, σ²::LinearAlgebra.UniformScaling{T}, ::Val{track}) where {T <: Real, track}
    univariate_normal_quote(
                    1, T, false, nothing, false,
                    (track[1], false, track[2]), false, false)
end

@generated function Normal(y::T, μ::T, σ::T, ::Val{track}) where {T <: Real, track}
    univariate_normal_quote( 1, T, false, false, false, track, false, true)
end
@generated function Normal(
                    y::AbstractFixedSizePaddedVector{M,T},
                    μ::Union{T,Int,<:AbstractFixedSizePaddedVector{M,T}},
                    σ::Union{T,Int,<:AbstractFixedSizePaddedVector{M,T}},
                    ::Val{track}) where {M, T <: Real, track}
    univariate_normal_quote( M, T, true,
        μ <: AbstractFixedSizePaddedVector, σ <: AbstractFixedSizePaddedVector, track, false, true)
end
@generated function Normal(
                    y::T,
                    μ::AbstractFixedSizePaddedVector{M,T},
                    σ::Union{T,Int,<:AbstractFixedSizePaddedVector{M,T}},
                    ::Val{track}) where {M, T <: Real, track}
    univariate_normal_quote( M, T, false,
        true, σ <: AbstractFixedSizePaddedVector, track, false, true)
end
@generated function Normal(
                    y::T,
                    μ::Union{T,Int},
                    σ::AbstractFixedSizePaddedVector{M,T},
                    ::Val{track}) where {M, T <: Real, track}
    univariate_normal_quote( M, T, false, false, true, track, false, true)
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


using DistributionParameters: AbstractFixedSizeCovarianceMatrix, AbstractCovarianceMatrix
function logdet_triangle(A::AbstractMatrix{T}) where {T}
    N = size(A,1)
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    Nrep = N >> Wshift

    vout = SIMDPirates.vbroadcast(Vec{W,T}, zero(T))
    @inbounds for i ∈ 0:Nrep-1
        x = ntuple(w -> Core.VecElement(A[W*i+w,W*i+w]), Val(W))
        vout = SIMDPirates.vadd(
            vout,
            SLEEFPirates.log( x )
        )
    end
#    rem = N & (W-1)
 #   log
    out = SIMDPirates.vsum(vout)
    @inbounds for i ∈ 1 + (N & -W):N
        out += log(A[i,i])
    end
    out
end
function vlogdet_triangle(A::AbstractMatrix{T}) where {T}
    N = size(A,1)
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    Nrep = N >> Wshift

    vout = vbroadcast(Vec{W,T}, zero(T))
    @inbounds for i ∈ 0:Nrep-1
        x = ntuple(w -> Core.VecElement(A[W*i+w,W*i+w]), Val(W))
        vout = SIMDPirates.vadd(
            vout,
            SLEEFPirates.log( x )
        )
    end
    offset = N & -W
    x = ntuple(Val(W)) do w
        i = w + offset
        @inbounds i > N ? Core.VecElement(one(T)) : Core.VecElement(A[i,i])
    end
    SIMDPirates.vadd( vout, SLEEFPirates.log( x ) )
end


@generated function Normal!(
    δ::AbstractMatrix{T},
    Y::NTuple{K},
    μ::NTuple{K,V},
    Σ::AbstractFixedSizeCovarianceMatrix{KT,T},
    ::Val{track}
) where {T, K, P, V <: PaddedMatrices.AbstractFixedSizePaddedVector{P,T}, KT, track}
    W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
    track_Y, track_μ, track_Σ = track
#    reps, rem = divrem(P, W)
#    if rem == 0 && reps < VectorizationBase.REGISTER_COUNT - 1
#        fill_δ = quote end
    quote
#=        ss = zero(T)
        @inbounds @fastmath for c ∈ 1:$KT, r ∈ c:$KT
            ss += Σ[r,c]
        end
        return sum(μ[1]) + sum(μ[2]) + ss=#


        coffset = 0
        for k ∈ 1:$K
            Yₖ = Y[k]
            μₖ = μ[k]
            ncol = size(Yₖ, 2)
            # Could avoid reloading μₖ
            # but performance difference seemed negligible in benchmarks
            # so, I figured I'd go for smaller code size.
            for c ∈ 1:ncol
                @vectorize $T for p ∈ 1:$P
                    δ[p,c+coffset] = μₖ[p] - Yₖ[p,c]
                end
            end
            coffset += ncol
        end
        L = Σ#.data
        LinearAlgebra.LAPACK.potrf!('L', L)
        LinearAlgebra.LAPACK.trtrs!('L', 'N', 'N', L, δ)
        #=
        Base.Cartesian.@nexprs 4 j -> target_j = SIMDPirates.vbroadcast(Vec{$(VectorizationBase.pick_vector_width(KT,T)),$T}, zero($T))
        δ_ptr = pointer(δ)
        Nδ = length(δ)
        for i ∈ 0:Nδ >> $(Wshift + 2) - 1
            Base.Cartesian.@nexprs 4 j -> begin
                vδ_j = SIMDPirates.vload(Vec{$W,$T}, δ_ptr + $(sizeof(T)*W) * ((j-1)+4i) )
                target_j = vmuladd( vδ_j, vδ_j, target )
            end
#            target = SIMDPirates.vmuladd( δ[i], δ[i], target )
        end
        for i ∈ 
        target_1 = vadd(target_1, target_3)
        target_2 = vadd(target_2, target_4)
            target = vadd(target_1, target_2)
        =#
        
        starget = vbroadcast(SVec{$(VectorizationBase.pick_vector_width(T)),T}, zero($T))
        @vectorize for i ∈ 1:length(δ)
            vδ = δ[i]
            starget = vmuladd( vδ, vδ, starget )
        end
        target = extract_data(starget)
        $(track_Σ ? :(vmuladd($(T(-0.5)), target, vmul(-one($T)*coffset, vlogdet_triangle(L)))) : :(vmul($(T(-0.5)), target)))

    end
end

# Inlined because of:
# https://github.com/JuliaLang/julia/issues/32414
# Stop forcing inlining when the issue is fixed.
@inline function Normal(sp::StackPointer, Y::NTuple{K}, μ::NTuple{K}, Σ::AbstractFixedSizeCovarianceMatrix{KT,T}, ::Val{track}) where {K, KT, T, track}
    Wm1 = VectorizationBase.pick_vector_width(KT,T) - 1
    cols = 0
    @inbounds for k ∈ 1:K
        cols += size(Y[k], 2)
    end
    δ = DynamicPtrMatrix(pointer(sp, T), (KT, cols), KT)# (KT+Wm1) & ~Wm1)
    # return sp, to reuse memory
    sp, Normal!(δ, Y, μ, Σ, Val{track}())
end
@generated function ∂Normal!(
#    ∂μ::Union{Nothing,NTuple{K,V}},
#    ∂Σ::Union{Nothing,AbstractMatrix{T}},
    Σ⁻¹δ::AbstractMatrix{T},
    δ::AbstractMatrix{T},
    Y::NTuple{K},
    μ::NTuple{K,V},
    Σ::AbstractFixedSizeCovarianceMatrix{KT,T,R},
    ::Val{track}
) where {T, K, P, R, V <: PaddedMatrices.AbstractFixedSizePaddedVector{P,T}, KT, track}
    W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
    track_Y, track_μ, track_Σ = track
#    track_μ = ∂μ != Nothing
#    track_Σ = ∂Σ != Nothing
#    track_Y, track_μ, track_Σ = track
#    @assert track_Y == false
    if !(track_μ | track_Σ)
        ret = :target
    else
        ret = Expr(:tuple, :target)
    end
    track_Y && push!(ret.args, :∂Y)
    track_μ && push!(ret.args, :∂μ)
    track_Σ && push!(ret.args, :∂Σ)
    if track_Σ
        q = quote
#=            ss = zero(T)
            @inbounds @fastmath for c ∈ 1:$KT, r ∈ c:$KT
                ss += Σ[r,c]
            end
            return (sum(μ[1]) + sum(μ[2]) + ss), DistributionParameters.One(), DistributionParameters.One()=#

            
            coffset = 0
            for k ∈ 1:$K
                Yₖ = Y[k]
                μₖ = μ[k]
                ncol = size(Yₖ, 2)
                # Could avoid reloading μₖ
                # but performance difference seemed negligible in benchmarks
                # so, I figured I'd go for smaller code size.
                for c ∈ 1:ncol
#                for p ∈ 1:$P
                    @vectorize $T for p ∈ 1:$P
                        δₚ = μₖ[p] - Yₖ[p,c]
                        δ[p,c+coffset] = δₚ
#                        Σ⁻¹δ[p,c+coffset] = δₚ
                    end
                end
                coffset += ncol
            end
            L, info = LinearAlgebra.LAPACK.potrf!('L', Σ)
            if info != 0
                ∂Σ = Σ
                ptr_δ = pointer(δ)
                $(track_μ ? Expr(:(=), :∂μ, Expr(:tuple, [:(PtrVector{$P,$T,$R,$R}( ptr_δ + $(sizeof(T)*R*(k-1)) )) for k ∈ 1:K]...)) : nothing)
                # TODO: support track_Y
   #             $(track_Y ? :() : nothing)
                target = vbroadcast(Vec{$W,$T}, $(T(-Inf)))
                return $ret
            end
            logdetL = vlogdet_triangle(L)
            Σ⁻¹ = L
            LinearAlgebra.LAPACK.potri!('L', Σ⁻¹)
            LinearAlgebra.BLAS.symm!('L', 'L', one($T), Σ, δ, zero($T), Σ⁻¹δ)
#            LinearAlgebra.LAPACK.potrs!('L', L, Σ⁻¹δ)
            starget = vbroadcast(SVec{$(VectorizationBase.pick_vector_width(T)),$T}, zero($T))
            @vectorize for i ∈ 1:length(δ)
                starget = vmuladd( Σ⁻¹δ[i], δ[i], starget )
            end
            target = extract_data(starget)
        end
    else
        q = quote
            coffset = 0
            for k ∈ 1:$K
                Yₖ = Y[k]
                μₖ = μ[k]
                ncol = size(Yₖ, 2)
                # Could avoid reloading μₖ
                # but performance difference seemed negligible in benchmarks
                # so, I figured I'd go for smaller code size.
                for c ∈ 1:ncol
#                for p ∈ 1:$P
                    @vectorize $T for p ∈ 1:$P
                        δₚ = μₖ[p] - Yₖ[p,c]
                        δ[p,c+coffset] = δₚ
                        Σ⁻¹δ[p,c+coffset] = δₚ
                    end
                end
                coffset += ncol
            end
            L = Σ#.data
            LinearAlgebra.LAPACK.potrf!('L', Σ)
            LinearAlgebra.LAPACK.potrs!('L', L, Σ⁻¹δ)
            starget = vbroadcast(SVec{$(VectorizationBase.pick_vector_width(T)),$T}, zero($T))
            @vectorize for i ∈ 1:length(δ)
                starget = vmuladd( Σ⁻¹δ[i], δ[i], starget )
            end
            target = extract_data(starget)
        end
    end
    
#=    if track_Y
    # Need to make these negative. Figure out best way to handle stack pointers.
        ∂μq = Expr(:tuple,)
        push!(q.args, :(coffset = 0))
        for k ∈ 1:K
            ∂μₖ = Symbol(:∂μ_, k)
            push!(∂μq.args, ∂μₖ)
            push!(q.args, quote
                  ncol = size(Y[$k],2)
                  $∂μₖ = view( Σ⁻¹δ, :, coffset+1:coffset+ncol )
                  coffset += ncol
                  end)
        end
        push!(q.args, :(∂μ = $∂μq))
    end =#
    if track_μ
        ∂μq = Expr(:tuple,)
        push!(q.args, :(coffset = 0))
        @assert !track_Y
        # Note that Σ⁻¹δ would be ∂Yₖ
        # so once we support that, we'll instead
        # of overwriting the first columns of δ
        # we'll write after the end of the array.
        push!(q.args, :(ptr_δ = pointer(δ)))
        for k ∈ 1:K
            ∂μₖ = Symbol(:∂μ_, k)
            ∂Yₖ = Symbol(:∂Y_, k)
            push!(∂μq.args, ∂μₖ)
            push!(q.args, quote
                  ncol = size(Y[$k],2)
                  $∂Yₖ = view( Σ⁻¹δ, :, coffset+1:coffset+ncol )
                  $∂μₖ = PtrVector{$P,$T,$R,$R}( ptr_δ + $(sizeof(T)*R*(k-1)) ) #view( Σ⁻¹δ, :, coffset+1:coffset+ncol )
                  PaddedMatrices.negative_sum!($∂μₖ, $∂Yₖ)
                  coffset += ncol
                  end)
        end
        push!(q.args, :(∂μ = $∂μq))
    end
    if track_Σ
        push!(q.args, quote
              ldfactor = -one($T)*coffset
              ∂Σ = Σ⁻¹
              LinearAlgebra.BLAS.syrk!('L', 'N', one($T), Σ⁻¹δ, ldfactor, ∂Σ)
#              LinearAlgebra.BLAS.syrk!('L', 'N', $(T(0.5)), Σ⁻¹δ, $(T(0.5)) * ldfactor, ∂Σ)
              @inbounds for i ∈ 1:$KT
                  ∂Σ[i,i] *= 0.5
              end
              #              target = vmul($(T(-0.5)), target)
              target = vmuladd($(T(-0.5)), target, vmul(ldfactor, logdetL) )
        end)
    else
        push!(q.args, :(target = vmul($(T(-0.5)),target)))
    end
    push!(q.args, ret)
    q
end
@generated function ∂Normal(
    sp::StackPointer,
    Y::NTuple{K},
    μ::NTuple{K},
    Σ::AbstractFixedSizeCovarianceMatrix{KT,T,KT},
    ::Val{track}
) where {KT, K, T, track}
#) where {K, T, KT, track}
    track_Y, track_μ, track_Σ = track
#    Wm1 = VectorizationBase.pick_vector_width(KT,T)-1
    #    R = (KT + Wm1) & ~Wm1
    R = KT
    @assert !track_Y
    q = quote
        # Inlined because of:
        # https://github.com/JuliaLang/julia/issues/32414
        # Stop forcing inlining when the issue is fixed.
        $(Expr(:meta,:inline))

        cols = 0
        @inbounds for k ∈ 1:$K
            cols += size(Y[k], 2)
        end
    end
#    if track_Σ
#        push!(q.args, :((sp, ∂Σ) = DistributionParameters.PtrFixedSizeCovarianceMatrix{$KT,$T}(sp)))#, ($KT,$KT), $((KT+Wm1) & ~Wm1))))
##        push!(q.args, :((sp, ∂Σ) = PtrMatrix{$KT,$T}(sp)))#, ($KT,$KT), $((KT+Wm1) & ~Wm1))))
##        push!(q.args, :((sp, ∂Σ) = DynamicPtrMatrix{$T}(sp, ($KT,$KT), $((KT+Wm1) & ~Wm1))))
#    else
#        push!(q.args, :(∂Σ = nothing))
#    end
    # This needs to be changed once we add support for track_Y == true
    # once we do that, we'll have to change where the pointers go, and
    # where sp ends up. Ideally, we calculate the best place at compile time
#    push!(q.args, :((sp, Σ⁻¹δ) = DynamicPtrMatrix{$T}(sp, ($KT,cols), $R)))
    push!(q.args, :(stack_pointer = pointer(sp,$T)))
    push!(q.args, :(δ = DynamicPtrMatrix(stack_pointer, ($KT, cols), $R)))
    push!(q.args, :(Σ⁻¹δ = DynamicPtrMatrix(stack_pointer + $(sizeof(T)*R)*cols, ($KT,cols), $R)))
    push!(q.args, :(sp = sp + $(K*sizeof(T)*R) ))
    push!(q.args, :(sp, ∂Normal!(Σ⁻¹δ, δ, Y, μ, Σ, Val{$track}()) ))
    q
end

#=

function Normal(δ::AbstractPaddedMatrix{T}, Y::NTuple{K}, μ::NTuple{K}, Σ::DynamicCovarianceMatrix{T}, ::Val{track}) where {T, K, track}
    track_Y, track_Σ = track
    target = zero(T)
    U = cholesky!(Σ).U
    U⁻¹Y = U \ Y
    @inbounds @simd for i ∈ eachindex(U⁻¹Y)
        target += U⁻¹Y[i] * U⁻¹Y[i]
        end
    if track_Σ
        return muladd(-0.5, target,  - size(Y,2) * logdet(U))
    else
        return -0.5target
    end
end
function Normal(Y::NTuple{N,MultivariateNormalVariate{T}}, Σ::DynamicCovarianceMatrix{T}, ::Val{track}) where {N, T, track}
    track_Y, track_Σ = track
    U = cholesky!(Σ).U
    target = zero(T)
    Ny = 0
    for n ∈ 1:N
        Ny -= size(Y[n], 2)
        U⁻¹Y = U \ Y[n]
        @inbounds @simd for i ∈ eachindex(U⁻¹Y)
            target += U⁻¹Y[i] * U⁻¹Y[i]
        end
    end
    if track_Σ
        return muladd(-0.5, target, Ny*logdet_triangle(U))
    else
        return -0.5target
    end
end



function ∂Normal(Y::MultivariateNormalVariate{T}, Σ::DynamicCovarianceMatrix{T}, ::Val{(true,true)}) where {T}
    cholΣ = cholesky!(Σ)
    target = zero(T)
    Ny = 0

    Ny -= size(Y, 2)
    δ = Y.δ
    Σ⁻¹δ = cholΣ \ Y
    @fastmath @inbounds @simd ivdep for i ∈ eachindex(Σ⁻¹δ)
        target += δ[i] * Σ⁻¹δ[i]
    end
    LinearAlgebra.BLAS.syrk!('L', 'N', T(0.5), Σ⁻¹δ.data, zero(T), Σ.∂Σ)
    
    muladd(-0.5, target, -size(Y, 2)*logdet_triangle(cholΣ)), Y.Σ⁻¹δ, Σ.∂Σ
end
function ∂Normal(Y::NTuple{N,MultivariateNormalVariate{T}}, Σ::DynamicCovarianceMatrix{T}, ::Val{(true,true)}) where {N, T}
    
    cholΣ = cholesky!(Σ)
    target = zero(T)
    Ny = 0
    for n ∈ 1:N
        Ny -= size(Y[n], 2)
        δ = Y[n].δ
        Σ⁻¹δ = cholΣ \ Y[n]
        @fastmath @inbounds @simd ivdep for i ∈ eachindex(Σ⁻¹δ)
            target += δ[i] * Σ⁻¹δ[i]
        end
        LinearAlgebra.BLAS.syrk!('L', 'N', T(0.5), Σ⁻¹δ, (n == 1 ? zero(T) : one(T)), Σ.∂Σ)
    end
    
    muladd(-0.5, target, Ny*logdet_triangle(cholΣ)), ntuple(n -> Y[n].Σ⁻¹δ, Val(N)), Σ.∂Σ
end
=#

@generated function ∂Normal(y::AbstractFixedSizePaddedVector{M,T}, ::Val{track}) where {M,T,track}
    univariate_normal_quote( M, T, true, nothing, nothing, (track[1], false, false), true, true)
end
@generated function ∂Normal(y::T, ::Val{track}) where {track,T <: Real}
    univariate_normal_quote(1, T, false, nothing, nothing, (track[1], false, false), true, true)
end
@generated function ∂Normal(y::AbstractFixedSizePaddedVector{M,T}, σ::Union{T,Int}, ::Val{track}) where {M, T <: Real, track}
    univariate_normal_quote(
                    M, T, true, nothing, false,
                    (track[1], false, track[2]), true, true)
end
@generated function ∂Normal(y::T, σ::Union{T,Int}, ::Val{track}) where {T <: Real, track}
    univariate_normal_quote(
                    1, T, false, nothing, false,
                    (track[1], false, track[2]), true, true)
end
@generated function ∂Normal(
                y::AbstractFixedSizePaddedVector{M,T},
                σ²::Union{LinearAlgebra.UniformScaling{T},LinearAlgebra.UniformScaling{Int}}, ::Val{track}) where {M, T <: Real, track}
    univariate_normal_quote(
                    M, T, true, nothing, false,
                    (track[1], false, track[2]), true, false)
end
@generated function ∂Normal(y::T, σ²::Union{LinearAlgebra.UniformScaling{T},LinearAlgebra.UniformScaling{Int}}, ::Val{track}) where {T <: Real, track}
    univariate_normal_quote(
                    1, T, false, nothing, false,
                    (track[1], false, track[2]), true, false)
end

@generated function ∂Normal(y::T, μ::Union{T,Int}, σ::Union{T,Int}, ::Val{track}) where {T <: Real, track}
    univariate_normal_quote( 1, T, false, false, false, track, true, true)
end
@generated function ∂Normal(
                    y::AbstractFixedSizePaddedVector{M,T},
                    μ::Union{T,Int,<:AbstractFixedSizePaddedVector{M,T}},
                    σ::Union{T,Int,<:AbstractFixedSizePaddedVector{M,T}},
                    ::Val{track}) where {M, T <: Real, track}
    univariate_normal_quote( M, T, true,
        μ <: AbstractFixedSizePaddedVector, σ <: AbstractFixedSizePaddedVector, track, true, true)
end
@generated function ∂Normal(
                    y::T,
                    μ::AbstractFixedSizePaddedVector{M,T},
                    σ::Union{T,Int,<:AbstractFixedSizePaddedVector{M,T}},
                    ::Val{track}) where {M, T <: Real, track}
    univariate_normal_quote( M, T, false,
        true, σ <: AbstractFixedSizePaddedVector, track, true, true)
end
@generated function ∂Normal(
                    y::T,
                    μ::Union{T,Int},
                    σ::AbstractFixedSizePaddedVector{M,T},
                    ::Val{track}) where {M, T <: Real, track}
    univariate_normal_quote( M, T, false, false, true, track, true, true)
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



