push!(DISTRIBUTION_DIFF_RULES, :Normal)
function univariate_normal_quote(
    M::Int, T::DataType, yisvec::Bool,
    μisvec::Union{Bool,Nothing}, σisvec::Union{Bool,Nothing},
    (track_y, track_μ, track_σ)::NTuple{3,Bool},
    partial::Bool, stddev::Bool, sp::Bool = false, S = Tuple{M}
)
    P = first(S.parameters); N = length(S.parameters)
 #   return_scalar = true
    return_scalar = false
    # q = quote end
    if M > 1
        pre_quote = quote
            qf = SIMDPirates.vbroadcast(Vec{$(VectorizationBase.pick_vector_width(M,T)),$T}, zero($T))
        end
    else
        pre_quote = quote
            qf = zero($T)
        end
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
            qf = LoopVectorization.SIMDPirates.vmuladd(δ, δ, qf)
        end
        if return_scalar
            push!(return_expr.args, :( SIMDPirates.vsum((vmul($(T(-0.5)),qf) ))))
        else
            push!(return_expr.args, :( vmul($(T(-0.5)),qf) ))
        end            
    elseif σisvec# == true
        push!(pre_quote.args, :(logdetσ = zero($T)))
        if stddev
            loop_expr = quote
                δ = $δexpr
                σ⁻¹ = 1 / σ[i]
                δσ⁻¹ = δ * σ⁻¹
                δσ⁻² = δσ⁻¹ * δσ⁻¹
                qf = LoopVectorization.SIMDPirates.vadd(qf, δσ⁻²)
            end
            if track_σ
                if M > 1
                    push!(pre_quote.args, :(logdetσ = vbroadcast(Vec{$(VectorizationBase.pick_vector_width(M,T)),$T}, zero($T)) ))
                else
                    push!(pre_quote.args, :(logdetσ = zero($T)) )
                end
                push!(loop_expr.args, :(logdetσ = vsub( logdetσ, SLEEFPirates.log(σ[i]))))
                if return_scalar
                    push!(return_expr.args, :( SIMDPirates.vsum(vmuladd($(T(-0.5)), qf, logdetσ) )))
                else
                    push!(return_expr.args, :( vmuladd($(T(-0.5)), qf, logdetσ) ))
                end
            else
                if return_scalar
                    push!(return_expr.args, :( SIMDPirates.vsum(vmul($(T(-0.5)), qf ))))
                else
                  push!(return_expr.args, :( vmul($(T(-0.5)), qf )))
                end
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
                qf = LoopVectorization.SIMDPirates.vadd(δσ⁻², qf)
            end
            if M > 1
                if return_scalar
                    push!(return_expr.args, :( SIMDPirates.vsum(DistributionParameters.Target(  vmul($(T(-0.5)), qf ), - $M * Base.log(σ) ))))
                else
                    push!(return_expr.args, :( DistributionParameters.Target( vmul($(T(-0.5)), qf ), - $M * Base.log(σ) )))
                end
            else
                if return_scalar
                    push!(return_expr.args, :( SIMDPirates.vsum(vmul($(T(-0.5)), qf) ) -  Base.log(σ) ))
                else
                    push!(return_expr.args, :( vmul($(T(-0.5)), qf) ) -  Base.log(σ) )
                end
            end
        else
            loop_expr = quote
                δ = $δexpr
                δσ⁻¹ = δ * σ⁻¹
                qf = LoopVectorization.SIMDPirates.vmuladd(δσ⁻¹, δσ⁻¹, qf)
            end
            if return_scalar
                push!(return_expr.args, :( SIMDPirates.vsum( vmul($(T(-0.5)), qf ))) )
            else
                push!(return_expr.args, :( vmul($(T(-0.5)), qf ))) 
            end
        end
    else #σisvec == false
        # we do not need to keep track of δ / σ
        loop_expr = quote
            δ = $δexpr
            qf = LoopVectorization.SIMDPirates.vmuladd(δ, δ, qf)
        end
        if track_σ
            if M > 1
                if stddev
                    if return_scalar
                        push!(return_expr.args, :( SIMDPirates.vsum(DistributionParameters.Target( vmul($(T(-0.5))/(σ*σ),qf),  - $M * Base.log(σ) ))))
                    else
                        push!(return_expr.args, :( DistributionParameters.Target( vmul($(T(-0.5))/(σ*σ),qf),  - $M * Base.log(σ) )))
                    end
                else # variance parameter
                    if return_scalar
                        push!(return_expr.args, :( SIMDPirates.vsum(DistributionParameters.Target( vmul($(T(-0.5))/σ,qf ), $(T(-0.5M)) * Base.log(σ) ))))
                    else
                        push!(return_expr.args, :( DistributionParameters.Target( extract_data( vmul($(T(-0.5))/σ,qf) ), $(T(-0.5M)) * Base.log(σ) )))
                    end
                end
            else
                if stddev
                    if return_scalar
                        push!(return_expr.args, :( SIMDPirates.vsum(vmul($(T(-0.5))/(σ*σ),qf)) - Base.log(σ) ))
                    else
                        push!(return_expr.args, :( DistributionParameters.Target(vmul($(T(-0.5))/(σ*σ),qf), - Base.log(σ) )))
                    end
                else # variance parameter
                    if return_scalar
                        push!(return_expr.args, :( SIMDPirates.vsum( vmul($(T(-0.5))/σ,qf) ) - $(T(-0.5)) * Base.log(σ) ))
                    else
                        push!(return_expr.args, :( DistributionParameters.Target( vmul($(T(-0.5))/σ,qf), - $(T(-0.5)) * Base.log(σ) )))
                    end
                end
            end
        else # σ not tracked, so we drop the constant term
            if stddev
                if return_scalar
                    push!(return_expr.args, :( SIMDPirates.vsum(extract_data( vmul($(T(-0.5))/(σ*σ), qf) )) ))
                else
                    push!(return_expr.args, :( extract_data( vmul($(T(-0.5))/(σ*σ), qf) )) )
                end
            else # variance parameter
                if return_scalar
                    push!(return_expr.args, :( SIMDPirates.vsum(vmul($(T(-0.5))/σ, qf) )) )
                else
                    push!(return_expr.args, :( vmul($(T(-0.5))/σ, qf) ))
                end
            end
        end
    end
    sp && push!(pre_quote.args, :(_sptr = pointer(sp, $T)))
    if partial
        if track_y
            if yisvec
                push!(loop_expr.args, :(∂y[i] = - δσ⁻¹ * σ⁻¹))
                if sp
                    push!(pre_quote.args, :(∂y = PtrArray{$S,$T,$N,$P}(_sptr)))
                    push!(pre_quote.args, :(_sptr += $(VectorizationBase.align(M*sizeof(T)))))
                    push!(return_expr.args, :(∂y'))
                else
                    push!(pre_quote.args, :(∂y = MutableFixedSizePaddedArray{$S,$T,$N,$P}(undef) ))
                    push!(return_expr.args, :(ConstantFixedSizePaddedArray(∂y)'))
                end
            else
                push!(pre_quote.args, :(∂y = zero($T)))
                push!(loop_expr.args, :(∂y -= δσ⁻¹ * σ⁻¹))
                push!(return_expr.args, :(∂y))
#                push!(return_expr.args, :(∂y))
            end
        end
        if track_μ
            if μisvec == true
                push!(loop_expr.args, :(∂μ[i] = δσ⁻¹ * σ⁻¹))
                if sp
                    push!(pre_quote.args, :(∂μ = PtrArray{$S,$T,$N,$P}(_sptr)))
                    push!(pre_quote.args, :(_sptr += $(VectorizationBase.align(M*sizeof(T)))))
                    push!(return_expr.args, :(∂μ'))
                else
                    push!(pre_quote.args, :(∂μ = MutableFixedSizePaddedArray{$S,$T,$N,$P}(undef) ))
                    push!(return_expr.args, :(ConstantFixedSizePaddedArray(∂μ)'))
                end
            elseif μisvec == false
                push!(pre_quote.args, :(∂μ = zero($T)))
                push!(loop_expr.args, :(∂μ += δσ⁻¹ * σ⁻¹))
                push!(return_expr.args, :∂μ)
            end
        end
        if track_σ
            if σisvec == true
                push!(loop_expr.args, :(∂σ[i] = δσ⁻² * σ⁻¹ - σ⁻¹ ))
                if sp
                    push!(pre_quote.args, :(∂σ = PtrArray{$S,$T,$N,$P}(_sptr)))
                    push!(pre_quote.args, :(_sptr += $(VectorizationBase.align(M*sizeof(T)))))
                    push!(return_expr.args, :(∂σ'))
                else
                    push!(pre_quote.args, :(∂σ = MutableFixedSizePaddedArray{$S,$T,$N,$P}(undef) ))
#                push!(loop_expr.args, :(∂σ[i] = δσ⁻² * σ⁻¹ ))
                    push!(return_expr.args, :(ConstantFixedSizePaddedArray(∂σ)'))
                end
            elseif σisvec == false
                push!(pre_quote.args, :(∂σ = zero($T)))
                push!(loop_expr.args, :(∂σ += δσ⁻² * σ⁻¹ ))
#                push!(loop_expr.args, :(∂σ += δσ⁻² * σ⁻¹ - σ⁻¹ ))
                if stddev
                    push!(return_expr.args, :(∂σ - $M*σ⁻¹ ) )
                else
                    push!(return_expr.args, :(∂σ * ∂σ∂σ² - 0.5*$M*σ⁻²))
                end
            end
        end
    end
    spexpr = :(PaddedMatrices.StackPointer(_sptr))
    q = if loop
        quote
            $(Expr(:meta,:inline))
            @fastmath begin
                $pre_quote
            end
            $(macroexpand(ProbabilityDistributions, quote
                          LoopVectorization.@vvectorize $T for i ∈ 1:$M
                          $loop_expr
                          end
                          end))
            @fastmath begin
                $(return_expression(return_expr, sp, spexpr))
            end
        end
    else
        quote
            $(Expr(:meta,:inline))
            @fastmath begin
                $pre_quote
                $loop_expr
                $(return_expression(return_expr, sp, spexpr))
            end
        end
    end
    simplify_expr(q)
end

@generated function Normal(y::T, ::Val{track}) where {track,T <: Real}
    univariate_normal_quote(1, T, false, nothing, nothing, (track[1], false, false), false, true)
end

@generated function Normal(y::T, σ::Union{T,Int}, ::Val{track}) where {T <: Real, track}
    univariate_normal_quote(
        1, T, false, nothing, false,
        (track[1], false, track[2]), false, true
    )
end

@generated function Normal(
    y::T, σ²::LinearAlgebra.UniformScaling{T}, ::Val{track}
) where {T <: Real, track}
    univariate_normal_quote(
        1, T, false, nothing, false,
        (track[1], false, track[2]), false, false
    )
end

@generated function Normal(
    y::T, μ::T, σ::T, ::Val{track}
) where {track, T <: Real}
#) where {T <: Real, track}
    univariate_normal_quote( 1, T, false, false, false, track, false, true)
end

@generated function ∂Normal(y::T, ::Val{track}) where {track,T <: Real}
    univariate_normal_quote(1, T, false, nothing, nothing, (track[1], false, false), true, true)
end

@generated function ∂Normal(y::T, σ::Union{T,Int}, ::Val{track}) where {T <: Real, track}
    univariate_normal_quote(
        1, T, false, nothing, false,
        (track[1], false, track[2]), true, true
    )
end

@generated function ∂Normal(
    y::T, σ²::Union{LinearAlgebra.UniformScaling{T},LinearAlgebra.UniformScaling{Int}}, ::Val{track}
) where {T <: Real, track}
    univariate_normal_quote(
        1, T, false, nothing, false,
        (track[1], false, track[2]), true, false
    )
end

@generated function ∂Normal(
    y::T, μ::Union{T,Int}, σ::Union{T,Int}, ::Val{track}
) where {track,T <: Real}
#) where {T <: Real,track}
    univariate_normal_quote( 1, T, false, false, false, track, true, true)
end

@noinline function univariate_normal_length(SV::Core.SimpleVector, N, R, L)
    fsv = first(SV)::Int
    if N == 1
        M = fsv
    else
        fsv == R || throw("Arrays with more than 1 dimension cannot be padded.")
        M = L
    end
    M
end

@generated function Normal(y::AbstractFixedSizePaddedArray{S,T,N,R,L}, ::Val{track}) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R, L)
    univariate_normal_quote( M, T, true, nothing, nothing, (track[1], false, false), false, true, false )
end
@generated function Normal(y::AbstractFixedSizePaddedArray{S,T,N,R,L}, σ::Union{T,Int}, ::Val{track}) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R, L)
    univariate_normal_quote(
        M, T, true, nothing, false,
        (track[1], false, track[2]), false, true, false
    )
end
@generated function Normal(
                y::AbstractFixedSizePaddedArray{S,T,N,R,L},
                σ²::Union{LinearAlgebra.UniformScaling{T},LinearAlgebra.UniformScaling{Int}},
                ::Val{track}
)::T where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R, L)
    univariate_normal_quote(
        M, T, true, nothing, false,
        (track[1], false, track[2]), false, false, false
    )
end
@generated function Normal(
    y::AbstractFixedSizePaddedArray{S,T,N,R,L},
    μ::Union{T,Int,<:AbstractFixedSizePaddedArray{S,T,N,R,L}},
    σ::Union{T,Int,<:AbstractFixedSizePaddedArray{S,T,N,R,L}},
    ::Val{track}
) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R, L)
    univariate_normal_quote( M, T, true,
        μ <: AbstractFixedSizePaddedMatrix, σ <: AbstractFixedSizePaddedMatrix, track, false, true, false
    )
end
@generated function Normal(
    y::T,
    μ::AbstractFixedSizePaddedArray{S,T,N,R,L},
    σ::Union{T,Int,<:AbstractFixedSizePaddedArray{S,T,N,R,L}},
    ::Val{track}
) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R, L)
    univariate_normal_quote( M, T, false,
        true, σ <: AbstractFixedSizePaddedMatrix, track, false, true, false
    )
end

@generated function ∂Normal(y::AbstractFixedSizePaddedArray{S,T,N,R,L}, ::Val{track}) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R, L)
    univariate_normal_quote( M, T, true, nothing, nothing, (track[1], false, false), true, true, false, S)
end
@generated function ∂Normal(
    y::AbstractFixedSizePaddedArray{S,T,N,R,L}, σ::Union{T,Int}, ::Val{track}
#) where {M, T <: Real, track}
) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R, L)
    univariate_normal_quote(
        M, T, true, nothing, false,
        (track[1], false, track[2]), true, true, false, S
    )
end
@generated function ∂Normal(
    y::AbstractFixedSizePaddedArray{S,T,N,R,L},
    σ²::Union{LinearAlgebra.UniformScaling{T},LinearAlgebra.UniformScaling{Int}}, ::Val{track}
) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R, L)
    univariate_normal_quote(
        M, T, true, nothing, false,
        (track[1], false, track[2]), true, false, false, S
    )
end
@generated function ∂Normal(
    y::AbstractFixedSizePaddedArray{S,T,N,R,L},
    μ::Union{T,Int,<:AbstractFixedSizePaddedArray{S,T,N,R,L}},
    σ::Union{T,Int,<:AbstractFixedSizePaddedArray{S,T,N,R,L}},
    ::Val{track}
) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R, L)
    univariate_normal_quote(
        M, T, true,
        μ <: AbstractFixedSizePaddedMatrix, σ <: AbstractFixedSizePaddedMatrix, track, true, true, false, S
    )
end
@generated function ∂Normal(
    y::T,
    μ::AbstractFixedSizePaddedArray{S,T,N,R,L},
    σ::Union{T,Int,<:AbstractFixedSizePaddedArray{S,T,N,R,L}},
    ::Val{track}
) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R, L)
    univariate_normal_quote(
        M, T, false,
        true, σ <: AbstractFixedSizePaddedMatrix, track, true, true, false, S
    )
end


@generated function ∂Normal(
    sp::StackPointer,
    y::AbstractFixedSizePaddedArray{S,T,N,R,L},
    ::Val{track}
) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R, L)
    univariate_normal_quote( M, T, true, nothing, nothing, (track[1], false, false), true, true, true, S)
end
@generated function ∂Normal(
    sp::StackPointer,
    y::AbstractFixedSizePaddedArray{S,T,N,R,L},
    σ::Union{T,Int},
    ::Val{track}
) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R, L)
    univariate_normal_quote(
        M, T, true, nothing, false,
        (track[1], false, track[2]), true, true, true, S
    )
end
@generated function ∂Normal(
    sp::StackPointer,
    y::AbstractFixedSizePaddedArray{S,T,N,R,L},
    σ²::Union{LinearAlgebra.UniformScaling{T},LinearAlgebra.UniformScaling{Int}},
    ::Val{track}
) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R, L)
    univariate_normal_quote(
        M, T, true, nothing, false,
        (track[1], false, track[2]), true, false, true, S
    )
end
@generated function ∂Normal(
    sp::StackPointer,
    y::AbstractFixedSizePaddedArray{S,T,N,R,L},
    μ::Union{T,Int,<:AbstractFixedSizePaddedArray{S,T,N,R,L}},
    σ::Union{T,Int,<:AbstractFixedSizePaddedArray{S,T,N,R,L}},
    ::Val{track}
) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R, L)
    univariate_normal_quote(
        M, T, true,
        μ <: AbstractFixedSizePaddedVector, σ <: AbstractFixedSizePaddedVector, track, true, true, true, S
    )
end
@generated function ∂Normal(
    sp::StackPointer,
    y::T,
    μ::AbstractFixedSizePaddedArray{S,T,N,R,L},
    σ::Union{T,Int,<:AbstractFixedSizePaddedArray{S,T,N,R,L}},
    ::Val{track}
) where {S,T,N,R,L,track}
    M = univariate_normal_length(S.parameters, N, R, L)
    univariate_normal_quote(
        M, T, false,
        true, σ <: AbstractFixedSizePaddedVector, track, true, true, true, S
    )
end





