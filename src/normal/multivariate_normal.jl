

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
@inline function vlogdet_triangle(A::AbstractMatrix{T}) where {T}
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
    offset == N && return vout
    x = ntuple(Val(W)) do w
        i = w + offset
        @inbounds i > N ? Core.VecElement(one(T)) : Core.VecElement(A[i,i])
    end
    SIMDPirates.vadd( vout, SLEEFPirates.log( x ) )
end
@generated function vlogdet_triangle(A::StructuredMatrices.AbstractTriangularMatrix{P,T}) where {P,T}
    W = VectorizationBase.pick_vector_width(P,T)
    quote
        out = SIMDPirates.vbroadcast(Vec{$W,$T}, zero($T))
        @vvectorize $T for p in 1:$P
            out = LoopVectorization.SIMDPirates.vadd(SLEEFPirates.log(A[p]), out)
        end
        out
    end
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
    q = quote
        $(Expr(:meta,:inline))
        coffset = 0
        for k ∈ 1:$K
            Yₖ = Y[k]
            μₖ = μ[k]
            ncol = size(Yₖ, 2)
            # Could avoid reloading μₖ
            # but performance difference seemed negligible in benchmarks
            # so, I figured I'd go for smaller code size.
            for c ∈ 1:ncol
                @vvectorize $T for p ∈ 1:$P
                    δ[p,c+coffset] = μₖ[p] - Yₖ[p,c]
                end
            end
            coffset += ncol
        end
        L = Σ#.data
        LinearAlgebra.LAPACK.potrf!('L', L)
        LinearAlgebra.LAPACK.trtrs!('L', 'N', 'N', L, δ)
        
        starget = vbroadcast(SVec{$(VectorizationBase.pick_vector_width(T)),T}, zero($T))
        @vvectorize for i ∈ 1:length(δ)
            vδ = δ[i]
            starget = vmuladd( vδ, vδ, starget )
        end
        target = extract_data(starget)
        $(track_Σ ? :(vmuladd($(T(-0.5)), target, vmul(-one($T)*coffset, vlogdet_triangle(L)))) : :(vmul($(T(-0.5)), target)))
    end
    simplify_expr(q)
end

# Inlined because of:
# https://github.com/JuliaLang/julia/issues/32414
# Stop forcing inlining when the issue is fixed.
@inline function Normal(sp::StackPointer, Y::NTuple{K}, μ::NTuple{K}, Σ::AbstractFixedSizeCovarianceMatrix{R,T}, ::Val{track}) where {K, R, T, track}
#    Wm1 = VectorizationBase.pick_vector_width(R,T) - 1
    cols = 0
    @inbounds for k ∈ 1:K
        cols += size(Y[k], 2)
    end
    δ = DynamicPtrMatrix(pointer(sp, T), (R, cols), R)# (KT+Wm1) & ~Wm1)
    # return sp, to reuse memory
    sp, Normal!(δ, Y, μ, Σ, Val{track}())
end
@inline function Normal(
    sp::StackPointer,
    Y::AbstractMatrix{T},
    μ::PaddedMatrices.AbstractFixedSizePaddedVector{R,T},
    Σ::AbstractFixedSizeCovarianceMatrix{R,T},
    ::Val{track}
) where {R,T,track}
    Normal(sp, (Y,), (μ,), Σ, Val{track}())
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
    if any((track_Y, track_μ, track_Σ))
        ret = Expr(:tuple, :target)
    else
        ret = :target
    end
    track_Y && push!(ret.args, :∂Y)
    track_μ && push!(ret.args, :∂μ)
    track_Σ && push!(ret.args, :∂Σ)


    q = quote
        $(Expr(:meta,:inline)) # work-around for SIMD-corruption bug
        coloffset_0 = 0
        ptr_δ = pointer(δ)
    end
    for k ∈ 1:K
        setup_quote = quote
            @inbounds $(Symbol(:Y_,k)) = Y[$k]
            #                @inbounds $(Symbol(:μ_,k)) = μ[$k]
            $(Symbol(:ncol_,k)) = size($(Symbol(:Y_,k)),2)
            $(Symbol(:coloffset_,k)) = $(Symbol(:coloffset_,k-1)) + $(Symbol(:ncol_,k))
            $(Symbol(:∂Y_,k)) = view( Σ⁻¹δ, :, $(Symbol(:coloffset_,k-1))+1:$(Symbol(:coloffset_,k)) )
            $(Symbol(:∂μ_,k)) = PtrVector{$P,$T,$R}( ptr_δ + $(sizeof(T)*R*(k-1)))
        end
        push!(q.args, setup_quote)
    end
    track_Y && push!(q.args, Expr(:(=), :∂Y, Expr(:tuple, [Symbol(:∂Y_,k) for k ∈ 1:K]...)))
    track_μ && push!(q.args, Expr(:(=), :∂μ, Expr(:tuple, [Symbol(:∂μ_,k) for k ∈ 1:K]...)))
    push!(q.args, Expr(:(=), :ncoltup, Expr(:tuple, [Symbol(:ncol_,k) for k ∈ 1:K]...)))
    push!(q.args, Expr(:(=), :coloffsettup, Expr(:tuple, [Symbol(:coloffset_,k-1) for k ∈ 1:K+1]...)))
    if track_Σ
        # if first real calculation fails, abort
        cholesky_check_quote = quote
            L, info = LinearAlgebra.LAPACK.potrf!('L', Σ) # Cholesky factor
            if info != 0
                ∂Σ = Σ
                target = vbroadcast(Vec{$W,$T}, $(typemin(T)))
                return $ret
            end
            # while still hot in memory, we proceed to calculate the determinant and inverse
            logdetL = vlogdet_triangle(L)
            Σ⁻¹ = L
            LinearAlgebra.LAPACK.potri!('L', Σ⁻¹) # calculates Σ⁻¹ from cholesky factor of Σ
        end
        push!(q.args, cholesky_check_quote)
        δloopquote = quote
            @inbounds for k ∈ 1:$K
                Yₖ = Y[k]
                μₖ = μ[k]
                for c ∈ 1:ncoltup[k]
                    coffset = c + coloffsettup[k]
                    @vvectorize $T for p ∈ 1:$P
                        δ[p,coffset] = μₖ[p] - Yₖ[p,c]
                    end
                end
            end
        end
        push!(q.args, δloopquote)
        target_calc_quote = quote
            LinearAlgebra.BLAS.symm!('L', 'L', one($T), Σ, δ, zero($T), Σ⁻¹δ)
            starget = vbroadcast(SVec{$(VectorizationBase.pick_vector_width(T)),$T}, zero($T))
            @vvectorize for i ∈ 1:length(δ)
                starget = vmuladd( Σ⁻¹δ[i], δ[i], starget )
            end
            target = extract_data(starget)
        end
        push!(q.args, target_calc_quote)
    else
        δloopquote = quote
            @inbounds for k ∈ 1:$K
                Yₖ = Y[k]
                μₖ = μ[k]
                for c ∈ 1:ncoltup[k]
                    coffset = c + coloffsettup[k]
                    @vvectorize $T for p ∈ 1:$P
                        δₚ = μₖ[p] - Yₖ[p,c]
                        δ[p,coffset] = δₚ
                        Σ⁻¹δ[p,coffset] = δₚ
                    end
                end
            end
        end
        push!(q.args, δloopquote)
        target_calc_quote = quote
            L = Σ#.data
            LinearAlgebra.LAPACK.potrf!('L', Σ) # really? We're assuming it is constant here...
            LinearAlgebra.LAPACK.potrs!('L', L, Σ⁻¹δ)
            starget = vbroadcast(SVec{$(VectorizationBase.pick_vector_width(T)),$T}, zero($T))
            @vvectorize for i ∈ 1:length(δ)
                starget = vmuladd( Σ⁻¹δ[i], δ[i], starget )
            end
            target = extract_data(starget)
        end
    end
    if track_μ || track_Y
        for k ∈ 1:K
            ∂Yₖ = Symbol(:∂Y_, k)
            ∂μₖ = Symbol(:∂μ_, k)
            push!(q.args, :( PaddedMatrices.negative_sum!($∂μₖ, $∂Yₖ) ))
        end
    end
    if track_Σ
        push!(q.args, quote
              ldfactor = -one($T)*$(Symbol(:coloffset_,K))
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
    simplify_expr(q)
end
@generated function ∂Normal(
    sp::StackPointer,
    Y::NTuple{K},
    μ::NTuple{K},
    Σ::AbstractFixedSizeCovarianceMatrix{R,T,R},
    ::Val{track}
) where {R, K, T, track}
#) where {K, T, KT, track}
    track_Y, track_μ, track_Σ = track

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
    push!(q.args, :(stack_pointer = pointer(sp,$T)))
    if track_Y
        # We are tracking Y, so we cannot drop Σ⁻¹δ, because this will be returned as ∂Yₖ
        push!(q.args, :(Σ⁻¹δ = DynamicPtrMatrix(stack_pointer, ($R,cols), $R)))
        push!(q.args, :(stack_pointer += $(sizeof(T)*R)*cols))
        push!(q.args, :(δ = DynamicPtrMatrix(stack_pointer, ($R, cols), $R)))
    else
        # because we are not tracking Y, we can drop Σ⁻¹δ, which will contain ∂Y
        # we therefore allocate it on top of δ on the stack.
        push!(q.args, :(δ = DynamicPtrMatrix(stack_pointer, ($R, cols), $R)))
        push!(q.args, :(stack_pointer += $(sizeof(T)*R)*cols))
        push!(q.args, :(Σ⁻¹δ = DynamicPtrMatrix(stack_pointer, ($R,cols), $R)))
    end
    if track_μ && track_Y
        push!(q.args, :(sp = PaddedMatrices.StackPointer(Base.unsafe_convert(Ptr{Cvoid}, stack_pointer + $(K*sizeof(T)*R)) )))
    elseif track_μ
        push!(q.args, :(sp = sp + $(K*sizeof(T)*R) ))
    elseif track_Y
        push!(q.args, :(sp = PaddedMatrices.StackPointer(Base.unsafe_convert(Ptr{Cvoid}, stack_pointer) )))
    end
    push!(q.args, :(sp, ∂Normal!(Σ⁻¹δ, δ, Y, μ, Σ, Val{$track}()) ))
    simplify_expr(q)
end

@generated function ∂Normal(
    sp::StackPointer,
    Y::AbstractMatrix{T},
    μ::AbstractFixedSizePaddedVector{R,T},
    Σ::AbstractFixedSizeCovarianceMatrix{R,T,R},
    ::Val{track}
) where {R, K, T, track}
    track_Y, track_μ, track_Σ = track
    ∂retin = Expr(:tuple,:target)
    ∂retout = Expr(:tuple,:target)
    track_Y && push!(∂retin.args, :∂Y)
    track_μ && push!(∂retin.args, :∂μ)
    track_Σ && push!(∂retin.args, :∂Σ)
    track_Y && push!(∂retout.args, :(∂Y[1]))
    track_μ && push!(∂retout.args, :(∂μ[1]))
    track_Σ && push!(∂retout.args, :∂Σ)
    quote
        $(Expr(:meta,:inline))
        (sp, $∂retin) = ∂Normal(sp, (Y,), (μ,), Σ, Val{$track}())
        @inbounds (sp, $∂retout)
    end
end


function loadδ_quote(
    R::Int, C::Int, K::Union{Symbol,Int}, T::DataType,
    Bstride::Union{Int,Symbol}, Bsym::Symbol,
    μdim::Int, μstride::Union{Int,Symbol},
    μsym::Union{Symbol,Nothing} = :μptr, maskload::Bool = true, μmy::Bool = true, μtransposed::Bool = false
)
    size_T = sizeof(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(R, T)
    V = Vec{W,T}
    Wm1 = W - 1
    Riter = R >> Wshift
    Rrem = R & Wm1
    mask = VectorizationBase.mask_from_remainder(T, Rrem)
    # if K isa Symbol
    q = quote
        BsymK = $Bsym + $(size_T)*$K*$Bstride
    end
    μdim == 2 && push!(q.args, :(μsumK = $μsym + $size_T*$μstride*$K))
    # else
        # q = quote
            # BsymK = $Bsym + $(size_T*K)*$Bstride
        # end
        # μdim == 2 && push!(q.args, :(μsumK = $μsym + $(size_T*μstride*K)))
    # end
#    @show μsym, typeof(μsym), μdim, μtransposed
    if μsym isa Symbol
        if μdim == 1 && !μtransposed
            for r ∈ 0:Riter-1
                push!(q.args, Expr(:(=), Symbol(:vμ_,r), :(SIMDPirates.vload($V, $μsym + $(size_T*W*r)))))
            end
            if Rrem > 0
                if maskload
                    push!(q.args, Expr(:(=), Symbol(:vμ_,r), :(SIMDPirates.vload($V, $μsym + $(size_T*W*Riter),$mask))))
                else
                    push!(q.args, Expr(:(=), Symbol(:vμ_,r), :(SIMDPirates.vload($V, $μsym + $(size_T*W*Riter)))))
                end
            end
        end
        for c ∈ 0:C-1            
            vμ_c = μdim == 0 ? μsym : Symbol(:vμ_, c)
            if μdim == 1
                if μstride isa Symbol
                    if K isa Symbol
                        push!(q.args, Expr(:(=), vμ_c, :(SIMDPirates.vbroadcast($V, $μsym + $size_T*$μstride*($c+$K)))))
                    else
                        push!(q.args, Expr(:(=), vμ_c, :(SIMDPirates.vbroadcast($V, $μsym + $size_T*$(c + K)*$μstride))))
                    end
                else
                    if K isa Symbol
                        push!(q.args, Expr(:(=), vμ_c, :(SIMDPirates.vbroadcast($V, $μsym + $(size_T*μstride)*($c+$K)))))
                    else
                        push!(q.args, Expr(:(=), vμ_c, :(SIMDPirates.vbroadcast($V, $μsym + $(size_T*μstride)*$(c + K)))))
                    end
                end
            end
            if (μdim == 1 && μtransposed) || μdim == 0
                for r ∈ 0:Riter-1
                    yloadexpr = :(SIMDPirates.vload($V, BsymK + $size_T * ($(r*W) + $c*$Bstride)))
                    if μmy
                        push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vsub($vμ_c, $yloadexpr)))
                    else
                        push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vsub($yloadexpr, $vμ_c)))
                    end
                end
                if Rrem > 0
                    # Only need to mask if we're on last column
                    if maskload && c == C-1
                        yloadexpr = :(SIMDPirates.vload($V, BsymK + $size_T * ($(Riter*W) + $c*$Bstride), $mask ))
                    else
                        yloadexpr = :(SIMDPirates.vload($V, BsymK + $size_T * ($(Riter*W) + $c*$Bstride)) )
                    end
                    if μmy
                        push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vsub($vμ_c, $yloadexpr)))
                    else
                        push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vsub($yloadexpr, $vμ_c)))
                    end
                end
            elseif μdim == 1 && !μtransposed
                for r ∈ 0:Riter-1
                    yloadexpr = :(SIMDPirates.vload($V, BsymK + $size_T * ($(r*W) + $c*$Bstride)))
                    vμ_r = Symbol(:vμ_,r)
                    if μmy
                        push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vsub($vμ_r, $yloadexpr)))
                    else
                        push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vsub($yloadexpr, $vμ_r)))
                    end
                end
                if Rrem > 0
                    # Only need to mask if we're on last column
                    vμ_r = Symbol(:vμ_,Riter)
                    if maskload && c == C-1
                        yloadexpr = :(SIMDPirates.vload($V, BsymK + $size_T * ($(Riter*W) + $c*$Bstride), $mask ))
                    else
                        yloadexpr = :(SIMDPirates.vload($V, BsymK + $size_T * ($(Riter*W) + $c*$Bstride) ))
                    end
                    if μmy
                        push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vsub($vμ_r, $yloadexpr)))
                    else
                        push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vsub($yloadexpr, $vμ_r)))
                    end
                end
            elseif μdim == 2
                for r ∈ 0:Riter-1
                    μloadexpr = :(SIMDPirates.vload($V, usymK + $size_T * ($(r*W) + $c*$μstride)))
                    yloadexpr = :(SIMDPirates.vload($V, BsymK + $size_T * ($(r*W) + $c*$Bstride)))
                    if μmy
                        push!(q.args, Expr(:(=), Symbol(:A_,r,:_,c), Expr(:call, :(SIMDPirates.vsub), μloadexpr, yloadexpr)))
                    else
                        push!(q.args, Expr(:(=), Symbol(:A_,r,:_,c), Expr(:call, :(SIMDPirates.vsub), yloadexpr, μloadexpr)))
                    end        
                end
                if Rrem > 0
                    # Only need to mask if we're on last column
                    if maskload && c == C-1
                        μloadexpr = :(SIMDPirates.vload($V, μsymK + $size_T * ($(Riter*W) + $c*$μstride), $mask))
                        yloadexpr = :(SIMDPirates.vload($V, BsymK + $size_T * ($(Riter*W) + $c*$Bstride), $mask))
                    else
                        μloadexpr = :(SIMDPirates.vload($V, μsymK + $size_T * ($(Riter*W) + $c*$μstride)) )
                        yloadexpr = :(SIMDPirates.vload($V, BsymK + $size_T * ($(Riter*W) + $c*$Bstride)) )
                    end
                    if μmy
                        push!(q.args, Expr(:(=), Symbol(:A_,Riter,:_,c)), Expr(:call,:(SIMDPirates.vsub), μloadexpr, yloadexpr))
                    else
                        push!(q.args, Expr(:(=), Symbol(:A_,Riter,:_,c)), Expr(:call,:(SIMDPirates.vsub), yloadexpr, μloadexpr))
                    end
                end
            else #μ assumed not to exist
                for r ∈ 0:Riter-1
                    push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vload($V, BsymK + $size_T * ($(r*W) + $c*$Bstride) ) ))
                end
                if Rrem > 0
                    # Only need to mask if we're on last column
                    if maskload && c == C-1
                        push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vload($V, BsymK + $size_T * ($(Riter*W) + $c*$Bstride), $mask ) ))
                    else
                        push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vload($V, BsymK + $size_T * ($(Riter*W) + $c*$Bstride)) ) )
                    end
                end
            end
        end
    else
        for c ∈ 0:C-1
            for r ∈ 0:Riter-1
                push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vload($V, BsymK + $size_T * ($(r*W) + $c*$Bstride)) ) )
            end
            if Rrem > 0
                # Only need to mask if we're on last column
                if maskload && c == C-1
                    push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vload($V, BsymK + $size_T * ($(Riter*W) + $c*$Bstride), $mask ) ))
                else
                    push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vload($V, BsymK + $size_T * ($(Riter*W) + $c*$Bstride)) ) )
                end
            end
        end
    end
    q
end
function loadδ_quote(
    R::Symbol, C::Int, K::Union{Symbol,Int}, T::DataType,
    Bstride::Symbol, Bsym::Symbol,
    μdim::Int, μstride::Union{Int,Symbol},
    μsym::Union{Symbol,Nothing} = :μptr, maskload::Bool = true,
    μmy::Bool = true, μtransposed::Bool = false, masksym = :__mask__
)
    size_T = sizeof(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    V = Vec{W,T}
    Wm1 = W - 1
    q = quote
            BsymK = $Bsym + $size_T*$Bstride*$K
        end
    μdim == 2 && push!(q.args, :(μsumK = $μsym + $size_T*$μstride*$K))
    if μsym isa Symbol
        if μdim == 1 && !μtransposed
            push!(q.args, Expr(:(=), :vμ_0, :(SIMDPirates.vload($V, $μsym,$masksym))))
        end
        for c ∈ 0:C-1            
            vμ_c = μdim == 0 ? μsym : Symbol(:vμ_, c)
            if μdim == 1
                push!(q.args, Expr(:(=), vμ_c, :(SIMDPirates.vbroadcast($V, $μsym + $size_T*$μstride*($c+$K)))))
            end
            if (μdim == 1 && μtransposed) || μdim == 0
                # Only need to mask if we're on last column
                yloadexpr = :(SIMDPirates.vload($V, BsymK + $size_T * $c*$Bstride, $masksym ))
                if μmy
                    push!(q.args, :($(Symbol(:A_0_,c)) = SIMDPirates.vsub($vμ_c, $yloadexpr)))
                else
                    push!(q.args, :($(Symbol(:A_0_,c)) = SIMDPirates.vsub($yloadexpr, $vμ_c)))
                end
            elseif μdim == 1 && !μtransposed
                # Only need to mask if we're on last column
                yloadexpr = :(SIMDPirates.vload($V, BsymK + $size_T * $c*$Bstride, $masksym ))
                if μmy
                    push!(q.args, :($(Symbol(:A_0_,c)) = SIMDPirates.vsub(vμ_0, $yloadexpr)))
                else
                    push!(q.args, :($(Symbol(:A_0_,c)) = SIMDPirates.vsub($yloadexpr, vμ_0)))
                end
            elseif μdim == 2
                # Only need to mask if we're on last column
                μloadexpr = :(SIMDPirates.vload($V, μsymK + $size_T * $c*$μstride, $masksym))
                yloadexpr = :(SIMDPirates.vload($V, BsymK + $size_T * $c*$Bstride, $masksym))
                if μmy
                    push!(q.args, Expr(:(=), Symbol(:A_0_,c)), Expr(:call,:(SIMDPirates.vsub), μloadexpr, yloadexpr))
                else
                    push!(q.args, Expr(:(=), Symbol(:A_0_,c)), Expr(:call,:(SIMDPirates.vsub), yloadexpr, μloadexpr))
                end
            else #μ assumed not to exist
                    # Only need to mask if we're on last column
                push!(q.args, :($(Symbol(:A_0_,c)) = SIMDPirates.vload($V, BsymK + $size_T * $c*$Bstride, $masksym ) ))
            end
        end
    else
        for c ∈ 0:C-1
            push!(q.args, :($(Symbol(:A_0_,c)) = SIMDPirates.vload($V, BsymK + $size_T * $c*$Bstride, $masksym ) ))
        end
    end
    q
end
function loadδfnmadd_quote(
    R::Int, C::Int, K::Union{Int,Symbol}, T::DataType,
    Ystride::Union{Int,Symbol}, Xstride::Union{Int,Symbol}, βstride::Int, βdim::Int,
    ysym::Symbol = :ptrY, xsym::Symbol = :ptrX, βsym::Symbol = :ptrβ, μsym::Symbol = :ptrμ,
    maskload::Bool = true, μmy::Bool = true, XP::Int = -1,
    μstride::Union{Int,Symbol} = -1, μdim::Int = -1, μtransposed::Bool = false
)
    size_T = sizeof(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(R, T)
    V = Vec{W,T}
    Wm1 = W - 1
    Riter = R >> Wshift
    Rrem = R & Wm1
    Riterl = Rrem > 0 ? Riter : Riter - 1
    maskload = maskload & (Rrem > 0)
    mask = VectorizationBase.mask_from_remainder(T, Rrem)
    # if K isa Symbol
    q = quote
        YsymK = $ysym + $size_T*$Ystride*$K
    end
    βdim == 2 && push!(q.args, :(βsymK = $βsym + $size_T*$βstride*$K))
    μdim == 2 && push!(q.args, :(μsymK = $μsym + $size_T*$μstride*$K))
    # else
        # q = quote
            # YsymK = $ysym + $(size_T*Ystride*K)
        # end
        # βdim == 2 && push!(q.args, :(βsymK = $βsym + $(size_T*βstride*K)))
        # μdim == 2 && push!(q.args, :(μsymK = $μsym + $(size_T*μstride*K)))
    # end
    # if μdim == 1, we assume that vμ_r has been computed for r = 0,...,Riter
    peel_first_iter = μmy & (βdim == 2) & (μstride == -1)
    # we will peel the first iteration if these are all true, because
    # if μmy then we want to make y negative, otherwise we don't
    # given that we want to make y negative, if...
    # μstride == -1, then we do have an intercept term which we can use for (y - μ)
    # if βdim == 1, then we can reorder the single subtraction
    if peel_first_iter
        for r ∈ 0:Riterl
            if r == Riterl && maskload
                xloadexpr = :(SIMDPirates.vload($V, $xsym + ($size_T * $(r*W)), $mask))
            else
                xloadexpr = :(SIMDPirates.vload($V, $xsym + ($size_T * $(r*W))))
            end
            push!(q.args, Expr(:(=), Symbol(:vx_,r), xloadexpr))
        end
    end
    f = μmy ? :(SIMDPirates.vfmsub) : :(SIMDPirates.vfnmadd)
    if μstride != -1 && μdim == 1
        if μtransposed
            for c ∈ 0:C-1
                push!(q.args, Expr(:(=), Symbol(:vμbase_,c), :(SIMDPirates.vbroadcast($V, $μsym + $size_T*$μstride*($K+$c) ))))
            end
        else
            for r ∈ 0:Riterl
                if r == Riterl && maskload
                    push!(q.args, Expr(:(=), Symbol(:vμbase_,r), :(SIMDPirates.vload($V, $μsym + $size_T * $(r*W),$mask))))
                else
                    push!(q.args, Expr(:(=), Symbol(:vμbase_,r), :(SIMDPirates.vload($V, $μsym + $size_T * $(r*W)))))
                end
            end
        end
    end
    for c ∈ 0:C-1
        if βdim == 1
            for r ∈ 0:Riterl
                vμ_r = Symbol(:vμ_,r)
                if r == Riterl && maskload && c == C - 1
                    yloadexpr = :(SIMDPirates.vload($V, YsymK + $size_T * ($(r*W) + $c*$Ystride),$mask))
                else
                    yloadexpr = :(SIMDPirates.vload($V, YsymK + $size_T * ($(r*W) + $c*$Ystride)))
                end
                if μstride != -1
                    if μdim == 1
                        yloadexpr = :(SIMDPirates.vsub($yloadexpr,$(Symbol(:vμbase_, μtransposed ? c : r))))
                    else#if μdim == 2
                        if r == Riterl && maskload && c == C - 1
                            αloadexpr = :(SIMDPirates.vload($V, $μsym + $size_T * ($(r*W) + $c*$μstride),$mask))
                        else
                            αloadexpr = :(SIMDPirates.vload($V, $μsym + $size_T * ($(r*W) + $c*$μstride)))
                        end
                        yloadexpr = :(SIMDPirates.vsub($yloadexpr,$αloadexpr))
                    end
                end
                if μmy
                    push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vsub($vμ_r, $yloadexpr)))
                else
                    push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vsub($yloadexpr, $vμ_r)))
                end
                # push!(q.args, :(@show getfield.($(Symbol(:A_,r,:_,c)), :value)))
            end
        elseif βdim == 2
            # we load first block, before the XP loop
            # if  μmy, that is X*β - Y # aka SIMDPirates.vfmsub  # loop vmuladd to this answer
            # if !μmy, that is Y - X*β # aka SIMDPirates.vfnmadd # loop vfnmadd to this answer
            if peel_first_iter
                β_c = Symbol(:β_,c)
                push!(q.args, Expr(:(=), β_c, :(SIMDPirates.vbroadcast($V, βsymK + ($size_T * $(c*βstride))))))
            end
            for r ∈ 0:Riterl
                if r == Riterl && maskload && c == C-1
                    yloadexpr = :(SIMDPirates.vload($V, YsymK + $size_T * ($(r*W) + $c*$Ystride),$mask))
                else
                    yloadexpr = :(SIMDPirates.vload($V, YsymK + $size_T * ($(r*W) + $c*$Ystride)))
                end
                # What is happening here is that we want to make y negative
                if peel_first_iter
                    yloadexpr = Expr(:call, f, Symbol(:vx_,r), β_c, yloadexpr)
                elseif μstride != -1
                    if μdim == 1
                        if μmy
                            yloadexpr = :(SIMDPirates.vsub($(Symbol(:vμbase_, μtransposed ? c : r)),$yloadexpr))
                        else
                            yloadexpr = :(SIMDPirates.vsub($yloadexpr,$(Symbol(:vμbase_, μtransposed ? c : r))))
                        end
                    else#if αdim == 2
                        if r == Riterl && maskload && c == C - 1
                            μloadexpr = :(SIMDPirates.vload($V, μsymK + $size_T * ($(r*W) + $c*$μstride),$mask))
                        else
                            μloadexpr = :(SIMDPirates.vload($V, μsymK + $size_T * ($(r*W) + $c*$μstride)))
                        end
                        yloadexpr = if μmy
                            :(SIMDPirates.$vsub($μloadexpr,$yloadexpr))
                        else
                            :(SIMDPirates.$vsub($yloadexpr,$μloadexpr))
                        end
                    end
                end
                # push!(q.args, Expr(:(=), Symbol(:A_,r,:_,c), Expr(:call, f, Symbol(:vx_,r), β_c, yloadexpr)))
                push!(q.args, Expr(:(=), Symbol(:A_,r,:_,c), yloadexpr))
            end
        end
    end
    f = μmy ? :(SIMDPirates.vmuladd) : :(SIMDPirates.vfnmadd)
    if βdim == 2
        p = gensym(:p)
        loopbody = quote end
        for r ∈ 0:Riterl
            if r == Riterl && maskload
                xloadexpr = :(SIMDPirates.vload($V, $xsym + $size_T * ($(r*W) + $p*$Xstride),$mask))
            else
                xloadexpr = :(SIMDPirates.vload($V, $xsym + $size_T * ($(r*W) + $p*$Xstride)))
            end
            push!(loopbody.args, Expr(:(=), Symbol(:vx_,r), xloadexpr))
        end
        for c ∈ 0:C-1
            β_c = Symbol(:β_,c)
            push!(loopbody.args, Expr(:(=), β_c, :(SIMDPirates.vbroadcast($V, βsymK + $size_T * ($(c*βstride)+$p)))))
            for r ∈ 0:Riterl
                push!(loopbody.args, Expr(:(=), Symbol(:A_,r,:_,c), Expr(:call, f, Symbol(:vx_,r), β_c, Symbol(:A_,r,:_,c))))
            end
        end
        loop = quote
            for $p ∈ $(peel_first_iter ? 1 : 0):$(XP-1)
                $loopbody
            end
        end
        push!(q.args, loop)
    end
    q
end
function loadδfnmadd_quote(
    R::Symbol, C::Int, K::Union{Symbol,Int}, T::DataType, Ystride::Symbol, Xstride::Symbol, βstride::Int, βdim::Int,
    ysym::Symbol = :ptrY, xsym::Symbol = :ptrX, βsym::Symbol = :ptrβ, μsym::Symbol = :ptrμ,
    maskload::Bool = true, μmy::Bool = true, XP::Int = -1,
    μstride::Union{Int,Symbol} = -1, μdim::Int = -1, μtransposed::Bool = false, masksym = :__mask__
)
    size_T = sizeof(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    V = Vec{W,T}
    Wm1 = W - 1
    # if K isa Symbol
    q = quote
        YsymK = $ysym + $size_T*$Ystride*$K
    end
    βdim == 2 && push!(q.args, :(βsymK = $βsym + $size_T*$βstride*$K))
    μdim == 2 && push!(q.args, :(μsymK = $μsym + $size_T*$μstride*$K))
    # else
        # q = quote
            # YsymK = $ysym + $(size_T*Ystride*K)
        # end
        # βdim == 2 && push!(q.args, :(βsymK = $βsym + $(size_T*βstride*K)))
        # μdim == 2 && push!(q.args, :(μsymK = $μsym + $(size_T*μstride*K)))
    # end
    # if μdim == 1, we assume that vμ_r has been computed for r = 0,...,Riter
    peel_first_iter = μmy & (βdim == 2) & (μstride == -1)
    # we will peel the first iteration if these are all true, because
    # if μmy then we want to make y negative, otherwise we don't
    # given that we want to make y negative, if...
    # μstride == -1, then we do have an intercept term which we can use for (y - μ)
    # if βdim == 1, then we can reorder the single subtraction
    if peel_first_iter
        xloadexpr = :(SIMDPirates.vload($V, $xsym, $masksym))
        push!(q.args, :(vx_0 = $xloadexpr))
    end
    f = μmy ? :(SIMDPirates.vfmsub) : :(SIMDPirates.vfnmadd)
    if μstride != -1 && μdim == 1
        if μtransposed
            for c ∈ 0:C-1
                push!(q.args, Expr(:(=), Symbol(:vμbase_,c), :(SIMDPirates.vbroadcast($V, $μsym + $size_T*$μstride*($K+$c) ))))
            end
        else
            push!(q.args, :(vμbase_0 = SIMDPirates.vload($V, $μsym, $masksym)))
        end
    end
    for c ∈ 0:C-1
        if βdim == 1
            yloadexpr = :(SIMDPirates.vload($V, YsymK + $size_T * $c*$Ystride, $masksym))
            if μstride != -1
                if μdim == 1
                    yloadexpr = :(SIMDPirates.vsub($yloadexpr,$(Symbol(:vμbase_, μtransposed ? c : 0))))
                else#if μdim == 2
                    αloadexpr = :(SIMDPirates.vload($V, $μsym + $size_T * $c*$μstride, $masksym))
                    yloadexpr = :(SIMDPirates.vsub($yloadexpr, $αloadexpr))
                end
            end
            if μmy
                push!(q.args, :($(Symbol(:A_0_,c)) = SIMDPirates.vsub(vμ_0, $yloadexpr)))
            else
                push!(q.args, :($(Symbol(:A_0_,c)) = SIMDPirates.vsub($yloadexpr, vμ_0)))
            end
        elseif βdim == 2
            # we load first block, before the XP loop
            # if  μmy, that is X*β - Y # aka SIMDPirates.vfmsub  # loop vmuladd to this answer
            # if !μmy, that is Y - X*β # aka SIMDPirates.vfnmadd # loop vfnmadd to this answer
            if peel_first_iter
                β_c = Symbol(:β_,c)
                push!(q.args, Expr(:(=), β_c, :(SIMDPirates.vbroadcast($V, βsymK + $size_T * $c*$βstride))))
            end
            yloadexpr = :(SIMDPirates.vload($V, YsymK + $size_T * $c*$Ystride, $masksym))
            # What is happening here is that we want to make y negative
            if peel_first_iter
                yloadexpr = Expr(:call, f, :vx_0, β_c, yloadexpr)
            elseif μstride != -1
                if μdim == 1
                    if μmy
                        yloadexpr = :(SIMDPirates.vsub($(Symbol(:vμbase_, μtransposed ? c : 0)),$yloadexpr))
                    else
                        yloadexpr = :(SIMDPirates.vsub($yloadexpr,$(Symbol(:vμbase_, μtransposed ? c : 0))))
                    end
                else#if αdim == 2
                    μloadexpr = :(SIMDPirates.vload($V, μsymK + $size_T * $c*$μstride, $masksym))
                    yloadexpr = if μmy
                        :(SIMDPirates.$vsub($μloadexpr,$yloadexpr))
                    else
                        :(SIMDPirates.$vsub($yloadexpr,$μloadexpr))
                    end
                end
            end
            push!(q.args, Expr(:(=), Symbol(:A_0_,c), yloadexpr))
        end
    end
    f = μmy ? :(SIMDPirates.vmuladd) : :(SIMDPirates.vfnmadd)
    if βdim == 2
        p = gensym(:p)
        loopbody = quote end
        xloadexpr = :(SIMDPirates.vload($V, $xsym + $size_T * $p*$Xstride, $masksym))
        push!(loopbody.args, Expr(:(=), :vx_0, xloadexpr))
        for c ∈ 0:C-1
            β_c = Symbol(:β_,c)
            push!(loopbody.args, Expr(:(=), β_c, :(SIMDPirates.vbroadcast($V, βsymK + $size_T * ($(c*βstride)+$p)))))
            push!(loopbody.args, Expr(:(=), Symbol(:A_0_,c), Expr(:call, f, :vx_0, β_c, Symbol(:A_0_,c))))
        end
        loop = quote
            for $p ∈ $(peel_first_iter ? 1 : 0):$(XP-1)
                $loopbody
            end
        end
        push!(q.args, loop)
    end
    q
end
function Xβ_load_quote(
    R::Int, T::DataType, Xstride::Union{Int,Symbol}, βstride::Int, μmy::Bool = true, XP::Int = -1, 
    xsym::Symbol = :ptrX, βsym::Symbol = :ptrβ, maskload::Bool = true
)
    size_T = sizeof(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(R, T)
    V = Vec{W,T}
    Wm1 = W - 1
    Riter = R >> Wshift
    Rrem = R & Wm1
    Riterl = Rrem > 0 ? Riter : Riter - 1
    maskload = maskload & (Rrem > 0)
    mask = VectorizationBase.mask_from_remainder(T, Rrem)
    q = quote end
    # Initial load
    push!(q.args, Expr(:(=), :vβ, :(SIMDPirates.vbroadcast($V, $βsym))))
    for r ∈ 0:Riterl
        if r == Riterl && maskload && XP == 1
            xloadexpr = :(SIMDPirates.vload($V, $xsym + $size_T * $(r*W),$mask))
        else
            xloadexpr = :(SIMDPirates.vload($V, $xsym + $size_T * $(r*W)))
        end
        push!(q.args, Expr(:(=), Symbol(:vμ_,r), Expr(:call, :(SIMDPirates.vmul), xloadexpr, :vβ)))
    end
    p = gensym(:p)
    # update through loop
    loopbody = quote
        vβ = SIMDPirates.vbroadcast($V, $βsym + $size_T*$p)
        # vβ = SIMDPirates.vbroadcast($V, $βsym + $(size_T*βstride)*$p)
    end
    for r ∈ 0:Riterl
        if r == Riterl && maskload
            xloadexpr = :(SIMDPirates.vload($V, $xsym + $size_T * ($(r*W) + $p*$Xstride),$mask))
        else
            xloadexpr = :(SIMDPirates.vload($V, $xsym + $size_T * ($(r*W) + $p*$Xstride)))
        end
        push!(loopbody.args, Expr(:(=), Symbol(:vμ_,r), Expr(:call, :(SIMDPirates.vmuladd), xloadexpr, :vβ, Symbol(:vμ_,r))))
    end
    loop = quote
        for $p ∈ 1:$(XP-1)
            $loopbody
        end
    end
    push!(q.args, loop)
    # for r ∈ 0:Riterl
    #     push!(q.args, :(@show $(Symbol(:vμ_,r))))
    # end
    q
end
# The symbol version assumes R < W (the SIMD width)
function Xβ_load_quote(
    R::Symbol, T::DataType, Xstride::Symbol, βstride::Int, μmy::Bool = true, XP::Int = -1, 
    xsym::Symbol = :ptrX, βsym::Symbol = :ptrβ, maskload::Bool = true, masksym = :__mask__
)
    size_T = sizeof(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    V = Vec{W,T}
    Wm1 = W - 1
    q = quote end
    # Initial load
    push!(q.args, Expr(:(=), :vβ, :(SIMDPirates.vbroadcast($V, $βsym))))
    xloadexpr = :(SIMDPirates.vload($V, $xsym, $masksym))
    push!(q.args, Expr(:(=), :vμ_0, Expr(:call, :(SIMDPirates.vmul), xloadexpr, :vβ)))
    p = gensym(:p)
    # update through loop
    loopbody = quote
        vβ = SIMDPirates.vbroadcast($V, $βsym + $size_T*$p)
    end
    xloadexpr = :(SIMDPirates.vload($V, $xsym + $size_T * $p*$Xstride, $masksym))
    push!(loopbody.args, Expr(:(=), :vμ_0, :(SIMDPirates.vmuladd($xloadexpr, vβ, vμ_0))))
    loop = quote
        for $p ∈ 1:$(XP-1)
            $loopbody
        end
    end
    push!(q.args, loop)
    q
end

function mutlivariate_normal_SMLT_rowiter(
    Mk::Union{Int,Symbol}, Nk::Int, col_rem::Int, T::DataType, Ystride::Union{Int,Symbol}, n_col_reps::Int, μdim::Int = -1, μstride::Union{Int,Symbol} = -1,
    μsym::Symbol = :ptrμ, XP::Int = -1, βstride::Int = -1, Xstride::Union{Int,Symbol} = -1, βdim::Int = -1, μtransposed::Bool = false
)
    #TODO: NOTE, WE DO NEED TO STORE THE SOLUTION MATRIX (at least 1 row set amd up to the last column block)
    # because this is used for calculating the next iteration.
    N = Nk * n_col_reps + col_rem
    size_T = sizeof(T)
    row_iter = (βdim == 1 && XP > 0) ? Xβ_load_quote(Mk, T, Xstride, βstride, false, XP, :ptrX, :ptrβ) : quote end
    if col_rem > 0
        if XP > 0
            loadδ_expr = loadδfnmadd_quote(
                Mk, col_rem, 0, T, Ystride, Xstride, βstride, βdim,
                :ptrY, :ptrX, :ptrβ, :ptrμ, true, false, XP, μstride, μdim, μtransposed
            )
        else
            loadδ_expr = loadδ_quote(Mk, col_rem, 0, T, Ystride, :ptrY, μdim, μstride, μsym, true, true, μtransposed)
        end
        iter_quote = StructuredMatrices.A_rdiv_U_kernel_quote(
            Mk, col_rem, 0, T, Mk, Ystride, N, true, true, storeA = n_col_reps > 0, loadB = false, reduce_sym = :δ²
        )
        #pushfirst!(row_iter.args, :(ptrUtri = ptrUtribase))
        push!(row_iter.args, loadδ_expr)
        push!(row_iter.args, iter_quote)
        push!(row_iter.args, :(ptrUdiag += $(col_rem*size_T)))
        base_K = col_rem
        KmZ = false
    else
        base_K = 0
        KmZ = true
    end
    if n_col_reps > 1
        if XP > 0
            loadδ_expr = loadδfnmadd_quote(
                Mk, Nk, :K, T, Ystride, Xstride, βstride, βdim,
                :ptrY, :ptrX, :ptrβ, :ptrμ, true, false, XP, μstride, μdim, μtransposed
            )
        else
            loadδ_expr = loadδ_quote(Mk, Nk, :K, T, Ystride, :ptrY, μdim, μstride, μsym, true, true, μtransposed)
        end
        iterquote = StructuredMatrices.A_rdiv_U_kernel_quote(
            Mk, Nk, :K, T, Mk, Ystride, N, true, true, storeA = true, loadB = false, reduce_sym = :δ²
        )
        row_iter_loop = quote
            K = $base_K
            for crep ∈ 0:$(n_col_reps-1)
                ptrUtri = ptrUtribase + K*$size_T
                $loadδ_expr
                $iterquote
                ptrUdiag += $(size_T*Nk)
                K += $Nk
            end
        end
        push!(row_iter.args, row_iter_loop)
    elseif n_col_reps == 1
        if XP > 0
            loadδ_expr = loadδfnmadd_quote(
                Mk, Nk, col_rem, T, Ystride, Xstride, βstride, βdim,
                :ptrY, :ptrX, :ptrβ, :ptrμ, true, false, XP, μstride, μdim, μtransposed
            )
        else
            loadδ_expr = loadδ_quote(Mk, Nk, col_rem, T, Ystride, :ptrY, μdim, μstride, μsym, true, true, μtransposed)
        end
        push!(row_iter.args, loadδ_expr)
        row_iter_single = StructuredMatrices.A_rdiv_U_kernel_quote(
            Mk, Nk, col_rem, T, Mk, Ystride, N, true, true, storeA = false, loadB = false, reduce_sym = :δ²
        )
        push!(row_iter.args, row_iter_single)
    end
    row_iter
end

## StructuredMatrices.jl Lower Triangular (SMLT) quote
## M is the sample size
function multivariate_normal_SMLT_quote(
    M::Union{Symbol,Int}, P::Int, track::NTuple{D,Bool}, T::DataType = Float64;
    Ystride::Union{Symbol,Int} = M, βstride::Int = -1, Xstride::Union{Symbol,Int} = -1, βdim::Int = -1,
    μdim::Int = -1, μstride::Union{Int,Symbol}= -1, sp::Bool = false, XP::Int = -1, μtransposed::Bool = false
) where {D}
    if D == 5
        track_Y, track_X, track_β, track_μ, track_L = track
    else
        if D == 4
            track_Y, track_X, track_β, track_L = track
            track_μ = false
        else
            track_X = track_β = false
            if D == 3
                track_Y, track_μ, track_L = track
            elseif D == 2
                track_Y, track_L = track
                track_μ = false
            else
                throw("Unknown number of arguments ($D) to normal.")
            end
        end
    end
    q = quote end
    maxM = M isa Symbol ? typemax(Int) : M
    
    W, Mk, Nk = StructuredMatrices.div_triangle_blocking_structure(maxM, P, T)
    #@show Mk, Nk
    V = Vec{W,T}
    Wm1 = W - 1
    n_col_reps, col_rem = divrem(P, Nk)
    total_col_iterations = n_col_reps + (col_rem > 0)
    #Nl = ( N + W - 1 ) & ~Wm1
    size_T = sizeof(T)
    loopbody = quote
        Bₚ = L[p]
        invdiag[p] = one($T) / Bₚ
    end
    track_L && push!(loopbody.args, :(δ²_0 = LoopVectorization.SIMDPirates.vadd(δ²_0, SLEEFPirates.log(Bₚ))))
    Mk2 = min(4, M isa Symbol ? cld(Mk,W) : cld(min(Mk,M),W) )
    q = quote
        $(Expr(:meta,:inline)) # because of allignment bug
#        B = Badj.parent
        $([Expr(:(=), Symbol(:δ²_,m), :(SIMDPirates.vbroadcast($V, zero($T)))) for m ∈ 0:Mk2-1]...)
        invdiag = $(sp ? :(PtrVector{$P,$T,$P,true}(pointer(sptr,$T))) : :(MutableFixedSizePaddedVector{$P,$T,$P,$P}(undef)))
        $(macroexpand(LoopVectorization, quote
                      @vvectorize $T for p ∈ 1:$P
                      $loopbody
                      end
                      end))
        ptrY = pointer(Y)
        ptrUtribase = pointer(L) + $(P*size_T)
    end
    track_L && push!(q.args, :(δ²_0 = SIMDPirates.vmul(δ²_0, SIMDPirates.vbroadcast($V,$(M isa Integer ? T(2M) : :($(T(2))*$T($M)))))))
    D >= 4 && push!(q.args, :(ptrX = pointer(X); ptrβ = pointer(β)))
    Aquote = quote
        A = $(sp ? :(PtrMatrix{$Mk,$P,$T,$Mk}(pointer(sptr,$T) + $(VectorizationBase.align(size_T*P)))) : :(MutableFixedSizePaddedMatrix{$Mk,$P,$T,$Mk}(undef)))
        ptrA = pointer(A)
    end
    total_col_iterations > 1 && push!(q.args, Aquote)
    if μdim == 0
        push!(q.args, Expr(:(=), :ptrμ, :(SIMDPirates.vbroadcast($V, μ))))
    elseif μdim > 0
        push!(q.args, Expr(:(=), :ptrμ, μtransposed ? :(pointer(μ.parent)) : :(pointer(μ))))
    end
    loop_increments = quote ptrY += $(size_T*Mk) end
    XP > 0 && push!(loop_increments.args, :( ptrX += $(size_T*Mk) ))
    uniqueμbyrow = μdim == 2 || (μdim == 1 && !μtransposed)
    uniqueμbyrow && push!(loop_increments.args, :( ptrμ += $(size_T*Mk) ))
    if M isa Integer
        n_row_reps, row_rem = divrem(M, Mk)
        total_row_iterations = n_row_reps + (row_rem > 0)
        Mk1 = n_row_reps == 0 ? row_rem : Mk
        row_iter = mutlivariate_normal_SMLT_rowiter(
            Mk1, Nk, col_rem, T, Ystride, n_col_reps, μdim, μstride, :ptrμ, XP, βstride, Xstride, βdim, μtransposed
        )
        if n_row_reps > 1
            row_loops = quote
                for rrep ∈ 1:$n_row_reps
                    ptrUdiag = pointer(invdiag); ptrUtri = ptrUtribase
                    $row_iter
                    $loop_increments
                end
            end
            push!(q.args, row_loops)
        else
            push!(q.args, :(ptrUdiag = pointer(invdiag); ptrUtri = ptrUtribase))
            push!(q.args, row_iter)
        end
        if row_rem > 0 && n_row_reps > 0
            push!(q.args, :(ptrUdiag = pointer(invdiag); ptrUtri = ptrUtribase))
            push!(q.args, mutlivariate_normal_SMLT_rowiter( row_rem, Nk, col_rem, T, Ystride, n_col_reps, μdim, μstride, :ptrμ, XP, βstride, Xstride, βdim, μtransposed ))
        end
    else # Unknown number of iterations.
        row_iter = mutlivariate_normal_SMLT_rowiter(
            Mk, Nk, col_rem, T, Ystride, n_col_reps, μdim, μstride, :ptrμ, XP, βstride, Xstride, βdim, μtransposed
        )
        Wrem, Mkrem, Nkrem = StructuredMatrices.div_triangle_blocking_structure(W, P, T)
        n_col_repsrem, col_remrem = divrem(P, Nkrem)
        row_iter_onevec = mutlivariate_normal_SMLT_rowiter(
            W, Nkrem, col_remrem, T, Ystride, n_col_repsrem, μdim, μstride, :ptrμ, XP, βstride, Xstride, βdim, μtransposed
        )
        row_iter_onevecmask = mutlivariate_normal_SMLT_rowiter(
            :row_rem_final, Nkrem, col_remrem, T, Ystride, n_col_repsrem, μdim, μstride, :ptrμ, XP, βstride, Xstride, βdim, μtransposed
        )
        loop_increments_onevec = quote ptrY += $(size_T*W) end 
        XP > 0 && push!(loop_increments_onevec.args, :(ptrX += $(size_T*W)))
        uniqueμbyrow && push!(loop_increments_onevec.args, :(ptrμ += $(size_T*W)))
        row_loops = quote
            Mkrep, Mkrem = divrem($M, $Mk)
            for rrep ∈ 1:Mkrep
                ptrUdiag = pointer(invdiag); ptrUtri = ptrUtribase
                $row_iter
                $loop_increments
            end
            for rrep ∈ 1:Mkrem >> $(VectorizationBase.intlog2(W))
                ptrUdiag = pointer(invdiag); ptrUtri = ptrUtribase
                $row_iter_onevec
                $loop_increments_onevec
            end
            ptrUdiag = pointer(invdiag); ptrUtri = ptrUtribase
            row_rem_final = Mkrem & $Wm1
            __mask__ = VectorizationBase.mask($T, row_rem_final)
            $row_iter_onevecmask
        end
        push!(q.args, row_loops)
    end
    # Reduce the Mk δ² into a single vector.
    R = Mk2
    while R > 1
        Risodd = isodd(R)
        Rh = R >> 1
        for r ∈ 0:(Rh-1)
            dl = Symbol(:δ²_,r)
            dh = Symbol(:δ²_,r+Rh)
            push!(q.args, :($dl = SIMDPirates.vadd($dl,$dh)))
        end
        Risodd && push!(q.args, Expr(:(=), :δ²_0, :(SIMDPirates.vadd(δ²_0, $(Symbol(:δ²_,R-1))))))
        R = Rh
    end
    push!(q.args, Expr(:(=), :δ²_0, :(SIMDPirates.vmul(SIMDPirates.vbroadcast($V, $(T(-0.5))), δ²_0))))
    sp ? push!(q.args, :((sptr,δ²_0))) : push!(q.args, :δ²_0)
    simplify_expr(q)
    # q
end


@generated function Normal(
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,PY},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true)}()
) where {M,P,T,track,PY}
    multivariate_normal_SMLT_quote(M, P, track, T, Ystride = PY, sp = false)
end
@generated function Normal(
    sptr::StackPointer,
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,PY},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true)}()
) where {M,P,T,track,PY}
    multivariate_normal_SMLT_quote(M, P, track, T, Ystride = PY, sp = true)
end


@generated function Normal(
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,PY},
    μ::Tμ,
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true)}()
) where {M,P,T,PY,Tμ,track}
# ) where {M,P,T,track,PY,Tμ}
    if Tμ === T
        multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, μdim = 0, μstride = 0)
    elseif Tμ <: LinearAlgebra.Adjoint
        multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, μdim = 1, μstride = 1, μtransposed = true)
    elseif Tμ <: AbstractFixedSizePaddedVector
        multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, μdim = 1, μstride = 1)
    elseif Tμ <: AbstractFixedSizePaddedMatrix
        multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, μdim = 2, μstride = Tμ.parameters[4])
    else
        throw("Type of μ == $A is not recognized.")
    end
end
@generated function Normal(
    sptr::StackPointer,
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,PY},
    μ::Tμ,
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true)}()
) where {M,P,T,track,PY,Tμ}
    if Tμ === T
        multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, μdim = 0, μstride = 0)
    elseif Tμ <: LinearAlgebra.Adjoint{T,<:PaddedMatrices.AbstractMutableFixedSizePaddedVector}
        multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, μdim = 1, μstride = 1, μtransposed = true)
    elseif Tμ <: AbstractFixedSizePaddedVector
        multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, μdim = 1, μstride = 1)
    elseif Tμ <: AbstractFixedSizePaddedMatrix
        multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, μdim = 2, μstride = Tμ.parameters[4])
    else
        throw("Type of μ == $A is not recognized.")
    end
end

@generated function Normal(
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,PY},
    X::AbstractMutableFixedSizePaddedMatrix{M,K_,T,PX},
    β::AbstractMutableFixedSizePaddedArray{Sβ,T,Nβ,Pβ},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true,true)}()
# ) where {M,P,T,track,PY,PX,PK,K_,Sβ,Nβ,Pβ}
) where {M,P,T,track,PY,PX,PK,Sβ,Nβ,Pβ,K_}
    @assert Sβ.parameters[1] == K_
    multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_)
end
@generated function Normal(
    sptr::StackPointer,
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,PY},
    X::AbstractMutableFixedSizePaddedMatrix{M,K_,T,PX},
    β::AbstractMutableFixedSizePaddedArray{Sβ,T,Nβ,Pβ},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true,true)}()
) where {M,P,T,track,PY,PX,PK,K_,Sβ,Nβ,Pβ}
# ) where {M,P,T,track,PY,PX,PK,Sβ,Nβ,Pβ,K_}
    @assert Sβ.parameters[1] == K_
    multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_)
end

    # M::Union{Symbol,Integer}, P::Int, track::NTuple{D,Bool}, T::DataType = Float64;
    # Ystride::Int = M, βstride::Int = -1, Xstride::Int = -1, βdim::Int = -1,
    # μdim::Int, μstride::Int, sp::Bool, XP::Int = -1, μtransposed::Bool = false
@generated function Normal_fmadd(
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,PY},
    X::AbstractMutableFixedSizePaddedMatrix{M,K_,T,PX},
    β::AbstractMutableFixedSizePaddedArray{Sβ,T,Nβ,Pβ},
    μ::Tμ,
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true,true,true)}()
) where {M,P,T,track,PY,PX,K_,Sβ,Nβ,Pβ,Tμ}
# ) where {M,P,T,track,PY,PX,Sβ,Nβ,Pβ,Tμ,K_}
    @assert Sβ.parameters[1] == K_
    if Tμ === T
        multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_, μdim = 0, μstride = 0)
    elseif Tμ <: LinearAlgebra.Adjoint{T,<:PaddedMatrices.AbstractMutableFixedSizePaddedVector}
        multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_, μdim = 1, μstride = 1, μtransposed = true)
    elseif Tμ <: AbstractFixedSizePaddedVector
        multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_, μdim = 1, μstride = 1)
    elseif Tμ <: AbstractFixedSizePaddedMatrix
        multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_, μdim = 2, μstride = Tμ.parameters[4])
    else
        throw("Type of μ == $A is not recognized.")
    end
end
@generated function Normal_fmadd(
    sptr::StackPointer,
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,PY},
    X::AbstractMutableFixedSizePaddedMatrix{M,K_,T,PX},
    β::AbstractMutableFixedSizePaddedArray{Sβ,T,Nβ,Pβ},
    μ::Tμ,
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true,true,true)}()
) where {M,P,T,track,PY,PX,K_,Sβ,Nβ,Pβ,Tμ}
# ) where {M,P,T,track,PY,PX,Sβ,Nβ,Pβ,Tμ,K_}
    @assert Sβ.parameters[1] == K_
    if Tμ === T
        multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_, μdim = 0, μstride = 0)
    elseif Tμ <: LinearAlgebra.Adjoint{T,<:PaddedMatrices.AbstractMutableFixedSizePaddedVector}
        multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_, μdim = 1, μstride = 1, μtransposed = true)
    elseif Tμ <: AbstractFixedSizePaddedVector
        multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_, μdim = 1, μstride = 1)
    elseif Tμ <: AbstractFixedSizePaddedMatrix
        multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_, μdim = 2, μstride = Tμ.parameters[4])
    else
        throw("Type of μ == $A is not recognized.")
    end
end



@generated function Normal(
    Y::AbstractMatrix{T},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true)}()
) where {P,T,track}
    M, PY = gensym(:M), gensym(:PY)
    quote
        $M = size(Y,1)
        $PY = $(Y <: Array ? M : :(stride(Y,2)))
        $(multivariate_normal_SMLT_quote(M, P, track, T, Ystride = PY, sp = false))
    end
end
@generated function Normal(
    sptr::StackPointer,
    Y::AbstractMatrix{T},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true)}()
) where {P,T,track}
    M, PY = gensym(:M), gensym(:PY)
    quote
        $M = size(Y,1)
        $PY = $(Y <: Array ? M : :(stride(Y,2)))
        $(multivariate_normal_SMLT_quote(M, P, track, T, Ystride = PY, sp = true))
    end
end


@generated function Normal(
    Y::AbstractMatrix{T},
    μ::Tμ,
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true)}()
# ) where {P,T,track,Tμ}
) where {P,T,Tμ,track}
    M, PY = gensym(:M), gensym(:PY)
    defs_quote = quote
        $M = size(Y,1)
        $PY = $(Y <: Array ? M : :(stride(Y,2)))
    end
    q = if Tμ === T
        multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, μdim = 0, μstride = 0)
    elseif Tμ <: LinearAlgebra.Adjoint{T,<:AbstractVector{T}}
        # if Tμ <:  LinearAlgebra.Adjoint{Float64,<: AbstractMutableFixedSizePaddedVector}
            multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, μdim = 1, μstride = 1, μtransposed = true)
    elseif Tμ <: AbstractVector# AbstractFixedSizePaddedVector
        multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, μdim = 1, μstride = 1)
    elseif Tμ <: AbstractFixedSizePaddedMatrix
        multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, μdim = 2, μstride = Tμ.parameters[4])
    elseif Tμ <: AbstractMatrix
        μstride = gensym(:μstride)
        push!(defs_quote.args, :($μstride = stride(μ,2)))
        multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, μdim = 2, μstride = μstride)
    else
        throw("Type of μ == $A is not recognized.")
    end
    quote
        $defs_quote
        $q
    end
end
@generated function Normal(
    sptr::StackPointer,
    Y::AbstractMatrix{T},
    μ::Tμ,
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true)}()
) where {P,T,track,Tμ}
    M, PY = gensym(:M), gensym(:PY)
    defs_quote = quote
        $M = size(Y,1)
        $PY = $(Y <: Array ? M : :(stride(Y,2)))
    end
    q = if Tμ === T
        multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, μdim = 0, μstride = 0)
    elseif Tμ <: LinearAlgebra.Adjoint{T,<:PaddedMatrices.AbstractMutableFixedSizePaddedVector}
        multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, μdim = 1, μstride = 1, μtransposed = true)
    elseif Tμ <: AbstractFixedSizePaddedVector
        multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, μdim = 1, μstride = 1)
    elseif Tμ <: AbstractFixedSizePaddedMatrix
        multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, μdim = 2, μstride = Tμ.parameters[4])
    elseif Tμ <: AbstractMatrix
        μstride = gensym(:μstride)
        push!(defs_quote.args, :($μstride = stride(μ,2)))
        multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, μdim = 2, μstride = μstride)
    else
        throw("Type of μ == $A is not recognized.")
    end
    quote
        $defs_quote
        $q
    end
end

@generated function Normal(
    Y::AbstractMatrix{T},
    X::AbstractMatrix{T},
    β::AbstractMutableFixedSizePaddedArray{Sβ,T,Nβ,Pβ},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true,true)}()
) where {P,T,track,PK,Sβ,Nβ,Pβ}
    K_ = Sβ.parameters[1]
    M, PY, PX = gensym(:M), gensym(:PY), gensym(:PX)
    q = multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_)
    quote
        $M = size(Y,1)
        $PY = $(Y <: Array ? M : :(stride(Y,2)))
        $PX = $(X <: Array ? M : :(stride(X,2)))
        $q
    end
end
@generated function Normal(
    sptr::StackPointer,
    Y::AbstractMatrix{T},
    X::AbstractMatrix{T},
    β::AbstractMutableFixedSizePaddedArray{Sβ,T,Nβ,Pβ},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true,true)}()
) where {P,T,track,PK,Sβ,Nβ,Pβ}
    K_ = Sβ.parameters[1]
    M, PY, PX = gensym(:M), gensym(:PY), gensym(:PX)
    q = multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_)
    quote
        $M = size(Y,1)
        $PY = $(Y <: Array ? M : :(stride(Y,2)))
        $PX = $(X <: Array ? M : :(stride(X,2)))
        $q
    end
end

    # M::Union{Symbol,Integer}, P::Int, track::NTuple{D,Bool}, T::DataType = Float64;
    # Ystride::Int = M, βstride::Int = -1, Xstride::Int = -1, βdim::Int = -1,
    # μdim::Int, μstride::Int, sp::Bool, XP::Int = -1, μtransposed::Bool = false
@generated function Normal_fmadd(
    Y::AbstractMatrix{T},
    X::AbstractMatrix{T},
    β::AbstractMutableFixedSizePaddedArray{Sβ,T,Nβ,Pβ},
    μ::Tμ,
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true,true,true)}()
) where {P,T,track,Sβ,Nβ,Pβ,Tμ}
    K_ = Sβ.parameters[1]
    M, PY, PX = gensym(:M), gensym(:PY), gensym(:PX)
    defs_quote = quote
        $M = size(Y,1)
        $PY = $(Y <: Array ? M : :(stride(Y,2)))
        $PX = $(X <: Array ? M : :(stride(X,2)))
    end
    q = if Tμ === T
        multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_, μdim = 0, μstride = 0)
    elseif Tμ <: LinearAlgebra.Adjoint{T,<:PaddedMatrices.AbstractMutableFixedSizePaddedVector}
        multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_, μdim = 1, μstride = 1, μtransposed = true)
    elseif Tμ <: AbstractFixedSizePaddedVector
        multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_, μdim = 1, μstride = 1)
    elseif Tμ <: AbstractFixedSizePaddedMatrix
        multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_, μdim = 2, μstride = Tμ.parameters[4])
    elseif Tμ <: AbstractMatrix
        μstride = gensym(:μstride)
        push!(defs_quote.args, :($μstride = stride(μ,2)))
        multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_, μdim = 2, μstride = μstride)
    else
        throw("Type of μ == $A is not recognized.")
    end
    quote
        $defs_quote
        $q
    end
end
@generated function Normal_fmadd(
    sptr::StackPointer,
    Y::AbstractMatrix{T},
    X::AbstractMatrix{T},
    β::AbstractMutableFixedSizePaddedArray{Sβ,T,Nβ,Pβ},
    μ::Tμ,
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true,true,true)}()
) where {P,T,track,Sβ,Nβ,Pβ,Tμ}
    K_ = Sβ.parameters[1]
    M, PY, PX = gensym(:M), gensym(:PY), gensym(:PX)
    defs_quote = quote
        $M = size(Y,1)
        $PY = $(Y <: Array ? M : :(stride(Y,2)))
        $PX = $(X <: Array ? M : :(stride(X,2)))
    end
    q = if Tμ === T
        multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_, μdim = 0, μstride = 0)
    elseif Tμ <: LinearAlgebra.Adjoint{T,<:PaddedMatrices.AbstractMutableFixedSizePaddedVector}
        multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_, μdim = 1, μstride = 1, μtransposed = true)
    elseif Tμ <: AbstractFixedSizePaddedVector
        multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_, μdim = 1, μstride = 1)
    elseif Tμ <: AbstractFixedSizePaddedMatrix
        multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_, μdim = 2, μstride = Tμ.parameters[4])
    elseif Tμ <: AbstractMatrix
        μstride = gensym(:μstride)
        push!(defs_quote.args, :($μstride = stride(μ,2)))
        multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_, μdim = 2, μstride = μstride)
    else
        throw("Type of μ == $A is not recognized.")
    end
    quote
        $defs_quote
        $q
    end
end


function track_mu_store(Mk::Int,Nk,T,μdim,μmy,W,Wshift,μstride,track_Y,μtransposed,initialize::Bool=false)
    size_T = sizeof(T)
    V = Vec{W,T}
    Riter = Mk >> Wshift
    Rrem = Mk & (W-1)
    Riterl = Rrem > 0 ? Riter : Riter-1
    mask = VectorizationBase.mask_from_remainder(T, Rrem)
    row_iter = quote end
    f = μmy ? :vsub : :vadd
    if μdim == 0
        iter = 0
        for c ∈ 0:(Nk-1), m ∈ 0:Riterl
            mask_this_iter = m == Riterl && Rrem > 0
            pm = Symbol(:∂μ_,iter & 3)
            A_m_c = Symbol(:A_,m,:_,c)
            if mask_this_iter
                push!(row_iter.args, Expr(:(=), pm, :(SIMDPirates.vifelse($mask,SIMDPirates.$f($pm, $A_m_c),$p_m))))
            else
                push!(row_iter.args, Expr(:(=), pm, :(SIMDPirates.$f($pm, $A_m_c))))
            end
            iter += 1
        end
    elseif μdim == 1
        if μtransposed
            for c ∈ 0:(Nk-1)
                mc = Symbol(:vμ_,c)
                push!(row_iter.args, Expr(:(=), mc, :(SIMDPirates.vload($V, ptrv∂μ + $(c*W*size_T)))))
            end
            for m ∈ 0:Riterl
                mask_this_iter = m == Riterl && Rrem > 0
                for c ∈ 0:(Nk-1)
                    mc = Symbol(:vμ_,c)
                    if mask_this_iter
                        push!(row_iter.args, Expr(:(=), mc, :(SIMDPirates.vifelse($mask,SIMDPirates.$f($mc, $(Symbol(:A_,m,:_,c))),$mc))))
                    else
                        push!(row_iter.args, Expr(:(=), mc, :(SIMDPirates.$f($mc, $(Symbol(:A_,m,:_,c))))))
                    end
                end
            end
            for c ∈ 0:(Nk-1)
                mc = Symbol(:vμ_,c)
                push!(row_iter.args, :(SIMDPirates.vstore!(ptrv∂μ + $(c*W)*$size_T, $mc)))
            end
        else
            if initialize
                if μmy
                    for m ∈ 0:Riterl
                        push!(row_iter.args, Expr(:(=), Symbol(:v∂μ_, m), :(SIMDPirates.vsub($(Symbol(:A_,m,:_0))))))
                    end
                else
                    for m ∈ 0:Riterl
                        push!(row_iter.args, Expr(:(=), Symbol(:v∂μ_, m), Symbol(:A_,m,:_0)))
                    end
                end
                firstc = 1
            else
                for m ∈ 0:Riterl
                    # We don't mask here, because beyond the end of the vector is junk, and since things are padded to allignment, the vector wont encroach on data we care about.
                    # if m == Riterl
                    push!(row_iter.args, Expr(:(=), Symbol(:v∂μ_, m), :(SIMDPirates.vload($V, ptr∂μ + $(m*W*size_T) ) )))
                end
                firstc = 0
            end
            for c ∈ firstc:Nk-1
                for m ∈ 0:Riterl
                    pm = Symbol(:v∂μ_,m)
                    push!(row_iter.args, Expr(:(=), pm, :(SIMDPirates.$f($pm, $(Symbol(:A_,m,:_,c))))))
                end
            end
            for m ∈ 0:Riterl
                # if m == Riterl && Rrem > 0
                    # push!(row_iter.args, :(SIMDPirates.vstore!(ptr∂μ + $(m*W*size_T), $(Symbol(:v∂μ_, m)), $mask )))
                # else
                    push!(row_iter.args, :(SIMDPirates.vstore!(ptr∂μ + $(m*W)*$size_T, $(Symbol(:v∂μ_, m)) )))
                # end
            end
        end
    elseif μdim == 2
        if track_Y
            for c ∈ 0:(Nk-1)
                for r ∈ 0:Riter-1
                    push!(row_iter.args, :(SIMDPirates.vstore!(ptr∂μ + $size_T*($(r*W)+$c*$μstride), SIMDPirates.vsub($(Symbol(:A_,r,:_,c))))))
                end
                if Rrem > 0
                    index = :(ptr∂μ + $size_T*($(Riter*W)+$c*$μstride))
                    nAsym = :(SIMDPirates.vsub($(Symbol(:A_,Riter,:_,c))))
                    if c == Nk-1
                        push!(row_iter.args, :(SIMDPirates.vstore!($index, $nAsym, $mask)))
                    else
                        push!(row_iter.args, :(SIMDPirates.vstore!($index, $nAsym)))
                    end
                end
            end        
        end # else, ptrA holds partial_mu, so we don't have to do anything
    end
    row_iter
end

function track_mu_store(Mk::Symbol,Nk,T,μdim,μmy,W,Wshift,μstride,track_Y,μtransposed,initialize::Bool=false,masksym = :__mask__)
    size_T = sizeof(T)
    V = Vec{W,T}
    row_iter = quote end
    f = μmy ? :vsub : :vadd
    if μdim == 0
        iter = 0
        for c ∈ 0:(Nk-1)
            pm = Symbol(:∂μ_,iter & 3)
            A_m_c = Symbol(:A_0_,c)
            push!(row_iter.args, Expr(:(=), pm, :(SIMDPirates.vifelse($masksym,SIMDPirates.$f($pm, $A_m_c),$p_m))))
            iter += 1
        end
    elseif μdim == 1
        if μtransposed
            for c ∈ 0:(Nk-1)
                mc = Symbol(:vμ_,c)
                push!(row_iter.args, Expr(:(=), mc, :(SIMDPirates.vload($V, ptrv∂μ + $(c*W*size_T)))))
            end
            for c ∈ 0:(Nk-1)
                mc = Symbol(:vμ_,c)
                push!(row_iter.args, Expr(:(=), mc, :(SIMDPirates.vifelse($masksym,SIMDPirates.$f($mc, $(Symbol(:A_0_,c))),$mc))))
            end
            for c ∈ 0:(Nk-1)
                mc = Symbol(:vμ_,c)
                push!(row_iter.args, :(SIMDPirates.vstore!(ptrv∂μ + $(c*W)*$size_T, $mc)))
            end
        else
            if initialize
                if μmy
                    push!(row_iter.args, :(v∂μ_0 = SIMDPirates.vsub(A_0_0)))
                else
                    push!(row_iter.args, :(v∂μ_0 = A_0_0))
                end
                firstc = 1
            else
                push!(row_iter.args, Expr(:(=), Symbol(:v∂μ_, m), :(SIMDPirates.vload($V, ptr∂μ, $masksym ) )))
                firstc = 0
            end
            for c ∈ firstc:Nk-1
                pm = :v∂μ_0
                push!(row_iter.args, Expr(:(=), pm, :(SIMDPirates.$f($pm, $(Symbol(:A_0_,c))))))
            end
            push!(row_iter.args, :(SIMDPirates.vstore!(ptr∂μ, v∂μ_0, $masksym )))
        end
    elseif μdim == 2
        if track_Y
            for c ∈ 0:(Nk-1)
                push!(row_iter.args, :(SIMDPirates.vstore!(ptr∂μ + $size_T*$c*$μstride, SIMDPirates.vsub($(Symbol(:A_0_,c))),$masksym)))
            end        
        end # else, ptrA holds partial_mu, so we don't have to do anything
    end
    row_iter
end

"""
Sets pointers back columns during the reverse pass over rows.
"""
function loop_pointer_increments(track_Y, track_μ, track_X, track_β, track_L, Nk, Nk2, K, size_T, W, Astride, μstride, μdim, μtransposed)
    b2Nk = StructuredMatrices.binomial2(Nk)
    loop_ptr_increments = quote
        ptrLdiag -= $(size_T*Nk2)
        ptrLtri -= $size_T*($Nk*$K + $b2Nk)
    end
    if track_Y || (track_μ && μdim == 2) || track_X || track_β
        push!(loop_ptr_increments.args, Expr(:(-=), :ptrA_rev, (Astride isa Symbol || Nk isa Symbol) ? :($Astride*$Nk*$size_T) : Nk*Astride*size_T))
        if track_Y && track_μ && μdim == 2  # then 
            push!(loop_ptr_increments.args, Expr(:(-=), :ptr∂μ, (μstride isa Symbol || Nk isa Symbol) ? :($μstride*$Nk*$size_T) : Nk*μstride*size_T))
        end
    end
    if track_μ && μdim == 1 && μtransposed
        push!(loop_ptr_increments.args, Expr(:(-=), :ptrv∂μ, Nk isa Symbol ? :($Nk*$(size_T*W)) : Nk*size_T*W))
    end
    if track_L
        push!(loop_ptr_increments.args, :(ptrv∂Ldiag -= $(size_T*W)*$Nk; ptrv∂Ltri -= $(size_T*W)*($Nk*$K+$b2Nk)))
    end
    loop_ptr_increments
end

function ∂mutlivariate_normal_SMLT_rowiter(
    Mk::Union{Int,Symbol}, Nk::Int, col_rem::Int, T::DataType, Ystride::Union{Int,Symbol},
    n_col_reps::Int, μdim::Int, μstride::Union{Int,Symbol}, track::NTuple{D,Bool},
    μmy::Bool, μsym::Symbol = :μptr,
    Astride::Union{Int,Symbol} = Ystride, XP::Int = -1, βstride::Int=-1,
    Xstride::Union{Int,Symbol} = -1, βdim::Int = -1, μtransposed::Bool = false
) where {D}
    if D == 5
        track_Y, track_X, track_β, track_μ, track_L = track
    else
        if D == 4
            track_Y, track_X, track_β, track_L = track
            track_μ = false
        else
            track_X = track_β = false
            if D == 3
                track_Y, track_μ, track_L = track
            elseif D == 2
                track_Y, track_L = track
                track_μ = false
            else
                throw("Unknown number of arguments ($D) to normal.")
            end
        end
    end
    if Mk isa Int
        W, Wshift = VectorizationBase.pick_vector_width_shift(Mk, T)
    else
        W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    end
    V = Vec{W,T}
    #TODO: NOTE, WE DO NEED TO STORE THE SOLUTION MATRIX (at least 1 row set amd up to the last column block)
    # because this is used for calculating the next iteration.
    N = Nk * n_col_reps + col_rem
    size_T = sizeof(T)
    if βdim == 1 && XP > 0
        row_iter = Xβ_load_quote(Mk, T, Xstride, βstride, μmy, XP, :ptrX, :ptrβ)
    else
        row_iter = quote end
    end
    if col_rem > 0
        if XP > 0
            loadδ_expr = loadδfnmadd_quote(
                Mk, col_rem, 0, T, Ystride, Xstride, βstride, βdim,
                :ptrY, :ptrX, :ptrβ, :ptrμ, true, μmy, XP, μstride, μdim, μtransposed
            )
        else
            loadδ_expr = loadδ_quote(Mk, col_rem, 0, T, Ystride, :ptrY, μdim, μstride, μsym, true, μmy, μtransposed)
        end
        iter_quote = StructuredMatrices.A_rdiv_U_kernel_quote(
            Mk, col_rem, 0, T, Astride, Ystride, N, true, true, storeA = true, loadB = false, reduce_sym = :δ² # storeA = n_col_reps > 0
        )
        #pushfirst!(row_iter.args, :(ptrUtri = ptrUtribase))
        push!(row_iter.args, loadδ_expr)
        push!(row_iter.args, iter_quote)
        push!(row_iter.args, :(ptrUdiag += $(col_rem*size_T)))
        base_K = col_rem
        KmZ = false
    else
        base_K = 0
        KmZ = true
    end
    if n_col_reps > 1
        if XP > 0
            loadδ_expr = loadδfnmadd_quote(
                Mk, Nk, :K, T, Ystride, Xstride, βstride, βdim,
                :ptrY, :ptrX, :ptrβ, :ptrμ, true, μmy, XP, μstride, μdim, μtransposed
            )
        else
            loadδ_expr = loadδ_quote(Mk, Nk, :K, T, Ystride, :ptrY, μdim, μstride, μsym, true, μmy, μtransposed)
        end
        iterquote = StructuredMatrices.A_rdiv_U_kernel_quote(
            Mk, Nk, :K, T, Astride, Ystride, N, true, true, storeA = true, loadB = false, reduce_sym = :δ²
        )
        row_iter_loop = quote
            K = $base_K
            for crep ∈ 0:$(n_col_reps-1)
                ptrUtri = ptrUtribase + K*$size_T
                $loadδ_expr
                $iterquote
                ptrUdiag += $(size_T*Nk)
                K += $Nk
            end
        end
        push!(row_iter.args, row_iter_loop)
    elseif n_col_reps == 1
        if XP > 0
            loadδ_expr = loadδfnmadd_quote(
                Mk, Nk, col_rem, T, Ystride, Xstride, βstride, βdim,
                :ptrY, :ptrX, :ptrβ, :ptrμ, true, μmy, XP, μstride, μdim, μtransposed
            )
        else
            loadδ_expr = loadδ_quote(Mk, Nk, col_rem, T, Ystride, :ptrY, μdim, μstride, μsym, true, μmy, μtransposed)
        end
        push!(row_iter.args, loadδ_expr)
        row_iter_single = StructuredMatrices.A_rdiv_U_kernel_quote(
            Mk, Nk, col_rem, T, Astride, Ystride, N, true, true, storeA = true, loadB = false, reduce_sym = :δ² # storeA = col_rem > 0
        )
        push!(row_iter.args, row_iter_single)
    end
    ########################
    ### now time for ÷ L ###
    ########################
    # set starting pointers for reverse pass
    push!(row_iter.args, :(ptrLdiag = ptrLdiagbase; ptrLtri = ptrLtribase; ptrA_rev = ptrA + _A_offset_))
    # mu has to change columnwise for us to set the pointer here;
    # if we aren't tracking y, then ptrA aliases it, so setting it is unnecessary.
    if track_μ && ((μdim == 1 && μtransposed) || (!track_Y && μdim == 2))
        push!(row_iter.args, :(ptrv∂μ = ptrv∂μbase))
    end
    if track_L
        push!(row_iter.args, :(ptrv∂Ltri = ptrv∂Ltribase; ptrv∂Ldiag = ptrv∂Ldiagbase))
    end
    if col_rem > 0
        row_iter_rev = StructuredMatrices.A_rdiv_L_kernel_quote(
            Mk, col_rem, col_rem, T, Astride, Astride, false, true,
            Bsym = :ptrA_rev, Asym = :ptrA_rev, Ltrisym = :ptrLtri, Ldiagsym = :ptrLdiag,
            loadB = true, storeA = true, calc_product = track_L ? N : 0
        )
        fullcols = Nk * n_col_reps
        # handle following in A_rdiv_L_quote
        append!(row_iter.args, row_iter_rev.args)
        track_μ && push!(row_iter.args, track_mu_store(Mk,col_rem,T,μdim,μmy,W,Wshift,μstride,track_Y,μtransposed,true))
        push!(row_iter.args, loop_pointer_increments(track_Y, track_μ, track_X, track_β, track_L, Nk, col_rem, col_rem, size_T, W, Astride, μstride, μdim, μtransposed))
        base_K = col_rem
        KmZ = false
    else
        base_K = 0
        KmZ = true
    end
    loop_ptr_increments = loop_pointer_increments(track_Y, track_μ, track_X, track_β, track_L, Nk, Nk, :K, size_T, W, Astride, μstride, μdim, μtransposed)
    if n_col_reps > 1
        iterquote = StructuredMatrices.A_rdiv_L_kernel_quote(
            Mk, Nk, :K, T, Astride, Astride, false, true,
            Bsym = :ptrA_rev, Asym = :ptrA_rev, Ltrisym = :ptrLtri, Ldiagsym = :ptrLdiag,
            loadB = true, storeA = true, calc_product = track_L ? N : 0
        )
        if col_rem == 0 && !μtransposed && track_μ # then we need to zero-initialize these rows before entering the loop
            Riter = Mk >> Wshift
            Rrem = Mk & (W-1)
            Riterl = Rrem > 0 ? Riter : Riter-1
            for r ∈ 0:Riterl
                push!(row_iter.args, :(SIMDPirates.vstore!(ptr∂μ + $(r*W*size_T), SIMDPirates.vbroadcast($V, zero($T)))))
            end
        end
        track_μ && push!(iterquote.args, track_mu_store(Mk,Nk,T,μdim,μmy,W,Wshift,μstride,track_Y,μtransposed,false))
        row_iter_rev_loop = quote
            K = $col_rem
            for crep ∈ 0:$(n_col_reps-1)
                K += $Nk
                $iterquote
                $loop_ptr_increments
            end
        end
        push!(row_iter.args, row_iter_rev_loop)
    elseif n_col_reps == 1
        row_iter_rev_single = StructuredMatrices.A_rdiv_L_kernel_quote(
            Mk, Nk, N, T, Astride, Astride, false, true,
            Bsym = :ptrA_rev, Asym = :ptrA_rev, Ltrisym = :ptrLtri, Ldiagsym = :ptrLdiag,
            loadB = true, storeA = true, calc_product = track_L ? N : 0
        )
        push!(row_iter.args, row_iter_rev_single)
        track_μ && push!(row_iter.args, track_mu_store(Mk,Nk,T,μdim,μmy,W,Wshift,μstride,track_Y,μtransposed,col_rem == 0))
    end
    row_iter
end

## StructuredMatrices.jl Lower Triangular (SMLT) quote
## M is the sample size
function ∂multivariate_normal_SMLT_quote(
    M::Union{Int,Symbol}, P::Int, track::NTuple{D,Bool}, T::DataType = Float64;
    βstride::Int = -1, Xstride::Union{Symbol,Int} = -1, Ystride::Union{Symbol,Int} = M, μstride::Union{Int,Symbol} = -1,
    μdim::Int = -1, sp::Bool = false, βdim::Int = -1,  XP::Int = -1, μtransposed::Bool = false
) where {D}
    if D == 5
        track_Y, track_X, track_β, track_μ, track_L = track
    else
        if D == 4
            track_Y, track_X, track_β, track_L = track
            track_μ = false
        else
            track_X = track_β = false
            if D == 3
                track_Y, track_μ, track_L = track
            elseif D == 2
                track_Y, track_L = track
                track_μ = false
            else
                throw("Unknown number of arguments ($D) to normal.")
            end
        end
    end
    q = quote end
    maxM = M isa Symbol ? typemax(Int) : M
    W, Mk, Nk = StructuredMatrices.div_ul_blocking_structure(maxM, P, T)
    V = Vec{W,T}
    Wm1 = W - 1
    n_col_reps, col_rem = divrem(P, Nk)
    total_col_iterations = n_col_reps + (col_rem > 0)
    #Nl = ( N + W - 1 ) & ~Wm1
    size_T = sizeof(T)
    loopbody = quote
        Bₚ = L[p]
        invdiag[p] = one($T) / Bₚ
    end
    track_L && push!(loopbody.args, :(δ²_0 = LoopVectorization.SIMDPirates.vadd(δ²_0, SLEEFPirates.log(Bₚ))))
    # need to allocate invdiag, ∂Y, ∂μ, ∂L, and v∂L
    # also writing into ptrA
    # Q: do we increment ptrA alongside Y?
    # Q: if yes, is ptrA ∂Y or ∂μ ?
    invdiagL = VectorizationBase.align(P, W)
    array_allocations = sp ? quote _sptr = pointer(sptr,$T) end : quote end
    if track_L
        ∂LL = VectorizationBase.align(StructuredMatrices.binomial2(P + 1), W)
        if sp
            push!(array_allocations.args, :(∂L = StructuredMatrices.PtrLowerTriangularMatrix{$P,$T,$∂LL}(_sptr)))
            push!(array_allocations.args, :(invdiag = PtrVector{$P,$T,$P}(_sptr)))
            push!(array_allocations.args, :(_sptr += $(∂LL*size_T)))
        else
            push!(array_allocations.args, :(v∂L = StructuredMatrices.MutableLowerTriangularMatrix{$P,$V,$∂LL}(undef)))
            push!(array_allocations.args, :(∂L = StructuredMatrices.MutableLowerTriangularMatrix{$P,$T,$∂LL}(undef)))
            push!(array_allocations.args, :(invdiag = PtrVector{$P,$T,$P}(pointer(∂L))))
        end
    elseif !sp
        push!(array_allocations.args, :(invdiag = MutableFixedSizePaddedVector{$P,$T,$invdiagL,$invdiagL}(undef)))
    end
    row_increments = quote
        ptrY += $(size_T*Mk)
    end
    row_increments_rem = quote
        ptrY += $(size_T*W)
    end
    if XP != -1
        push!(row_increments.args, :(ptrX += $(size_T*Mk)))
        push!(row_increments_rem.args, :(ptrX += $(size_T*W)))
    end
    return_expr = Expr(:tuple,:δ²_0)
    track_Y && push!(return_expr.args, :(A'))
    track_X && push!(return_expr.args, :(∂X'))
    track_β && push!(return_expr.args, βdim == 1 ? :(∂βv') : :(∂β'))
    track_μ && push!(return_expr.args, (!track_Y && (μdim == 2)) ? :(A') : :(∂μ'))
    track_L && push!(return_expr.args, :∂L)
    # this increments _sptr
    sptroff = 0
    sptroffexpr = quote end
    nonempty_sptroff_expr = false
    μmy = track_Y
    if sp # define sptroff, the offset of the sptr relative to the end of the last returned object (where a non-returned object would start)
        if !(track_Y || track_μ || track_X || track_β)# don't need to track A
            Aquote = quote
                A = PtrMatrix{$Mk,$P,$T,$Mk}(_sptr)
                ptrA = pointer(A)
            end
            sptroff = VectorizationBase.align(Mk*P*size_T)
            Astride = Mk
        else # We track at least one of the four
            if (μdim == 1) && !(track_Y || track_X || track_β) # We do not track or store all of A, so we make it a MK x P block to hold a single set of iterations across columns
                if μtransposed
                    Aquote = quote
                        ∂μ = PtrVector{$P,$T}(_sptr)
                        ptr∂μ = _sptr
                        _sptr += $(invdiagL*size_T)
                        v∂μ = PtrMatrix{$W,$P,$T,$W,$(W*P)}(_sptr) # accmulate in v∂μ; reduce at end
                        ptrv∂μ = _sptr
                    end
                    sptroff = W*P*size_T # aligned because of W
                else
                    Aquote = if M isa Integer
                        ML = VectorizationBase.align(M, W)
                        quote
                            ∂μ = PtrVector{$M,$T}(_sptr)
                            ptr∂μ = _sptr
                            _sptr += $(ML*size_T)
                        end
                    else
                        quote
                            MalignedtoW = VectorizationBase.align($M, $W)
                            ∂μ = DynamicPtrVector{$T}(_sptr, ($M,), MalignedtoW)
                            ptr∂μ = _sptr
                            _sptr += MalignedtoW*$size_T
                        end
                    end
                end
                push!(Aquote.args, :(A = PtrMatrix{$Mk,$P,$T,$Mk}(_sptr + $sptroff); ptrA = pointer(A)))
                Astride = Mk
                sptroff += VectorizationBase.align(size_T*Mk*P)
            else# We do create a full-sized (size(A) == size(Y)) A-matrix
                Astride = M isa Integer ? VectorizationBase.align(M, W) : :_A_stride_
                # Therefore, we must increment through row iterations
                push!(row_increments.args, :(ptrA += $(size_T*Mk)))
                push!(row_increments_rem.args, :(ptrA += $(size_T*W)))
                Aquote = M isa Integer ? quote end : quote _A_stride_ = VectorizationBase.align($M,$W) end
                must_still_allocate_A = true

                if track_Y || (track_μ && (μdim == 2))
                    # if track_Y, A is ∂Y
                    # otherwise, if track_μ && (μdim == 2), A is ∂μ
                    # if none of these, we allocate A later.
                    A_init_quote = if M isa Integer # Y
                        quote
                            A = PtrMatrix{$M,$P,$T,$Astride}(_sptr)
                            ptrA = _sptr
                            _sptr += $(Astride*P*size_T)
                        end
                    else#if !(M isa Symbol)
                        quote
                            A = DynamicPtrMatrix{$T}(_sptr, ($M,$P), _A_stride_)
                            ptrA = _sptr
                            _sptr += $(size_T*P) * _A_stride_
                        end
                    end
                    push!(Aquote.args, A_init_quote)
                    must_still_allocate_A = false
                end
                if track_X              
                    if M isa Integer
                        push!(Aquote.args, :(∂X = PtrMatrix{$M,$XP,$T,$Astride}(_sptr)))
                        push!(Aquote.args, :( ptr∂X = _sptr; _sptr += $(VectorizationBase.align(size_T*Astride*XP)) ))
                    else
                        push!(Aquote.args, :(∂X = DynamicPtrMatrix{$T}(_sptr, ($M,$XP), _A_stride_)))
                        push!(Aquote.args, :( ptr∂X = _sptr; _sptr += $(size_T*XP) * _A_stride_ ))
                    end
                end
                delayed_allocation_quote = quote end
                delay_alloc = false
                if track_μ
                    if μdim == 1
                        if μtransposed
                            delay_alloc = true
                            push!(Aquote.args, :(∂μ = PtrVector{$P,$T}(_sptr); ptr∂μ = _sptr))
                            push!(Aquote.args, :(_sptr += $(invdiagL*size_T)))
                            push!(delayed_allocation_quote.args, :(v∂μ = PtrMatrix{$W,$P,$T,$W,$(W*P)}(_sptr))) # accmulate in v∂μ; reduce at end
                            push!(delayed_allocation_quote.args, :(ptrv∂μ = _sptr))
                            sptroff = W*P*size_T
                        else#if !μtransposed
                            if M isa Integer
                                push!(Aquote.args, :(∂μ = PtrVector{$M,$T}(_sptr); ptr∂μ = _sptr))
                                push!(Aquote.args, :(_sptr += $(Astride*size_T)))
                            else#if M isa Symbol
                                push!(Aquote.args, :(∂μ = DynamicPtrVector{$T}(_sptr, ($M,), _A_stride_); ptr∂μ = _sptr))
                                push!(Aquote.args, :(_sptr += $(size_T) * _A_stride_))
                            end
                        end
                    elseif track_Y# && μdim == 2
                        if M isa Integer # Y
                            push!(Aquote.args, :(∂μ = PtrMatrix{$M,$P,$T,$Astride}(_sptr); ptr∂μ = _sptr))
                            push!(Aquote.args, :(_sptr += $(Astride*P*size_T)))
                        else#if !(M isa Symbol)
                            push!(Aquote.args, :(∂μ = DynamicPtrMatrix{$T}(_sptr, ($M,$P), _A_stride_); ptr∂μ = _sptr))
                            push!(Aquote.args, :(_sptr += $(size_T*P) * _A_stride_))
                        end
                    end
                    if ((μdim == 1) && !μtransposed) || ((μdim == 2) && track_Y)
                        push!(row_increments.args, :(ptr∂μ += $(size_T*Mk)))
                        push!(row_increments_rem.args, :(ptr∂μ += $(size_T*W)))
                    end
                end
                if track_β # we vbroadcast from β rather than load, so no point alligning columns
                    XPL = VectorizationBase.align(XP, T)
                    push!(Aquote.args, :(∂β = PtrMatrix{$XP,$P,$T}(_sptr); ptr∂β = _sptr))
                    alignβoffset = XPL*P*size_T
                    if βdim == 1
                        push!(Aquote.args, :(_sptr += $(XPL*size_T))) # impacts the pointer we ultimately return
                        # first increment (because of if/else statements), so we could (and did) turn the += into an =
                        # gives extra offset for future allocations
                        sptroff += alignβoffset - XPL * size_T
                        push!(Aquote.args, Expr(:(=), :∂βv, :(PtrVector{$XP,$T}(ptr∂β))))
                    else
                        push!(Aquote.args, :(_sptr += $alignβoffset))
                    end
                end
                delay_alloc && push!(Aquote.args, delayed_allocation_quote)

                if must_still_allocate_A
                    if M isa Integer
                        if sptroff == 0
                            push!(Aquote.args, :(A = PtrMatrix{$M,$P,$T,$Astride}(_sptr) ))
                        else#if sptroff != 0
                            push!(Aquote.args, :(A = PtrMatrix{$M,$P,$T,$Astride}(_sptr + $sptroff) ))
                        end
                        sptroff += Astride*P*size_T
                    else#if M isnot an Integer
                        if sptroff == 0
                            push!(Aquote.args, :(A = DynamicPtrMatrix{$T}(_sptr, ($M,$P), _A_stride_) ))
                        else#if sptroff != 0
                            push!(Aquote.args, :(A = DynamicPtrMatrix{$T}(_sptr + $sptroff, ($M,$P), _A_stride_) ))
                        end
                        sptroffexpr = :( $(size_T*P) * _A_stride_)
                        nonempty_sptroff_expr = true
                    end
                    push!(Aquote.args, sptroff == 0 ? :(ptrA = _sptr) : :(ptrA = pointer(A)))
                end
            end
        end
        final_offset_expr = if nonempty_sptroff_expr
            sptroff == 0 ? :(_sptr + $sptroffexpr) : :(_sptr + $sptroff + $sptroffexpr)
        else
            sptroff == 0 ? :(_sptr) : :(_sptr + $sptroff)
        end
        if track_L
            push!(Aquote.args, :(v∂L = StructuredMatrices.PtrLowerTriangularMatrix{$P,$V,$∂LL}( $final_offset_expr )))
        else # allocate invdiagL at the end
            push!(Aquote.args, :(invdiag = PtrVector{$P,$T,$invdiagL}( $final_offset_expr )))
        end        
    else#if !sp
        # Life is easier if we don't use our own stack, because
        # now we don't have to bother sorting the parameters on said stack ourselves.
        # Nor do we have to worry about keeping the stack (REGISTER_SIZE)-bytes alligned
        if !(track_Y || track_μ || track_X || track_β)# don't need to track A
            Aquote = quote
                A = MutableFixedSizePaddedMatrix{$Mk,$P}(undef)
                ptrA = pointer(A)
            end
            Astride = Mk
        else # We track at least one of the four
            if (μdim == 1) && !(track_Y || track_X || track_β) # We do not track or store all of A, so we make it a MK x P block to hold a single set of iterations across columns
                if μtransposed
                    Aquote = quote
                        ∂μ = MutableFixedSizePaddedVector{$P,$T}(undef)
                        v∂μ = MutableFixedSizePaddedMatrix{$W,$P,$T,$W,$(W*P)}(undef) # accmulate in v∂μ; reduce at end
                    end
                    sptroff = W*P*size_T # aligned because of W
                else
                    Aquote = if M isa Integer
                        quote ∂μ = MutableFixedSizePaddedVector{$M,$T}(undef) end
                    else
                        quote ∂μ = Vector{$T}(undef, $M) end
                    end
                end
                push!(Aquote.args, :(ptr∂μ = pointer(∂μ)))
                push!(Aquote.args, :(A = MutableFixedSizePaddedMatrix{$Mk,$P,$T,$Mk}(undef); ptrA = pointer(A)))
                Astride = Mk
                sptroff += VectorizationBase.align(size_T*Mk*P)
            else# We do create a full-sized (size(A) == size(Y)) A-matrix
                Astride = M isa Integer ? VectorizationBase.align(M, W) : M
                # Therefore, we must increment through row iterations
                push!(row_increments.args, :(ptrA += $(size_T*Mk)))
                push!(row_increments_rem.args, :(ptrA += $(size_T*W)))
                Aquote = quote end
                push!(Aquote.args, M isa Integer ? :(A = MutableFixedSizePaddedMatrix{$M,$P,$T,$Astride}(undef)) : :(A = Matrix{$T}(undef, $M,$P)) )
                push!(Aquote.args, :(ptrA = pointer(A)))
                #end
                if track_X              
                    push!(Aquote.args, M isa Integer ? :(∂X = MutableFixedSizePaddedMatrix{$M,$XP,$T,$Astride}(undef)) : :(∂X = Matrix{$T}(undef, $M,$XP))  )
                    push!(Aquote.args, :(ptr∂X = pointer(∂X)))
                end
                if track_μ
                    if μdim == 1
                        if μtransposed
                            PL = VectorizationBase.align(P, W) # align the number of columns to SIMD width
                            push!(Aquote.args, :(∂μ = MutableFixedSizePaddedVector{$P,$T}(undef)))
                            push!(Aquote.args, :(v∂μ = MutableFixedSizePaddedMatrix{$W,$P,$T,$W,$(W*P)}(undef))) # accmulate in v∂μ; reduce at end
                            push!(Aquote.args, :(ptrv∂μ = pointer(v∂μ)))
                        else#if !μtransposed
                            if M isa Integer
                                push!(Aquote.args, :(∂μ = MutableFixedSizePaddedVector{$M,$T,$Astride}(undef)))
                            else#if M isa Symbol
                                push!(Aquote.args, :(∂μ = Vector{$T}(undef, $M)))
                            end
                        end
                    elseif track_Y# && μdim == 2
                        if M isa Integer # Y
                            push!(Aquote.args, :(∂μ = MutableFixedSizePaddedMatrix{$M,$P,$T,$Astride}(undef)))
                        else#if !(M isa Symbol)
                            push!(Aquote.args, :(∂μ = Matrix{$T}(undef, $M,$P)))
                        end
                    end
                    if ((μdim == 1) && !μtransposed) || ((μdim == 2) && track_Y)
                        push!(row_increments.args, :(ptr∂μ += $(size_T*Mk)))
                        push!(row_increments_rem.args, :(ptr∂μ += $(size_T*W)))
                    end
                    push!(Aquote.args, :(ptr∂μ = pointer(∂μ)))
                end
                if track_β # we vbroadcast from β rather than load, so no point alligning columns
                    push!(Aquote.args, :(∂β = MutableFixedSizePaddedMatrix{$XP,$P,$T}(undef)))
                    if βdim == 1
                        push!(Aquote.args, Expr(:(=), :∂βv, :(MutableFixedSizePaddedVector{$XP,$T}(undef))))
                    end
                end
            end
        end
    end
    push!(array_allocations.args, Aquote)
    Mk2 = min(4, M isa Symbol ? cld(Mk,W) : cld(min(Mk,M),W))
    startoffset = (total_col_iterations-1) * Nk
    loopexpr = quote
        @vvectorize $T for p ∈ 1:$P
            $loopbody # fills out invdiag and calculates logdetL if necessary
        end
    end
    q = quote
        $(Expr(:meta,:inline)) # because of allignment bug
        $array_allocations
        $([Expr(:(=), Symbol(:δ²_,m), :(SIMDPirates.vbroadcast($V, zero($T)))) for m ∈ 0:Mk2-1]...)
        #$Aquote
        $(macroexpand(LoopVectorization, loopexpr))
        ptrY = pointer(Y)
        ptrUtribase = pointer(L) + $(P*size_T)
        _A_offset_ = $size_T*$Astride*$startoffset
        ptrLtribase = pointer(L) + $size_T * $(P + StructuredMatrices.binomial2(startoffset) + startoffset * (P - startoffset)) # diag + triangle + subtriangle
        ptrLdiagbase = pointer(invdiag) + $(size_T * startoffset)
    end
    if track_L
        push!(q.args, :(δ²_0 = SIMDPirates.vmul(δ²_0, SIMDPirates.vbroadcast($V,$(M isa Integer ? T(2M) : :($(T(2))*$T($M)))))))
        set_v∂L_to_zero_quote = quote
            ptrv∂L = pointer(v∂L)
            for p ∈ 0:$(StructuredMatrices.binomial2(P+1)-1)
                SIMDPirates.vstore!(ptrv∂L + p *$(W*size_T), SIMDPirates.vbroadcast($V, zero($T)))
            end
        end
        push!(q.args, set_v∂L_to_zero_quote)
        push!(q.args, :(ptrv∂Ltribase = pointer(v∂L) + $(W*size_T * (P + StructuredMatrices.binomial2(startoffset) + startoffset * (P - startoffset))))) # diag + triangle + subtriangle
        push!(q.args, :(ptrv∂Ldiagbase = pointer(v∂L) + $(W*size_T*startoffset)))
    end
    D >= 4 && push!(q.args, :(ptrX = pointer(X); ptrβ = pointer(β)))
    if track_μ
        if μdim == 0
            for m ∈ 0:3
                push!(q.args, Expr(:(=), Symbol(:v∂μ_,m), :(SIMDPirates.vbroadcast($V, zero($T)))))
            end
        else
            if μdim == 1 && μtransposed
                push!(q.args, :(ptrv∂μbase = pointer(v∂μ) + $(size_T*W*startoffset)))
                set_ptr_vmu_zero_expr = quote
                    ptrv∂μ = pointer(v∂μ)
                    for p ∈ 0:$(P-1)
                        SIMDPirates.vstore!(ptrv∂μ + p*$(W*size_T), SIMDPirates.vbroadcast($V, zero($T)))
                    end
                end
                push!(q.args, set_ptr_vmu_zero_expr)
            elseif μdim == 2 && track_Y
                # if M isa Symbol
                    # push!(q.args, :(ptrv∂μbase = pointer(v∂μ) + $(size_T*startoffset)*$M))
                # else
                    # push!(q.args, :(ptrv∂μbase = pointer(v∂μ) + $(size_T*startoffset*M)))
                # end
                push!(q.args, :(ptrv∂μbase = ptrv∂μ + $(M isa Symbol ? :($(size_T*startoffset)*$M) : size_T*startoffset*M)))
            end
        end
    end
    if μdim == 0
        push!(q.args, Expr(:(=), :ptrμ, :(SIMDPirates.vbroadcast($V, μ))))
    elseif μdim > 0
        push!(q.args, Expr(:(=), :ptrμ, μtransposed ? :(pointer(μ.parent)) : :(pointer(μ))))
    end
    if M isa Integer
        n_row_reps, row_rem = divrem(M, Mk)
        total_row_iterations = n_row_reps + (row_rem > 0)
        Mk1 = n_row_reps == 0 ? row_rem : Mk
            # Mk::Union{Int,Symbol}, Nk::Int, col_rem::Int, T::DataType, Ystride::Int,
        # n_col_reps::Int, μdim::Int, μstride::Int, track::NTuple{D,Bool},
    # μmy::Bool, μsym::Symbol = :μptr,
    # Astride::Int = Ystride, XP::Int = -1, βstride::Int=-1,
    # Xstride::Int = -1, βdim::Int = -1, μtransposed::Bool = false
        row_iter = ∂mutlivariate_normal_SMLT_rowiter(
            Mk1, Nk, col_rem, T, Ystride, n_col_reps, μdim, μstride, track, μmy, :ptrμ, Astride, XP, βstride, Xstride, βdim, μtransposed
        )
        if n_row_reps > 1
            row_loops = quote
                for rrep ∈ 1:$n_row_reps
                    ptrUdiag = pointer(invdiag); ptrUtri = ptrUtribase
                    $row_iter
                    $row_increments
                end
            end
            push!(q.args, row_loops)
        else
            push!(q.args, :(ptrUdiag = pointer(invdiag); ptrUtri = ptrUtribase))
            push!(q.args, row_iter)
        end
        if row_rem > 0 && n_row_reps > 0
            push!(q.args, :(ptrUdiag = pointer(invdiag); ptrUtri = ptrUtribase))
            push!(q.args, ∂mutlivariate_normal_SMLT_rowiter( row_rem, Nk, col_rem, T, Ystride, n_col_reps, μdim, μstride, track, μmy, :ptrμ, Astride, XP, βstride, Xstride, βdim, μtransposed ))
        end
    else # Unknown number of iterations.
        row_iter = ∂mutlivariate_normal_SMLT_rowiter(
            Mk, Nk, col_rem, T, Ystride, n_col_reps, μdim, μstride, track, μmy, :ptrμ, Astride, XP, βstride, Xstride, βdim, μtransposed
        )
        Wrem, Mkrem, Nkrem = StructuredMatrices.div_triangle_blocking_structure(W, P, T)
        n_col_repsrem, col_remrem = divrem(P, Nkrem)
        row_iter_onevec = ∂mutlivariate_normal_SMLT_rowiter(
            W, Nkrem, col_remrem, T, Ystride, n_col_repsrem, μdim, μstride, track, μmy, :ptrμ, Astride, XP, βstride, Xstride, βdim, μtransposed
        )
        row_iter_onevecmask = ∂mutlivariate_normal_SMLT_rowiter(
            :row_rem_final, Nkrem, col_remrem, T, Ystride, n_col_repsrem, μdim, μstride, track, μmy, :ptrμ, Astride, XP, βstride, Xstride, βdim, μtransposed
        )
        row_loops = quote
            Mkrep, Mkrem = divrem($M, $Mk)
            for rrep ∈ 1:Mkrep
                ptrUdiag = pointer(invdiag); ptrUtri = ptrUtribase
                $row_iter
                $row_increments
            end
            for rrep ∈ 1:Mkrem >> $(VectorizationBase.intlog2(W))
                ptrUdiag = pointer(invdiag); ptrUtri = ptrUtribase
                $row_iter_onevec
                $row_increments_rem
            end
            row_rem_final = Mkrem & $Wm1
            ptrUdiag = pointer(invdiag); ptrUtri = ptrUtribase
            __mask__ = VectorizationBase.mask($T, row_rem_final)
            $row_iter_onevecmask
        end
        push!(q.args, row_loops)
    end
    # Reduce the Mk δ² into a single vector.
    R = Mk2
    while R > 1
        Risodd = isodd(R)
        Rh = R >> 1
        for r ∈ 0:(Rh-1)
            dl = Symbol(:δ²_,r)
            dh = Symbol(:δ²_,r+Rh)
            push!(q.args, :($dl = SIMDPirates.vadd($dl,$dh)))
        end
        Risodd && push!(q.args, Expr(:(=), :δ²_0, :(SIMDPirates.vadd(δ²_0, $(Symbol(:δ²_,R-1))))))
        R = Rh
    end
    push!(q.args, Expr(:(=), :δ²_0, :(SIMDPirates.vmul(SIMDPirates.vbroadcast($V, $(T(-0.5))), δ²_0))))
    if track_L
        loopheader = quote ptrv∂L = pointer(v∂L); ptr∂L = pointer(∂L) end
        loop1body = quote
            VectorizationBase.store!(
                    ptr∂L + p*$size_T, Base.FastMath.sub_fast(SIMDPirates.vsum(SIMDPirates.vload($V, ptrv∂L + p*$(W*size_T))), Base.FastMath.mul_fast($(M isa Symbol ? :($T($M)) : T(M)), VectorizationBase.load(ptr∂L + p*$size_T)))
                )
        end
        if track_μ && μdim == 1 && μtransposed
            push!(loopheader.args, :(ptr∂μ = pointer(∂μ); ptrv∂μ = pointer(v∂μ)))
            push!(loop1body.args, :(VectorizationBase.store!( ptr∂μ + p*$size_T, SIMDPirates.vsum(SIMDPirates.vload($V, ptrv∂μ + p*$(W*size_T))))))
        end
        vsum_L_expr = quote
            $loopheader    
            for p in 0:$(P-1)
                $loop1body
#                ∂L[p] = SIMDPirates.vsum(v∂L[p]) - ∂L[p]# subtract inverse diag of L
            end
            for p in $(P):$(StructuredMatrices.binomial2(P+1)-1)
                VectorizationBase.store!(
                    ptr∂L + p*$size_T, SIMDPirates.vsum(SIMDPirates.vload($V, ptrv∂L + p*$(W*size_T)))
                )
#                ∂L[p] = SIMDPirates.vsum(v∂L[p])
            end
        end
        push!(q.args, vsum_L_expr)
    end
    if track_μ
        if μdim == 1 && !track_L
            vsum_mu_expr = quote
                ptr∂μ = pointer(∂μ); ptrv∂μ = pointer(v∂μ)
                for p in 0:$(P-1)
                    VectorizationBase.store!(
                        ptr∂μ + p*$size_T, SIMDPirates.vsum(SIMDPirates.vload($V, ptrv∂μ + p*$(W*size_T)))
                    )
                    # ∂μ[p] = SIMDPirates.vsum(v∂μ[p])
                end
            end
            push!(q.args, vsum_mu_expr)
        elseif μdim == 0
            push!(q.args, Expr(:(=), :v∂μ_0, :(SIMDPirates.vadd(SIMDPirates.vadd(v∂μ_0,v∂μ_2),SIMDPirates.vadd(v∂μ_1,v∂μ_3)))))
            push!(q.args, Expr(:(=), :∂μ, :(SIMDPirates.vsum(v∂μ_0))))
        end
    end
    if track_X | track_β
        # push!(q.args, :(@show A))
        # push!(q.args, :(@show X))
        f = μmy ?  :(PaddedMatrices.nmul!) : :(LinearAlgebra.mul!)
        track_X && push!(q.args, Expr(:call, f, :∂X, :A, :(β')))
        if track_β
            push!(q.args, Expr(:call, f, :∂β, :(X'), :A))
            # push!(q.args, :(@show ∂β))
            if βdim == 1
                push!(q.args, Expr(:call, :sum!, :∂βv, :∂β))
            end
        end
    end
    if sp
        push!(q.args, :(PaddedMatrices.StackPointer(_sptr),$return_expr))
    else
        push!(q.args, return_expr)
    end
    simplify_expr(q)
end



@generated function ∂Normal(
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,PY},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true)}()
) where {M,P,T,track,PY}
    ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY)
end
@generated function ∂Normal(
    sptr::StackPointer,
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,PY},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true)}()
) where {M,P,T,track,PY}
    ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY)
end
@generated function ∂Normal(
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,PY},
    μ::Tμ,
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true)}()
) where {M,P,T,PY,Tμ,track}
# ) where {M,P,T,track,PY,Tμ}
    if Tμ === T
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, μdim = 0, μstride = 0)
    elseif Tμ <: LinearAlgebra.Adjoint{T,<:AbstractMutableFixedSizePaddedVector}
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, μdim = 1, μstride = 1, μtransposed = true)
    elseif Tμ <: AbstractMutableFixedSizePaddedVector
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, μdim = 1, μstride = 1)
    elseif Tμ <: AbstractMutableFixedSizePaddedMatrix
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, μdim = 2, μstride = Tμ.parameters[4])
    else
        throw("Type of μ = $(Tμ) was not recognized.")
    end
end
@generated function ∂Normal(
    sptr::StackPointer,
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,PY},
    μ::Tμ,
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true)}()
) where {M,P,T,PY,Tμ,track}
# ) where {M,P,T,track,PY,Tμ}
    if Tμ === T
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, μdim = 0, μstride = 0)
    elseif Tμ <: LinearAlgebra.Adjoint{T,<:AbstractMutableFixedSizePaddedVector}
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, μdim = 1, μstride = 1, μtransposed = true)
    elseif Tμ <: AbstractMutableFixedSizePaddedVector
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, μdim = 1, μstride = 1)
    elseif Tμ <: AbstractMutableFixedSizePaddedMatrix
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, μdim = 2, μstride = Tμ.parameters[4])
    else
        throw("Type of μ = $(Tμ) was not recognized.")
    end
end


@generated function ∂Normal(
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,PY},
    X::AbstractMutableFixedSizePaddedMatrix{M,K_,T,PX},
    β::AbstractMutableFixedSizePaddedArray{Sβ,T,Nβ,Pβ},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true,true)}()
# ) where {M,P,T,track,PY,PX,Sβ,Nβ,Pβ,K_}
) where {M,P,T,track,K_,PY,PX,Sβ,Nβ,Pβ}
    @assert Sβ.parameters[1] == K_
    ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_)
end
@generated function ∂Normal(
    sptr::StackPointer,
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,PY},
    X::AbstractMutableFixedSizePaddedMatrix{M,K_,T,PX},
    β::AbstractMutableFixedSizePaddedArray{Sβ,T,Nβ,Pβ},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true,true)}()
# ) where {M,P,T,track,PY,PX,Sβ,Nβ,Pβ,K_}
) where {M,P,T,track,K_,PY,PX,Sβ,Nβ,Pβ}
    @assert Sβ.parameters[1] == K_
    ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_)
end


    # M::Union{Symbol,Integer}, P::Int, track::NTuple{D,Bool}, T::DataType = Float64;
    # βstride::Int = -1, Xstride::Int = -1, Ystride::Int = M, μstride::Int = -1,
    # μdim::Int = -1, sp::Bool = false, βdim::Int = -1,  XP::Int = -1, μtransposed::Bool = false
@generated function ∂Normal_fmadd(
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,PY},
    X::AbstractMutableFixedSizePaddedMatrix{M,K_,T,PX},
    β::AbstractMutableFixedSizePaddedArray{Sβ,T,Nβ,Pβ},
    μ::Tμ,
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true,true,true)}()
# ) where {M,P,T,track,K_,PY,PX,Sβ,Nβ,Pβ,Tμ}
) where {M,P,T,track,K_,PY,PX,Tμ,Sβ,Nβ,Pβ}
    @assert Sβ.parameters[1] == K_
    if Tμ === T
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_, μdim = 0, μstride = 0)
    elseif Tμ <: LinearAlgebra.Adjoint{T,<:AbstractMutableFixedSizePaddedVector}
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_, μdim = 1, μstride = 1, μtransposed = true)
    elseif Tμ <: AbstractMutableFixedSizePaddedVector
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_, μdim = 1, μstride = 1)
    elseif Tμ <: AbstractMutableFixedSizePaddedMatrix
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_, μdim = 2, μstride = Tμ.parameters[4])
    else
        throw("Type of μ = $(Tμ) was not recognized.")
    end
end
@generated function ∂Normal_fmadd(
    sptr::StackPointer,
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,PY},
    X::AbstractMutableFixedSizePaddedMatrix{M,K_,T,PX},
    β::AbstractMutableFixedSizePaddedArray{Sβ,T,Nβ,Pβ},
    μ::Tμ,
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true,true,true)}()
) where {M,P,T,track,K_,PY,PX,Sβ,Nβ,Pβ,Tμ}
# ) where {M,P,T,track,K_,PY,PX,Tμ,Sβ,Nβ,Pβ}
    @assert Sβ.parameters[1] == K_
    if Tμ === T
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_, μdim = 0, μstride = 0)
    elseif Tμ <: LinearAlgebra.Adjoint{T,<:AbstractMutableFixedSizePaddedVector}
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_, μdim = 1, μstride = 1, μtransposed = true)
    elseif Tμ <: AbstractMutableFixedSizePaddedVector
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_, μdim = 1, μstride = 1)
    elseif Tμ <: AbstractMutableFixedSizePaddedMatrix
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_, μdim = 2, μstride = Tμ.parameters[4])
    else
        throw("Type of μ = $(Tμ) was not recognized.")
    end
end



@generated function ∂Normal(
    Y::AbstractMatrix{T},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true)}()
) where {P,T,track}
    M, PY = gensym(:M), gensym(:PY)
    quote
        $M = size(Y,1)
        $PY = $(Y <: Array ? M : :(stride(Y,2)))
        $(∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY))
    end
end
@generated function ∂Normal(
    sptr::StackPointer,
    Y::AbstractMatrix{T},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true)}()
) where {P,T,track}
    M, PY = gensym(:M), gensym(:PY)
    quote
        $M = size(Y,1)
        $PY = $(Y <: Array ? M : :(stride(Y,2)))
        $(∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY))
    end
end
@generated function ∂Normal(
    Y::AbstractMatrix{T},
    μ::Tμ,
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true)}()
# ) where {P,T,Tμ,track}
) where {T,P,Tμ,track}
    M, PY = gensym(:M), gensym(:PY)
    defs_quote = quote
        $M = size(Y,1)
        $PY = $(Y <: Array ? M : :(stride(Y,2)))
    end
    q = if Tμ === T
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, μdim = 0, μstride = 0)
    elseif Tμ <: LinearAlgebra.Adjoint{T,<:AbstractMutableFixedSizePaddedVector}
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, μdim = 1, μstride = 1, μtransposed = true)
    elseif Tμ <: AbstractMutableFixedSizePaddedVector
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, μdim = 1, μstride = 1)
    elseif Tμ <: AbstractMutableFixedSizePaddedMatrix
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, μdim = 2, μstride = Tμ.parameters[4])
    elseif Tμ <: AbstractMatrix
        μstride = gensym(:μstride)
        push!(defs_quote.args, :($μstride = stride(μ,2)))
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, μdim = 2, μstride = μstride)
    else
        throw("Type of μ = $(Tμ) was not recognized.")
    end
    quote
        $defs_quote
        $q
    end
end
@generated function ∂Normal(
    sptr::StackPointer,
    Y::AbstractMatrix{T},
    μ::Tμ,
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true)}()
# ) where {P,T,Tμ,track}
) where {T,P,Tμ,track}
    M, PY = gensym(:M), gensym(:PY)
    defs_quote = quote
        $M = size(Y,1)
        $PY = $(Y <: Array ? M : :(stride(Y,2)))
    end
    q = if Tμ === T
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, μdim = 0, μstride = 0)
    elseif Tμ <: LinearAlgebra.Adjoint{T,<:AbstractMutableFixedSizePaddedVector}
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, μdim = 1, μstride = 1, μtransposed = true)
    elseif Tμ <: AbstractMutableFixedSizePaddedVector
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, μdim = 1, μstride = 1)
    elseif Tμ <: AbstractMutableFixedSizePaddedMatrix
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, μdim = 2, μstride = Tμ.parameters[4])
    elseif Tμ <: AbstractMatrix
        μstride = gensym(:μstride)
        push!(defs_quote.args, :($μstride = stride(μ,2)))
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, μdim = 2, μstride = μstride)
    else
        throw("Type of μ = $(Tμ) was not recognized.")
    end
    quote
        $defs_quote
        $q
    end
end

# @generated function Normal(
    # Y::AbstractMatrix{T},
    # X::AbstractMatrix{T},
    # β::AbstractMutableFixedSizePaddedArray{Sβ,T,Nβ,Pβ},
    # L::AbstractLowerTriangularMatrix{P,T},
    # ::Val{track} = Val{(true,true,true,true)}()
# ) where {P,T,track,PK,Sβ,Nβ,Pβ}
    # K_ = Sβ.parameters[1]
    # M, PY, PX = gensym(:M), gensym(:PY), gensym(:PX)
    # q = multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, βstride = Pβ, Xstride = PX, βdim = Nβ, XP = K_)
    # quote
        # $M = size(Y,1)
        # $PY = $(Y <: Array ? M : :(stride(Y,2)))
        # $PX = $(X <: Array ? M : :(stride(X,R)))
        # $q
    # end
# end

@generated function ∂Normal(
    Y::AbstractMatrix{T},
    X::AbstractMatrix{T},
    β::AbstractMutableFixedSizePaddedArray{Sβ,T,Nβ,Pβ},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true,true)}()
# ) where {P,T,track,Sβ,Nβ,Pβ}
) where {T,P,track,Sβ,Nβ,Pβ}
    M, PY, PX = gensym(:M), gensym(:PY), gensym(:PX)
    K_ = Sβ.parameters[1]
    q = ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_)
    quote
        $M = size(Y,1)
        $PY = $(Y <: Array ? M : :(stride(Y,2)))
        $PX = $(X <: Array ? M : :(stride(X,2)))
        $q
    end
end
@generated function ∂Normal(
    sptr::StackPointer,
    Y::AbstractMatrix{T},
    X::AbstractMatrix{T},
    β::AbstractMutableFixedSizePaddedArray{Sβ,T,Nβ,Pβ},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true,true)}()
# ) where {P,T,track,Sβ,Nβ,Pβ}
) where {T,P,track,Sβ,Nβ,Pβ}
    M, PY, PX = gensym(:M), gensym(:PY), gensym(:PX)
    K_ = Sβ.parameters[1]
    q = ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_)
    quote
        $M = size(Y,1)
        $PY = $(Y <: Array ? M : :(stride(Y,2)))
        $PX = $(X <: Array ? M : :(stride(X,2)))
        $q
    end
end


    # M::Union{Symbol,Integer}, P::Int, track::NTuple{D,Bool}, T::DataType = Float64;
    # βstride::Int = -1, Xstride::Int = -1, Ystride::Int = M, μstride::Int = -1,
    # μdim::Int = -1, sp::Bool = false, βdim::Int = -1,  XP::Int = -1, μtransposed::Bool = false
@generated function ∂Normal_fmadd(
    Y::AbstractMatrix{T},
    X::AbstractMatrix{T},
    β::AbstractMutableFixedSizePaddedArray{Sβ,T,Nβ,Pβ},
    μ::Tμ,
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true,true,true)}()
) where {P,T,track,Tμ,Sβ,Nβ,Pβ}
# ) where {T,P,track,Tμ,Sβ,Nβ,Pβ}
    M, PY, PX = gensym(:M), gensym(:PY), gensym(:PX)
    K_ = Sβ.parameters[1]
    defs_quote = quote
        $M = size(Y,1)
        $PY = $(Y <: Array ? M : :(stride(Y,2)))
        $PX = $(X <: Array ? M : :(stride(X,2)))
    end
    q = if Tμ === T
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_, μdim = 0, μstride = 0)
    elseif Tμ <: LinearAlgebra.Adjoint{T,<:AbstractVector}
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_, μdim = 1, μstride = 1, μtransposed = true)
    elseif Tμ <: AbstractMutableFixedSizePaddedVector
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_, μdim = 1, μstride = 1)
    elseif Tμ <: AbstractMutableFixedSizePaddedMatrix
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_, μdim = 2, μstride = Tμ.parameters[4])
    elseif Tμ <: AbstractMatrix
        μstride = gensym(:μstride)
        push!(defs_quote.args, :($μstride = stride(μ,2)))
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = false, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_, μdim = 2, μstride = μstride)
    else
        throw("Type of μ = $(Tμ) was not recognized.")
    end
    quote
        $defs_quote
        $q
    end
end
@generated function ∂Normal_fmadd(
    sptr::StackPointer,
    Y::AbstractMatrix{T},
    X::AbstractMatrix{T},
    β::AbstractMutableFixedSizePaddedArray{Sβ,T,Nβ,Pβ},
    μ::Tμ,
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true,true,true)}()
# ) where {P,T,track,Sβ,Nβ,Pβ,Tμ}
) where {T,P,track,Tμ,Sβ,Nβ,Pβ}
    M, PY, PX = gensym(:M), gensym(:PY), gensym(:PX)
    K_ = Sβ.parameters[1]
    defs_quote = quote
        $M = size(Y,1)
        $PY = $(Y <: Array ? M : :(stride(Y,2)))
        $PX = $(X <: Array ? M : :(stride(X,2)))
    end
    q = if Tμ === T
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_, μdim = 0, μstride = 0)
    elseif Tμ <: LinearAlgebra.Adjoint{T,<:AbstractVector{T}}
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_, μdim = 1, μstride = 1, μtransposed = true)
    elseif Tμ <: AbstractMutableFixedSizePaddedVector
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_, μdim = 1, μstride = 1)
    elseif Tμ <: AbstractMutableFixedSizePaddedMatrix
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_, μdim = 2, μstride = Tμ.parameters[4])
    elseif Tμ <: AbstractMatrix
        μstride = gensym(:μstride)
        push!(defs_quote.args, :($μstride = stride(μ,2)))
        ∂multivariate_normal_SMLT_quote(M, P, track, T, sp = true, Ystride = PY, Xstride = PX, βstride = Pβ, βdim = Nβ, XP = K_, μdim = 2, μstride = μstride)
    else
        throw("Type of μ = $(Tμ) was not recognized.")
    end
    quote
        $defs_quote
        $q
    end
end

push!(DISTRIBUTION_DIFF_RULES, :Normal_fmadd)
push!(FMADD_DISTRIBUTIONS, :Normal)

function ldnorm!(dy, dm, dl, b, y, m, l)
    b .= m .- y
    LAPACK.trtrs!('L', 'N', 'N', l, b)
    lp = dot(b, b); dy .= b
    LAPACK.trtrs!('L', 'T', 'N', l, dy)
    dm .= 0
    @inbounds for n ∈ 1:size(y,2)
        @simd for p ∈ 1:size(y,1)
            dm[p] -= dy[p,n]
        end
    end
    BLAS.gemm!('N', 'T', 1.0, dy, b, 0.0, dl)
    N = size(y,2); lp *= -0.5
    @inbounds for p ∈ 1:size(dl,1)
        dl[p,p] -= N * l[p,p]
        lp -= N * log(l[p,p])
    end
    lp
end

