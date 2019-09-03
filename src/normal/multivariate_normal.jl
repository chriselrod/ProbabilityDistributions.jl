

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
            $(Symbol(:∂μ_,k)) = PtrVector{$P,$T,$R,$R}( ptr_δ + $(sizeof(T)*R*(k-1)))
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
    R::Int, C::Int, K::Union{Symbol,Int}, T::DataType, Bstride::Int, Bsym::Symbol, μdim::Int, μstride::Int,
    μsym::Union{Symbol,Nothing} = :μptr, maskload::Bool = true, μmy::Bool = true
)
    size_T = sizeof(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(R, T)
    V = Vec{W,T}
    Wm1 = W - 1
    Riter = R >> Wshift
    Rrem = R & Wm1
    mask = VectorizationBase.mask_from_remainder(T, Rrem)
    if K isa Symbol
        q = quote
            BsymK = $Bsym + $(size_T*Bstride)*$K
        end
        μdim == 2 && push!(q.args, :(μsumK = $μsym + $(size_T*μstride)*$K))
    else
        q = quote
            BsymK = $Bsym + $(size_T*Bstride*K)
        end
        μdim == 2 && push!(q.args, :(μsumK = $μsym + $(size_T*μstride*K)))
    end
    if μsym isa Symbol
        for c ∈ 0:C-1            
            vμ_c = μdim == 0 ? μsym : Symbol(:vμ_, c)
            if μdim == 1
                if K isa Symbol
                    push!(q.args, Expr(:(=), vμ_c, :(SIMDPirates.vbroadcast($V, $μsym + $(size_T*μstride)*($c+$K)))))
                else
                    push!(q.args, Expr(:(=), vμ_c, :(SIMDPirates.vbroadcast($V, $μsym + $(size_T*(c + K)*μstride)))))
                end
            end
            if μdim == 1 || μdim == 0
                for r ∈ 0:Riter-1
                    yloadexpr = :(SIMDPirates.vload($V, BsymK + $(size_T * (r*W + c*Bstride))))
                    if μmy
                        push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vsub($vμ_c, $yloadexpr)))
                    else
                        push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vsub($yloadexpr, $vμ_c)))
                    end
                end
                if Rrem > 0
                    # Only need to mask if we're on last column
                    if maskload && c == C-1
                        yloadexpr = :(SIMDPirates.vload($V, BsymK + $(size_T * (Riter*W + c*Bstride)), $mask ))
                    else
                        yloadexpr = :(SIMDPirates.vload($V, BsymK + $(size_T * (Riter*W + c*Bstride)) ))
                    end
                    if μmy
                        push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vsub($vμ_c, $yloadexpr)))
                    else
                        push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vsub($yloadexpr, $vμ_c)))
                    end
                end
            elseif μdim == 2
                for r ∈ 0:Riter-1
                    μloadexpr = :(SIMDPirates.vload($V, usymK + $(size_T * (r*W + c*μstride))))
                    yloadexpr = :(SIMDPirates.vload($V, BsymK + $(size_T * (r*W + c*Bstride))))
                    if μmy
                        push!(q.args, Expr(:(=), Symbol(:A_,r,:_,c), Expr(:call, :(SIMDPirates.vsub), μloadexpr, yloadexpr)))
                    else
                        push!(q.args, Expr(:(=), Symbol(:A_,r,:_,c), Expr(:call, :(SIMDPirates.vsub), yloadexpr, μloadexpr)))
                    end        
                end
                if Rrem > 0
                    # Only need to mask if we're on last column
                    if maskload && c == C-1
                        μloadexpr = :(SIMDPirates.vload($V, μsymK + $(size_T * (Riter*W + c*μstride)), $mask))
                        yloadexpr = :(SIMDPirates.vload($V, BsymK + $(size_T * (Riter*W + c*Bstride)), $mask))
                    else
                        μloadexpr = :(SIMDPirates.vload($V, μsymK + $(size_T * (Riter*W + c*μstride)) ))
                        yloadexpr = :(SIMDPirates.vload($V, BsymK + $(size_T * (Riter*W + c*Bstride)) ))
                    end
                    if μmy
                        push!(q.args, Expr(:(=), Symbol(:A_,Riter,:_,c)), Expr(:call,:(SIMDPirates.vsub), μloadexpr, yloadexpr))
                    else
                        push!(q.args, Expr(:(=), Symbol(:A_,Riter,:_,c)), Expr(:call,:(SIMDPirates.vsub), yloadexpr, μloadexpr))
                    end
                end
            else #μ assumed not to exist
                for r ∈ 0:Riter-1
                    push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vload($V, BsymK + $(size_T * (r*W + c*Bstride)) ) ))
                end
                if Rrem > 0
                    # Only need to mask if we're on last column
                    if maskload && c == C-1
                        push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vload($V, BsymK + $(size_T * (Riter*W + c*Bstride)), $mask ) ))
                    else
                        push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vload($V, BsymK + $(size_T * (Riter*W + c*Bstride)) ) ))
                    end
                end
            end
        end
    else
        for c ∈ 0:C-1
            for r ∈ 0:Riter-1
                push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vload($V, BsymK + $(size_T * (r*W + c*Bstride)) ) ))
            end
            if Rrem > 0
                # Only need to mask if we're on last column
                if maskload && c == C-1
                    push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vload($V, BsymK + $(size_T * (Riter*W + c*Bstride)), $mask ) ))
                else
                    push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vload($V, BsymK + $(size_T * (Riter*W + c*Bstride)) ) ))
                end
            end
        end
    end
    q
end
function loadδfnmadd_quote(
    R::Int, C::Int, K::Union{Symbol,Int}, T::DataType, Ystride::Int, Xstride::Int, βstride::Int, μdim::Int,
    ysym::Symbol = :ptrY, xsym::Symbol = :ptrX, βsym::Symbol = :ptrβ, maskload::Bool = true, μmy::Bool = true, XP::Int = -1
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
    if K isa Symbol
        q = quote
            BsymK = $ysym + $(size_T*Ystride)*$K
        end
        μdim == 2 && push!(q.args, :(βsymK = $βsym + $(size_T*βstride)*$K))
    else
        q = quote
            BsymK = $ysym + $(size_T*Ystride*K)
        end
        μdim == 2 && push!(q.args, :(βsymK = $βsym + $(size_T*βstride*K)))
    end
    # if μdim == 1, we assume that vμ_r has been computed for r = 0,...,Riter
    if μdim == 2
        for r ∈ 0:Riterl
            if r == Riterl && maskload
                xloadexpr = :(SIMDPirates.vload($V, $xsym + ($size_T * $(r*W)),$mask))
            else
                xloadexpr = :(SIMDPirates.vload($V, $xsym + ($size_T * $(r*W))))
            end
            push!(q.args, Expr(:(=), Symbol(:vx_,r), xloadexpr))
        end
    end
    f = μmy ? :(SIMDPirates.vfmsub) : :(SIMDPirates.vfnmadd)
    for c ∈ 0:C-1
        if μdim == 1
            for r ∈ 0:Riterl
                vμ_r = Symbol(:vμ_,r)
                if r == Riterl && maskload && c == C - 1
                    yloadexpr = :(SIMDPirates.vload($V, BsymK + $(size_T * (r*W + c*Ystride)),$mask))
                else
                    yloadexpr = :(SIMDPirates.vload($V, BsymK + $(size_T * (r*W + c*Ystride))))
                end
                if μmy
                    push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vsub($vμ_r, $yloadexpr)))
                else
                    push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vsub($yloadexpr, $vμ_r)))
                end
                # push!(q.args, :(@show getfield.($(Symbol(:A_,r,:_,c)), :value)))
            end
        elseif μdim == 2
            # we load first block, before the XP loop
            # if  μmy, that is X*β - Y # aka SIMDPirates.vfmsub  # loop vmuladd to this answer
            # if !μmy, that is Y - X*β # aka SIMDPirates.vfnmadd # loop vfnmadd to this answer
            β_c = Symbol(:β_,c)
            push!(q.args, Expr(:(=), β_c, :(SIMDPirates.vbroadcast($V, βsymk + ($size_T * $(c*βstride))))))
            for r ∈ 0:Riterl
                if r == Riterl && maskload && c == C-1
                    yloadexpr = :(SIMDPirates.vload($V, YsymK + $(size_T * (r*W + c*Ystride)),$mask))
                else
                    yloadexpr = :(SIMDPirates.vload($V, YsymK + $(size_T * (r*W + c*Ystride))))
                end
                push!(q.args, Expr(:(=), Symbol(:A_,r,:_,c), Expr(:call, f, Symbol(:vx_,r), β_c, yloadexpr)))
            end
        end
    end
    f = μmy ? :(SIMDPirates.vmuladd) : :(SIMDPirates.vfnmadd)
    if μdim == 2
        p = gensym(:p)
        loopbody = quote end
        for r ∈ 0:Riterl
            if r == Riterl && maskload
                xloadexpr = :(SIMDPirates.vload($V, $xsym + $size_T * ($(r*W) + $p*$Xstride)),$mask)
            else
                xloadexpr = :(SIMDPirates.vload($V, $xsym + $size_T * ($(r*W) + $p*$Xstride)))
            end
            push!(loopbody.args, Expr(:(=), Symbol(:vx_,r), xloadexpr))
        end
        for c ∈ 0:C-1
            β_c = Symbol(:β_,c)
            push!(loopbody.args, Expr(:(=), β_c, :(SIMDPirates.vbroadcast($V, βsymk + $size_T * ($(c*βstride)+$p)))))
            for r ∈ 0:Riterl
                push!(loopbody.args, Expr(:(=), Symbol(:A_,r,:_,c), Expr(:call, f, Symbol(:vx_,r), β_c, Symbol(:A_,r,:_,c))))
            end
        end
        loop = quote
            for $p ∈ 1:$(XP-1)
                $loopbody
            end
        end
        push!(q.args, loop)
    end
    q
end
function Xβ_load_quote(
    R::Int, T::DataType, Xstride::Int, βstride::Int, μmy::Bool = true, XP::Int = -1, 
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
    push!(q.args, Expr(:(=), :vβ, :(SIMDPirates.vbroadcast($V, $βsym))))
    for r ∈ 0:Riterl
        if r == Riterl && maskload && XP == 1
            xloadexpr = :(SIMDPirates.vload($V, $xsym + ($size_T * $(r*W)),$mask))
        else
            xloadexpr = :(SIMDPirates.vload($V, $xsym + ($size_T * $(r*W))))
        end
        push!(q.args, Expr(:(=), Symbol(:vμ_,r), Expr(:call, :(SIMDPirates.vmul), xloadexpr, :vβ)))
    end
    p = gensym(:p)
    loopbody = quote
        vβ = SIMDPirates.vbroadcast($V, $βsym + $(size_T*βstride)*$p)
    end
    for r ∈ 0:Riterl
        if r == Riterl && maskload
            xloadexpr = :(SIMDPirates.vload($V, $xsym + $size_T * ($(r*W) + $p*$Xstride)),$mask)
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



function mutlivariate_normal_SMLT_rowiter(
    Mk::Int, Nk::Int, col_rem::Int, T::DataType, Ystride::Int, n_col_reps::Int, μdim::Int, μstride::Int, μsym::Symbol = :μptr, XP::Int = -1, Xstride = -1, βstride = -1
)
    #TODO: NOTE, WE DO NEED TO STORE THE SOLUTION MATRIX (at least 1 row set amd up to the last column block)
    # because this is used for calculating the next iteration.

    N = Nk * n_col_reps + col_rem
    
    size_T = sizeof(T)
    row_iter = (μdim == 1 && XP > 0) ? Xβ_load_quote(Mk, T, Xstride, βstride, true, XP) : quote end
    if col_rem > 0
        if XP > 0
            loadδ_expr = loadδfnmadd_quote(
                Mk, col_rem, 0, T, Ystride, Xstride, βstride, μdim,
                :ptrY, :Xptr, :βptr, true, true, XP
            )
        else
            loadδ_expr = loadδ_quote(Mk, col_rem, 0, T, Ystride, :ptrY, μdim, μstride, μsym)
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
                Mk, Nk, :K, T, Ystride, Xstride, βstride, μdim,
                :ptrY, :Xptr, :βptr, true, true, XP
            )
        else
            loadδ_expr = loadδ_quote(Mk, Nk, :K, T, Ystride, :ptrY, μdim, μstride, μsym)
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
                Mk, Nk, col_rem, T, Ystride, Xstride, βstride, μdim,
                :ptrY, :Xptr, :βptr, true, true, XP
            )
        else
            loadδ_expr = loadδ_quote(Mk, Nk, col_rem, T, Ystride, :ptrY, μdim, μstride, μsym)
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
function multivariate_normal_SMLT_quote(M::Union{Symbol,Integer}, P, track::NTuple{D,Bool}, μdim::Int, μstride::Int, sp::Bool, Ystride = M, T::DataType = Float64) where {D}
    if D == 4
        track_Y, track_X, track_β, track_L = track
        track_μ = track_X | track_β
        Xstride = μstride
    else
        track_X = track_β = false
        Xstride = -1
        if D == 3
            track_Y, track_μ, track_L = track
        elseif D == 2
            track_Y, track_L = track
            track_μ = false
        else
            throw("Unknown number of arguments ($D) to normal.")
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
        invdiag = $(sp ? :(PtrVector{$P,$T,$P,$P,true}(pointer(sptr,$T))) : :(MutableFixedSizePaddedVector{$P,$T,$P,$P}(undef)))
        $(macroexpand(LoopVectorization, quote
                      @vvectorize $T for p ∈ 1:$P
                      $loopbody
                      end
                      end))
        ptrY = pointer(Y)
        ptrUtribase = pointer(L) + $(P*size_T)
    end
    track_L && push!(q.args, :(δ²_0 = SIMDPirates.vmul(δ²_0, SIMDPirates.vbroadcast($V,$(M isa Integer ? T(2M) : :($(T(2))*$T($M)))))))
    D == 4 && push!(q.args, :(ptrX = pointer(X); ptrβ = pointer(β)))
    Aquote = quote
        A = $(sp ? :(PtrMatrix{$Mk,$P,$T,$Mk}(pointer(sptr,$T) + $(VectorizationBase.align(size_T*P)))) : :(MutableFixedSizePaddedMatrix{$Mk,$P,$T,$Mk}(undef)))
        ptrA = pointer(A)
    end
    total_col_iterations > 1 && push!(q.args, Aquote)
    if μdim == 0
        push!(q.args, Expr(:(=), :μptr, :(SIMDPirates.vbroadcast($V, μ))))
    elseif μdim > 0
        push!(q.args, Expr(:(=), :μptr, :(pointer(μ))))
    end
    if M isa Integer
        n_row_reps, row_rem = divrem(M, Mk)
        total_row_iterations = n_row_reps + (row_rem > 0)
        Mk1 = n_row_reps == 0 ? row_rem : Mk
        row_iter = mutlivariate_normal_SMLT_rowiter(
            Mk1, Nk, col_rem, T, Ystride, n_col_reps, μdim, μstride, :μptr
        )
        if n_row_reps > 1
            row_loops = quote
                for rrep ∈ 1:$n_row_reps
                    ptrUdiag = pointer(invdiag)
                    ptrUtri = ptrUtribase#pointer(B) + $(size_T * N)
                    $row_iter
                    $(μdim == 2 ? :(ptrY += $(size_T*Mk); ptrμ += $(size_T*Mk)) :  :(ptrY += $(size_T*Mk)))
                    #ptrA += $(size_T*Mk)
                end
            end
            push!(q.args, row_loops)
        else
            push!(q.args, :(ptrUdiag = pointer(invdiag)))
            push!(q.args, :(ptrUtri = ptrUtribase))
            push!(q.args, row_iter)
            if total_row_iterations == 2 # then n_row_reps == 1 and row_rem > 0
                push!(q.args, mutlivariate_normal_SMLT_rowiter( row_rem, Nk, col_rem, T, Ystride, n_col_reps, μdim, μstride, :μptr ))
            end
        end
    else # Unknown number of iterations.
        row_iter = mutlivariate_normal_SMLT_rowiter(
            Mk, Nk, col_rem, T, Ystride, n_col_reps, μdim, μstride, :μptr
        )
        Wrem, Mkrem, Nkrem = StructuredMatrices.div_triangle_blocking_structure(W, P, T)
        n_col_repsrem, col_remrem = divrem(P, Nkrem)
        row_iter_onevec = mutlivariate_normal_SMLT_rowiter(
            W, Nkrem, col_remrem, T, Ystride, n_col_repsrem, μdim, μstride, :μptr
        )
        row_iter_onevecmask = mutlivariate_normal_SMLT_rowiter(
            :row_rem_final, Nkrem, col_remrem, T, Ystride, n_col_repsrem, μdim, μstride, :μptr
        )
        row_loops = quote
            Mkrep, Mkrem = divrem($M, $Mk)
            for rrep ∈ 1:Mkrep
                ptrUdiag = pointer(invdiag)
                ptrUtri = ptrUtribase#pointer(B) + $(size_T * N)
                $row_iter
                (μdim == 2 ? :(ptrY += $(size_T*Mk); μptr += (size_T*Mk)) : :(ptrY += $(size_T*Mk)))
                #ptrA += $(size_T*Mk)
            end
            for rrep ∈ 1:Mkrem >> $(VectorizationBase.intlog2(W))
                ptrUdiag = pointer(invdiag)
                ptrUtri = ptrUtribase#pointer(B) + $(size_T * N)
                $row_iter_onevec
                (μdim == 2 ? :(ptrY += $(size_T*W); μptr += (size_T*W)) : :(ptrY += $(size_T*W)))
                #ptrA += $(size_T*W)
            end
            row_rem_final = Mkrem & $Wm1
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
    push!(q.args, :δ²_0)
    simplify_expr(q)
end


@generated function Normal(
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,MP},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true)}()
) where {M,P,T,track,MP}
    multivariate_normal_SMLT_quote(M, P, track, -1, -1, false, MP, T)
end
@generated function Normal(
    sp::StackPointer,
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,MP},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true)}()
) where {M,P,T,track,MP}
    multivariate_normal_SMLT_quote(M, P, track, -1, -1, true, MP, T)
end
@generated function Normal(
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,MP},
    μ::T,
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true)}()
) where {M,P,T,track,MP}
    multivariate_normal_SMLT_quote(M, P, track, 0, 0, false, MP, T)
end
@generated function Normal(
    sptr::StackPointer,
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,MP},
    μ::T,
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true)}()
) where {M,P,T,track,MP}
    multivariate_normal_SMLT_quote(M, P, track, 0, 0, true, MP, T)
end
@generated function Normal(
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,MP},
    μ::AbstractMutableFixedSizePaddedVector{P,T},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true)}()
) where {M,P,T,MP,track}
#) where {M,P,T,track,MP}
    multivariate_normal_SMLT_quote(M, P, track, 1, 1, false, MP, T)
end
@generated function Normal(
    sptr::StackPointer,
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,MP},
    μ::AbstractMutableFixedSizePaddedVector{P,T},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true)}()
) where {M,P,T,track,MP}
#) where {M,P,T,MP,track}
    multivariate_normal_SMLT_quote(M, P, track, 1, 1, true, MP, T)
end
@generated function Normal(
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,MP},
    μ::AbstractMutableFixedSizePaddedMatrix{M,P,T,MM},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true)}()
) where {M,P,T,track,MP,MM}
    multivariate_normal_SMLT_quote(M, P, track, 2, MM, false, MP, T)
end
@generated function Normal(
    sptr::StackPointer,
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,MP},
    μ::AbstractMutableFixedSizePaddedMatrix{M,P,T,MM},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true)}()
) where {M,P,T,track,MP,MM}
    multivariate_normal_SMLT_quote(M, P, track, 2, MM, true, MP, T)
end


function track_mu_store(Mk,Nk,T,μdim,μmy,W,Wshift,μstride,track_Y)
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
        for c ∈ 0:(Nk-1), m ∈ 0:((Mk>>Wshift)-1)
            pm = Symbol(:∂μ_,iter & 3)
            push!(row_iter.args, Expr(:(=), pm, :(SIMDPirates.$f($pm, $(Symbol(:A_,m,:_,c))))))
            iter += 1
        end
    elseif μdim == 1
        for c ∈ 0:(Nk-1)
            mc = Symbol(:vμ_,c)
            push!(row_iter.args, Expr(:(=), mc, :(SIMDPirates.vload($V, ptrv∂μ + $(c*W*size_T)))))
        end
        for m ∈ 0:Riterl, c ∈ 0:(Nk-1)
            mc = Symbol(:vμ_,c)
            push!(row_iter.args, Expr(:(=), mc, :(SIMDPirates.$f($mc, $(Symbol(:A_,m,:_,c))))))
        end
        for c ∈ 0:(Nk-1)
            mc = Symbol(:vμ_,c)
            push!(row_iter.args, :(SIMDPirates.vstore!(ptrv∂μ + $(c*W*size_T), $mc)))
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

function loop_pointer_increments(track_Y, track_μ, track_X, track_β, track_L, Nk, Nk2, K, size_T, W, Astride, μstride, μdim)
    b2Nk = StructuredMatrices.binomial2(Nk)
    loop_ptr_increments = quote
        ptrLdiag -= $(size_T*Nk2)
        ptrLtri -= $size_T*($Nk*K + $b2Nk)
    end
    if track_Y || (track_μ && μdim == 2) || track_X || track_β
        push!(loop_ptr_increments.args, Expr(:(-=), :ptrA_rev, Astride isa Symbol ? :($Astride*$(Nk*size_T)) : Nk*Astride*size_T))
        if track_Y && track_μ && μdim == 2  # then 
            push!(loop_ptr_increments.args, Expr(:(-=), :ptr∂μ, μstride isa Symbol ? :($μstride*$(Nk*size_T)) : Nk*μstride*size_T))
        end
    end
    if track_μ && !(track_X | track_β)
        push!(loop_ptr_increments.args, Expr(:(-=), :ptrv∂μ, Nk*size_T*W))
    end
    if track_L
        push!(loop_ptr_increments.args, :(ptrv∂Ldiag -= $(size_T*W*Nk); ptrv∂Ltri -= $(size_T*W)*($Nk*K+$b2Nk)))
    end
    loop_ptr_increments
end

function ∂mutlivariate_normal_SMLT_rowiter(
    Mk::Int, Nk::Int, col_rem::Int, T::DataType, Ystride::Int, n_col_reps::Int, μdim::Int, μstride::Int, track::NTuple{D,Bool}, μmy::Bool, μsym::Symbol = :μptr,
    Astride::Int = Ystride, XP::Int = -1, Xstride::Int=-1, βstride::Int=-1
) where {D}
    if D == 4
        track_Y, track_X, track_β, track_L = track
        track_μ = track_X | track_β
        Xstride = μstride
        no_Xorβ = !track_μ
    else
        no_Xorβ = true
        track_X = track_β = false
        Xstride = -1
        if D == 3
            track_Y, track_μ, track_L = track
        elseif D == 2
            track_Y, track_L = track
            track_μ = false
        else
            throw("Unknown number of arguments ($D) to normal.")
        end
    end
    W, Wshift = VectorizationBase.pick_vector_width_shift(Mk, T)
    V = Vec{W,T}
    #TODO: NOTE, WE DO NEED TO STORE THE SOLUTION MATRIX (at least 1 row set amd up to the last column block)
    # because this is used for calculating the next iteration.
    N = Nk * n_col_reps + col_rem
    size_T = sizeof(T)
    if μdim == 1 && XP > 0
        row_iter = Xβ_load_quote(Mk, T, Xstride, βstride, μmy, XP)
    else
        row_iter = quote end
    end
    store_μ = track_μ && XP == -1
    if col_rem > 0
        if XP > 0
            loadδ_expr = loadδfnmadd_quote(
                Mk, col_rem, 0, T, Ystride, Xstride, βstride, μdim,
                :ptrY, :Xptr, :βptr, true, true, XP
            )
        else
            loadδ_expr = loadδ_quote(Mk, col_rem, 0, T, Ystride, :ptrY, μdim, μstride, μsym, true, μmy)
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
                Mk, Nk, :K, T, Ystride, Xstride, βstride, μdim,
                :ptrY, :Xptr, :βptr, true, true, XP
            )
        else
            loadδ_expr = loadδ_quote(Mk, Nk, :K, T, Ystride, :ptrY, μdim, μstride, μsym, true, μmy)
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
             col_remloadδ_expr = loadδfnmadd_quote(
                Mk, Nk, col_rem, T, Ystride, Xstride, βstride, μdim,
                :ptrY, :Xptr, :βptr, true, true, XP
            )
        else
            loadδ_expr = loadδ_quote(Mk, Nk, col_rem, T, Ystride, :ptrY, μdim, μstride, μsym, true, μmy)
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
    if track_μ && (μdim == 1 || (!track_Y && μdim == 2)) && no_Xorβ
        push!(row_iter.args, :(ptrv∂μ = ptrv∂μbase))
    end
    if track_L
        push!(row_iter.args, :(ptrv∂Ltri = ptrv∂Ltribase; ptrv∂Ldiag = ptrv∂Ldiagbase))
    end
    if col_rem > 0
        row_iter_rev = StructuredMatrices.A_rdiv_L_kernel_quote(
            Mk, col_rem, col_rem, T, Astride, Ystride, false, true,
            Bsym = :ptrA_rev, Asym = :ptrA_rev, Ltrisym = :ptrLtri, Ldiagsym = :ptrLdiag,
            loadB = true, storeA = true, calc_product = track_L ? N : 0
        )
        fullcols = Nk * n_col_reps
        # handle following in A_rdiv_L_quote
        append!(row_iter.args, row_iter_rev.args)
        store_μ && push!(row_iter.args, track_mu_store(Mk,col_rem,T,μdim,μmy,W,Wshift,μstride,track_Y))
        push!(row_iter.args, loop_pointer_increments(track_Y, track_μ, track_X, track_β, track_L, Nk, col_rem, col_rem, size_T, W, Astride, μstride, μdim))
        base_K = col_rem
        KmZ = false
    else
        base_K = 0
        KmZ = true
    end
    loop_ptr_increments = loop_pointer_increments(track_Y, track_μ, track_X, track_β, track_L, Nk, Nk, :K, size_T, W, Astride, μstride, μdim)
    if n_col_reps > 1
        iterquote = StructuredMatrices.A_rdiv_L_kernel_quote(
            Mk, Nk, :K, T, Astride, Ystride, false, true,
            Bsym = :ptrA_rev, Asym = :ptrA_rev, Ltrisym = :ptrLtri, Ldiagsym = :ptrLdiag,
            loadB = true, storeA = true, calc_product = track_L ? N : 0
        )
        store_μ && push!(iterquote.args, track_mu_store(Mk,Nk,T,μdim,μmy,W,Wshift,μstride,track_Y))
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
            Mk, Nk, N, T, Astride, Ystride, false, true,
            Bsym = :ptrA_rev, Asym = :ptrA_rev, Ltrisym = :ptrLtri, Ldiagsym = :ptrLdiag,
            loadB = true, storeA = true, calc_product = track_L ? N : 0
        )
        push!(row_iter.args, row_iter_rev_single)
        store_μ && push!(row_iter.args, track_mu_store(Mk,Nk,T,μdim,μmy,W,Wshift,μstride,track_Y))
    end
    row_iter
end

## StructuredMatrices.jl Lower Triangular (SMLT) quote
## M is the sample size
function ∂multivariate_normal_SMLT_quote(
    M::Union{Symbol,Integer}, P::Int, track::NTuple{D,Bool}, μdim::Int, μstride::Int, sp::Bool, Ystride::Int = M, T::DataType = Float64, XP::Int = -1, βstride::Int = -1
) where {D}
    if D == 4
        track_Y, track_X, track_β, track_L = track
        track_μ = track_X | track_β
        Xstride = μstride
        no_Xorβ = !track_μ
    else
        track_X = track_β = false
        no_Xorβ = true
        Xstride = -1
        if D == 3
            track_Y, track_μ, track_L = track
        elseif D == 2
            track_Y, track_L = track
            track_μ = false
        else
            throw("Unknown number of arguments ($D) to normal.")
        end
    end
    q = quote end
    maxM = M isa Symbol ? typemax(Int) : M
    W, Mk, Nk = StructuredMatrices.div_ul_blocking_structure(maxM, P, T)
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
    # need to allocate invdiag, ∂Y, ∂μ, ∂L, and v∂L
    # also writing into ptrA
    # Q: do we increment ptrA alongside Y?
    # Q: if yes, is ptrA ∂Y or ∂μ ?
    invdiagL = VectorizationBase.align(P, W)
    array_allocations = sp ? quote _sptr = pointer(sptr,$T) end : quote end
    if track_L
        ∂LL = StructuredMatrices.binomial2(P+1)
        if sp
            sptroff = VectorizationBase.align(∂LL, W)
            push!(array_allocations.args, :(∂L = StructuredMatrices.PtrLowerTriangularMatrix{$P,$T,$sptroff}(_sptr)))
            push!(array_allocations.args, :(_sptr += $(sptroff*size_T)))
        else
            push!(array_allocations.args, :(v∂L = StructuredMatrices.MutableLowerTriangularMatrix{$P,$V,$∂LL}(undef)))
            push!(array_allocations.args, :(∂L = StructuredMatrices.MutableLowerTriangularMatrix{$P,$T,$((∂LL+Wm1)&~Wm1)}(undef)))
        end
        push!(array_allocations.args, :(invdiag = PtrVector{$P,$T,$P,$P}(pointer(∂L))))
    elseif !sp
        push!(array_allocations.args, :(invdiag = MutableFixedSizePaddedVector{$P,$T,$invdiagL,$invdiagL}(undef)))
    end
    row_increments = quote
        ptrY += $(size_T*Mk)
    end
    row_increments_rem = quote
        ptrY += $(size_T*W)
    end
    return_expr = Expr(:tuple,:δ²_0)
    # this increments _sptr
    sptroff = 0
    local Astride::Union{Int,Symbol}
    if sp # define sptroff, the offset of the sptr relative to the end of the last returned object (where a non-returned object would start)
        if !track_Y && !track_μ# don't need to track A
            Aquote = quote
                A = PtrMatrix{$Mk,$P,$T,$Mk}(_sptr)
                ptrA = pointer(A)
            end
            sptroff = VectorizationBase.align(Mk*P*size_T)
            push!(array_allocations.args, Aquote)
            μmy = true
            Astride = Mk
        else
            # if track_X | track_β
            # end
            if track_Y # we do need to track A; A will be partialY
                push!(return_expr.args, :A)
                push!(row_increments.args, :(ptrA += $(size_T*Mk)))
                push!(row_increments_rem.args, :(ptrA += $(size_T*W)))
                if M isa Integer # Y
                    Aquote = quote
                        A = PtrMatrix{$M,$P,$T,$M}(_sptr)
                        ptrA = _sptr
                        _sptr += $(VectorizationBase.align(M*P*size_T))
                    end
                else#if M is a Symbol
                    Aquote = quote
                        A = DynamicPtrMatrix{$T}(_sptr, ($M,$P), $M)
                        ptrA = _sptr
                        _sptr += VectorizationBase.align($(size_T*P) * $M)
                    end
                end
                Astride = M
                if track_μ
                    if track_X | track_β
                        if track_X
                            push!(return_expr.args, :∂X)                                  
                            if M isa Integer
                                push!(Aquote.args, :(∂X = PtrMatrix{$M,$XP,$T,$M}(_sptr)))
                            else
                                push!(Aquote.args, :(∂X = DynamicPtrMatrix{$T}(_sptr, ($M,$XP), $M)))
                            end
                            push!(Aquote.args, :( ptr∂X = _sptr; _sptr += $(VectorizationBase.align(size_T * M*XP)) ))
                        end
                        if track_β
                            push!(Aquote.args, :(∂β = PtrMatrix{$XP,$P,$T,$XP}(_sptr); ptr∂β = _sptr))
                            alignβoffset = VectorizationBase.align(size_T * XP * P)
                            if μdim == 1
                                push!(return_expr.args, :∂βv)
                                push!(Aquote.args, :(_sptr += $(VectorizationBase.align(size_T * XP))))
                                sptroff += alignβoffset - VectorizationBase.align(size_T * XP)
                                push!(Aquote.args, Expr(:(=), :∂βv, :(PtrVector{$P,$T,$P,$P}(ptr∂β))))
                            else
                                push!(return_expr.args, :∂β)
                                push!(Aquote.args, :(_sptr += $alignβoffset))
                            end
                        end
                        push!(row_increments.args, :(ptrX += $(size_T*Mk)))
                        push!(row_increments_rem.args, :(ptrX += $(size_T*W)))
                    else
                        push!(return_expr.args, :∂μ)
                        if μdim == 1
                            push!(Aquote.args, :(∂μ = PtrVector{$P,$T,$P,$P}(_sptr)))
                            push!(Aquote.args, :(ptr∂μ = _sptr))
                            push!(Aquote.args, :(_sptr += $(VectorizationBase.align(P*size_T))))
                            push!(Aquote.args, :(v∂μ = PtrMatrix{$W,$P,$T,$W,$(W*P)}(_sptr))) # accmulate in v∂μ; reduce at end
                            #push!(Aquote.args, :(ptrv∂μ = _sptr))
                            sptroff = W*P*size_T
                            #push!(Aquote.args, :(_sptr += W * P))
                        elseif μdim == 2
                            muquote = if M isa Integer
                                quote
                                    ∂μ = PtrMatrix{$M,$P,$T,$M}(_sptr)
                                    ptr∂μ = _sptr
                                    _sptr += $(VectorizationBase.align(M*P*size_T))
                                end
                            else#if M is a Symbol
                                quote
                                    ∂μ = DynamicPtrMatrix{$T}(_sptr, ($M,$P), $M)
                                    ptr∂μ = _sptr
                                    _sptr += VectorizationBase.align($(size_T*P) * $M)
                                end
                            end
                            push!(Aquote.args, muquote)
                            push!(row_increments.args, :(ptrμ += $(size_T*Mk)))
                            push!(row_increments_rem.args, :(ptrμ += $(size_T*W)))
                            #push!(Aquote.args, :(invdiag = PtrVector{$P,$T,$invdiagL,$invdiagL}(_sptr)))
                        else# μdim == 0
                            push!(Aquote.args, :(ptrμ = SIMDPirates.vbroadcast($V,zero($T))))
                            #push!(Aquote.args, :(invdiag = PtrVector{$P,$T,$invdiagL,$invdiagL}(_sptr)))
                        end
                    end
                end
                μmy = true
                push!(array_allocations.args, Aquote)
            else#if track_μ we are only tacking μ; A is ∂μ
                if track_X | track_β
                    Aquote = quote end
                    if track_X
                        push!(return_expr.args, :∂X)                                  
                        if M isa Integer
                            push!(Aquote.args, :(∂X = PtrMatrix{$M,$XP,$T,$M}(_sptr)))
                        else
                            push!(Aquote.args, :(∂X = DynamicPtrMatrix{$T}(_sptr, ($M,$XP), $M)))
                        end
                        push!(Aquote.args, :( ptr∂X = _sptr; _sptr += $(VectorizationBase.align(size_T * M*XP)) ))
                    end
                    if track_β
                        push!(Aquote.args, :(∂β = PtrMatrix{$XP,$P,$T,$XP}(_sptr); ptr∂β = _sptr))
                        alignβoffset = VectorizationBase.align(size_T * XP * P)
                        if μdim == 1
                            push!(return_expr.args, :∂βv)
                            push!(Aquote.args, :(_sptr += $(VectorizationBase.align(size_T * XP))))
                            sptroff += alignβoffset - VectorizationBase.align(size_T * XP)
                            push!(Aquote.args, Expr(:(=), :∂βv, :(PtrVector{$P,$T,$P,$P}(ptr∂β))))
                        else
                            push!(return_expr.args, :∂β)
                            push!(Aquote.args, :(_sptr += $alignβoffset))
                        end
                    end
                    if M isa Integer
                        push!(Aquote.args, :(A = PtrMatrix{$M,$P,$T,$M}(_sptr + $sptroff)))
                    else
                        push!(Aquote.args, :(A = DynamicPtrMatrix{$T}(_sptr + $sptroff, ($M,$P), $M)))
                    end
                    Astride = M
                    push!(Aquote.args, :(ptrA = pointer(A)))
                    push!(row_increments.args, :(ptrA += $(size_T*Mk)))
                    push!(row_increments_rem.args, :(ptrA += $(size_T*W)))
                    push!(row_increments.args, :(ptrX += $(size_T*Mk)))
                    push!(row_increments_rem.args, :(ptrX += $(size_T*W)))
                    sptroff += VectorizationBase.align(M*P*size_T)
                else
                    if μdim == 1
                        Aquote = quote
                            ∂μ = PtrVector{$P,$T,$P,$P}(_sptr)
                            ptr∂μ = _sptr
                            _sptr += $(VectorizationBase.align(P*size_T))
                            v∂μ = PtrMatrix{$W,$P,$T,$W,$(W*P)}(_sptr) # accmulate in v∂μ; reduce at end
                            #ptrv∂μ = _sptr
                            #invdiag = PtrVector{$P,$T,$invdiagL,$invdiagL}(_sptr + $(W*P))
                        end
                        sptroff = W*P*size_T
                    elseif μdim == 2
                        Aquote = if M isa Integer
                            quote
                                A = PtrMatrix{$M,$P,$T,$M}(_sptr)
                                ptrA = _sptr
                                _sptr += $(VectorizationBase.align(M*P*size_T))
                            end
                        else#if M is a Symbol
                            quote
                                A = DynamicPtrMatrix{$T}(_sptr, ($M,$P), $M)
                                ptrA = _sptr
                                _sptr += VectorizationBase.align($(size_T*P) * $M)
                            end
                        end
                        push!(row_increments.args, :(ptrA += $(size_T*Mk)))
                        push!(row_increments_rem.args, :(ptrA += $(size_T*W)))
                        #push!(Aquote.args, :(invdiag = PtrVector{$P,$T,$invdiagL,$invdiagL}(_sptr)))
                    else# μdim == 0
                        Aquote.args = quote ptrμ = SIMDPirates.vbroadcast($V,zero($T)) end
                    end
                    if μdim == 2
                        push!(return_expr.args, :A)
                        Astride = M
                    else
                        #push!(Aquote.args, :(invdiag = PtrVector{$P,$T,$invdiagL,$invdiagL}(_sptr)))
                        push!(Aquote.args, :(A = PtrMatrix{$Mk,$P,$T,$Mk}(_sptr + $sptroff)))
                        Astride = Mk
                        sptroff += VectorizationBase.align(size_T*Mk*P)
                        push!(Aquote.args, :(ptrA = pointer(A)))
                    end
                    push!(return_expr.args, :∂μ)
                end
                μmy = false
                push!(array_allocations.args, Aquote)
            end
        end
        if track_L
            push!(array_allocations.args, :(v∂L = StructuredMatrices.PtrLowerTriangularMatrix{$P,$V,$∂LL}(_sptr + $sptroff)))
        else
            push!(array_allocations.args, :(invdiag = PtrVector{$P,$T,$invdiagL,$invdiagL}(_sptr + $sptroff)))
            #sptroff += invdiagL*size_T
        end        
    else#if !sp
        if !track_Y && !track_μ# don't need to track A
            Aquote = quote
                A = MutableFixedSizePaddedMatrix{$Mk,$P,$T,$Mk}(undef)
                ptrA = pointer(A)
            end
            push!(array_allocations.args, Aquote)
            μmy = true
            Astride = Mk
        else
            if track_Y # we do need to track A; A will be partialY
                push!(row_increments.args, :(ptrA += $(size_T*Mk)))
                push!(row_increments_rem.args, :(ptrA += $(size_T*W)))
                push!(return_expr.args, :A)
                if M isa Integer # Y
                    # @show M,P,T,M
                    Aquote = quote
                        A = MutableFixedSizePaddedMatrix{$M,$P,$T,$M}(undef)
                    end
                else#if M is a Symbol
                    Aquote = quote
                        A = Matrix{$T}(undef, $M, $P) # could do a DynamicPaddedMatrix...but why that over a regular old array?
                    end
                end
                Astride = M
                push!(Aquote.args, :(ptrA = pointer(A)))
                if track_μ
                    if track_X | track_β
                        if track_X
                            push!(return_expr.args, :∂X)                                  
                            if M isa Integer
                                push!(Aquote.args, :(∂X = MutableFixedSizePaddedMatrix{$M,$XP,$T,$M}(undef)))
                            else
                                push!(Aquote.args, :(∂X = Matrix{$T}(undef, $M,$XP)))
                            end
                            push!(Aquote.args, :( ptr∂X = pointer(∂X)))
                        end
                        if track_β
                            push!(Aquote.args, :(∂β = MutableFixedSizePaddedMatrix{$XP,$P,$T,$XP}(undef); ptr∂β = pointer(∂β)))
                            if μdim == 1
                                push!(Aquote.args, :(∂βv = MutableFixedSizePaddedVector{$XP,$T}(undef)))
                                push!(return_expr.args, :∂βv)
                            else
                                push!(return_expr.args, :∂β)
                            end
                        end
                        push!(row_increments.args, :(ptrX += $(size_T*Mk)))
                        push!(row_increments_rem.args, :(ptrX += $(size_T*W)))
                    else
                        push!(return_expr.args, :∂μ)
                        if μdim == 1
                            push!(Aquote.args, :(∂μ = MutableFixedSizePaddedVector{$P,$T,$P,$P}(undef)))
                            push!(Aquote.args, :(ptr∂μ = pointer(∂μ)))
                            push!(Aquote.args, :(v∂μ = MutableFixedSizePaddedVector{$P,$V,$P,$P}(undef)))
                            #push!(Aquote.args, :(ptrv∂μ = pointer(v∂μ)))
                        elseif μdim == 2
                            muquote = if M isa Integer
                                :(∂μ = MutableFixedSizePaddedMatrix{$M,$P,$T,$M}(undef))
                            else#if M is a Symbol
                                :(∂μ = Matrix{$T}(undef, $M,$P))
                            end
                            push!(Aquote.args, muquote)
                            push!(Aquote.args, :(ptr∂μ = pointer(∂μ)))
                            push!(row_increments.args, :(ptrμ += $(size_T*Mk)))
                            push!(row_increments_rem.args, :(ptrμ += $(size_T*W)))
                        else# μdim == 0
                            push!(Aquote.args, :(ptrμ = SIMDPirates.vbroadcast($V,zero($T))))
                        end
                    end
                end
                # push!(Aquote.args, :(invdiag = MutableFixedSizePaddedVector{$P,$T,$invdiagL,$invdiagL}(undef)))
                μmy = true
                push!(array_allocations.args, Aquote)
            else#if track_μ we are only tacking μ; A is ∂μ
                if track_X | track_β
                    Aquote = quote end
                    if track_X
                        push!(return_expr.args, :∂X)                                  
                        if M isa Integer
                            push!(Aquote.args, :(∂X = MutableFixedSizePaddedMatrix{$M,$XP,$T,$M}(undef)) )
                        else
                            push!(Aquote.args, :(∂X = Matrix{$T}(undef, $M,$XP)))
                        end
                        push!(Aquote.args, :( ptr∂X = pointer(∂X)))
                    end
                    if track_β
                        push!(Aquote.args, :(∂β = MutableFixedSizePaddedMatrix{$XP,$P,$T,$XP}(undef); ptr∂β = pointer(∂β)))
                        if μdim == 1
                            push!(return_expr.args, :∂βv)
                            push!(Aquote.args, Expr(:(=), :∂βv, :(MutableFixedSizePaddedVector{$P,$T,$P,$P}(undef))))
                        else
                            push!(return_expr.args, :∂β)
                        end
                    end
                    if M isa Integer
                        push!(Aquote.args, :(A = MutableFixedSizePaddedMatrix{$M,$P,$T,$M}(undef)))
                    else
                        push!(Aquote.args, :(A = Matrix{$T}(undef, $M,$P)))
                    end
                    push!(Aquote.args, :(ptrA = pointer(A)))
                    push!(row_increments.args, :(ptrX += $(size_T*Mk)))
                    push!(row_increments_rem.args, :(ptrX += $(size_T*W)))
                    push!(row_increments.args, :(ptrA += $(size_T*Mk)))
                    push!(row_increments_rem.args, :(ptrA += $(size_T*W)))
                    Astride = M
                else
                    if μdim == 1
                        Aquote = quote
                            ∂μ = MutableFixedSizePaddedVector{$P,$T,$P,$P}(undef)
                            ptr∂μ = pointer(∂μ)
                            v∂μ = MutableFixedSizePaddedVector{$P,$V,$P,$P}(undef) # accmulate in v∂μ; reduce at end
                            #ptrv∂μ = pointer(v∂μ)
                        end
                    elseif μdim == 2
                        Aquote = if M isa Integer
                            quote A = MutableFixedSizePaddedMatrix{$M,$P,$T,$M}(undef) end
                        else#if M is a Symbol
                            quote A = Matrix{$T}(undef, $M,$P) end
                        end
                        push!(Aquote.args, :(ptrA = pointer(A)))
                        push!(row_increments.args, :(ptrA += $(size_T*Mk)))
                        push!(row_increments_rem.args, :(ptrA += $(size_T*W)))
                        push!(return_expr.args, :A)
                        Astride = M
                    else# μdim == 0
                        Aquote = quote ptrμ = SIMDPirates.vbroadcast($V,zero($T)) end
                    end
                    #push!(Aquote.args, :(invdiag = MutableFixedSizePaddedVector{$P,$T,$invdiagL,$invdiagL}(undef)))
                    if μdim != 2
                        #push!(Aquote.args, :(invdiag = MutableFixedSizePaddedVector{$P,$T,$invdiagL,$invdiagL}(undef)))
                        push!(Aquote.args, :(A = MutableFixedSizePaddedMatrix{$Mk,$P,$T,$Mk}(undef)))
                        push!(Aquote.args, :(ptrA = pointer(A)))
                        Astride = Mk
                    end
                    push!(return_expr.args, :∂μ)
                end
                μmy = false
                push!(array_allocations.args, Aquote)
            end
        end
    end
    push!(return_expr.args, :(∂L))
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
        _A_offset_ = $(size_T*Astride*startoffset)
        ptrLtribase = pointer(L) + $(size_T * (P + StructuredMatrices.binomial2(startoffset) + startoffset * (P - startoffset))) # diag + triangle + subtriangle
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
    D == 4 && push!(q.args, :(ptrX = pointer(X); ptrβ = pointer(β)))
    if track_μ
        if μdim == 0
            for m ∈ 0:3
                push!(q.args, Expr(:(=), Symbol(:v∂μ_,m), :(SIMDPirates.vbroadcast($V, zero($T)))))
            end
        elseif no_Xorβ
            if μdim == 1
                push!(q.args, :(ptrv∂μbase = pointer(v∂μ) + $(size_T*W*startoffset)))
                set_ptr_vmu_zero_expr = quote
                    ptrv∂μ = pointer(v∂μ)
                    for p ∈ 0:$(P-1)
                        SIMDPirates.vstore!(ptrv∂μ + p*$(W*size_T), SIMDPirates.vbroadcast($V, zero($T)))
                    end
                end
                push!(q.args, set_ptr_vmu_zero_expr)
            elseif μdim == 2 && track_Y
                if M isa Symbol
                    push!(q.args, :(ptrv∂μbase = pointer(v∂μ) + $(size_T*startoffset)*$M))
                else
                    push!(q.args, :(ptrv∂μbase = pointer(v∂μ) + $(size_T*startoffset*M)))
                end
            end
        end
    end
    if μdim == 0
        push!(q.args, Expr(:(=), :μptr, :(SIMDPirates.vbroadcast($V, μ))))
    elseif μdim > 0 && no_Xorβ
        push!(q.args, Expr(:(=), :μptr, :(pointer(μ))))
    end
    if M isa Integer
        n_row_reps, row_rem = divrem(M, Mk)
        total_row_iterations = n_row_reps + (row_rem > 0)
        Mk1 = n_row_reps == 0 ? row_rem : Mk
        row_iter = ∂mutlivariate_normal_SMLT_rowiter(
            Mk1, Nk, col_rem, T, Ystride, n_col_reps, μdim, μstride, track, μmy, :μptr, Astride, XP, Xstride, βstride
        )
        if n_row_reps > 1
            row_loops = quote
                for rrep ∈ 1:$n_row_reps
                    ptrUdiag = pointer(invdiag)
                    ptrUtri = ptrUtribase#pointer(B) + $(size_T * N)
                    $row_iter
                    $row_increments
                end
            end
            push!(q.args, row_loops)
            # if row_rem > 0
            #     push!(q.args, ∂mutlivariate_normal_SMLT_rowiter( row_rem, Nk, col_rem, T, Ystride, n_col_reps, μdim, μstride, track, μmy, :μptr, Astride, XP, Xstride, βstride ))
            # end
        else
            push!(q.args, :(ptrUdiag = pointer(invdiag)))
            push!(q.args, :(ptrUtri = ptrUtribase))
            push!(q.args, row_iter)
            # if total_row_iterations == 2 # then n_row_reps == 1 and row_rem > 0
            #     push!(q.args, ∂mutlivariate_normal_SMLT_rowiter( row_rem, Nk, col_rem, T, Ystride, n_col_reps, μdim, μstride, track, μmy, :μptr, Astride, XP, Xstride, βstride ))
            # end
        end
        if row_rem > 0 # then n_row_reps == 1 and row_rem > 0
            push!(q.args, ∂mutlivariate_normal_SMLT_rowiter( row_rem, Nk, col_rem, T, Ystride, n_col_reps, μdim, μstride, track, μmy, :μptr, Astride, XP, Xstride, βstride ))
        end
    else # Unknown number of iterations.
        row_iter = ∂mutlivariate_normal_SMLT_rowiter(
            Mk, Nk, col_rem, T, Ystride, n_col_reps, μdim, μstride, track, μmy, :μptr, Astride, XP, Xstride, βstride
        )
        Wrem, Mkrem, Nkrem = StructuredMatrices.div_triangle_blocking_structure(W, P, T)
        n_col_repsrem, col_remrem = divrem(P, Nkrem)
        row_iter_onevec = ∂mutlivariate_normal_SMLT_rowiter(
            W, Nkrem, col_remrem, T, Ystride, n_col_repsrem, μdim, μstride, track, μmy, :μptr, Astride, XP, Xstride, βstride
        )
        row_iter_onevecmask = ∂mutlivariate_normal_SMLT_rowiter(
            :row_rem_final, Nkrem, col_remrem, T, Ystride, n_col_repsrem, μdim, μstride, track, μmy, :μptr, Astride, XP, Xstride, βstride
        )
        row_loops = quote
            Mkrep, Mkrem = divrem($M, $Mk)
            for rrep ∈ 1:Mkrep
                ptrUdiag = pointer(invdiag)
                ptrUtri = ptrUtribase#pointer(B) + $(size_T * N)
                $row_iter
                $row_increments
            end
            for rrep ∈ 1:Mkrem >> $(VectorizationBase.intlog2(W))
                ptrUdiag = pointer(invdiag)
                ptrUtri = ptrUtribase#pointer(B) + $(size_T * N)
                $row_iter_onevec
                $row_increments_rem
            end
            row_rem_final = Mkrem & $Wm1
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
        if track_μ && μdim == 1 && no_Xorβ
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
    if track_μ && no_Xorβ
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
        f = μmy ? :(PaddedMatrices.nmul!) : :(LinearAlgebra.mul!)
        track_X && push!(q.args, Expr(:call, f, :∂X, :A, :(β')))
        if track_β
            push!(q.args, Expr(:call, f, :∂β, :(X'), :A))
            if μdim == 1
                push!(q.args, Expr(:call, :sum!, :∂βv, :∂β))
            end
        end
    end
    if sp
        push!(q.args, :(_sptr,$return_expr))
    else
        push!(q.args, return_expr)
    end
    simplify_expr(q)
end



@generated function ∂Normal(
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,MP},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true)}()
) where {M,P,T,track,MP}
    ∂multivariate_normal_SMLT_quote(M, P, track, -1, -1, false, MP, T)
end
@generated function ∂Normal(
    sp::StackPointer,
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,MP},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true)}()
) where {M,P,T,track,MP}
    ∂multivariate_normal_SMLT_quote(M, P, track, -1, -1, true, MP, T)
end
@generated function ∂Normal(
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,MP},
    μ::T,
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true)}()
) where {M,P,T,track,MP}
    ∂multivariate_normal_SMLT_quote(M, P, track, 0, 0, false, MP, T)
end
@generated function ∂Normal(
    sptr::StackPointer,
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,MP},
    μ::T,
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true)}()
) where {M,P,T,track,MP}
    ∂multivariate_normal_SMLT_quote(M, P, track, 0, 0, true, MP, T)
end
@generated function ∂Normal(
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,MP},
    μ::AbstractMutableFixedSizePaddedVector{P,T},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true)}()
# ) where {M,P,T,MP,track}
) where {M,P,T,track,MP}
    ∂multivariate_normal_SMLT_quote(M, P, track, 1, 1, false, MP, T)
end
@generated function ∂Normal(
    sptr::StackPointer,
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,MP},
    μ::AbstractMutableFixedSizePaddedVector{P,T},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true)}()
# ) where {M,P,T,track,MP}
) where {M,P,T,MP,track}
    ∂multivariate_normal_SMLT_quote(M, P, track, 1, 1, true, MP, T)
end
@generated function ∂Normal(
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,MP},
    μ::AbstractMutableFixedSizePaddedMatrix{M,P,T,MM},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true)}()
) where {M,P,T,track,MP,MM}
    ∂multivariate_normal_SMLT_quote(M, P, track, 2, MM, false, MP, T)
end
@generated function ∂Normal(
    sptr::StackPointer,
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,MP},
    μ::AbstractMutableFixedSizePaddedMatrix{M,P,T,MM},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true)}()
) where {M,P,T,track,MP,MM}
    ∂multivariate_normal_SMLT_quote(M, P, track, 2, MM, true, MP, T)
end



@generated function ∂Normal(
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,PY},
    X::AbstractMutableFixedSizePaddedMatrix{M,K_,T,PX},
    β::AbstractFixedSizePaddedVector{K_,T,PK},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true,true)}()
# ) where {M,P,T,track,PY,PX,PK,K_}
) where {M,P,T,track,K_,PY,PX,PK}
    ∂multivariate_normal_SMLT_quote(M, P, track, 1, PX, false, PY, T, K_, 1)
end
@generated function ∂Normal(
    sptr::StackPointer,
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,PY},
    X::AbstractMutableFixedSizePaddedMatrix{M,K_,T,PX},
    β::AbstractFixedSizePaddedVector{K_,T,PK},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true,true)}()
) where {M,P,T,track,PY,PX,K_,PK}
# ) where {M,P,T,track,K_,PY,PX,PK}
    ∂multivariate_normal_SMLT_quote(M, P, track, 1, PX, true, PY, T, K_, 1)
end
@generated function ∂Normal(
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,PY},
    X::AbstractMutableFixedSizePaddedMatrix{M,K_,T,PX},
    β::AbstractFixedSizePaddedMatrix{K_,P,T,PK},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true,true)}()
) where {M,P,T,track,PY,PX,K_,PK}
# ) where {M,P,T,track,K_,PY,PX,PK}
    ∂multivariate_normal_SMLT_quote(M, P, track, 2, PX, false, PY, T, K_, PK)
end
@generated function ∂Normal(
    sptr::StackPointer,
    Y::AbstractMutableFixedSizePaddedMatrix{M,P,T,PY},
    X::AbstractMutableFixedSizePaddedMatrix{M,K_,T,PX},
    β::AbstractFixedSizePaddedMatrix{K_,P,T,PK},
    L::AbstractLowerTriangularMatrix{P,T},
    ::Val{track} = Val{(true,true,true,true)}()
) where {M,P,T,track,PY,PX,K_,PK}
# ) where {M,P,T,track,K_,PY,PX,PK}
    ∂multivariate_normal_SMLT_quote(M, P, track, 2, PX, true, PY, T, K_, PK)
end




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

