
@inline canonicalize_Σ(L::AbstractCovarCholesky) = L

using DistributionParameters: AbstractFixedSizeCovarianceMatrix, AbstractCovarianceMatrix
function logdet_triangle(A::AbstractMatrix{T}) where {T}
    N = size(A,1)
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    Nrep = N >>> Wshift

    vout = vbroadcast(Vec{W,T}, zero(T))
    @inbounds for i ∈ 0:Nrep-1
        x = ntuple(w -> Core.VecElement(A[W*i+w,W*i+w]), Val(W))
        vout = vadd(
            vout,
            SLEEFPirates.log( x )
        )
    end
    out = vsum(vout)
    @inbounds for i ∈ 1 + (N & -W):N
        out += log(A[i,i])
    end
    out
end
@inline function vlogdet_triangle(A::AbstractMatrix{T}) where {T}
    N = size(A,1)
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    Nrep = N >>> Wshift

    vout = vbroadcast(Vec{W,T}, zero(T))
    @inbounds for i ∈ 0:Nrep-1
        x = ntuple(w -> Core.VecElement(A[W*i+w,W*i+w]), Val(W))
        vout = vadd(
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
    vadd( vout, SLEEFPirates.log( x ) )
end
@generated function vlogdet_triangle(A::StructuredMatrices.AbstractTriangularMatrix{P,T}) where {P,T}
    W = VectorizationBase.pick_vector_width(P,T)
    quote
        out = vbroadcast(Vec{$W,$T}, zero($T))
        @vvectorize $T for p in 1:$P
            out = LoopVectorization.vadd(SLEEFPirates.log(A[p]), out)
        end
        out
    end
end

function info_trtrs!(uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar, A::AbstractMatrix{Float64}, B::AbstractVecOrMat{Float64})
    n = size(B,1)
    info = Ref{LinearAlgebra.BlasInt}()
    ccall(
        (BLAS.@blasfunc(dtrtrs_), LAPACK.liblapack), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt},
        Ptr{Float64}, Ref{LinearAlgebra.BlasInt}, Ptr{Float64}, Ref{LinearAlgebra.BlasInt}, Ptr{LinearAlgebra.BlasInt}),
        uplo, trans, diag, n, size(B,2), A, max(1,stride(A,2)),
        B, max(1,stride(B,2)), info)
    info[]
end

@generated function Normal!(
    ::Val{track},
    δ::AbstractMatrix{T},
    Y::NTuple{K},
    μ::NTuple{K,V},
    Σ::AbstractFixedSizeCovarianceMatrix{KT,T}
) where {T, K, P, V <: PaddedMatrices.AbstractFixedSizeVector{P,T}, KT, track}
    W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
    track_Y, track_μ, track_Σ = track
    q = quote
        # $(Expr(:meta,:inline))
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
        info = last(LinearAlgebra.LAPACK.potrf!('L', L))
        info == 0 || return $T(-Inf)
        # LinearAlgebra.LAPACK.trtrs!('L', 'N', 'N', L, δ)
        info = info_trtrs!('L', 'N', 'N', L, δ)
        info == 0 || return $T(-Inf)
        
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
@inline function Normal(sp::StackPointer, ::Val{track}, Y::NTuple{K}, μ::NTuple{K}, Σ::AbstractFixedSizeCovarianceMatrix{R,T}) where {K, R, T, track}
#    Wm1 = VectorizationBase.pick_vector_width(R,T) - 1
    cols = 0
    @inbounds for k ∈ 1:K
        cols += size(Y[k], 2)
    end
    δ = DynamicPtrMatrix(pointer(sp, T), (R, cols), R)# (KT+Wm1) & ~Wm1)
    # return sp, to reuse memory
    Normal!(Val{track}(), δ, Y, μ, Σ)
end
@inline function Normal(
    sp::StackPointer,
    ::Val{track},
    Y::AbstractMatrix{T},
    μ::PaddedMatrices.AbstractFixedSizeVector{R,T},
    Σ::AbstractFixedSizeCovarianceMatrix{R,T}
) where {R,T,track}
    Normal(sp, Val{track}(), (Y,), (μ,), Σ)
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
) where {T, K, P, R, V <: PaddedMatrices.AbstractFixedSizeVector{P,T}, KT, track}
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
        # $(Expr(:meta,:inline)) # work-around for SIMD-corruption bug
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
    ::Val{track},
    Y::NTuple{K},
    μ::NTuple{K},
    Σ::AbstractFixedSizeCovarianceMatrix{R,T,R}
) where {R, K, T, track}
#) where {K, T, KT, track}
    track_Y, track_μ, track_Σ = track

    q = quote
        # Inlined because of:
        # https://github.com/JuliaLang/julia/issues/32414
        # Stop forcing inlining when the issue is fixed.
        # $(Expr(:meta,:inline))

        cols = 0
        @inbounds for k ∈ 1:$K
            cols += size(Y[k], 2)
        end
    end
    push!(q.args, :(stack_pointer = pointer(sp,$T)))
    if track_Y
        # We are tracking Y, so we cannot drop Σ⁻¹δ, because this will be returned as ∂Yₖ
        push!(q.args, :(Σ⁻¹δ = DynamicPtrMatrix(stack_pointer, ($R,cols), $R)))
        push!(q.args, :(stack_pointer += $(VectorizationBase.align(sizeof(T)*R)*cols)))
        push!(q.args, :(δ = DynamicPtrMatrix(stack_pointer, ($R, cols), $R)))
    else
        # because we are not tracking Y, we can drop Σ⁻¹δ, which will contain ∂Y
        # we therefore allocate it on top of δ on the stack.
        push!(q.args, :(δ = DynamicPtrMatrix(stack_pointer, ($R, cols), $R)))
        push!(q.args, :(stack_pointer += $(VectorizationBase.align(sizeof(T)*R))*cols))
        push!(q.args, :(Σ⁻¹δ = DynamicPtrMatrix(stack_pointer, ($R,cols), $R)))
    end
    if track_μ && track_Y
        push!(q.args, :(sp = PaddedMatrices.StackPointer(Base.unsafe_convert(Ptr{Cvoid}, stack_pointer + $(VectorizationBase.align(K*sizeof(T)*R)) ))))
    elseif track_μ
        push!(q.args, :(sp = sp + $(VectorizationBase.align(K*sizeof(T)*R) )))
    elseif track_Y
        push!(q.args, :(sp = PaddedMatrices.StackPointer(Base.unsafe_convert(Ptr{Cvoid}, stack_pointer) )))
    end
    push!(q.args, :(sp, ∂Normal!(Σ⁻¹δ, δ, Y, μ, Σ, Val{$track}()) ))
    simplify_expr(q)
end

@generated function ∂Normal(
    sp::StackPointer,
    ::Val{track},
    Y::AbstractMatrix{T},
    μ::AbstractFixedSizeVector{R,T},
    Σ::AbstractFixedSizeCovarianceMatrix{R,T,R}
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
        # $(Expr(:meta,:inline))
        (sp, $∂retin) = ∂Normal(sp, (Y,), (μ,), Σ, Val{$track}())
        @inbounds (sp, $∂retout)
    end
end

# Type unstable to avoid compiling more than one ::NormalCholeskyConfiguration{Float64} method.
# Realistically, T === Float64, but could move that into a field if it seems like it will ever actually be something else.
# This should probalby be factored into pieces.
Base.@kwdef mutable struct NormalCholeskyConfiguration{T}
    M::Union{Int,Symbol} = -1
    P::Int = -1
    βstride::Int = -1
    Xstride::Union{Symbol,Int} = -1
    Ystride::Union{Symbol,Int} = M
    μstride::Union{Int,Symbol} = -1
    μdim::Int = -1
    sp::Bool = false
    βdim::Int = -1
    XP::Int = -1
    LL::Int = -1
    μtransposed::Bool = false
    arity::Int = -1
    track_Y::Bool = false
    track_X::Bool = false
    track_β::Bool = false
    track_μ::Bool = false
    track_L::Bool = false
    allocate_partials::Bool = false
    initY::Bool = true
    initX::Bool = true
    initβ::Bool = true
    initμ::Bool = true
    initL::Bool = true
    calclogdet::Bool = true
end


@noinline function loadδ_quote(
    R::Int, C::Int, K::Union{Symbol,Int}, T::DataType,
    Bstride::Union{Int,Symbol}, Bsym::Symbol,
    μdim::Int, μstride::Union{Int,Symbol},
    μsym::Union{Symbol,Nothing} = :μptr, maskload::Bool = true, μmy::Bool = true, μtransposed::Bool = false
)
    size_T = sizeof(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(R, T)
    V = Vec{W,T}
    Wm1 = W - 1
    Riter = R >>> Wshift
    Rrem = R & Wm1
    mask = VectorizationBase.mask(T, Rrem)
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
                push!(q.args, Expr(:(=), Symbol(:vμ_,r), :(vload($V, $μsym + $(size_T*W*r)))))
            end
            if Rrem > 0
                if maskload
                    push!(q.args, Expr(:(=), Symbol(:vμ_,r), :(vload($V, $μsym + $(size_T*W*Riter),$mask))))
                else
                    push!(q.args, Expr(:(=), Symbol(:vμ_,r), :(vload($V, $μsym + $(size_T*W*Riter)))))
                end
            end
        end
        for c ∈ 0:C-1            
            vμ_c = μdim == 0 ? μsym : Symbol(:vμ_, c)
            if μdim == 1
                if μstride isa Symbol
                    if K isa Symbol
                        push!(q.args, Expr(:(=), vμ_c, :(vbroadcast($V, $μsym + $size_T*$μstride*($c+$K)))))
                    else
                        push!(q.args, Expr(:(=), vμ_c, :(vbroadcast($V, $μsym + $size_T*$(c + K)*$μstride))))
                    end
                else
                    if K isa Symbol
                        push!(q.args, Expr(:(=), vμ_c, :(vbroadcast($V, $μsym + $(size_T*μstride)*($c+$K)))))
                    else
                        push!(q.args, Expr(:(=), vμ_c, :(vbroadcast($V, $μsym + $(size_T*μstride)*$(c + K)))))
                    end
                end
            end
            if (μdim == 1 && μtransposed) || μdim == 0
                for r ∈ 0:Riter-1
                    yloadexpr = :(vload($V, BsymK + $size_T * ($(r*W) + $c*$Bstride)))
                    if μmy
                        push!(q.args, :($(Symbol(:A_,r,:_,c)) = vsub($vμ_c, $yloadexpr)))
                    else
                        push!(q.args, :($(Symbol(:A_,r,:_,c)) = vsub($yloadexpr, $vμ_c)))
                    end
                end
                if Rrem > 0
                    # Only need to mask if we're on last column
                    if maskload && c == C-1
                        yloadexpr = :(vload($V, BsymK + $size_T * ($(Riter*W) + $c*$Bstride), $mask ))
                    else
                        yloadexpr = :(vload($V, BsymK + $size_T * ($(Riter*W) + $c*$Bstride)) )
                    end
                    if μmy
                        push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = vsub($vμ_c, $yloadexpr)))
                    else
                        push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = vsub($yloadexpr, $vμ_c)))
                    end
                end
            elseif μdim == 1 && !μtransposed
                for r ∈ 0:Riter-1
                    yloadexpr = :(vload($V, BsymK + $size_T * ($(r*W) + $c*$Bstride)))
                    vμ_r = Symbol(:vμ_,r)
                    if μmy
                        push!(q.args, :($(Symbol(:A_,r,:_,c)) = vsub($vμ_r, $yloadexpr)))
                    else
                        push!(q.args, :($(Symbol(:A_,r,:_,c)) = vsub($yloadexpr, $vμ_r)))
                    end
                end
                if Rrem > 0
                    # Only need to mask if we're on last column
                    vμ_r = Symbol(:vμ_,Riter)
                    if maskload && c == C-1
                        yloadexpr = :(vload($V, BsymK + $size_T * ($(Riter*W) + $c*$Bstride), $mask ))
                    else
                        yloadexpr = :(vload($V, BsymK + $size_T * ($(Riter*W) + $c*$Bstride) ))
                    end
                    if μmy
                        push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = vsub($vμ_r, $yloadexpr)))
                    else
                        push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = vsub($yloadexpr, $vμ_r)))
                    end
                end
            elseif μdim == 2
                for r ∈ 0:Riter-1
                    μloadexpr = :(vload($V, usymK + $size_T * ($(r*W) + $c*$μstride)))
                    yloadexpr = :(vload($V, BsymK + $size_T * ($(r*W) + $c*$Bstride)))
                    if μmy
                        push!(q.args, Expr(:(=), Symbol(:A_,r,:_,c), Expr(:call, :(vsub), μloadexpr, yloadexpr)))
                    else
                        push!(q.args, Expr(:(=), Symbol(:A_,r,:_,c), Expr(:call, :(vsub), yloadexpr, μloadexpr)))
                    end        
                end
                if Rrem > 0
                    # Only need to mask if we're on last column
                    if maskload && c == C-1
                        μloadexpr = :(vload($V, μsymK + $size_T * ($(Riter*W) + $c*$μstride), $mask))
                        yloadexpr = :(vload($V, BsymK + $size_T * ($(Riter*W) + $c*$Bstride), $mask))
                    else
                        μloadexpr = :(vload($V, μsymK + $size_T * ($(Riter*W) + $c*$μstride)) )
                        yloadexpr = :(vload($V, BsymK + $size_T * ($(Riter*W) + $c*$Bstride)) )
                    end
                    if μmy
                        push!(q.args, Expr(:(=), Symbol(:A_,Riter,:_,c)), Expr(:call,:(vsub), μloadexpr, yloadexpr))
                    else
                        push!(q.args, Expr(:(=), Symbol(:A_,Riter,:_,c)), Expr(:call,:(vsub), yloadexpr, μloadexpr))
                    end
                end
            else #μ assumed not to exist
                for r ∈ 0:Riter-1
                    push!(q.args, :($(Symbol(:A_,r,:_,c)) = vload($V, BsymK + $size_T * ($(r*W) + $c*$Bstride) ) ))
                end
                if Rrem > 0
                    # Only need to mask if we're on last column
                    if maskload && c == C-1
                        push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = vload($V, BsymK + $size_T * ($(Riter*W) + $c*$Bstride), $mask ) ))
                    else
                        push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = vload($V, BsymK + $size_T * ($(Riter*W) + $c*$Bstride)) ) )
                    end
                end
            end
        end
    else
        for c ∈ 0:C-1
            for r ∈ 0:Riter-1
                push!(q.args, :($(Symbol(:A_,r,:_,c)) = vload($V, BsymK + $size_T * ($(r*W) + $c*$Bstride)) ) )
            end
            if Rrem > 0
                # Only need to mask if we're on last column
                if maskload && c == C-1
                    push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = vload($V, BsymK + $size_T * ($(Riter*W) + $c*$Bstride), $mask ) ))
                else
                    push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = vload($V, BsymK + $size_T * ($(Riter*W) + $c*$Bstride)) ) )
                end
            end
        end
    end
    q
end
@noinline function loadδ_quote(
    R::Symbol, C::Int, K::Union{Symbol,Int}, T::DataType,
    Bstride::Symbol, Bsym::Symbol,
    μdim::Int, μstride::Union{Int,Symbol},
    μsym::Union{Symbol,Nothing} = :μptr, maskload::Bool = true,
    μmy::Bool = true, μtransposed::Bool = false, masksym::Union{Unsigned,Symbol} = :__mask__
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
            push!(q.args, Expr(:(=), :vμ_0, :(vload($V, $μsym,$masksym))))
        end
        for c ∈ 0:C-1            
            vμ_c = μdim == 0 ? μsym : Symbol(:vμ_, c)
            if μdim == 1
                push!(q.args, Expr(:(=), vμ_c, :(vbroadcast($V, $μsym + $size_T*$μstride*($c+$K)))))
            end
            if (μdim == 1 && μtransposed) || μdim == 0
                # Only need to mask if we're on last column
                yloadexpr = :(vload($V, BsymK + $size_T * $c*$Bstride, $masksym ))
                if μmy
                    push!(q.args, :($(Symbol(:A_0_,c)) = vsub($vμ_c, $yloadexpr)))
                else
                    push!(q.args, :($(Symbol(:A_0_,c)) = vsub($yloadexpr, $vμ_c)))
                end
            elseif μdim == 1 && !μtransposed
                # Only need to mask if we're on last column
                yloadexpr = :(vload($V, BsymK + $size_T * $c*$Bstride, $masksym ))
                if μmy
                    push!(q.args, :($(Symbol(:A_0_,c)) = vsub(vμ_0, $yloadexpr)))
                else
                    push!(q.args, :($(Symbol(:A_0_,c)) = vsub($yloadexpr, vμ_0)))
                end
            elseif μdim == 2
                # Only need to mask if we're on last column
                μloadexpr = :(vload($V, μsymK + $size_T * $c*$μstride, $masksym))
                yloadexpr = :(vload($V, BsymK + $size_T * $c*$Bstride, $masksym))
                if μmy
                    push!(q.args, Expr(:(=), Symbol(:A_0_,c)), Expr(:call,:(vsub), μloadexpr, yloadexpr))
                else
                    push!(q.args, Expr(:(=), Symbol(:A_0_,c)), Expr(:call,:(vsub), yloadexpr, μloadexpr))
                end
            else #μ assumed not to exist
                    # Only need to mask if we're on last column
                push!(q.args, :($(Symbol(:A_0_,c)) = vload($V, BsymK + $size_T * $c*$Bstride, $masksym ) ))
            end
        end
    else
        for c ∈ 0:C-1
            push!(q.args, :($(Symbol(:A_0_,c)) = vload($V, BsymK + $size_T * $c*$Bstride, $masksym ) ))
        end
    end
    q
end
@noinline function loadδfnmadd_quote(
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
    Riter = R >>> Wshift
    Rrem = R & Wm1
    Riterl = Rrem > 0 ? Riter : Riter - 1
    maskload = maskload & (Rrem > 0)
    mask = VectorizationBase.mask(T, Rrem)
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
                xloadexpr = :(vload($V, $xsym + ($size_T * $(r*W)), $mask))
            else
                xloadexpr = :(vload($V, $xsym + ($size_T * $(r*W))))
            end
            push!(q.args, Expr(:(=), Symbol(:vx_,r), xloadexpr))
        end
    end
    f = μmy ? :(vfmsub) : :(vfnmadd)
    if μstride != -1 && μdim == 1
        if μtransposed
            for c ∈ 0:C-1
                push!(q.args, Expr(:(=), Symbol(:vμbase_,c), :(vbroadcast($V, $μsym + $size_T*$μstride*($K+$c) ))))
            end
        else
            for r ∈ 0:Riterl
                if r == Riterl && maskload
                    push!(q.args, Expr(:(=), Symbol(:vμbase_,r), :(vload($V, $μsym + $size_T * $(r*W),$mask))))
                else
                    push!(q.args, Expr(:(=), Symbol(:vμbase_,r), :(vload($V, $μsym + $size_T * $(r*W)))))
                end
            end
        end
    end
    for c ∈ 0:C-1
        if βdim == 1
            for r ∈ 0:Riterl
                vμ_r = Symbol(:vμ_,r)
                if r == Riterl && maskload && c == C - 1
                    yloadexpr = :(vload($V, YsymK + $size_T * ($(r*W) + $c*$Ystride),$mask))
                else
                    yloadexpr = :(vload($V, YsymK + $size_T * ($(r*W) + $c*$Ystride)))
                end
                if μstride != -1
                    if μdim == 1
                        yloadexpr = :(vsub($yloadexpr,$(Symbol(:vμbase_, μtransposed ? c : r))))
                    else#if μdim == 2
                        if r == Riterl && maskload && c == C - 1
                            αloadexpr = :(vload($V, $μsym + $size_T * ($(r*W) + $c*$μstride),$mask))
                        else
                            αloadexpr = :(vload($V, $μsym + $size_T * ($(r*W) + $c*$μstride)))
                        end
                        yloadexpr = :(vsub($yloadexpr,$αloadexpr))
                    end
                end
                if μmy
                    push!(q.args, :($(Symbol(:A_,r,:_,c)) = vsub($vμ_r, $yloadexpr)))
                else
                    push!(q.args, :($(Symbol(:A_,r,:_,c)) = vsub($yloadexpr, $vμ_r)))
                end
                # push!(q.args, :(@show getfield.($(Symbol(:A_,r,:_,c)), :value)))
            end
        elseif βdim == 2
            # we load first block, before the XP loop
            # if  μmy, that is X*β - Y # aka vfmsub  # loop vmuladd to this answer
            # if !μmy, that is Y - X*β # aka vfnmadd # loop vfnmadd to this answer
            if peel_first_iter
                β_c = Symbol(:β_,c)
                push!(q.args, Expr(:(=), β_c, :(vbroadcast($V, βsymK + ($size_T * $(c*βstride))))))
            end
            for r ∈ 0:Riterl
                if r == Riterl && maskload && c == C-1
                    yloadexpr = :(vload($V, YsymK + $size_T * ($(r*W) + $c*$Ystride),$mask))
                else
                    yloadexpr = :(vload($V, YsymK + $size_T * ($(r*W) + $c*$Ystride)))
                end
                # What is happening here is that we want to make y negative
                if peel_first_iter
                    yloadexpr = Expr(:call, f, Symbol(:vx_,r), β_c, yloadexpr)
                elseif μstride != -1
                    if μdim == 1
                        if μmy
                            yloadexpr = :(vsub($(Symbol(:vμbase_, μtransposed ? c : r)),$yloadexpr))
                        else
                            yloadexpr = :(vsub($yloadexpr,$(Symbol(:vμbase_, μtransposed ? c : r))))
                        end
                    else#if αdim == 2
                        if r == Riterl && maskload && c == C - 1
                            μloadexpr = :(vload($V, μsymK + $size_T * ($(r*W) + $c*$μstride),$mask))
                        else
                            μloadexpr = :(vload($V, μsymK + $size_T * ($(r*W) + $c*$μstride)))
                        end
                        yloadexpr = if μmy
                            :($vsub($μloadexpr,$yloadexpr))
                        else
                            :($vsub($yloadexpr,$μloadexpr))
                        end
                    end
                end
                # push!(q.args, Expr(:(=), Symbol(:A_,r,:_,c), Expr(:call, f, Symbol(:vx_,r), β_c, yloadexpr)))
                push!(q.args, Expr(:(=), Symbol(:A_,r,:_,c), yloadexpr))
            end
        end
    end
    f = μmy ? :(vmuladd) : :(vfnmadd)
    if βdim == 2
        p = gensym(:p)
        loopbody = quote end
        for r ∈ 0:Riterl
            if r == Riterl && maskload
                xloadexpr = :(vload($V, $xsym + $size_T * ($(r*W) + $p*$Xstride),$mask))
            else
                xloadexpr = :(vload($V, $xsym + $size_T * ($(r*W) + $p*$Xstride)))
            end
            push!(loopbody.args, Expr(:(=), Symbol(:vx_,r), xloadexpr))
        end
        for c ∈ 0:C-1
            β_c = Symbol(:β_,c)
            push!(loopbody.args, Expr(:(=), β_c, :(vbroadcast($V, βsymK + $size_T * ($(c*βstride)+$p)))))
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
@noinline function loadδfnmadd_quote(
    R::Symbol, C::Int, K::Union{Symbol,Int}, T::DataType, Ystride::Symbol, Xstride::Symbol, βstride::Int, βdim::Int,
    ysym::Symbol = :ptrY, xsym::Symbol = :ptrX, βsym::Symbol = :ptrβ, μsym::Symbol = :ptrμ,
    maskload::Bool = true, μmy::Bool = true, XP::Int = -1,
    μstride::Union{Int,Symbol} = -1, μdim::Int = -1, μtransposed::Bool = false, masksym::Union{Unsigned,Symbol} = :__mask__
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
        xloadexpr = :(vload($V, $xsym, $masksym))
        push!(q.args, :(vx_0 = $xloadexpr))
    end
    f = μmy ? :(vfmsub) : :(vfnmadd)
    if μstride != -1 && μdim == 1
        if μtransposed
            for c ∈ 0:C-1
                push!(q.args, Expr(:(=), Symbol(:vμbase_,c), :(vbroadcast($V, $μsym + $size_T*$μstride*($K+$c) ))))
            end
        else
            push!(q.args, :(vμbase_0 = vload($V, $μsym, $masksym)))
        end
    end
    for c ∈ 0:C-1
        if βdim == 1
            yloadexpr = :(vload($V, YsymK + $size_T * $c*$Ystride, $masksym))
            if μstride != -1
                if μdim == 1
                    yloadexpr = :(vsub($yloadexpr,$(Symbol(:vμbase_, μtransposed ? c : 0))))
                else#if μdim == 2
                    αloadexpr = :(vload($V, $μsym + $size_T * $c*$μstride, $masksym))
                    yloadexpr = :(vsub($yloadexpr, $αloadexpr))
                end
            end
            if μmy
                push!(q.args, :($(Symbol(:A_0_,c)) = vsub(vμ_0, $yloadexpr)))
            else
                push!(q.args, :($(Symbol(:A_0_,c)) = vsub($yloadexpr, vμ_0)))
            end
        elseif βdim == 2
            # we load first block, before the XP loop
            # if  μmy, that is X*β - Y # aka vfmsub  # loop vmuladd to this answer
            # if !μmy, that is Y - X*β # aka vfnmadd # loop vfnmadd to this answer
            if peel_first_iter
                β_c = Symbol(:β_,c)
                push!(q.args, Expr(:(=), β_c, :(vbroadcast($V, βsymK + $size_T * $c*$βstride))))
            end
            yloadexpr = :(vload($V, YsymK + $size_T * $c*$Ystride, $masksym))
            # What is happening here is that we want to make y negative
            if peel_first_iter
                yloadexpr = Expr(:call, f, :vx_0, β_c, yloadexpr)
            elseif μstride != -1
                if μdim == 1
                    if μmy
                        yloadexpr = :(vsub($(Symbol(:vμbase_, μtransposed ? c : 0)),$yloadexpr))
                    else
                        yloadexpr = :(vsub($yloadexpr,$(Symbol(:vμbase_, μtransposed ? c : 0))))
                    end
                else#if αdim == 2
                    μloadexpr = :(vload($V, μsymK + $size_T * $c*$μstride, $masksym))
                    yloadexpr = if μmy
                        :($vsub($μloadexpr,$yloadexpr))
                    else
                        :($vsub($yloadexpr,$μloadexpr))
                    end
                end
            end
            push!(q.args, Expr(:(=), Symbol(:A_0_,c), yloadexpr))
        end
    end
    f = μmy ? :(vmuladd) : :(vfnmadd)
    if βdim == 2
        p = gensym(:p)
        loopbody = quote end
        xloadexpr = :(vload($V, $xsym + $size_T * $p*$Xstride, $masksym))
        push!(loopbody.args, Expr(:(=), :vx_0, xloadexpr))
        for c ∈ 0:C-1
            β_c = Symbol(:β_,c)
            push!(loopbody.args, Expr(:(=), β_c, :(vbroadcast($V, βsymK + $size_T * ($(c*βstride)+$p)))))
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
@noinline function Xβ_load_quote(
    R::Int, T::DataType, Xstride::Union{Int,Symbol}, βstride::Int, μmy::Bool = true, XP::Int = -1, 
    xsym::Symbol = :ptrX, βsym::Symbol = :ptrβ, maskload::Bool = true
)
    size_T = sizeof(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(R, T)
    V = Vec{W,T}
    Wm1 = W - 1
    Riter = R >>> Wshift
    Rrem = R & Wm1
    Riterl = Rrem > 0 ? Riter : Riter - 1
    maskload = maskload & (Rrem > 0)
    mask = VectorizationBase.mask(T, Rrem)
    q = quote end
    # Initial load
    push!(q.args, Expr(:(=), :vβ, :(vbroadcast($V, $βsym))))
    for r ∈ 0:Riterl
        if r == Riterl && maskload && XP == 1
            xloadexpr = :(vload($V, $xsym + $size_T * $(r*W),$mask))
        else
            xloadexpr = :(vload($V, $xsym + $size_T * $(r*W)))
        end
        push!(q.args, Expr(:(=), Symbol(:vμ_,r), Expr(:call, :(vmul), xloadexpr, :vβ)))
    end
    p = gensym(:p)
    # update through loop
    loopbody = quote
        vβ = vbroadcast($V, $βsym + $size_T*$p)
        # vβ = vbroadcast($V, $βsym + $(size_T*βstride)*$p)
    end
    for r ∈ 0:Riterl
        if r == Riterl && maskload
            xloadexpr = :(vload($V, $xsym + $size_T * ($(r*W) + $p*$Xstride),$mask))
        else
            xloadexpr = :(vload($V, $xsym + $size_T * ($(r*W) + $p*$Xstride)))
        end
        push!(loopbody.args, Expr(:(=), Symbol(:vμ_,r), Expr(:call, :(vmuladd), xloadexpr, :vβ, Symbol(:vμ_,r))))
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
@noinline function Xβ_load_quote(
    R::Symbol, T::DataType, Xstride::Symbol, βstride::Int, μmy::Bool = true, XP::Int = -1, 
    xsym::Symbol = :ptrX, βsym::Symbol = :ptrβ, maskload::Bool = true, masksym::Union{Unsigned,Symbol} = :__mask__
)
    size_T = sizeof(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    V = Vec{W,T}
    Wm1 = W - 1
    q = quote end
    # Initial load
    push!(q.args, Expr(:(=), :vβ, :(vbroadcast($V, $βsym))))
    xloadexpr = :(vload($V, $xsym, $masksym))
    push!(q.args, Expr(:(=), :vμ_0, Expr(:call, :(vmul), xloadexpr, :vβ)))
    p = gensym(:p)
    # update through loop
    loopbody = quote
        vβ = vbroadcast($V, $βsym + $size_T*$p)
    end
    xloadexpr = :(vload($V, $xsym + $size_T * $p*$Xstride, $masksym))
    push!(loopbody.args, Expr(:(=), :vμ_0, :(vmuladd($xloadexpr, vβ, vμ_0))))
    loop = quote
        for $p ∈ 1:$(XP-1)
            $loopbody
        end
    end
    push!(q.args, loop)
    q
end

@noinline function mutlivariate_normal_SMLT_rowiter(
    config::NormalCholeskyConfiguration{T}, Mk::Int, Nk::Int, col_rem::Int, n_col_reps::Int, μsym::Symbol = :ptrμ, Astride::Int = Mk
) where {T}
    @unpack Ystride, XP, βstride, Xstride, βdim, μtransposed, μstride, μdim = config
    N = Nk * n_col_reps + col_rem
    size_T = sizeof(T)
    if Mk == -1
        Mk = Astride
        maskrowiter = true
    else
        maskrowiter = false
    end
    row_iter = if (βdim == 1 && XP > 0)
        if maskrowiter
            Xβ_load_quote(:row_rem_final, T, Xstride, βstride, false, XP, :ptrX, :ptrβ)
        else
            Xβ_load_quote(Mk, T, Xstride, βstride, false, XP, :ptrX, :ptrβ)
        end
    else
        quote end
    end
    if col_rem > 0# col_rem > 0 may be the failpoint??? TODO try different row_rem to try and isolate failure
        loadδ_expr = load_δ_expr(config, Mk, col_rem, 0, μsym, true, maskrowiter)
        iter_quote = if maskrowiter
            StructuredMatrices.A_rdiv_U_kernel_quote(
                :row_rem_final, col_rem, 0, T, Astride, Ystride, N, true, true, storeA = n_col_reps > 0, loadB = false, reduce_sym = :δ²
            )
        else
            StructuredMatrices.A_rdiv_U_kernel_quote(
                Mk, col_rem, 0, T, Astride, Ystride, N, true, true, storeA = n_col_reps > 0, loadB = false, reduce_sym = :δ²
            )
        end
        #pushfirst!(row_iter.args, :(ptrUtri = ptrUtribase))
        push!(row_iter.args, loadδ_expr)
        push!(row_iter.args, iter_quote)
        push!(row_iter.args, :(ptrUdiag += $(col_rem*size_T)))
        base_K = col_rem
    else
        base_K = 0
    end
    if n_col_reps > 1
        loadδ_expr = load_δ_expr(config, Mk, Nk, :K, μsym, true, maskrowiter)
        iterquote = if maskrowiter
            StructuredMatrices.A_rdiv_U_kernel_quote(
                :row_rem_final, Nk, :K, T, Astride, Ystride, N, true, true, storeA = true, loadB = false, reduce_sym = :δ²
            )
        else
            StructuredMatrices.A_rdiv_U_kernel_quote(
                Mk, Nk, :K, T, Astride, Ystride, N, true, true, storeA = true, loadB = false, reduce_sym = :δ²
            )
        end
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
        push!(row_iter.args, :(ptrUtri = ptrUtribase + $col_rem*$size_T)) # not sure if this is correct. Bug somewhere
        loadδ_expr = load_δ_expr(config, Mk, Nk, col_rem, μsym, true, maskrowiter)
        push!(row_iter.args, loadδ_expr)
        row_iter_single = if maskrowiter
            StructuredMatrices.A_rdiv_U_kernel_quote(
                :row_rem_final, Nk, col_rem, T, Astride, Ystride, N, true, true, storeA = false, loadB = false, reduce_sym = :δ²
            )
        else
            StructuredMatrices.A_rdiv_U_kernel_quote(
                Mk, Nk, col_rem, T, Astride, Ystride, N, true, true, storeA = false, loadB = false, reduce_sym = :δ²
            )
        end
        push!(row_iter.args, row_iter_single)
    end
    row_iter
end

## StructuredMatrices.jl Lower Triangular (SMLT) quote
## M is the sample size
@noinline function multivariate_normal_SMLT_quote(
    config::NormalCholeskyConfiguration{T}
) where {T}
    @unpack M, P, track_Y, track_X, track_β, track_μ, track_L, Ystride, βstride, Xstride, βdim, μdim, μstride, sp, XP, μtransposed, calclogdet, LL, arity = config
    q = quote end
    maxM = M isa Symbol ? typemax(Int) : M
    W, Mk, Nk = StructuredMatrices.div_triangle_blocking_structure(maxM, P, T)
    V = Vec{W,T}
    Wm1 = W - 1
    n_col_reps, col_rem = divrem(P, Nk)
    size_T = sizeof(T)
    caches_invdiag = first(DistributionParameters.caches_invdiag(P, LL, T))
    preloopquote = if caches_invdiag
        quote invdiagL = invdiag(L) end
    elseif !(calclogdet && track_L)
        # alloc_invdiagL = sp ? :(PtrVector{$P,$T,$P,true}(pointer(sptr,$T))) : :(FixedSizeVector{$P,$T,$P}(undef))
        alloc_invdiagL = sp ? :(PtrVector{$P,$T,$P,true}(pointer(sptr,$T))) : :(PtrVector{$P,$T,$P}(SIMDPirates.alloca(Val{$P}(), $T)))
        quote invdiagL = copyto!($alloc_invdiagL, invdiag(L)) end
    else
        quote invdiagLlazy = invdiag(L) end
    end
    Mk2 = min(4, M isa Symbol ? cld(Mk,W) : cld(min(Mk,M),W) )
    q = quote
        # $(Expr(:meta,:inline)) # because of allignment bug
        $([Expr(:(=), Symbol(:δ²_,m), :(vbroadcast($V, zero($T)))) for m ∈ 0:Mk2-1]...)
        ptrY = pointer(Y)
        ptrUtribase = pointer(L) + $(P*size_T)
        $preloopquote
    end
    if (track_L && calclogdet) # we'll calculate invdiagL and the logdet in a single loop
        loopbody = quote δ²_0 = LoopVectorization.vadd(δ²_0, logdiagL[p]) end
        if !caches_invdiag
            alloc_invdiagL = sp ? :(PtrVector{$P,$T,$P,true}(pointer(sptr,$T))) : :(PtrVector{$P,$T,$P}(SIMDPirates.alloca(Val{$P}(),$T)))
            push!(q.args, :(invdiagL = $alloc_invdiagL))
            push!(loopbody.args, :(invdiagL[p] = invdiagLlazy[p]))
        end
        push!(q.args, :(logdiagL = logdiag(L)))
        loopq = quote
            @vvectorize $T for p ∈ 1:$P
                $loopbody
            end
            δ²_0 = vmul(δ²_0, vbroadcast($V,$(M isa Integer ? T(2M) : :($(T(2))*$T($M)))))
        end
        push!(q.args, macroexpand(LoopVectorization, loopq))
    end
    arity >= 4 && push!(q.args, :(ptrX = pointer(X); ptrβ = pointer(β)))
    if (n_col_reps + (col_rem > 0) > 1)
        Aquote = quote
            # A = $(sp ? :(PtrMatrix{$Mk,$P,$T,$Mk}(pointer(sptr,$T) + $(VectorizationBase.align(size_T*P)))) : :(FixedSizeMatrix{$Mk,$P,$T,$Mk}(undef)))
            A = $(sp ? :(PtrMatrix{$Mk,$P,$T,$Mk}(pointer(sptr,$T) + $(VectorizationBase.align(size_T*P)))) : :(PtrMatrix{$Mk,$P,$T,$Mk}(SIMDPirates.alloca(Val{$(P*Mk)}(),$T))))
            ptrA = pointer(A)
        end
        push!(q.args, Aquote) # more than one col iter, and need to store cumulative results
    end
    if μdim == 0
        push!(q.args, Expr(:(=), :ptrμ, :(vbroadcast($V, μ))))
    elseif μdim > 0
        push!(q.args, Expr(:(=), :ptrμ, μtransposed ? :(pointer(μ.parent)) : :(pointer(μ))))
    end
    loop_increments = quote ptrY += $(size_T*Mk) end
    XP > 0 && push!(loop_increments.args, :( ptrX += $(size_T*Mk) ))
    uniqueμbyrow = μdim == 2 || (μdim == 1 && !μtransposed)
    uniqueμbyrow && push!(loop_increments.args, :( ptrμ += $(size_T*Mk) ))
    if M isa Integer
        n_row_reps, row_rem = divrem(M, Mk)
        Mk1 = n_row_reps == 0 ? row_rem : Mk
        row_iter = mutlivariate_normal_SMLT_rowiter(
            config, Mk1, Nk, col_rem, n_col_reps, :ptrμ, Mk
        )
        if n_row_reps > 1
            row_loops = quote
                for rrep ∈ 1:$n_row_reps
                    ptrUdiag = pointer(invdiagL); ptrUtri = ptrUtribase
                    $row_iter
                    $loop_increments
                end
            end
            push!(q.args, row_loops)
        else
            push!(q.args, :(ptrUdiag = pointer(invdiagL); ptrUtri = ptrUtribase))
            push!(q.args, row_iter)
        end
        if row_rem > 0 && n_row_reps > 0
            push!(q.args, :(ptrUdiag = pointer(invdiagL); ptrUtri = ptrUtribase))
            push!(q.args, mutlivariate_normal_SMLT_rowiter( config, row_rem, Nk, col_rem, n_col_reps, :ptrμ, Mk ))
        end
    else # Unknown number of iterations.
        row_iter = mutlivariate_normal_SMLT_rowiter(
            config, Mk, Nk, col_rem, n_col_reps, :ptrμ, Mk
        )
        Wrem, Mkrem, Nkrem = StructuredMatrices.div_triangle_blocking_structure(W, P, T)
        n_col_repsrem, col_remrem = divrem(P, Nkrem)
        row_iter_onevec = mutlivariate_normal_SMLT_rowiter(
            config, W, Nkrem, col_remrem, n_col_repsrem, :ptrμ, W
        )
        row_iter_onevecmask = mutlivariate_normal_SMLT_rowiter(
            config, -1, Nkrem, col_remrem, n_col_repsrem, :ptrμ, W
        )
        loop_increments_onevec = quote ptrY += $(size_T*W) end 
        XP > 0 && push!(loop_increments_onevec.args, :(ptrX += $(size_T*W)))
        uniqueμbyrow && push!(loop_increments_onevec.args, :(ptrμ += $(size_T*W)))
        row_loops = quote
            Mkrep, Mkrem = divrem($M, $Mk)
            for rrep ∈ 1:Mkrep
                ptrUdiag = pointer(invdiagL); ptrUtri = ptrUtribase
                $row_iter
                $loop_increments
            end
            for rrep ∈ 1:Mkrem >>> $(VectorizationBase.intlog2(W))
                ptrUdiag = pointer(invdiagL); ptrUtri = ptrUtribase
                $row_iter_onevec
                $loop_increments_onevec
            end
            row_rem_final = Mkrem & $Wm1
            if row_rem_final != 0
                ptrUdiag = pointer(invdiagL); ptrUtri = ptrUtribase
                __mask__ = VectorizationBase.mask($T, row_rem_final)
                $row_iter_onevecmask
            end
        end
        push!(q.args, row_loops)
    end
    # Reduce the Mk δ² into a single vector.
    R = Mk2
    while R > 1
        Risodd = isodd(R)
        Rh = R >>> 1
        for r ∈ 0:(Rh-1)
            dl = Symbol(:δ²_,r)
            dh = Symbol(:δ²_,r+Rh)
            push!(q.args, :($dl = vadd($dl,$dh)))
        end
        Risodd && push!(q.args, Expr(:(=), :δ²_0, :(vadd(δ²_0, $(Symbol(:δ²_,R-1))))))
        R = Rh
    end
    push!(q.args, Expr(:(=), :δ²_0, :(vmul(vbroadcast($V, $(T(-0.5))), δ²_0))))
    # sp ? push!(q.args, :((sptr,δ²_0))) : push!(q.args, :δ²_0)
    push!(q.args, :δ²_0)
    simplify_expr(q)
    # q
end

typeset!(s::Set{Symbol}, sym::Symbol) = push!(s, sym)
function typeset!(s::Set{Symbol}, arg::Expr)
    postwalk(arg) do ex
        if ex isa Expr
            if ex.head === :curly
                for i ∈ 2:length(ex.args)
                    push!(s, ex.args[i])
                end
            elseif ex.head === :(::)
                ex.args[2] isa Symbol && push!(s, ex.args[2])
            end
        end
        ex
    end
    s
end
calc_whereset(args::Vector{Expr}) = (s = Set{Symbol}(); foreach(arg -> typeset!(s, arg), args); s)
function modify_args!(args, sp, trackval)
    whereargs = calc_whereset(args)
    if trackval
        pushfirst!(args, :(::Val{track}))
        track_ret = :track
        push!(whereargs, :track)
    else
        track_ret = Expr(:tuple, [true for i ∈ 1:length(args)]...)
    end
    if sp
        pushfirst!(args, :(sptr::StackPointer))
    end
    track_ret, whereargs
end

for calclogdet ∈ (true,false)
    dist = calclogdet ? :Normal : :Normal_kernel
    for sp ∈ (true,false)
        for trackval ∈ (true,false)
            args = [:(Y::AbstractMutableFixedSizeMatrix{M,P,T,PY}), :(L::AbstractLowerTriangularMatrix{P,T,LL})]
            track, whereargs = modify_args!(args, sp, trackval)
            # println(whereargs)
            @eval @generated function $dist(
                $(args...)
            ) where {$(whereargs...)}
                track_Y, track_L = $track
                config = NormalCholeskyConfiguration{T}()
                config.arity = 2
                config.M = M; config.P = P; config.track_Y = track_Y; config.track_L = track_L
                config.Ystride = PY; config.sp = $sp; config.calclogdet = $calclogdet; config.LL = LL
                multivariate_normal_SMLT_quote(config)
            end

            args = [:(Y::AbstractMutableFixedSizeMatrix{M,P,T,PY}), :(μ::Tμ), :(L::AbstractLowerTriangularMatrix{P,T,LL})]
            track, whereargs = modify_args!(args, sp, trackval)
            # println(whereargs)
            @eval @generated function $dist(
                $(args...)
            ) where {$(whereargs...)}
                track_Y, track_μ, track_L = $track
                config = NormalCholeskyConfiguration{T}()
                config.M = M; config.P = P; config.track_Y = track_Y; config.track_μ = track_μ; config.track_L = track_L; config.sp = $sp; config.Ystride = PY;
                config.calclogdet = $calclogdet; config.LL = LL; config.arity = 3
                if Tμ === T
                    config.μstride = 0; config.μdim = 0
                elseif Tμ <: LinearAlgebra.Adjoint
                    config.μstride = 1; config.μdim = 1; config.μtransposed = true
                elseif Tμ <: AbstractFixedSizeVector
                    config.μstride = 0; config.μdim = 1
                elseif Tμ <: AbstractFixedSizeMatrix
                    config.μstride = (Tμ.parameters[4].parameters[2])::Int; config.μdim = 2
                else
                    throw("Type of μ == $A is not recognized.")
                end
                multivariate_normal_SMLT_quote(config)
            end

            args = [:(Y::AbstractMutableFixedSizeMatrix{M,P,T,PY}), :(X::AbstractMutableFixedSizeMatrix{M,K_,T,PX}), :(β::AbstractMutableFixedSizeArray{Sβ,T,Nβ,Pβ}), :(L::AbstractLowerTriangularMatrix{P,T,LL})]
            track, whereargs = modify_args!(args, sp, trackval)
            @eval @generated function $dist(
                $(args...)
            ) where {$(whereargs...)}
                track_Y, track_X, track_β, track_L = $track
                @assert Sβ.parameters[1] == K_
                config = NormalCholeskyConfiguration{T}()
                config.βstride = length(Pβ.parameters) == 1 ? K_ : (Pβ.parameters[2])::Int
                config.arity = 4
                @pack! config = M, P, track_Y, track_X, track_β, track_L, LL # pack! where names match
                config.sp = $sp; config.Ystride = PY; config.Xstride = PX; config.βdim = Nβ; config.XP = K_; config.calclogdet = $calclogdet
                multivariate_normal_SMLT_quote(config)
            end

            args = [:(Y::AbstractMutableFixedSizeMatrix{M,P,T,PY}), :(X::AbstractMutableFixedSizeMatrix{M,K_,T,PX}),
                    :(β::AbstractMutableFixedSizeArray{Sβ,T,Nβ,Pβ}), :(μ::Tμ), :(L::AbstractLowerTriangularMatrix{P,T,LL})]
            track, whereargs = modify_args!(args, sp, trackval)
            @eval @generated function $dist(
                $(args...)
            ) where {$(whereargs...)}
                @assert Sβ.parameters[1] == K_
                config = NormalCholeskyConfiguration{T}()
                config.arity = 5
                config.βstride = length(Pβ.parameters) == 1 ? K_ : (Pβ.parameters[2])::Int
                track_Y, track_X, track_β, track_μ, track_L = $track
                @pack! config = M, P, track_Y, track_X, track_β, track_μ, track_L, LL
                config.sp = $sp; config.Ystride = PY; config.Xstride = PX; config.βdim = Nβ; config.XP = K_; config.calclogdet = $calclogdet
                if Tμ === T
                    config.μdim = 0; config.μstride = 0
                elseif Tμ <: LinearAlgebra.Adjoint{T,<:PaddedMatrices.AbstractMutableFixedSizeVector}
                    config.μdim = 1; config.μstride = 1; config.μtransposed = true
                elseif Tμ <: AbstractFixedSizeVector
                    config.μdim = 1; config.μstride = 1
                elseif Tμ <: AbstractFixedSizeMatrix
                    config.μdim = 2; config.μstride = (Tμ.parameters[4].parameters[2])::Int
                else
                    throw("Type of μ == $A is not recognized.")
                end
                multivariate_normal_SMLT_quote(config)
            end
        end
    end
end

for calclogdet ∈ (true,false)
    dist = calclogdet ? :Normal : :Normal_kernel
    for sp ∈ (true,false)
        for trackval ∈ (true,false)        
            args = [:(Y::AbstractMatrix{T}), :(L::AbstractLowerTriangularMatrix{P,T,LL})]
            track, whereargs = modify_args!(args, sp, trackval)
            @eval @generated function $dist(
                $(args...)
            ) where {$(whereargs...)}
                M, PY = gensym(:M), gensym(:PY)
                track_Y, track_L = $track
                config = NormalCholeskyConfiguration{T}()
                config.arity = 2
                @pack! config = M, P, track_Y, track_L, LL
                config.Ystride = PY; config.sp = $sp; config.calclogdet = $calclogdet
                quote
                    $M = size(Y,1)
                    $PY = $(Y <: Array ? M : :(stride(Y,2)))
                    $(multivariate_normal_SMLT_quote(config))
                end
            end

            args = [:(Y::AbstractMatrix{T}), :(μ::Tμ), :(L::AbstractLowerTriangularMatrix{P,T,LL})]
            track, whereargs = modify_args!(args, sp, trackval)
            @eval @generated function $dist(
                $(args...)
            ) where {$(whereargs...)}
                M, PY = gensym(:M), gensym(:PY)
                track_Y, track_μ, track_L = $track
                config = NormalCholeskyConfiguration{T}()
                defs_quote = quote
                    $M = size(Y,1)
                    $PY = $(Y <: Array ? M : :(stride(Y,2)))
                end
                config.arity = 3
                @pack! config = M, P, track_Y, track_μ, track_L, LL
                config.sp = $sp; config.Ystride = PY; config.calclogdet = $calclogdet
                q = if Tμ === T
                    config.μdim = 0; config.μstride = 0
                elseif Tμ <: LinearAlgebra.Adjoint{T,<:AbstractVector{T}}
                    config.μdim = 1; config.μstride = 1; config.μtransposed = true
                elseif Tμ <: AbstractVector# AbstractFixedSizeVector
                    config.μdim = 1; config.μstride = 1
                elseif Tμ <: AbstractFixedSizeMatrix
                    config.μdim = 2; config.μstride = (Tμ.parameters[4].parameters[2])::Int
                elseif Tμ <: AbstractMatrix
                    μstride = gensym(:μstride)
                    config.μdim = 0; config.μstride = μstride
                    push!(defs_quote.args, :($μstride = stride(μ,2)))
                else
                    throw("Type of μ == $A is not recognized.")
                end
                q = multivariate_normal_SMLT_quote(config)
                quote
                    $defs_quote
                    $q
                end
            end

            args = [:(Y::AbstractMatrix{T}), :(X::AbstractMatrix{T}), :(β::AbstractMutableFixedSizeArray{Sβ,T,Nβ,Pβ}), :(L::AbstractLowerTriangularMatrix{P,T,LL})]
            track, whereargs = modify_args!(args, sp, trackval)
            @eval @generated function $dist(
                $(args...)
            ) where {$(whereargs...)}
                K_ = Sβ.parameters[1]
                βstride = length(Pβ.parameters) == 1 ? K_ : (Pβ.parameters[2])::Int
                M, PY, PX = gensym(:M), gensym(:PY), gensym(:PX)
                track_Y, track_X, track_β, track_L = $track
                config = NormalCholeskyConfiguration{T}()
                config.arity = 4
                @pack! config = track_Y, track_X, track_β, track_L, M, P, βstride, LL
                config.calclogdet = $calclogdet; config.sp = $sp; config.Ystride = PY; config.Xstride = PX; config.βdim = Nβ; config.XP = K_
                q = multivariate_normal_SMLT_quote(config)
                quote
                    $M = size(Y,1)
                    $PY = $(Y <: Array ? M : :(stride(Y,2)))
                    $PX = $(X <: Array ? M : :(stride(X,2)))
                    $q
                end
            end
            
            args = [:(Y::AbstractMatrix{T}), :(X::AbstractMatrix{T}), :(β::AbstractMutableFixedSizeArray{Sβ,T,Nβ,Pβ}), :(μ::Tμ), :(L::AbstractLowerTriangularMatrix{P,T,LL})]
            track, whereargs = modify_args!(args, sp, trackval)
            @eval @generated function $dist(
                $(args...)
            ) where {$(whereargs...)}
                K_ = Sβ.parameters[1]
                βstride = length(Pβ.parameters) == 1 ? K_ : (Pβ.parameters[2])::Int
                M, PY, PX = gensym(:M), gensym(:PY), gensym(:PX)
                track_Y, track_X, track_β, track_μ, track_L = $track
                defs_quote = quote
                    $M = size(Y,1)
                    $PY = $(Y <: Array ? M : :(stride(Y,2)))
                    $PX = $(X <: Array ? M : :(stride(X,2)))
                end
                config = NormalCholeskyConfiguration{T}()
                config.arity = 5
                @pack! config = track_Y, track_X, track_β, track_μ, track_L, M, P, βstride, LL
                config.Ystride = PY; config.Xstride = PX; config.βdim = Nβ; config.XP = K_; config.sp = $sp; config.calclogdet = $calclogdet
                q = if Tμ === T
                    config.μdim = 0; config.μstride = 0
                elseif Tμ <: LinearAlgebra.Adjoint{T,<:PaddedMatrices.AbstractMutableFixedSizeVector}
                    config.μdim = 1; config.μstride = 1; config.μtransposed = true
                elseif Tμ <: AbstractFixedSizeVector
                    config.μdim = 1; config.μstride = 1
                elseif Tμ <: AbstractFixedSizeMatrix
                    config.μdim = 2; config.μstride = (Tμ.parameters[4].parameters[2])::Int
                elseif Tμ <: AbstractMatrix
                    μstride = gensym(:μstride)
                    config.μdim = 2; config.μstride = μstride
                    push!(defs_quote.args, :($μstride = stride(μ,2)))
                else
                    throw("Type of μ == $A is not recognized.")
                end
                q = multivariate_normal_SMLT_quote(config)
                quote
                    $defs_quote
                    $q
                end
            end
        end
    end
end

@noinline function store_A_kernel!(q, Mk, Nk, init, sym, W, Wshift, T, stride, negative::Bool = true)#, K = :K)
    Riter = Mk >>> Wshift
    Rrem = Mk & (W - 1)
    mask = VectorizationBase.mask(T, Rrem)
    V = Vec{W,T}; size_T = sizeof(T)
    func = negative ? :vsub : :vadd
    # symK = Symbol(sym, "#K#")
    # Kdiff = K isa Symbol ? :($K - $Nk) : K - Nk
    # push!(q.args, Expr(:(=), symK, :($sym + $size_T * $stride * $Kdiff)))
    for c ∈ 0:(Nk-1)
        for r ∈ 0:Riter-1
            Aexpr = if init
                negative ? :(vsub($(Symbol(:A_,r,:_,c)))) : Symbol(:A_,r,:_,c)
            else
                :($func(vload($V, $sym + $size_T*($(r*W)+$c*$stride)), $(Symbol(:A_,r,:_,c))))
            end
            push!(q.args, :(vstore!($sym + $size_T*($(r*W)+$c*$stride), $Aexpr)))
        end
        if Rrem > 0
            index = :($sym + $size_T*($(Riter*W)+$c*$stride))
            nAsym = if init
                negative ? :(vsub($(Symbol(:A_,Riter,:_,c)))) : Symbol(:A_,Riter,:_,c)
            elseif c == Nk - 1
                :($func(vload($V, $index, $mask), $(Symbol(:A_,Riter,:_,c))))
            else
                :($func(vload($V, $index), $(Symbol(:A_,Riter,:_,c))))
            end
            if c == Nk-1
                push!(q.args, :(vstore!($index, $nAsym, $mask)))
            else
                push!(q.args, :(vstore!($index, $nAsym)))
            end
        end
    end
end
@noinline function store_A_kernel!(q, Mk::Symbol, Nk, init, sym, W, Wshift, T, stride, negative::Bool = true)#, K = :K)
    mask = 
    V = Vec{W,T}; size_T = sizeof(T)
    func = negative ? :vsub : :vadd
    # symK = Symbol(sym, "#K#")
    # Kdiff = K isa Symbol ? :($K - $Nk) : K - Nk
    # push!(q.args, Expr(:(=), symK, :($sym + $size_T * $stride * $Kdiff)))
    for c ∈ 0:(Nk-1)
        index = :($sym + $size_T*($c*$stride))
        nAsym = if init
            negative ? :(vsub($(Symbol(:A_0_,c)))) : Symbol(:A_0_,c)
        elseif c == Nk - 1
            :($func(vload($V, $index, $mask), $(Symbol(:A_0_,c))))
        else
            :($func(vload($V, $index), $(Symbol(:A_0_,c))))
        end
        if c == Nk-1
            push!(q.args, :(vstore!($index, $nAsym, __mask__)))
        else
            push!(q.args, :(vstore!($index, $nAsym)))
        end
    end
end

@noinline function track_mu_store(
    Mk::Int,Nk,T,μdim,μmy,W,Wshift,μstride,track_Y,μtransposed,
    initialize::Bool=false,initμ::Bool=true,initY::Bool=true#,K = :K
)::Expr
    size_T = sizeof(T)
    V = Vec{W,T}
    Riter = Mk >>> Wshift
    Rrem = Mk & (W-1)
    Riterl = Rrem > 0 ? Riter : Riter-1
    mask = VectorizationBase.mask(T, Rrem)
    row_iter = quote end
    f = μmy ? :vsub : :vadd
    if μdim == 0
        iter = 0
        for c ∈ 0:(Nk-1), m ∈ 0:Riterl
            mask_this_iter = m == Riterl && Rrem > 0
            pm = Symbol(:∂μ_,iter & 3)
            A_m_c = Symbol(:A_,m,:_,c)
            if mask_this_iter
                push!(row_iter.args, Expr(:(=), pm, :(vifelse($mask,$f($pm, $A_m_c),$p_m))))
            else
                push!(row_iter.args, Expr(:(=), pm, :($f($pm, $A_m_c))))
            end
            iter += 1
        end
    elseif μdim == 1
        if μtransposed
            for c ∈ 0:(Nk-1)
                mc = Symbol(:vμ_,c)
                push!(row_iter.args, Expr(:(=), mc, :(vload($V, ptrv∂μ + $(c*W*size_T)))))
            end
            for m ∈ 0:Riterl
                mask_this_iter = m == Riterl && Rrem > 0
                for c ∈ 0:(Nk-1)
                    mc = Symbol(:vμ_,c)
                    if mask_this_iter
                        push!(row_iter.args, Expr(:(=), mc, :(vifelse($mask,$f($mc, $(Symbol(:A_,m,:_,c))),$mc))))
                    else
                        push!(row_iter.args, Expr(:(=), mc, :($f($mc, $(Symbol(:A_,m,:_,c))))))
                    end
                end
            end
            for c ∈ 0:(Nk-1)
                mc = Symbol(:vμ_,c)
                push!(row_iter.args, :(vstore!(ptrv∂μ + $(c*W)*$size_T, $mc)))
            end
        else
            if initialize
                if initμ
                    if μmy
                        for m ∈ 0:Riterl
                            push!(row_iter.args, Expr(:(=), Symbol(:v∂μ_, m), :(vsub($(Symbol(:A_,m,:_0))))))
                        end
                    else
                        for m ∈ 0:Riterl
                            push!(row_iter.args, Expr(:(=), Symbol(:v∂μ_, m), Symbol(:A_,m,:_0)))
                        end
                    end
                else
                    func = μmy ? :vsub : :vadd
                    for m ∈ 0:Riterl
                        push!(row_iter.args, Expr(:(=), Symbol(:v∂μ_, m), :($func(vload($V, ptr∂μ + $(m*W*size_T)),$(Symbol(:A_,m,:_0))))))
                    end
                end
                firstc = 1
            else
                for m ∈ 0:Riterl
                    # We don't mask here, because beyond the end of the vector is junk, and since things are padded to allignment, the vector wont encroach on data we care about.
                    # if m == Riterl
                    push!(row_iter.args, Expr(:(=), Symbol(:v∂μ_, m), :(vload($V, ptr∂μ + $(m*W*size_T) ) )))
                end
                firstc = 0
            end
            for c ∈ firstc:Nk-1
                for m ∈ 0:Riterl
                    pm = Symbol(:v∂μ_,m)
                    push!(row_iter.args, Expr(:(=), pm, :($f($pm, $(Symbol(:A_,m,:_,c))))))
                end
            end
            for m ∈ 0:Riterl
                # if m == Riterl && Rrem > 0
                    # push!(row_iter.args, :(vstore!(ptr∂μ + $(m*W*size_T), $(Symbol(:v∂μ_, m)), $mask )))
                # else
                    push!(row_iter.args, :(vstore!(ptr∂μ + $(m*W)*$size_T, $(Symbol(:v∂μ_, m)) )))
                # end
            end
        end
    elseif μdim == 2 # if ∂μ is not aliasing A, we need to store.
        need_to_store = if track_Y && initY # A aliases ∂Y, need to store ∂μ
            true
        else# 
            !initμ
        end
        need_to_store && store_A_kernel!(row_iter, Mk, Nk, initμ, :ptr∂μ_rev, W, Wshift, T, μstride, μmy)#, K)
    end
    row_iter
end
@noinline function track_mu_store(
    Mk::Symbol,Nk,T,μdim,μmy,W,Wshift,μstride,track_Y,μtransposed,
    initialize::Bool=false,initμ::Bool=true,initY::Bool=true#,K = :K
)::Expr
    size_T = sizeof(T)
    V = Vec{W,T}
    row_iter = quote end
    f = μmy ? :vsub : :vadd
    if μdim == 0
        iter = 0
        for c ∈ 0:(Nk-1)
            pm = Symbol(:∂μ_,iter & 3)
            A_m_c = Symbol(:A_0_,c)
            push!(row_iter.args, Expr(:(=), pm, :(vifelse(__mask__,$f($pm, $A_m_c),$p_m))))
            iter += 1
        end
    elseif μdim == 1
        if μtransposed
            for c ∈ 0:(Nk-1)
                mc = Symbol(:vμ_,c)
                push!(row_iter.args, Expr(:(=), mc, :(vload($V, ptrv∂μ + $(c*W*size_T)))))
            end
            for c ∈ 0:(Nk-1)
                mc = Symbol(:vμ_,c)
                push!(row_iter.args, Expr(:(=), mc, :(vifelse(__mask__,$f($mc, $(Symbol(:A_0_,c))),$mc))))
            end
            for c ∈ 0:(Nk-1)
                mc = Symbol(:vμ_,c)
                push!(row_iter.args, :(vstore!(ptrv∂μ + $(c*W)*$size_T, $mc)))
            end
        else
            if initialize
                if initμ
                    if μmy
                        push!(row_iter.args, :(v∂μ_0 = vsub(A_0_0)))
                    else
                        push!(row_iter.args, :(v∂μ_0 = A_0_0))
                    end
                else
                    func = μmy ? :vsub : :vadd
                    push!(row_iter.args, :(v∂μ_0 = $func(vload($V, ptr∂μ, A_0_0))))
                end
                firstc = 1
            else
                push!(row_iter.args, :(v∂μ_0 = vload($V, ptr∂μ )))
                firstc = 0
            end
            for c ∈ firstc:Nk-1
                push!(row_iter.args, :(v∂μ_0 = $f(v∂μ_0, $(Symbol(:A_0_,c)))))
            end
            push!(row_iter.args, :(vstore!(ptr∂μ, v∂μ_0)))
        end
    elseif μdim == 2 # if ∂μ is not aliasing A, we need to store.
        need_to_store = if track_Y && initY # A aliases ∂Y, need to store ∂μ
            true
        else# 
            !initμ
        end
        need_to_store && store_A_kernel!(row_iter, Mk, Nk, initμ, :ptr∂μ_rev, W, Wshift, T, μstride, μmy)#, K)
    end
    row_iter
end

function ∂Y_distinct_from_A(config)::Bool
    @unpack track_Y, initY = config
    track_Y && !initY
end
function ∂μ2d_distinct_from_A(config)::Bool # returns true when track_fsμ && A != ∂μ
    @unpack track_Y, initY, track_μ, initμ, μdim = config
    (track_μ && (μdim == 2)) || return false
    if track_Y && initY
        true
    else
        !initμ
    end
end

"""
Sets pointers back columns during the reverse pass over rows.
"""
@noinline function loop_pointer_increments(config::NormalCholeskyConfiguration{T}, Nk, K, W, Astride) where {T}
    @unpack track_μ, track_L, Ystride, μstride, μdim, μtransposed = config
    @unpack initY, initX, initβ, initμ = config
    size_T = sizeof(T)

    b2Nk = StructuredMatrices.binomial2(Nk)
    loop_ptr_increments = quote
        ptrLdiag -= $(size_T*Nk)
        ptrLtri -= $size_T*($Nk*$K + $b2Nk)
        ptrA_rev -= $size_T*$Nk*$Astride 
    end
    ∂Y_distinct_from_A(config) && push!(loop_ptr_increments.args, :(ptr∂Y_rev -= $size_T*$Nk*$Ystride))
    ∂μ2d_distinct_from_A(config) && push!(loop_ptr_increments.args, :(ptr∂μ_rev -= $size_T*$Nk*$μstride))
    if track_μ && μdim == 1 && μtransposed
        push!(loop_ptr_increments.args, Expr(:(-=), :ptrv∂μ, Nk isa Symbol ? :($Nk*$(size_T*W)) : Nk*size_T*W))
    end
    if track_L
        push!(loop_ptr_increments.args, :(ptrv∂Ldiag -= $(size_T*W)*$Nk; ptrv∂Ltri -= $(size_T*W)*($Nk*$K+$b2Nk)))
    end
    loop_ptr_increments
end

@noinline function init_reverse_pointers!(row_iter, config)
    @unpack track_Y, track_X, track_β, track_μ, track_L, βstride, Xstride, Ystride, μstride, μdim, βdim, XP, μtransposed = config
    @unpack initY, initX, initβ, initμ = config
    track_Xβ = track_X | track_β
    track_fsμ = track_μ & (μdim == 2)

    # set starting pointers for reverse pass
    push!(row_iter.args, :(ptrLdiag = ptrLdiagbase; ptrLtri = ptrLtribase; ptrA_rev = ptrA + _A_offset_))
    # mu has to change columnwise for us to set the pointer here;
    # if we aren't tracking y, then ptrA aliases it, so setting it is unnecessary.
    if track_μ && (μdim == 1 && μtransposed)# || (!track_Y && μdim == 2))
        push!(row_iter.args, :(ptrv∂μ = ptrv∂μbase))
    end
    if track_L
        push!(row_iter.args, :(ptrv∂Ltri = ptrv∂Ltribase; ptrv∂Ldiag = ptrv∂Ldiagbase))
    end
    ∂Y_distinct_from_A(config) && push!(row_iter.args, :(ptr∂Y_rev = ptr∂Y + _Y_offset_))
    ∂μ2d_distinct_from_A(config) && push!(row_iter.args, :(ptr∂μ_rev = ptr∂μ + _μ_offset_))
    nothing
end


function load_δ_expr(
    config::NormalCholeskyConfiguration{T}, Mk, Nk, K, μsym, μmy, maskrowiter
) where {T}
    @unpack Ystride, Xstride, βstride, βdim, XP, μstride, μdim, μtransposed = config
    if XP > 0
        if maskrowiter
            loadδfnmadd_quote(
                :row_rem_final, Nk, K, T, Ystride, Xstride, βstride, βdim,
                :ptrY, :ptrX, :ptrβ, :ptrμ, true, μmy, XP, μstride, μdim, μtransposed
            )
        else
            loadδfnmadd_quote(
                Mk,             Nk, K, T, Ystride, Xstride, βstride, βdim,
                :ptrY, :ptrX, :ptrβ, :ptrμ, true, μmy, XP, μstride, μdim, μtransposed
            )
        end
    else
        if maskrowiter
            loadδ_quote(:row_rem_final, Nk, K, T, Ystride, :ptrY, μdim, μstride, μsym, true, μmy, μtransposed)
        else
            loadδ_quote(Mk,             Nk, K, T, Ystride, :ptrY, μdim, μstride, μsym, true, μmy, μtransposed)
        end
    end
end

@noinline function ∂mutlivariate_normal_SMLT_rowiter(
    config::NormalCholeskyConfiguration{T},
    Mk::Int, Nk::Int, col_rem::Int,
    n_col_reps::Int, μmy::Bool, μsym::Symbol = :μptr,
    Astride::Union{Int,Symbol} = Ystride
) where {T}
    @unpack track_Y, track_X, track_β, track_μ, track_L, βstride, Xstride, Ystride, μstride, μdim, βdim, XP, μtransposed = config
    @unpack initY, initX, initβ, initμ = config
    track_Xβ = track_X | track_β
    track_fsμ = track_μ & (μdim == 2)
    if Mk == -1
        W, Wshift = VectorizationBase.pick_vector_width_shift(T) # row_rem_final
        maskrowiter = true
        Mk = W
    else
        W, Wshift = VectorizationBase.pick_vector_width_shift(Mk, T)
        maskrowiter = false
    end
    V = Vec{W,T}
    N = Nk * n_col_reps + col_rem
    size_T = sizeof(T)
    if βdim == 1 && XP > 0
        if maskrowiter
            row_iter = Xβ_load_quote(:row_rem_final, T, Xstride, βstride, μmy, XP, :ptrX, :ptrβ)
        else
            row_iter = Xβ_load_quote(Mk, T, Xstride, βstride, μmy, XP, :ptrX, :ptrβ)
        end
    else
        row_iter = quote end
    end
    if col_rem > 0
        loadδ_expr = load_δ_expr(config, Mk, col_rem, 0, μsym, μmy, maskrowiter)
        iter_quote = if maskrowiter
            StructuredMatrices.A_rdiv_U_kernel_quote( :row_rem_final, col_rem, 0, T, Astride, Ystride, N, true, true, storeA = true, loadB = false, reduce_sym = :δ² )
        else
            StructuredMatrices.A_rdiv_U_kernel_quote( Mk, col_rem, 0, T, Astride, Ystride, N, true, true, storeA = true, loadB = false, reduce_sym = :δ² )
        end
        #pushfirst!(row_iter.args, :(ptrUtri = ptrUtribase))
        push!(row_iter.args, loadδ_expr)
        push!(row_iter.args, iter_quote)
        push!(row_iter.args, :(ptrUdiag += $(col_rem*size_T)))
        base_K = col_rem
    else
        base_K = 0
    end
    if n_col_reps > 1
        loadδ_expr = load_δ_expr(config, Mk, Nk, :K, μsym, μmy, maskrowiter)
        iterquote = if maskrowiter
            StructuredMatrices.A_rdiv_U_kernel_quote(
                :row_rem_final, Nk, :K, T, Astride, Ystride, N, true, true, storeA = true, loadB = false, reduce_sym = :δ²
            )
        else
            StructuredMatrices.A_rdiv_U_kernel_quote(
                Mk, Nk, :K, T, Astride, Ystride, N, true, true, storeA = true, loadB = false, reduce_sym = :δ²
            )
        end
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
        loadδ_expr = load_δ_expr(config, Mk, Nk, col_rem, μsym, μmy, maskrowiter)
        push!(row_iter.args, loadδ_expr)
        push!(row_iter.args, :(ptrUtri = ptrUtribase + $col_rem*$size_T))
        row_iter_single = if maskrowiter
            StructuredMatrices.A_rdiv_U_kernel_quote(
                :row_rem_final, Nk, col_rem, T, Astride, Ystride, N, true, true, storeA = true, loadB = false, reduce_sym = :δ² # storeA = col_rem > 0
            )
        else
            StructuredMatrices.A_rdiv_U_kernel_quote(
                Mk, Nk, col_rem, T, Astride, Ystride, N, true, true, storeA = true, loadB = false, reduce_sym = :δ² # storeA = col_rem > 0
            )
        end
        push!(row_iter.args, row_iter_single)
    end
    ########################
    ### now time for ÷ L ###
    ########################
    init_reverse_pointers!(row_iter, config)
    store_Y = ∂Y_distinct_from_A(config)
    if col_rem > 0
        row_iter_rev = if maskrowiter
            StructuredMatrices.A_rdiv_L_kernel_quote(
                :row_rem_final, col_rem, col_rem, T, Astride, Astride, false, true,
                Bsym = :ptrA_rev, Asym = :ptrA_rev, Ltrisym = :ptrLtri, Ldiagsym = :ptrLdiag,
                loadB = true, storeA = true, calc_product = track_L ? N : 0
            )
        else
            StructuredMatrices.A_rdiv_L_kernel_quote(
                Mk, col_rem, col_rem, T, Astride, Astride, false, true,
                Bsym = :ptrA_rev, Asym = :ptrA_rev, Ltrisym = :ptrLtri, Ldiagsym = :ptrLdiag,
                loadB = true, storeA = true, calc_product = track_L ? N : 0
            )
        end
        fullcols = Nk * n_col_reps
        # handle following in A_rdiv_L_quote
        append!(row_iter.args, row_iter_rev.args)
        if maskrowiter
            track_μ && push!(row_iter.args, track_mu_store(:row_rem_final,col_rem,T,μdim,μmy,W,Wshift,μstride,track_Y,μtransposed,true,initμ,initY))#, col_rem))
            store_Y && store_A_kernel!(row_iter, :row_rem_final, col_rem, initY, :ptr∂Y_rev, W, Wshift, T, Ystride, !μmy)#, col_rem)
        else
            track_μ && push!(row_iter.args, track_mu_store(Mk,col_rem,T,μdim,μmy,W,Wshift,μstride,track_Y,μtransposed,true,initμ,initY))#, col_rem))
            store_Y && store_A_kernel!(row_iter, Mk, col_rem, initY, :ptr∂Y_rev, W, Wshift, T, Ystride, !μmy)#, col_rem)
        end
        push!(row_iter.args, loop_pointer_increments(config, Nk, col_rem, W, Astride))
        base_K = col_rem
    else
        base_K = 0
    end
    loop_ptr_increments = loop_pointer_increments(config, Nk, :K, W, Astride)
    if n_col_reps > 1
        iterquote = if maskrowiter
            StructuredMatrices.A_rdiv_L_kernel_quote(
                :row_rem_final, Nk, :K, T, Astride, Astride, false, true,
                Bsym = :ptrA_rev, Asym = :ptrA_rev, Ltrisym = :ptrLtri, Ldiagsym = :ptrLdiag,
                loadB = true, storeA = true, calc_product = track_L ? N : 0
            )
        else
            StructuredMatrices.A_rdiv_L_kernel_quote(
                Mk, Nk, :K, T, Astride, Astride, false, true,
                Bsym = :ptrA_rev, Asym = :ptrA_rev, Ltrisym = :ptrLtri, Ldiagsym = :ptrLdiag,
                loadB = true, storeA = true, calc_product = track_L ? N : 0
            )
        end
        if col_rem == 0 && !μtransposed && track_μ && μdim == 1 # then we need to zero-initialize these rows before entering the loop
            if maskrowiter
                push!(row_iter.args, :(vstore!(ptr∂μ, vbroadcast($V, zero($T)), __mask__)))
            else
                Riter = Mk >>> Wshift
                Rrem = Mk & (W-1)
                Riterl = Rrem > 0 ? Riter : Riter-1
                for r ∈ 0:Riterl
                    push!(row_iter.args, :(vstore!(ptr∂μ + $(r*W*size_T), vbroadcast($V, zero($T)))))
                end
            end
        end
        if maskrowiter
            track_μ && push!(iterquote.args, track_mu_store(:row_rem_final,Nk,T,μdim,μmy,W,Wshift,μstride,track_Y,μtransposed,false,initμ,initY))#, :K))
            store_Y && store_A_kernel!(iterquote, :row_rem_final, Nk, initY, :ptr∂Y_rev, W, Wshift, T, Ystride, !μmy)#, :K)
        else
            track_μ && push!(iterquote.args, track_mu_store(Mk,Nk,T,μdim,μmy,W,Wshift,μstride,track_Y,μtransposed,false,initμ,initY))#, :K))
            store_Y && store_A_kernel!(iterquote, Mk, Nk, initY, :ptr∂Y_rev, W, Wshift, T, Ystride, !μmy)#, :K)
        end
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
        store_A = track_Xβ || (track_fsμ && initμ) || (track_Y && initY)
        row_iter_rev_single = if maskrowiter
            StructuredMatrices.A_rdiv_L_kernel_quote(
                :row_rem_final, Nk, N, T, Astride, Astride, false, true,
                Bsym = :ptrA_rev, Asym = :ptrA_rev, Ltrisym = :ptrLtri, Ldiagsym = :ptrLdiag,
                loadB = true, storeA = store_A, calc_product = track_L ? N : 0
            )
        else
            StructuredMatrices.A_rdiv_L_kernel_quote(
                Mk, Nk, N, T, Astride, Astride, false, true,
                Bsym = :ptrA_rev, Asym = :ptrA_rev, Ltrisym = :ptrLtri, Ldiagsym = :ptrLdiag,
                loadB = true, storeA = store_A, calc_product = track_L ? N : 0
            )
        end
        push!(row_iter.args, row_iter_rev_single)
        if maskrowiter
            track_μ && push!(row_iter.args, track_mu_store(:row_rem_final,Nk,T,μdim,μmy,W,Wshift,μstride,track_Y,μtransposed,col_rem == 0,initμ,initY))#, N))
            store_Y && store_A_kernel!(row_iter, :row_rem_final, Nk, initY, :ptr∂Y_rev, W, Wshift, T, Ystride, !μmy)#, N)
        else
            track_μ && push!(row_iter.args, track_mu_store(Mk,Nk,T,μdim,μmy,W,Wshift,μstride,track_Y,μtransposed,col_rem == 0,initμ,initY))#, N))
            store_Y && store_A_kernel!(row_iter, Mk, Nk, initY, :ptr∂Y_rev, W, Wshift, T, Ystride, !μmy)#, N)
        end
    end
    row_iter
end

@noinline function allocate_fullsize_A_stackpointer!(
    row_increments, row_increments_rem, sptroff, config::NormalCholeskyConfiguration{T}, W, Mk, Nk, invdiagLL
) where {T}
    @unpack M, P, track_Y, track_X, track_β, track_μ, track_L, βstride, Xstride, Ystride, μstride, μdim, sp, βdim, XP, μtransposed, arity = config
    size_T = sizeof(T)
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
                push!(Aquote.args, :(_sptr += $(invdiagLL*size_T)))
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
        if βdim == 1
            alignβoffset = VectorizationBase.align(PaddedMatrices.calc_padding(XP,T)*P*size_T)
            # if βdim is 1, we would prefer for the matrix to have its colums aligned
            push!(Aquote.args, :(∂β = PtrMatrix{$XP,$P,$T}(_sptr); ptr∂β = _sptr))
            XPL = VectorizationBase.align(XP, T)
            push!(Aquote.args, :(_sptr += $(XPL*size_T))) # impacts the pointer we ultimately return
            # first increment (because of if/else statements), so we could (and did) turn the += into an =
            # gives extra offset for future allocations
            sptroff += alignβoffset - XPL * size_T
            push!(Aquote.args, Expr(:(=), :∂βv, :(PtrVector{$XP,$T}(ptr∂β))))
        else
            alignβoffset = VectorizationBase.align(XP*P*size_T)
            # Because we return it (and it may map directly onto the returned gradient), we would prefer it to have no padding between columns
            push!(Aquote.args, :(∂β = PtrMatrix{$XP,$P,$T,$XP}(_sptr); ptr∂β = _sptr))
            push!(Aquote.args, :(_sptr += $alignβoffset))
        end
    end
    delay_alloc && push!(Aquote.args, delayed_allocation_quote)
    nonempty_sptroff_expr = false
    if must_still_allocate_A
        if M isa Integer
            if sptroff == 0
                push!(Aquote.args, :(A = PtrMatrix{$M,$P,$T,$Astride}(_sptr) ))
            else#if sptroff != 0
                push!(Aquote.args, :(A = PtrMatrix{$M,$P,$T,$Astride}(_sptr + $sptroff) ))
            end
            sptroffexpr = quote end
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
    else
        sptroffexpr = quote end        
    end
    Aquote, Astride, sptroff, sptroffexpr, nonempty_sptroff_expr
end

@noinline function allocate_partials_stackpointer!(row_increments, row_increments_rem, config::NormalCholeskyConfiguration{T}, W, Mk, Nk, invdiagLL, ∂LL) where {T}
    @unpack M, P, track_Y, track_X, track_β, track_μ, track_L, βstride, Xstride, Ystride, μstride, μdim, sp, βdim, XP, μtransposed, arity = config
    sptroff = 0
    size_T = sizeof(T)
    nonempty_sptroff_expr = false
    if !(track_Y || track_μ || track_X || track_β)# don't need to track A
        Aquote = quote
            A = PtrMatrix{$Mk,$P,$T,$Mk}(_sptr)
            ptrA = pointer(A)
        end
        sptroff = VectorizationBase.align(Mk*P*size_T)
        Astride = Mk
        sptroffexpr = quote end
    else # We track at least one of the four
        if (μdim == 1) && !(track_Y || track_X || track_β) # We do not track or store all of A, so we make it a MK x P block to hold a single set of iterations across columns
            if μtransposed
                Aquote = quote
                    ∂μ = PtrVector{$P,$T}(_sptr)
                    ptr∂μ = _sptr
                    _sptr += $(invdiagLL*size_T)
                    v∂μ = PtrMatrix{$W,$P,$T,$W,$(W*P)}(_sptr) # accmulate in v∂μ; reduce at end
                    ptrv∂μ = _sptr
                end
                sptroff = W*P*size_T # aligned because of W; we only care about alignment to nearest vector width, ie, 32 bytes is fine.
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
            sptroffexpr = quote end
        else# We do create a full-sized (size(A) == size(Y)) A-matrix
            Aquote, Astride, sptroff, sptroffexpr, nonempty_sptroff_expr = allocate_fullsize_A_stackpointer!(row_increments, row_increments_rem, sptroff, config, W, Mk, Nk, invdiagLL)
        end
    end
    final_offset_expr = if nonempty_sptroff_expr
        sptroff == 0 ? :(_sptr + $sptroffexpr) : :(_sptr + $sptroff + $sptroffexpr)
    else
        sptroff == 0 ? :(_sptr) : :(_sptr + $sptroff)
    end
    if track_L
        push!(Aquote.args, :(v∂L = StructuredMatrices.PtrLowerTriangularMatrix{$P,Vec{$W,$T},$∂LL}( $final_offset_expr )))
    else # allocate invdiagLL at the end
        push!(Aquote.args, :(invdiagL = PtrVector{$P,$T,$invdiagLL}( $final_offset_expr )))
    end        
    Aquote, Astride
end
function allocate_partials_no_stackpointer!(
    row_increments, row_increments_rem, config::NormalCholeskyConfiguration{T}, W, Mk, Nk, invdiagLL
) where {T}
    @unpack M, P, track_Y, track_X, track_β, track_μ, track_L, βstride, Xstride, Ystride, μstride, μdim, sp, βdim,  XP, μtransposed, arity = config
    # Life is easier if we don't use our own stack, because
    # now we don't have to bother sorting the parameters on said stack ourselves.
    # Nor do we have to worry about keeping the stack (REGISTER_SIZE)-bytes alligned
    size_T = sizeof(T)
    if !(track_Y || track_μ || track_X || track_β)# don't need to track A
        Aquote = quote
            A = FixedSizeMatrix{$Mk,$P}(undef)
            ptrA = pointer(A)
        end
        Astride = Mk
    else # We track at least one of the four
        if (μdim == 1) && !(track_Y || track_X || track_β) # We do not track or store all of A, so we make it a MK x P block to hold a single set of iterations across columns
            # To be here, we must track_μ
            if μtransposed
                Aquote = quote
                    ∂μ = FixedSizeVector{$P,$T}(undef)
                    v∂μ = FixedSizeMatrix{$W,$P,$T,$W,$(W*P)}(undef) # accmulate in v∂μ; reduce at end
                end
            else
                Aquote = if M isa Integer
                    quote ∂μ = FixedSizeVector{$M,$T}(undef) end
                else
                    quote ∂μ = Vector{$T}(undef, $M) end
                end
            end
            push!(Aquote.args, :(ptr∂μ = pointer(∂μ)))
            push!(Aquote.args, :(A = FixedSizeMatrix{$Mk,$P,$T,$Mk}(undef); ptrA = pointer(A)))
            Astride = Mk
        else# We do create a full-sized (size(A) == size(Y)) A-matrix
            Astride = M isa Integer ? VectorizationBase.align(M, W) : M
            # Therefore, we must increment through row iterations
            push!(row_increments.args, :(ptrA += $(size_T*Mk)))
            push!(row_increments_rem.args, :(ptrA += $(size_T*W)))
            Aquote = quote end
            Aexpr = M isa Integer ? :(FixedSizeMatrix{$M,$P,$T,$Astride}(undef)) :  :(Matrix{$T}(undef, $M, $P))
            push!(Aquote.args, :(A = $Aexpr))
            push!(Aquote.args, :(ptrA = pointer(A)))
            #end
            if track_X
                if M isa Integer
                    push!(Aquote.args, :(∂X = FixedSizeMatrix{$M,$XP,$T,$Astride}(undef)))
                else
                    push!(Aquote.args, :(∂X = Matrix{$T}(undef, $M,$XP))  )
                end
                push!(Aquote.args, :(ptr∂X = pointer(∂X)))
            end
            if track_μ
                if μdim == 1
                    if μtransposed
                        PL = VectorizationBase.align(P, W) # align the number of columns to SIMD width
                        push!(Aquote.args, :(∂μ = FixedSizeVector{$P,$T}(undef)))
                        push!(Aquote.args, :(v∂μ = FixedSizeMatrix{$W,$P,$T,$W,$(W*P)}(undef))) # accmulate in v∂μ; reduce at end
                        push!(Aquote.args, :(ptrv∂μ = pointer(v∂μ)))
                    else#if !μtransposed
                        if M isa Integer
                            push!(Aquote.args, :(∂μ = FixedSizeVector{$M,$T,$Astride}(undef)))
                        else#if M isa Symbol
                            push!(Aquote.args, :(∂μ = Vector{$T}(undef, $M)))
                        end
                    end
                elseif track_Y# && μdim == 2
                    if M isa Integer # Y
                        push!(Aquote.args, :(∂μ = FixedSizeMatrix{$M,$P,$T,$Astride}(undef)))
                    else#if !(M isa Symbol)
                        push!(Aquote.args, :(∂μ = Matrix{$T}(undef,$M,$P)))
                    end
                end
                if ((μdim == 1) && !μtransposed) || ((μdim == 2) && track_Y)
                    push!(row_increments.args, :(ptr∂μ += $(size_T*Mk)))
                    push!(row_increments_rem.args, :(ptr∂μ += $(size_T*W)))
                end
                push!(Aquote.args, :(ptr∂μ = pointer(∂μ)))
            end
            if track_β # we vbroadcast from β rather than load, so no point alligning columns
                push!(Aquote.args, :(∂β = FixedSizeMatrix{$XP,$P,$T}(undef)))
                if βdim == 1
                    push!(Aquote.args, Expr(:(=), :∂βv, :(FixedSizeVector{$XP,$T}(undef))))
                end
            end
        end
    end
    Aquote, Astride
end


@noinline function allocate_partials_quote!(row_increments, row_increments_rem, config::NormalCholeskyConfiguration{T}, W, Mk, Nk, invdiagLL) where {T}
    @unpack M, P, track_Y, track_X, track_β, track_μ, track_L, βstride, Xstride, Ystride, μstride, μdim, sp, βdim, XP, μtransposed, arity = config
    size_T = sizeof(T)
    array_allocations = sp ? quote _sptr = pointer(sptr,$T) end : quote end
    ∂LL = 0
    if track_L
        ∂LL = VectorizationBase.align(StructuredMatrices.binomial2(P + 1), W)
        if sp
            push!(array_allocations.args, :(∂L = StructuredMatrices.PtrLowerTriangularMatrix{$P,$T,$∂LL}(_sptr)))
            push!(array_allocations.args, :(invdiagL = PtrVector{$P,$T,$P}(_sptr)))
            push!(array_allocations.args, :(_sptr += $(VectorizationBase.align(∂LL*size_T))))
        else
            push!(array_allocations.args, :(v∂L = StructuredMatrices.MutableLowerTriangularMatrix{$P,Vec{$W,$T},$∂LL}(undef)))
            push!(array_allocations.args, :(∂L = StructuredMatrices.MutableLowerTriangularMatrix{$P,$T,$∂LL}(undef)))
            push!(array_allocations.args, :(invdiagL = PtrVector{$P,$T,$P}(pointer(∂L))))
        end
    elseif !sp
        push!(array_allocations.args, :(invdiagL = FixedSizeVector{$P,$T,$invdiagLL}(undef)))
    end
    Aquote, Astride = if sp # define sptroff, the offset of the sptr relative to the end of the last returned object (where a non-returned object would start)
        allocate_partials_stackpointer!(row_increments, row_increments_rem, config, W, Mk, Nk, invdiagLL, ∂LL)
    else#if !sp
        allocate_partials_no_stackpointer!(row_increments, row_increments_rem, config, W, Mk, Nk, invdiagLL)
    end
    push!(array_allocations.args, Aquote)
    array_allocations, Astride
end
@noinline function alloc_A_kernel(Mk::Int, P::Int, ::Type{T}) where {T}
    Aquote = quote
        A = PtrMatrix{$Mk,$P,$T,$Mk}(SIMDPirates.alloca(Val{$(Mk*P)}(), $T))
    end
    Astride = Mk
    Aquote, Astride
end
@noinline function alloc_A_fullsize!(row_increments, row_increments_rem, ::Type{T}, M, Mk, P, W, sp) where {T}
    size_T = sizeof(T)
    push!(row_increments.args, :(ptrA += $(size_T*Mk)))
    push!(row_increments_rem.args, :(ptrA += $(size_T*W)))
    if M isa Integer
        Astride = VectorizationBase.align(M, W)
        Aquote = quote end
        if sp #&& Astride * P > 100_000 # If the matrix has more than 100_000 elements, we'll use the StackPointer.
            push!(Aquote.args, :(A = PtrMatrix{$M,$P,$T,$Astride}(_sptr)))
            push!(Aquote.args, :(_sptr += $size_T*$(Astride*P)))
        else
            push!(Aquote.args, :(A = PtrMatrix{$M,$P,$T,$Astride}(SIMDPirates.alloca(Val{$(Astride*P)}(),$T))))
        end
    else
        Astride = :_A_stride_
        Aquote = quote _A_stride_ = VectorizationBase.align($M,$W) end
        if sp # If it is dynamically sized, we don't want to allocate an absurd amount of memory, so we use the StackPointer
            push!(Aquote.args, :(A = DynamicPtrMatrix{$T}(_sptr, ($M, $P), _A_stride_)))
            push!(Aquote.args, :(_sptr += $size_T*$P*_A_stride_))
        else
            push!(Aquote.args, :(A = DynamicPaddedMatrix{$T}(undef, ($M, $P), _A_stride_)))
        end
    end
    Aquote, Astride::Union{Symbol,Int}
end
@noinline function sym_aliases_A!(row_increments, row_increments_rem, sym, size_T, M, Mk, P, W, sp) # TODO: use stride of aliasing symbol
    push!(row_increments.args, :(ptrA += $(size_T*Mk)))
    push!(row_increments_rem.args, :(ptrA += $(size_T*W)))
    if M isa Integer
        Astride = VectorizationBase.align(M, W)
        Aquote = quote end
    else
        Astride = :_A_stride_
        Aquote = quote _A_stride_ = VectorizationBase.align($M,$W) end
    end
    push!(Aquote.args, :(A = $sym))
    Aquote, Astride::Union{Symbol,Int}
end
@noinline function allocate_temporaries_quote!(row_increments, row_increments_rem, config::NormalCholeskyConfiguration{T}, W, Mk, Nk, invdiagLL) where {T}
    @unpack M, P, track_Y, track_X, track_β, track_μ, track_L, βstride, Xstride, Ystride, μstride, μdim, sp, βdim, XP, μtransposed, arity, initY, initX, initβ, initμ, initL = config
    size_T = sizeof(T)
    array_allocations = sp ? quote _sptr = pointer(sptr,$T) end : quote end
    if track_L
        ∂LL = VectorizationBase.align(StructuredMatrices.binomial2(P + 1), W)
        push!(array_allocations.args, :(v∂L = StructuredMatrices.MutableLowerTriangularMatrix{$P,Vec{$W,$T},$∂LL}(undef)))
    end
    track_Xβ = track_X | track_β
    track_fsμ = track_μ & (μdim == 2)
    # This is horrible. There has got to be a better way.
    A_aliases_Y = false
    A_aliases_μ = false
    if track_Y 
        if track_fsμ
            if track_Xβ
                if initY # ∂Y used to calculate ∂X, ∂β, and ∂μ
                    Aquote, Astride = sym_aliases_A!(row_increments, row_increments_rem, :∂Y, size_T, M, Mk, P, W, sp)
                    A_aliases_Y = true
                else#if !initY
                    if initμ # ∂μ used to calculate ∂X, ∂β, and ∂μ
                        Aquote, Astride = sym_aliases_A!(row_increments, row_increments_rem, :∂μ, size_T, M, Mk, P, W, sp)
                        A_aliases_μ = true
                    else#if !initμ && !initY # allocate separate A, use for calculating ∂Y, ∂μ, and then ∂X and ∂β
                        Aquote, Astride = alloc_A_fullsize!(row_increments, row_increments_rem, T, M, Mk, P, W, sp)
                    end
                end
            else#if !track_Xβ
                if initY
                    Aquote, Astride = sym_aliases_A!(row_increments, row_increments_rem, :∂Y, size_T, M, Mk, P, W, sp)
                    A_aliases_Y = true
                else#if !initY
                    if initμ
                        Aquote, Astride = sym_aliases_A!(row_increments, row_increments_rem, :∂μ, size_T, M, Mk, P, W, sp)
                        A_aliases_μ = true
                    else
                        Aquote, Astride = alloc_A_kernel(Mk, P, T)
                    end
                end
            end
        else#if !track_fsμ
            if track_Xβ
                if initY
                    Aquote, Astride = sym_aliases_A!(row_increments, row_increments_rem, :∂Y, size_T, M, Mk, P, W, sp)
                    A_aliases_Y = true
                else
                    Aquote, Astride = alloc_A_fullsize!(row_increments, row_increments_rem, T, M, Mk, P, W, sp)
                end
            else#if !track_Xβ
                if initY
                    Aquote, Astride = sym_aliases_A!(row_increments, row_increments_rem, :∂Y, size_T, M, Mk, P, W, sp)
                    A_aliases_Y = true
                else
                    Aquote, Astride = alloc_A_kernel(Mk, P, T)
                end
            end
        end
    else#if !track_Y
        if track_fsμ
            if track_Xβ
                if initμ
                    Aquote, Astride = sym_aliases_A!(row_increments, row_increments_rem, :∂μ, size_T, M, Mk, P, W, sp)
                    A_aliases_μ = true
                else
                    Aquote, Astride = alloc_A_fullsize!(row_increments, row_increments_rem, T, M, Mk, P, W, sp)
                end
            else#if !track_Xβ
                if initμ
                    Aquote, Astride = sym_aliases_A!(row_increments, row_increments_rem, :∂μ, size_T, M, Mk, P, W, sp)
                    A_aliases_μ = true
                else
                    Aquote, Astride = alloc_A_kernel(Mk, P, T)
                end
            end
        else#if !track_fsμ
            if track_Xβ
                Aquote, Astride = alloc_A_fullsize!(row_increments, row_increments_rem, T, M, Mk, P, W, sp)
            else#if !track_Xβ
                Aquote, Astride = alloc_A_kernel(Mk, P, T)
            end
        end
    end
    if track_Y && !A_aliases_Y # We need to increment these pointers.
        push!(row_increments.args, :(ptr∂Y += $(size_T*Mk)))
        push!(row_increments_rem.args, :(ptr∂Y += $(size_T*W)))
        push!(Aquote.args, :(ptr∂Y = pointer(∂Y)))
    end
    if track_fsμ && !A_aliases_μ
        push!(row_increments.args, :(ptr∂μ += $(size_T*Mk)))
        push!(row_increments_rem.args, :(ptr∂μ += $(size_T*W)))
        push!(Aquote.args, :(ptr∂μ = pointer(∂μ)))
    end
    push!(Aquote.args, :(ptrA = pointer(A)))
    if track_μ
        if μdim == 1 && μtransposed
            PL = VectorizationBase.align(P, W) # align the number of columns to SIMD width
            push!(Aquote.args, :(v∂μ = PtrMatrix{$W,$P,$T,$W,$(W*P)}(SIMDPirates.alloca(Val{$(W*P)}(), $T)))) # accmulate in v∂μ; reduce at end
            push!(Aquote.args, :(ptrv∂μ = pointer(v∂μ)))
        end
        if ((μdim == 1) && !μtransposed) || ((μdim == 2) && track_Y) # if μ lines up with the rows of Y, we need to increment ptr∂μ
            push!(row_increments.args, :(ptr∂μ += $(size_T*Mk)))
            push!(row_increments_rem.args, :(ptr∂μ += $(size_T*W)))
            push!(Aquote.args, :(ptr∂μ = pointer(∂μ))) # otherwise we store in ptrv∂μ, and take the ptr∂μ later
        end        
    end
    if track_β && βdim == 1
        # push!(Aquote.args, Expr(:(=), :∂βv, :(FixedSizeVector{$XP,$T}(undef))))
        if sp
            push!(Aquote.args, :(∂βmat = PtrMatrix{$XP,$P,$T,$Xstride}(_sptr)))
            push!(Aquote.args, :(_sptr += $(VectorizationBase.align(Xstride * P * size_T))))
        else
            push!(Aquote.args, :(∂βmat = FixedSizeMatrix{$XP,$P,$T}(undef)))
        end
    end
    push!(array_allocations.args, Aquote)
    array_allocations, Astride::Union{Int,Symbol}
end

struct BlockingStructure{T}
    P::Int
    W::Int
    Mk::Int
    Nk::Int
    n_col_reps::Int
    col_rem::Int
    total_col_iterations::Int
    startoffset::Int
end
function BlockingStructure(maxM, P, ::Type{T}) where {T}
    W, Mk, Nk = StructuredMatrices.div_ul_blocking_structure(maxM, P, T)
    n_col_reps, col_rem = divrem(P, Nk)
    total_col_iterations = n_col_reps + (col_rem > 0)
    startoffset = (total_col_iterations-1) * Nk
    BlockingStructure{T}(
        P, W, Mk, Nk, n_col_reps, col_rem, total_col_iterations, startoffset
    )
end

function add_reversepass_base_offsets!(q, blocking_structure::BlockingStructure{T}, config::NormalCholeskyConfiguration{T}, Astride) where {T}
    size_T = sizeof(T)
    @unpack P, W, Mk, Nk, n_col_reps, col_rem, total_col_iterations, startoffset = blocking_structure
    @unpack Ystride, μstride, track_L, track_μ, μdim, μtransposed = config
    push!(q.args, :( ptrUtribase = pointer(L) + $(P*size_T) ) )
    push!(q.args, :( _A_offset_ = $size_T*$Astride*$startoffset ) )
    push!(q.args, :( ptrLtribase = pointer(L) + $size_T * $(P + StructuredMatrices.binomial2(startoffset) + startoffset * (P - startoffset)) ) ) # diag + triangle + subtriangle
    ∂Y_distinct_from_A(config) && push!(q.args, :(_Y_offset_ = $size_T * $Ystride * $startoffset))
    ∂μ2d_distinct_from_A(config) && push!(q.args, :(_μ_offset_ = $size_T * $μstride * $startoffset))
    push!(q.args, :(ptrLdiagbase = pointer(invdiagL) + $(size_T * startoffset)))
    if track_L
        push!(q.args, :(ptrv∂Ltribase = pointer(v∂L) + $(W*size_T * (P + StructuredMatrices.binomial2(startoffset) + startoffset * (P - startoffset))))) # diag + triangle + subtriangle
        push!(q.args, :(ptrv∂Ldiagbase = pointer(v∂L) + $(W*size_T*startoffset)))
    end
    if track_μ && μdim == 1 && μtransposed
        push!(q.args, :(ptrv∂μbase = pointer(v∂μ) + $(size_T*W*startoffset)))
    end
    nothing
end

## StructuredMatrices.jl Lower Triangular (SMLT) quote
## M is the sample size
@noinline function ∂multivariate_normal_SMLT_quote(
    config::NormalCholeskyConfiguration{T}
) where {T}
    @unpack M, P, track_Y, track_X, track_β, track_μ, track_L, βstride, Xstride, Ystride, μstride, μdim, sp, βdim,  XP, μtransposed, arity, allocate_partials, calclogdet, LL = config
    @unpack initY, initX, initβ, initμ, initL = config
    maxM = M isa Symbol ? typemax(Int) : M
    blockstructure = BlockingStructure(maxM, P, T)
    @unpack W, Mk, Nk = blockstructure
    V = Vec{W,T}
    Wm1 = W - 1
    n_col_reps, col_rem = divrem(P, Nk)
    size_T = sizeof(T)
    invdiagLL = VectorizationBase.align(P, W)
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
    # this increments _sptr
    if allocate_partials
        return_expr = Expr(:tuple,:δ²_0)
        track_Y && push!(return_expr.args, :(A'))
        track_X && push!(return_expr.args, :(∂X'))
        track_β && push!(return_expr.args, βdim == 1 ? :(∂βv') : :(∂β'))
        track_μ && push!(return_expr.args, (!track_Y && (μdim == 2)) ? :(A') : :(∂μ'))
        track_L && push!(return_expr.args, :∂L)
        array_allocations, Astride = allocate_partials_quote!(row_increments, row_increments_rem, config, W, Mk, Nk, invdiagLL)
    else
        array_allocations, Astride = allocate_temporaries_quote!(row_increments, row_increments_rem, config, W, Mk, Nk, invdiagLL)
    end
    cachedchol = first(DistributionParameters.caches_invdiag(P, LL, T))
    if cachedchol
        push!(array_allocations.args, :(invdiagL = invdiag(L)))
    elseif !(track_L && calclogdet)
        sptrexpr = allocate_partials ? :(StackPointer(pointer(∂L))) : :(StackPointer(_sptr))
        push!(array_allocations.args, :(invdiagL = last(Base.Broadcast.materialize($sptrexpr, invdiag(L)))))
    end
    Mk2 = min(4, M isa Symbol ? cld(Mk,W) : cld(min(Mk,M),W))
    q = quote
        $array_allocations
        $(Expr[Expr(:(=), Symbol(:δ²_,m), :(vbroadcast($V, zero($T)))) for m ∈ 0:Mk2-1]...)
        ptrY = pointer(Y)
    end
    local q::Expr
    if track_L && calclogdet
        loopprequote = quote logdiag_L = logdiag(L) end
        loopbody = quote δ²_0 = LoopVectorization.vadd(δ²_0, logdiag_L[p]) end
        if !cachedchol
            push!(loopprequote.args, :(invdiag_L_lazy = invdiag(L)))
            if !allocate_partials
                push!(loopprequote.args, :(invdiagL = PtrVector{$P,$T,$(PaddedMatrices.calc_padding(P,T))}(SIMDPirates.alloca(Val{$invdiagLL}(), $T))))
            end
            push!(loopbody.args, :(invdiagL[p] = invdiag_L_lazy[p]))
        end
        loopexpr = quote
            $loopprequote
            @vvectorize $T for p ∈ 1:$P
                $loopbody
            end
        end
        # println(loopexpr)
        push!(q.args, macroexpand(LoopVectorization, loopexpr))
    end
    # Workaround for Vec alignment issue
    # If M isa Symbol, the inline will be added by the func calling this one.
    # M isa Integer && pushfirst!(q.args, Expr(:meta, :inline))
    if track_L
        calclogdet && push!(q.args, :(δ²_0 = vmul(δ²_0, vbroadcast($V,$(M isa Integer ? T(2M) : :($(T(2))*$T($M)))))))
        set_v∂L_to_zero_quote = quote
            ptrv∂L = pointer(v∂L)
            for p ∈ 0:$(StructuredMatrices.binomial2(P+1)-1)
                vstore!(ptrv∂L + p *$(W*size_T), vbroadcast($V, zero($T)))
            end
        end
        push!(q.args, set_v∂L_to_zero_quote)
    end
    arity >= 4 && push!(q.args, :(ptrX = pointer(X); ptrβ = pointer(β)))
    if track_μ
        if μdim == 0
            for m ∈ 0:3
                push!(q.args, Expr(:(=), Symbol(:v∂μ_,m), :(vbroadcast($V, zero($T)))))
            end
        else
            if μdim == 1 && μtransposed
                set_ptr_vmu_zero_expr = quote
                    ptrv∂μ = pointer(v∂μ)
                    for p ∈ 0:$(P-1)
                        vstore!(ptrv∂μ + p*$(W*size_T), vbroadcast($V, zero($T)))
                    end
                end
                push!(q.args, set_ptr_vmu_zero_expr)
            end
        end
    end
    if μdim > 0
        push!(q.args, Expr(:(=), :ptrμ, μtransposed ? :(pointer(μ.parent)) : :(pointer(μ))))
    end
    μmy = track_Y && !(!initY && μdim == 2 && track_μ)
    add_reversepass_base_offsets!(q, blockstructure, config, Astride)
    if M isa Integer
        n_row_reps, row_rem = divrem(M, Mk)
        Mk1 = n_row_reps == 0 ? row_rem : Mk
        row_iter = ∂mutlivariate_normal_SMLT_rowiter(
            config, Mk1, Nk, col_rem, n_col_reps, μmy, :ptrμ, Astride
        )
        if n_row_reps > 1
            row_loops = quote
                for rrep ∈ 1:$n_row_reps
                    ptrUdiag = pointer(invdiagL); ptrUtri = ptrUtribase
                    $row_iter
                    $row_increments
                end
            end
            push!(q.args, row_loops)
        else
            push!(q.args, :(ptrUdiag = pointer(invdiagL); ptrUtri = ptrUtribase))
            push!(q.args, row_iter)
        end
        if row_rem > 0 && n_row_reps > 0
            push!(q.args, :(ptrUdiag = pointer(invdiagL); ptrUtri = ptrUtribase))
            onevecblock = BlockingStructure(row_rem, P, T) # a different blocking structure may be optimal for the remainder.
            if onevecblock.startoffset == blockstructure.startoffset
                Nkrem, col_remrem, n_col_repsrem = Nk, col_rem, n_col_reps
            else
                add_reversepass_base_offsets!(q, onevecblock, config, Astride)
                Nkrem = onevecblock.Nk; col_remrem = onevecblock.col_rem; n_col_repsrem = onevecblock.n_col_reps
            end
            push!(q.args, ∂mutlivariate_normal_SMLT_rowiter(
                config, row_rem, Nkrem, col_remrem, n_col_repsrem, μmy, :ptrμ, Astride
            ))
        end
    else # Unknown number of iterations.
        row_iter = ∂mutlivariate_normal_SMLT_rowiter(
            config, Mk, Nk, col_rem, n_col_reps, μmy, :ptrμ, Astride
        )
        row_loops_primary = quote
            Mkrep, Mkrem = divrem($M, $Mk)
            for rrep ∈ 1:Mkrep
                ptrUdiag = pointer(invdiagL); ptrUtri = ptrUtribase
                $row_iter
                $row_increments
            end
        end
        push!(q.args, row_loops_primary)
        onevecblock = BlockingStructure(W, P, T)
        Nkrem = onevecblock.Nk; col_remrem = onevecblock.col_rem; n_col_repsrem = onevecblock.n_col_reps
        row_iter_onevec = ∂mutlivariate_normal_SMLT_rowiter(
            config, W, Nkrem, col_remrem, n_col_repsrem, μmy, :ptrμ, Astride
        )
        row_iter_onevecmask = ∂mutlivariate_normal_SMLT_rowiter(
            config, -1, Nkrem, col_remrem, n_col_repsrem, μmy, :ptrμ, Astride
        )
        if blockstructure.startoffset != onevecblock.startoffset
            add_reversepass_base_offsets!(q, onevecblock, config, Astride)
        end
        row_loops_remainder = quote
            for rrep ∈ 1:Mkrem >>> $(VectorizationBase.intlog2(W))
                ptrUdiag = pointer(invdiagL); ptrUtri = ptrUtribase
                $row_iter_onevec
                $row_increments_rem
            end
            row_rem_final = Mkrem & $Wm1
            if row_rem_final != 0
                ptrUdiag = pointer(invdiagL); ptrUtri = ptrUtribase
                __mask__ = VectorizationBase.mask($T, row_rem_final)
                $row_iter_onevecmask
            end
        end
        push!(q.args, row_loops_remainder)
    end
    # Reduce the Mk δ² into a single vector.
    R = Mk2
    while R > 1
        Risodd = isodd(R)
        Rh = R >>> 1
        for r ∈ 0:(Rh-1)
            dl = Symbol(:δ²_,r)
            dh = Symbol(:δ²_,r+Rh)
            push!(q.args, :($dl = vadd($dl,$dh)))
        end
        Risodd && push!(q.args, Expr(:(=), :δ²_0, :(vadd(δ²_0, $(Symbol(:δ²_,R-1))))))
        R = Rh
    end
    push!(q.args, Expr(:(=), :δ²_0, :(vmul(vbroadcast($V, $(T(-0.5))), δ²_0))))
    if track_L
        loopheader = quote ptrv∂L = pointer(v∂L); ptr∂L = pointer(∂L); ptrinvdiag = pointer(invdiagL) end
        if calclogdet
            vsumexpr = :(Base.FastMath.sub_fast(
                    vsum(vload($V, ptrv∂L + p*$(W*size_T))),
                    Base.FastMath.mul_fast(
                        $(M isa Symbol ? :($T($M)) : T(M)),
                        VectorizationBase.load(ptrinvdiag + p*$size_T))
            ))
            if !initL
                vsumexpr = :(Base.FastMath.add_fast(VectorizationBase.load(ptr∂L + p*$size_T,), $vsumexpr))
            end
            loop1body = quote
                VectorizationBase.store!(
                    ptr∂L + p*$size_T, $vsumexpr
                )
            end
            if track_μ && μdim == 1 && μtransposed
                push!(loopheader.args, :(ptr∂μ = pointer(∂μ); ptrv∂μ = pointer(v∂μ)))
                if initμ
                    push!(loop1body.args, :(VectorizationBase.store!(
                        ptr∂μ + p*$size_T, vsum(vload($V, ptrv∂μ + p*$(W*size_T)))
                    )))
                else
                    push!(loop1body.args, :(VectorizationBase.store!(
                        ptr∂μ + p*$size_T, Base.FastMath.add_fast(load(ptr∂μ + p*$size_T), vsum(vload($V, ptrv∂μ + p*$(W*size_T))))
                    )))
                end
            end
            push!(loopheader.args, :(for p in 0:$(P-1); $loop1body; end))
            remloopstart = P
        else
            remloopstart = 0
        end # Consider using gathers and vload/stores
        rem_body = :(vsum(vload($V, ptrv∂L + p*$(W*size_T))))
        if !initL
            rem_body = :(Base.FastMath.add_fast(load(ptr∂L + p*$size_T), $rem_body))
        end
        vsum_L_expr = quote
            $loopheader    
            for p in $remloopstart:$(StructuredMatrices.binomial2(P+1)-1)
                VectorizationBase.store!(
                    ptr∂L + p*$size_T, $rem_body
                )
            end
        end
        push!(q.args, vsum_L_expr)
    end
    if track_μ
        if μdim == 1 && μtransposed && !(track_L && calclogdet)
            if initμ
                vsum_mu_expr = quote
                    ptr∂μ = pointer(∂μ); ptrv∂μ = pointer(v∂μ)
                    for p in 0:$(P-1)
                        VectorizationBase.store!(
                            ptr∂μ + p*$size_T, vsum(vload($V, ptrv∂μ + p*$(W*size_T)))
                        )
                        # ∂μ[p] = vsum(v∂μ[p])
                    end
                end
            else #TODO consider using gather instructions
                vsum_mu_expr = quote
                    ptr∂μ = pointer(∂μ); ptrv∂μ = pointer(v∂μ)
                    for p in 0:$(P-1)
                        VectorizationBase.store!(
                            ptr∂μ + p*$size_T, Base.FastMath.add_fast(load(ptr∂μ + p*$size_T), vsum(vload($V, ptrv∂μ + p*$(W*size_T))))
                        )
                        # ∂μ[p] += vsum(v∂μ[p])
                    end
                end
            end
            push!(q.args, vsum_mu_expr)
        elseif μdim == 0
            push!(q.args, Expr(:(=), :v∂μ_0, :(vadd(vadd(v∂μ_0,v∂μ_2),vadd(v∂μ_1,v∂μ_3)))))
            push!(q.args, Expr(initμ ? :(=) : :(+=), :∂μ, :(vsum(v∂μ_0))))
        end
    end
    if track_X | track_β
        # push!(q.args, :(@show A))
        # push!(q.args, :(@show X))
        f = if initX
            μmy ?  :(PaddedMatrices.nmul!) : :(LinearAlgebra.mul!)
        else
            μmy ?  :(PaddedMatrices.nmuladd!) : :(PaddedMatrices.muladd!)
        end
        track_X && push!(q.args, Expr(:call, f, :∂X, :A, :(β')))
        if track_β
            f = if initβ
                μmy ?  :(PaddedMatrices.nmul!) : :(LinearAlgebra.mul!)
            else
                μmy ?  :(PaddedMatrices.nmuladd!) : :(PaddedMatrices.muladd!)
            end
            # push!(q.args, :(@show ∂β))
            if βdim == 1
                push!(q.args, Expr(:call, f, :∂βmat, :(X'), :A))
                push!(q.args, Expr(:call, :sum!, :∂β, :∂βmat))
            else
                push!(q.args, Expr(:call, f, :∂β, :(X'), :A))
            end
        end
    end
    if allocate_partials
        if sp
            push!(q.args, :(PaddedMatrices.StackPointer(_sptr),$return_expr))
        else
            push!(q.args, return_expr)
        end
    else
        push!(q.args, :δ²_0)
    end
    simplify_expr(q)
end

function modify_args_∂!(args, allocate_partials, calclogdet)
    syms = [first(arg.args) for arg ∈ args]
    tracksyms = [Symbol(:track_,sym) for sym ∈ syms]
    whereargs = calc_whereset(args)
    if allocate_partials
        pushfirst!(args, :(::Val{track}))
        ret = quote
            $(Expr(:tuple, tracksyms...)) = track
        end
        push!(whereargs, :track)
    else
        ∂syms = [Symbol(:∂,sym) for sym ∈ syms]
        ∂arg_types = [Symbol(ps, :T) for ps ∈ ∂syms]
        foreach(pa -> push!(whereargs, pa), ∂arg_types)
        prepend!(args, [Expr(:(::), ∂s, ∂st) for (∂s,∂st) ∈ zip(∂syms, ∂arg_types)])
        ret = quote
            $([:($ts = $(∂a) !== Nothing) for (∂a,ts) ∈ zip(∂arg_types, tracksyms)]...)
        end
    end
    push!(ret.args, :(config = NormalCholeskyConfiguration{T}()))
    push!(ret.args, :(config.arity = $(length(syms))))
    push!(ret.args, :(config.M = M; config.P = P; config.LL = LL))
    push!(ret.args, :(config.calclogdet = $calclogdet))
    for (i,ts) ∈ enumerate(tracksyms)
        push!(ret.args, :(config.$ts = $ts))
        if !allocate_partials
            is = Symbol(:init, syms[i])
            push!(ret.args, :(config.$is = !isinitialized($(∂syms[i]))))
        end
    end
    ret, whereargs
end

for calclogdet ∈ (true, false)
    distbase = calclogdet ? :∂Normal : :∂Normal_kernel
    for allocate_partials ∈ (true, false)
        dist = allocate_partials ? distbase : Symbol(distbase, :!)
        args = [:(Y::AbstractMutableFixedSizeMatrix{M,P,T,PY}), :(L::AbstractLowerTriangularMatrix{P,T,LL})]
        setup, whereargs = modify_args_∂!(args, allocate_partials, calclogdet)
        for sp ∈ (false, true)
            sp && pushfirst!(args, :(sptr::StackPointer))
            # println(args)
            @eval @generated function $dist( $(args...) ) where {$(whereargs...)}
                $setup
                config.sp = $sp
                config.Ystride = PY
                ∂multivariate_normal_SMLT_quote(config)
            end
        end

        args = [:(Y::AbstractMutableFixedSizeMatrix{M,P,T,PY}), :(μ::Tμ), :(L::AbstractLowerTriangularMatrix{P,T,LL})]
        setup, whereargs = modify_args_∂!(args, allocate_partials, calclogdet)
        for sp ∈ (false,true)
            sp && pushfirst!(args, :(sptr::StackPointer))
            @eval @generated function $dist( $(args...) ) where {$(whereargs...)}
                $setup
                config.sp = $sp
                config.Ystride = PY
                if Tμ === T
                    config.μdim = 0; config.μstride = 0
                elseif Tμ <: LinearAlgebra.Adjoint{T,<:AbstractMutableFixedSizeVector}
                    config.μdim = 1; config.μstride = 1; config.μtransposed = true
                elseif Tμ <: AbstractMutableFixedSizeVector
                    config.μdim = 1; config.μstride = 1
                elseif Tμ <: AbstractMutableFixedSizeMatrix
                    config.μdim = 2; config.μstride = (Tμ.parameters[4].parameters[2])::Int
                else
                    throw("Type of μ = $(Tμ) was not recognized.")
                end
                ∂multivariate_normal_SMLT_quote(config)
            end
        end

        args = [:(Y::AbstractMutableFixedSizeMatrix{M,P,T,PY}), :(X::AbstractMutableFixedSizeMatrix{M,K_,T,PX}), :(β::AbstractMutableFixedSizeArray{Sβ,T,Nβ,Pβ}), :(L::AbstractLowerTriangularMatrix{P,T,LL})]
        setup, whereargs = modify_args_∂!(args, allocate_partials, calclogdet)
        for sp ∈ (false, true)
            sp && pushfirst!(args, :(sptr::StackPointer))
            @eval @generated function $dist( $(args...) ) where {$(whereargs...)}
                @assert Sβ.parameters[1] == K_
                βstride = length(Pβ.parameters) == 1 ? K_ : (Pβ.parameters[2])::Int
                $setup
                config.Ystride = PY; config.Xstride = PX; config.βstride = βstride; config.βdim = Nβ; config.XP = K_; config.sp = $sp
                ∂multivariate_normal_SMLT_quote(config)
            end
        end

        args = [:(Y::AbstractMutableFixedSizeMatrix{M,P,T,PY}), :(X::AbstractMutableFixedSizeMatrix{M,K_,T,PX}), :(β::AbstractMutableFixedSizeArray{Sβ,T,Nβ,Pβ}), :(μ::Tμ), :(L::AbstractLowerTriangularMatrix{P,T,LL})]
        setup, whereargs = modify_args_∂!(args, allocate_partials, calclogdet)
        for sp ∈ (false, true)
            sp && pushfirst!(args, :(sptr::StackPointer))
            @eval @generated function $dist( $(args...) ) where {$(whereargs...)}
                @assert Sβ.parameters[1] == K_
                βstride = length(Pβ.parameters) == 1 ? K_ : (Pβ.parameters[2])::Int
                $setup
                config.sp = $sp; config.Ystride = PY; config.Xstride = PX; config.βstride = βstride; config.βdim = Nβ; config.XP = K_
                if Tμ === T
                    config.μdim = 0; config.μstride = 0
                elseif Tμ <: LinearAlgebra.Adjoint{T,<:AbstractMutableFixedSizeVector}
                    config.μdim = 1; config.μstride = 1; config.μtransposed = true
                elseif Tμ <: AbstractMutableFixedSizeVector
                    config.μdim = 1; config.μstride = 1
                elseif Tμ <: AbstractMutableFixedSizeMatrix
                    config.μdim = 2; config.μstride = (Tμ.parameters[4].parameters[2])::Int
                else
                    throw("Type of μ = $(Tμ) was not recognized.")
                end
                ∂multivariate_normal_SMLT_quote(config)
            end
        end
    end
end

for calclogdet ∈ (true, false)
    distbase = calclogdet ? :∂Normal : :∂Normal_kernel
    for allocate_partials ∈ (true, false)
        dist = allocate_partials ? distbase : Symbol(distbase, :!)
        args = [:(Y::AbstractMatrix{T}), :(L::AbstractLowerTriangularMatrix{P,T,LL})]
        setup, whereargs = modify_args_∂!(args, allocate_partials, calclogdet)
        for sp ∈ (false, true)
            sp && pushfirst!(args, :(sptr::StackPointer))
            @eval @generated function $dist( $(args...) ) where {$(whereargs...)}
                $setup
                config.sp = $sp
                M, PY = gensym(:M), gensym(:PY)
                config.Ystride = PY
                quote
                    # $(Expr(:meta,:inline))
                    $M = size(Y,1)
                    $PY = $(Y <: Array ? M : :(stride(Y,2)))
                    $(∂multivariate_normal_SMLT_quote(config))
                end
            end
        end
        
        args = [:(Y::AbstractMatrix{T}), :(μ::Tμ), :(L::AbstractLowerTriangularMatrix{P,T,LL})]
        setup, whereargs = modify_args_∂!(args, allocate_partials, calclogdet)
        for sp ∈ (false,true)
            sp && pushfirst!(args, :(sptr::StackPointer))
            @eval @generated function $dist( $(args...) ) where {$(whereargs...)}
                M, PY = gensym(:M), gensym(:PY)
                defs_quote = quote
                    $M = size(Y,1)
                    $PY = $(Y <: Array ? M : :(stride(Y,2)))
                end
                $setup
                config.sp = $sp
                config.Ystride = PY
                if Tμ === T
                    config.μdim = 0; config.μstride = 0
                elseif Tμ <: LinearAlgebra.Adjoint{T,<:AbstractMutableFixedSizeVector}
                    config.μdim = 1; config.μstride = 1; config.μtransposed = true
                elseif Tμ <: AbstractVector
                    config.μdim = 1; config.μstride = 1
                elseif Tμ <: AbstractMutableFixedSizeMatrix
                    config.μdim = 2; config.μstride = (Tμ.parameters[4].parameters[2])::Int
                elseif Tμ <: AbstractMatrix
                    μstride = gensym(:μstride)
                    push!(defs_quote.args, :($μstride = stride(μ,2)))
                    config.μdim = 2; config.μstride = μstride
                else
                    throw("Type of μ = $(Tμ) was not recognized.")
                end
                quote
                    # $(Expr(:meta,:inline))
                    $defs_quote
                    $(∂multivariate_normal_SMLT_quote(config))
                end
            end
        end

        args = [:(Y::AbstractMatrix{T}), :(X::AbstractMatrix{T}), :(β::AbstractMutableFixedSizeArray{Sβ,T,Nβ,Pβ}), :(L::AbstractLowerTriangularMatrix{P,T,LL})]
        setup, whereargs = modify_args_∂!(args, allocate_partials, calclogdet)
        for sp ∈ (false, true)
            sp && pushfirst!(args, :(sptr::StackPointer))
            @eval @generated function $dist( $(args...) ) where {$(whereargs...)}
                M, PY, PX = gensym(:M), gensym(:PY), gensym(:PX)
                K_ = Sβ.parameters[1]
                $setup
                config.βstride = length(Pβ.parameters) == 1 ? K_ : (Pβ.parameters[2])::Int
                config.Ystride = PY; config.Xstride = PX; config.βdim = Nβ; config.XP = K_; config.sp = $sp
                quote
                    # $(Expr(:meta,:inline))
                    $M = size(Y,1)
                    $PY = $(Y <: Array ? M : :(stride(Y,2)))
                    $PX = $(X <: Array ? M : :(stride(X,2)))
                    $(∂multivariate_normal_SMLT_quote(config))
                end
            end
        end

        args = [:(Y::AbstractMatrix{T}), :(X::AbstractMatrix{T}), :(β::AbstractMutableFixedSizeArray{Sβ,T,Nβ,Pβ}), :(μ::Tμ), :(L::AbstractLowerTriangularMatrix{P,T,LL})]
        setup, whereargs = modify_args_∂!(args, allocate_partials, calclogdet)
        for sp ∈ (false, true)
            sp && pushfirst!(args, :(sptr::StackPointer))
            @eval @generated function $dist( $(args...) ) where {$(whereargs...)}
                M, PY, PX = gensym(:M), gensym(:PY), gensym(:PX)
                Sβv = Sβ.parameters; Pβv = Pβ.parameters
                K_ = (Sβv[1])::Int
                defs_quote = quote
                    $M = size(Y,1)
                    $PY = $(Y <: Array ? M : :(stride(Y,2)))
                    $PX = $(X <: Array ? M : :(stride(X,2)))
                end
                $setup
                config.sp = $sp; config.Ystride = PY; config.Xstride = PX; config.βdim = Nβ; config.XP = K_
                config.βstride = Nβ == 1 ? K_ : (Pβv[2])::Int
                if Tμ === T
                    config.μdim = 0; config.μstride = 0
                elseif Tμ <: LinearAlgebra.Adjoint{T,<:AbstractMutableFixedSizeVector}
                    config.μdim = 1; config.μstride = 1; config.μtransposed = true
                elseif Tμ <: AbstractVector
                    config.μdim = 1; config.μstride = 1
                elseif Tμ <: AbstractMutableFixedSizeMatrix
                    config.μdim = 2; config.μstride = (Tμ.parameters[4].parameters[2])::Int
                elseif Tμ <: AbstractMatrix
                    μstride = gensym(:μstride)
                    push!(defs_quote.args, :($μstride = stride(μ,2)))
                    config.μdim = 2; config.μstride = μstride
                else
                    throw("Type of μ = $(Tμ) was not recognized.")
                end
                quote
                    # $(Expr(:meta,:inline))
                    $defs_quote
                    $(∂multivariate_normal_SMLT_quote(config))
                end
            end
        end
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

