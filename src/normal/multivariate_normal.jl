

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
@generated function vlogdet_triangle(A::AbstractTriangularMatrix{P,T}) where {P,T}
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
    quote
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
    q
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
    q
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



function loadδ_quote(R, C, K::Union{Symbol,Int}, T::DataType, Bstride, Bsym, μsym::Union{Symbol,Nothing} = :μptr, maskload::Bool = true)
    size_T = sizeof(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(R, T)
    Wm1 = W - 1
    Riter = R >> Wshift
    Rrem = R & Wm1
    mask = VectorizationBase.mask_from_remainder(T, Rrem)
    q = if K isa Symbol
        quote
            BsymK = $Bsym + $(size_T*Bstride)*$K
        end
    else
        quote
            BsymK = $Bsym + $(size_T*Bstride*K)
        end
    end
    if μsym isa Symbol
        for c ∈ 0:C-1
            vμ_c = Symbol(:vμ_, c)
            if K isa Symbol
                push!(q.args, Expr(:(=), vμ_c, :(SIMDPirates.vbroadcast(Vec{$W,$T}, $μsym + $size_T*($c+$K*$C)))))
            else
                push!(q.args, Expr(:(=), vμ_c, :(SIMDPirates.vbroadcast(Vec{$W,$T}, $μsym + $(size_T*(c + K*C))))))
            end
            for r ∈ 0:Riter-1
                push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vsub($vμ_c, SIMDPirates.vload(Vec{$W,$T}, BsymK + $(size_T * (r*W + c*Bstride)) ) )))
            end
            if Rrem > 0
                # Only need to mask if we're on last column
                if maskload && c == C-1
                    push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vsub($vμ_c, SIMDPirates.vload(Vec{$W,$T}, BsymK + $(size_T * (Riter*W + c*Bstride)), $mask ) )))
                else
                    push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vsub($vμ_c, SIMDPirates.vload(Vec{$W,$T}, BsymK + $(size_T * (Riter*W + c*Bstride)) ) )))
                end
            end
        end
    else
        for c ∈ 0:C-1
            for r ∈ 0:Riter-1
                push!(q.args, :($(Symbol(:A_,r,:_,c)) = SIMDPirates.vload(Vec{$W,$T}, BsymK + $(size_T * (r*W + c*Bstride)) ) ))
            end
            if Rrem > 0
                # Only need to mask if we're on last column
                if maskload && c == C-1
                    push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vload(Vec{$W,$T}, BsymK + $(size_T * (Riter*W + c*Bstride)), $mask ) ))
                else
                    push!(q.args, :($(Symbol(:A_,Riter,:_,c)) = SIMDPirates.vload(Vec{$W,$T}, BsymK + $(size_T * (Riter*W + c*Bstride)) ) ))
                end
            end
        end
    end
    q
end

function mutlivariate_normal_SMLT_rowiter(Mk, Nk, col_rem, T, Ystride, n_col_reps, μsym = :μptr)
    #TODO: NOTE, WE DO NEED TO STORE THE SOLUTION MATRIX (at least 1 row set amd up to the last column block)
    # because this is used for calculating the next iteration.
    N = Nk * n_col_reps + col_rem
    size_T = sizeof(T)
    row_iter = quote end
    if col_rem > 0
        loadδ_expr = loadδ_quote(Mk, col_rem, 0, T, Ystride, :Yptr, μsym)
        iter_quote = StructuredMatrices.A_rdiv_U_kernel_quote(
            Mk, col_rem, 0, T, Mk, Ystride, N, true, true, storeA = true, loadB = false, reduce_sym = :δ²
        )
        #pushfirst!(row_iter.args, :(ptrUtri = ptrUtribase))
        push!(row_iter.args, iter_quote)
        push!(row_iter.args, :(ptrUdiag += $(col_rem*size_T)))
        base_K = col_rem
        KmZ = false
    else
        base_K = 0
        KmZ = true
    end
    if n_col_reps > 1
        loadδ_expr = loadδ_quote(Mk, Nk, 0, T, Ystride, :Yptr, μsym)
        iterquote = StructuredMatrices.A_rdiv_U_kernel_quote(
            Mk, Nk, :K, T, Mk, Ystride, N, true, true, storeA = true, loadB = false, reduceA = true, reduce_sym = :δ²
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
        row_iter_single = StructuredMatrices.A_rdiv_U_kernel_quote(
            Mk, Nk, col_rem, T, Mk, Ystride, N, true, true, storeA = false, loadB = false, reduceA = true, reduce_sym = :δ²
        )
        push!(row_iter.args, row_iter_single)
    end
    row_iter
end

## StructedMatrices.jl Lower Triangular (SMLT) quote
## M is the sample size
function multivariate_normal_SMLT_quote(M::Union{Symbol,Integer}, P, track, sp::Bool, Ystride = M, T::DataType = Float64)
    track_Y, track_μ, track_L = track
    q = quote end
    maxM = M isa Symbol ? typemax(Int) : M

    W, Mk, Nk = StructuredMatrices.div_triangle_blocking_structure(maxM, P, T)
    Wm1 = W - 1
    n_col_reps, col_rem = divrem(M, Nk)
    total_col_iterations = n_col_reps + (col_rem > 0)
    #Nl = ( N + W - 1 ) & ~Wm1
    size_T = sizeof(T)
    loopbody = quote
        Bₚ = B[p]
        invdiag[p] = one($T) / Bₚ
    end
    track_L && push!(loopbody.args, :(δ²_0 = LoopVectorization.SIMDPirates.vsub(δ²_0, SLEEFPirates.log(Bₚ))))
    q = quote
        Expr(:meta,:inline) # because of allignment bug
        B = Badj.parent
        $([Expr(:(=), Symbol(:δ²_,m), :(SIMDPirates.vbroadcast(Vec{$W,$T}, zero($T)))) for m ∈ 0:Mk-1]...)
        invdiag = $(sp ? :(PtrVector{$P,$T,$P,$P}(pointer(sptr,$T))) : :(MutableFixedSizePaddedVector{$P,$T,$P,$P}(undef)))
        LoopVectorization.@vvectorize $T for p ∈ 1:$P
            $loopbody
        end
        A = $(sp ? :(PtrMatrix{$Mk,$P,$T,$Mk}(pointer(sptr,$T) + $(VectorizationBase.align(size_T*P)))) : :(MutableFixedSizePaddedMatrix{$Mk,$P,$T,$Mk}(undef)))
        ptrA = pointer(A)
        ptrB = pointer(Y)
        ptrUtribase = pointer(L) + $(P*size_T)
    end
    if M isa Integer
        n_row_reps, row_rem = divrem(M, Mk)
        total_row_iterations = n_row_reps + (row_rem > 0)
        Mk1 = n_row_reps == 0 ? row_rem : Mk
        row_iter = mutlivariate_normal_SMLT_rowiter(
            Mk1, Nk, col_rem, T, Ystride, n_col_reps, :μptr
        )
        if n_row_reps > 1
            row_loops = quote
                for rrep ∈ 1:$n_row_reps
                    ptrUdiag = pointer(invdiag)
                    ptrUtri = ptrUtribase#pointer(B) + $(size_T * N)
                    $row_iter
                    ptrB += $(size_T*Mk)
                    #ptrA += $(size_T*Mk)
                end
            end
            push!(q.args, row_loops)
        else
            push!(q.args, :(ptrUdiag = pointer(invdiag)))
            push!(q.args, :(ptrUtri = ptrUtribase))
            push!(q.args, row_iter)
            if total_row_iterations == 2 # then n_row_reps == 1 and row_rem > 0
                push!(q.args, mutlivariate_normal_SMLT_rowiter( row_rem, Nk, col_rem, T, Ystride, n_col_reps ))
            end
        end
    else # Unknown number of iterations.
        row_iter = mutlivariate_normal_SMLT_rowiter(
            Mk, Nk, col_rem, T, Ystride, n_col_reps, :μptr
        )
        Wrem, Mkrem, Nkrem = StructuredMatrices.div_triangle_blocking_structure(W, P, T)
        n_col_repsrem, col_remrem = divrem(P, Nkrem)
        row_iter_onevec = mutlivariate_normal_SMLT_rowiter(
            W, Nkrem, col_remrem, T, Ystride, n_col_repsrem, :μptr
        )
        row_iter_onevecmask = mutlivariate_normal_SMLT_rowiter(
            :row_rem_final, Nkrem, col_remrem, T, Ystride, n_col_repsrem, :μptr
        )
        row_loops = quote
            Mkrep, Mkrem = divrem($M, $Mk)
            for rrep ∈ 1:Mkrep
                ptrUdiag = pointer(invdiag)
                ptrUtri = ptrUtribase#pointer(B) + $(size_T * N)
                $row_iter
                ptrB += $(size_T*Mk)
                #ptrA += $(size_T*Mk)
            end
            for rrep ∈ 1:Mkrem >> $(VectorizationBase.intlog2(W))
                ptrUdiag = pointer(invdiag)
                ptrUtri = ptrUtribase#pointer(B) + $(size_T * N)
                $row_iter_onevec
                ptrB += $(size_T*W)
                #ptrA += $(size_T*W)
            end
            row_rem_final = Mkrem & $Wm1
            $row_iter_onevecmask
        end
        push!(q.args, row_loops)
    end
    # Reduce the Mk δ² into a single vector.
    R = Mk
    while R > 0
        Risodd = isodd(R)
        Rh = R >> 1
        for r ∈ 0:(Rh-1)
            dl = Symbol(:δ²_,r)
            dh = Symbol(:δ²_,r+Rh)
            push!(q.args, :($dl = SIMDPirates.vadd($dl,$dh)))
        end
        Risodd && push!(q.args, Expr(:(=), :δ²_0, :(SIMDPirates.vadd(δ²_0, $(Symbol(:δ²_,R))))))
        R = Rh
    end
    push!(q.args, :δ²_0)
    q
end


function ∂mutlivariate_normal_SMLT_rowiter(Mk, Nk, col_rem, T, CP, AP, n_col_reps, track)
    track_Y, track_μ, track_L = track

end

## StructedMatrices.jl Lower Triangular (SMLT) quote
function ∂multivariate_normal_SMLT_quote(N::Union{Symbol,Integer}, P, track, sp::Bool, T::DataType = Float64)
    track_Y, track_μ, track_L = track
    q = quote end
    maxN = N isa Symbol ? typemax(Int) : N

    W, Mk, Nk, (colremiszero, firstthenlast) = StructuredMatrices.div_ul_blocking_structure(maxN, P)

    # Solve forward and then backward
    # Forward -> calc the logdensity
    # backward is grad for Y and mu
    # forward * backward is grad for L

end




