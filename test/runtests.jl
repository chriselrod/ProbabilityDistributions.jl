using ProbabilityDistributions
using Test

using ProbabilityDistributions: Normal, ∂Normal, Normalc, ∂Normalc
using DistributionParameters, PaddedMatrices, LinearAlgebra
using SIMDPirates: vsum

# quarter of a GiB
# up to 1<<(28-3) == 33_554_432 doubles
const STACK_POINTER = PaddedMatrices.StackPointer(Libc.malloc(1<<28));

function clean_ret(f::F, args::Vararg{<:Any,K}) where {F,K}
    track = Val{(ntuple(Val(K)) do k true end)}()
    sp2, vt = f(STACK_POINTER, args..., track)
    vsum(vt)
end
function clean_ret(f::F, args::Tuple, ::Val{track}) where {F,track}
    sp2, vt = f(STACK_POINTER, args..., Val{track}())
    vsum(vt)
end
@generated function ∂clean_ret(f::F, args::Vararg{<:Any,K}) where {F,K}
    ∂args = Expr(:tuple, (gensym() for k ∈ 1:K)...)
    ∂argsc = copy(∂args)
    pushfirst!(∂argsc.args, :vt)
    pushfirst!(∂args.args, :(vsum(vt)))
    quote
        track = Val{$(Expr(:tuple,(true for k ∈ 1:K)...))}()
        sp2, $∂argsc = f(STACK_POINTER, args..., track)
        $∂args
    end
end
@generated function ∂clean_ret(f::F, args::Tuple, ::Val{track}) where {F,track}
    K = length(args.parameters)
    num_track = sum(track)
    ∂args = Expr(:tuple, (gensym() for k ∈ 1:num_track)...)
    ∂argsc = copy(∂args)
    pushfirst!(∂argsc.args, :vt)
    pushfirst!(∂args.args, :(vsum(vt)))
    quote
        sp2, $∂argsc = f(STACK_POINTER, args..., Val{track}())
        $∂args
    end
end


rel_error(x, y) = (x == y ? zero(x) : (x - y) / y)
@generated function test_grad(f, ∂f, t1, a, t2, tol = 1e-5)
    K1 = length(t1.parameters)
    K2 = length(t2.parameters)
    track = Expr(:tuple,append!(push!(fill(false, K1), true), fill(false, K2))...)
    argstup = Expr(:tuple, [:(t1[$k]) for k ∈ 1:K1]..., :a, [:(t2[$k]) for k ∈ 1:K2]...)
    call = Expr(:call, :clean_ret, :f, :argstuple, :(Val{$track}()))
    ∂call = Expr(:call, :∂clean_ret, :∂f, :argstuple, :(Val{$track}()))
    quote
        @inbounds argstuple = $argstup
        dbase, gptr = $∂call
        g = copy(gptr)
        for i ∈ eachindex(a)
            aᵢ = a[i]
            step = max(cbrt(eps(aᵢ)), 1e-20)
            a[i] = aᵢ + step
            dforward = $call
            a[i] = aᵢ - step
            dback = $call

            gad = g[i]
            gforward = (dforward - dbase) / step
            gback = (dbase - dback) / step
            gcenter = (dforward - dback) / (2step)

            if abs(gcenter) < 1e-20 && abs(gad) < 1e-20
                continue
            end
            r1, r2, r3 = rel_error.(gad, (gforward, gback, gcenter))
            if abs(r3) < tol
            else
                if sign(r1) != sign(r2)
                    print("; i = $i: $r3")
                else
                    @show i, (gad, (gforward,gback,gcenter)),(r1,r2,r3)
                end
            end            
            a[i] = aᵢ
        end
    end
end

@testset "ProbabilityDistributions.jl" begin
    # Write your own tests here.

    M = 100
    Σ = DistributionParameters.MutableFixedSizeCovarianceMatrix{M,Float64}(undef);
    fill!(Σ, 0.0);
    Σcopy = DistributionParameters.MutableFixedSizeCovarianceMatrix{M,Float64}(undef);
    A = randn(M, 2M);
    BLAS.syrk!('L', 'N', 1.0, A, 0.0, Σ);

    L = MutableFixedSizePaddedMatrix{M,M,Float64,M}(undef);
    copyto!(L, Σ);
    LAPACK.potrf!('L', L);

    μ = @Mutable randn(M);
    Y = LowerTriangular(L) * randn(M, 50) .+ μ;

    function mvnormdirect(Y, μ, Σ0)
        Σ1 = Symmetric(Array(Σ), :L)
        Lc = cholesky!(Σ1)
        L = LowerTriangular(Lc.L)
        LldivY = L \ (Y .- μ)
        -size(Y,2)*logdet(L) - 0.5dot(LldivY, LldivY)
    end
    t0 = mvnormdirect(Y, μ, Σ)

    t1 = clean_ret(Normal, Y, μ, copyto!(Σcopy, Σ))
    t2, ∂Y, ∂μ, ∂Σ = ∂clean_ret(∂Normal, Y, μ, copyto!(Σcopy, Σ))

    t0, t1, t2
    
    test_grad(Normalc, ∂Normalc, (), Y, (μ, Σ, Σcopy))
    test_grad(Normalc, ∂Normalc, (Y,), μ, (Σ, Σcopy))
    test_grad(Normalc, ∂Normalc, (Y, μ), Σ, (Σcopy,), 1e-2)


end

using LoopVectorization, SIMDPirates, VectorizationBase
function vdot(δ::AbstractArray{T}) where {T}
    starget = vbroadcast(SVec{(VectorizationBase.pick_vector_width(T)),T}, zero(T))
    @vectorize for i ∈ 1:length(δ)
        vδ = δ[i]
        starget = vmuladd( vδ, vδ, starget )
    end
    target = extract_data(starget)
end

