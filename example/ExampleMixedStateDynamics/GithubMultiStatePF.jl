# https://github.com/JuliaPOMDP/ParticleFilters.jl/issues/57
using POMDPs, ParticleFilters, Distributions
const MIXED_S = Tuple{Bool, Float64}

struct TuplePOMDP <: POMDP{MIXED_S, Bool, MIXED_S} end

POMDPs.initialstate(::TuplePOMDP) = Dirac((false, 0.0))

POMDPs.observation(::TuplePOMDP, sp::MIXED_S) = product_distribution(
    [Bernoulli(sp[1] ? 0.9 : 0.1), Normal(sp[2], 1.0)]
)

function POMDPs.gen(m::TuplePOMDP, s, a, rng)
    sp = (
        (rand(rng) < 0.75) ? s[1] : !s[1],
        s[2] + a + randn(rng)
    )
    o = rand(rng, observation(m, sp))
    r = 0.0
    return (;sp, o, r)
end



pomdp = TuplePOMDP()
n_particles = 100
T = 50
pf = BootstrapFilter(pomdp, n_particles)
belief = initialize_belief(pf, initialstate(pomdp))
state = rand(initialstate(pomdp))
action = true
b_hist = [belief.particles]
s_hist = [state]

for i ∈ 1:T-1
    global belief
    global state
    state, observation, reward = @gen(:sp,:o,:r)(pomdp, state, action)
    belief = update(pf, belief, action, observation)
    push!(s_hist, state)
    push!(b_hist, belief.particles)
end

using Plots

X = reduce(vcat, fill(i, n_particles) for i ∈ eachindex(b_hist, s_hist))
Y_1 = reduce(vcat, map(belief->getindex.(belief,1), b_hist))
Y_2 = reduce(vcat, map(belief->getindex.(belief,2), b_hist))

p1 = scatter(X, Y_1, label="particle state", markeralpha=0.1, title="x₁ (discrete)")
plot!(p1,1:T, getindex.(s_hist, 1), label="true state", c=:red, lw=3)

p2 = scatter(X, Y_2, label="particle state", markeralpha=0.1, title="x₂ (continuous)")
plot!(p2, 1:T, getindex.(s_hist, 2), label="true state", c=:red, lw=3)
p = plot(p1, p2)
