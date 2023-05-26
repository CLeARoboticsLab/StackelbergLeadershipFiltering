# This file defines the dynamics and cost functions for an integrator game with up to N players.

# Two player dynamics
# Dynamics (Euler-discretized double integrator equations with Δt = 0.1s).
# State for each player is layed out as [x, ẋ, y, ẏ].

using LinearAlgebra

num_players = 2
num_states = 8
num_ctrls = [2, 2]

Ã(dt) = [1 dt;
         0  1]
integrator_A(N, dt) = I + Matrix(blockdiag([sparse(Ã(dt)) for _ in 1:N]...))

integrator_B(N, dt) = Matrix(blockdiag([sparse([0. dt]) for _ in 1:N]...))

B₂(dt) = vcat(zeros(4, 2),
              [0   0;
               dt  0;
               0   0;
               0   dt])

IntegratorDynamics(N::Int, dt::Float64) = LinearDynamics(integrator_A(N, dt), [integrator_B(N, dt)])


export IntegratorDynamics
