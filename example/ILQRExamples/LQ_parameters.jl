using StackelbergControlHypothesesFiltering

using LinearAlgebra

num_players = 1

T = 101
t0 = 0.0
dt = 0.05
horizon = T * dt
times = dt * cumsum(ones(T)) .- dt

x0 = [0.;0.;0.;0.]# for the double integrator dynamics
xf = [5.; 5.; 0.; 0.]

println("initial state: ", x0')
println("desired state at time T: ", round.(xf', sigdigits=6), " over ", round(horizon, sigdigits=4), " seconds.")


#####################################
#        Define the dynamics.       #
#####################################
# Ensure the dynamics and costs are both homogenized similarly.
dyn = LinearDynamics([1. 0. dt 0.;
                      0. 1. 0. dt;
                      0. 0. 1. 0.;
                      0. 0. 0. 1.],
                     [vcat(zeros(2,2),
                      [dt 0; 0 dt])]) # 2d double integrator [x y xdot ydot]


#####################################
#         Define the costs.         #
#####################################
Q = Matrix(Diagonal(1*[1., 1., 1., 1.]))
R = Matrix(Diagonal(1*[1., 1.]))
# quad_cost = make_quadratic_cost_with_offset(Q, xf)
quad_cost = QuadraticCost(Q)
add_control_cost!(quad_cost, 1, R)
quad_cost = QuadraticCostWithOffset(quad_cost, xf)


#####################################
#    Define the initial controls.   #
#####################################
us_1 = zeros(udim(dyn), T)
us_1[1,:] .= 0.1
us_1[2,:] .= 0.01
duration = (T-1) * dt
us_1[1, :] .= (xf[3] - x0[3]) / duration # omega
us_1[2, :] .= (xf[4] - x0[4]) / duration # accel

# lin_dyn_0 = linearize_dynamics(dyn, (t0, t0+dt), x0, [zeros(udim(dyn))])
# ctrl_strats, _ = solve_lqr_feedback(lin_dyn_0, quad_cost, T)
# _, us_1 = unroll_feedback(dyn, times, ctrl_strats, x0)
# us_1 = us_1[1] + randn(size(us_1[1])) * 0.1
