using StackelbergControlHypothesesFiltering

using LinearAlgebra

include("params_time.jl")

num_players = 1

x0 = [0.;0.;0.;0.]
xf = [5.; 5.; -pi/2; 0.]

println("System: 1-player unicycle dynamics with quadratic offset cost + exponential control cost")
println("initial state: ", x0')
println("desired state at time T: ", round.(xf', sigdigits=6), " over ", round(horizon, sigdigits=4), " seconds.")
println()


#####################################
#        Define the dynamics.       #
#####################################
dyn = UnicycleDynamics(num_players)


#####################################
#         Define the costs.         #
#####################################
Q = Matrix(Diagonal(1*[1., 1., 1., 1.]))
R = Matrix(Diagonal(1*[1., 1.]))
quad_cost = QuadraticCost(Q)
add_control_cost!(quad_cost, 1, R)
quad_w_offset_cost = QuadraticCostWithOffset(quad_cost, xf)

# TODO(hmzh) - Implement this particular non-quadratic cost and test.
const_multiplier = 1.0
max_accel = 5.
max_omega = 1.
example_ilqr_cost = ExampleILQRCost(quad_w_offset_cost, const_multiplier, max_accel, max_omega, xf, true)

println("setting cost to Example ILQR Cost")
selected_cost = example_ilqr_cost
