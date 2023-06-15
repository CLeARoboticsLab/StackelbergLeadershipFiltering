using StackelbergControlHypothesesFiltering

using LinearAlgebra

include("params_time.jl")

x0 = [0.;0.;0.;0.]# for the double integrator dynamics
xf = [5.; 5.; 0.; 0.]

println("System: 1-player double integrator dynamics with quadratic offset cost")
println("initial state: ", x0')
println("desired state at time T: ", round.(xf', sigdigits=6), " over ", round(horizon, sigdigits=4), " seconds.")
println()

#####################################
#        Define the dynamics.       #
#####################################
cont_dyn = ContinuousLinearDynamics([0. 0. 1. 0.;
                                     0. 0. 0. 1.;
                                     0. 0. 0. 0.;
                                     0. 0. 0. 0.],
                                    [vcat(zeros(2,2),
                                     [1. 0; 0 1.])]) # 2d double integrator [x y xdot ydot]
dyn = discretize(cont_dyn, dt)

#####################################
#         Define the costs.         #
#####################################
Q = Matrix(Diagonal(1*[1., 1., 1., 1.]))
R = Matrix(Diagonal(1*[1., 1.]))

# Make the quadratic cost to be used for LQR.
lqr_quad_cost = QuadraticCost(Q)
add_control_cost!(lqr_quad_cost, 1, R)

# Before adding offsets, copy it for use in the tracking quadratic cost.
quad_cost = deepcopy(lqr_quad_cost)

# Finish up the one for LQR.
add_offsets!(lqr_quad_cost, xf, get_zero_ctrls(dyn))

# default zero controls
us_1 = zeros(udim(dyn), T)

# constant inputs - same as Jake
us_1[1,:] .= 0.1
us_1[2,:] .= 0.01

# Zero-offsets for the one in the tracking cost.
add_offsets!(quad_cost, zeros(xdim(dyn)), get_zero_ctrls(dyn))
quad_w_offset_cost = make_quadratic_tracking_cost(quad_cost, xf, get_zero_ctrls(dyn))

println("setting cost to Quadratic Offset Cost")
selected_cost = quad_w_offset_cost
