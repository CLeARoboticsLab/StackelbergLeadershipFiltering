using StackelbergControlHypothesesFiltering

using LinearAlgebra
using Plots


# TODO(hamzah) - Run iLQR and get outputs.
T = 11
x0 = zeros(4)
t0 = 0.0

times = 0.5 * ones(T) .- 0.5

dyn = UnicycleDynamics(1)

xf = [5.; 5.; 3*pi/2; 0.]
max_accel = 5
max_omega = 1
Q = Diagonal([1., 1., 1., 1.])
R = Diagonal([1., 1.])
const_multiplier = 1.0
cost = ExampleILQRCost(Q, R, const_multiplier, max_accel, max_omega, xf)

xs_1 = zeros(xdim(dyn), T)
for tt in 1:T
    lambda = (tt-1)/T
    xs_1[:, tt] = lambda * (xf - x0)
end

us_1 = zeros(udim(dyn), T)
for tt in 1:T
    lambda = (tt-1)/T
    us_1[1, tt] = lambda * (xf[3] - x0[3])/(T-1) # omega
    us_1[2, tt] = lambda * (xf[4] - x0[4])/(T-1) # accel
end

xs_i, us_i, is_converged, num_iters = ilqr(T, x0, t0, times, dyn, cost, xs_1, us_1)

println(size(xs_i), size(us_i))

# TODO(hamzah) - Plot time vs. cost


