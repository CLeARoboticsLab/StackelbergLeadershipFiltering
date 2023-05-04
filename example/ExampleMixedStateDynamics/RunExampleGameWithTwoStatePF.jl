using StackelbergControlHypothesesFiltering

using Plots
using Printf
using StatsBase

include("ConfigSimpleKinematic1DDynamics.jl")
include("SimpleKinematic1DDynamics.jl")


### RUN THE MODEL

# Specify particle filter details.
Ns = 100

# Integrate process noise into the dynamics.
gen_process_noise(rng, num_samples) = zeros(2, num_samples)# rand(rng, process_noise_distribution, num_samples)
f_dynamics_1(t_range, x, u, rng) = f₁(t_range, x, u, gen_process_noise(rng, size(x, 2)))
f_dynamics_2(t_range, x, u, rng) = f₂(t_range, x, u, gen_process_noise(rng, size(x, 2)))
f_dynamics = [f_dynamics_1, f_dynamics_2]

# set controls
u_inputs = Dict(i => ones(1) for i in 1:num_data) # inputs should always be 1 or 0

# Run the normal particle filter if needed for sanity checking.
# x̂✶_PF, P_PF, z̄_PF, P̄_zz_PF, ϵ_bar, ϵ_hat, N̂s_n, particles =
#                                 PF(x̄_prior,P_prior,u_inputs,t,t0,z,meas_R,f_dynamics_1,h₁,
#                                 process_noise_distribution;seed=seed,Ns=Ns)


x̂✶_PF, P_PF, z̄_PF, P̄_zz_PF, ϵ_bar, ϵ_hat, N̂s_n, s_PF, s_probs, particles = 
    two_state_PF(x̄_prior,P_prior,u_inputs,s_prior_distrib,t,t0,z,meas_R,discrete_state_transition,f_dynamics,[h₁, h₂],
                 ;seed=seed,Ns=Ns)

# Run an iterative particle filter that can be used real-time.
rng = MersenneTwister(seed)
pf = initialize_particle_filter(x̄_prior, P_prior, s_prior_distrib, t0, Ns, num_data, size(meas_R, 1), rng)
for tt in 1:num_data
    prev_time = (tt == 1) ? t0 : t[tt-1]
    time_range = (prev_time, t[tt])
    step_pf(pf, time_range, f_dynamics,[h₁, h₂], discrete_state_transition, u_inputs[tt], z[tt, :], meas_R)
end
x̂✶_PF, P_PF, z̄_PF, P̄_zz_PF, ϵ_bar, ϵ_hat, N̂s_n, s_PF, s_probs, particles = pf.x̂, pf.P, pf.z̄, pf.P̄_zz, pf.ϵ_bar, pf.ϵ_hat, pf.N̂s, pf.s, pf.ŝ_probs, pf.particles
println("final PF: ", x̂✶_PF[:, num_data])

### PLOT THE RESULTS

x̂✶ = permutedims(x̂✶_PF,(2,1))
P = permutedims(P_PF,(3,2,1))
z̄ = permutedims(z̄_PF,(2,1))
P̄_zz = permutedims(P̄_zz_PF,(3,2,1))

# 1. plot velocity signals
plot_mult = 3.0

s=1
p1 = scatter(t, particles[1, :, :][:, :]', color=:black, markersize=0.15, label="", yrange=(-plot_mult*2.0,plot_mult*2.0), legend=:outertopright, ylabel="velocity (m/s)")
if !only_pos_measurements
    p1 = plot!(t, z[:, 1], label="meas.", color=:green)
end
p1 = plot!(t, x̂✶[:, 1], label="est. vel.", color=:red)
p1 = plot!(t, x_true[:, 1], label="true vel.", color=:blue)

# 2. plot position signals
s=2
p2 = scatter(t, particles[2, :, :][:, :]', color=:black, markersize=0.15, label="", yrange=(-plot_mult*5.0,plot_mult*5.0), legend=:outertopright, ylabel="position (m)")
z_idx = (only_pos_measurements) ? 1 : 2
p2 = plot!(t, z[:, z_idx], label="meas.", color=:green)
p2 = plot!(t, x̂✶[:, 2], label="est. pos.", color=:red)
p2 = plot!(t, x_true[:, 2], label="true pos.", color=:blue)

# 3. compute expected probabilities and plot discrete state signals
was_resampled = [resample_condition(Ns, N̂s) for N̂s in N̂s_n]
println(sum(was_resampled))
expected_probs = zeros(2, num_data)
expected_probs[:, 1] = [p_init; 1-p_init]
for i in 2:num_data
    expected_probs[:, i] = disc_state_matrix * expected_probs[:, i-1]
    if was_resampled[i]
        # This is slightly incorrect because
        # 1. It relies on an out put produced by the algorithm.
        # 2. it depends on whether s_probs is calculated before or after resampling.
        #    Right now, it is before so we use the next probabilities in time.
        ip1 = (i == num_data) ? num_data : i+1
        p_new = s_probs[ip1]
        expected_probs[:, i] = [p_new; 1-p_new]
    end
end

s=3
p3 = plot(t, expected_probs[1, :], label="Markov prob.\n w/ resamp.", title="Probability of positive accel.", yrange=(-0.1, 1.1), legend=:outertopright)
p3 = plot!(t, mean(expected_probs[1, :]) * ones(num_data), label="actual avg.")
# p3 = plot!(t, entropy.(expected_probs[1, :]), label="entropy")
p3 = plot!(t, s_probs', label="actual")
p3 = plot!(t, true_prob_state1 * ones(ℓ), label="true", color=:black, xlabel="time (s)", ylabel="p(pos. accel.)")

display(plot(p1,p2,p3,layout=(3,1),size=(800,800),
        plot_title="1D kinematics system velocity and position with state"))



# # RMS errors
# errors = x̂✶ - x_true
# position_rms_error = √(sum([errors[k,2]^2 for k in 1:ℓ])/ℓ)
# velocity_rms_error = √(sum([errors[k,1]^2 for k in 1:ℓ])/ℓ)

# println("RMS Errors - Normal Distribution Process Noise Assumption")
# println("Position RMS error (m): ",position_rms_error)
# println("Velocity RMS error (m/s): ",velocity_rms_error)
# println("")

# ylims_pos = (-1., 1.0)
# ylims_vel = (-1., 1.0)

# # part 1 - error plots
# s=1
# p1 = plot(t, errors[:,s],
#             linecolor=:gray,ylims=ylims_pos,
#             title="Position error (m)",label="Error")
# p1 = plot!(t, 3 .* .√P[:,s,s],label="3σ",linestyle=:dash,linecolor=:green)
# p1 = plot!(t, -3 .* .√P[:,s,s],label="",linestyle=:dash,linecolor=:green)

# s=2
# p2 = plot(t, errors[:,s],
#             linecolor=:gray,ylims=ylims_vel,
#             title="Velocity error (m/s)",label="Error",
#             xlabel="Time (sec)",legend=false)
# p2 = plot!(t, 3 .* .√P[:,s,s],label="3σ",linestyle=:dash,linecolor=:green)
# p2 = plot!(t, -3 .* .√P[:,s,s],label="",linestyle=:dash,linecolor=:green)

# display(plot(p1,p2,layout=(2,1),size=(800,800),
#         plot_title="Errors - Normal Distribution Process Noise Assumption"))


# ## part 2 - residuals

# # plots
# s=1
# p3 = plot(t, ϵ_bar[:,s], 
#         title="Velocity (m)",
#         label="Prediction",
#         legend=true,xlabel="Time (sec)")
# p3 = plot!(t, ϵ_hat[:,s],
#         label="Post-fit")

# s=2
# p4 = plot(t, ϵ_bar[:,s], 
#         title="Position (m/s)",
#         label="Prediction",
#         legend=true,xlabel="Time (sec)")
# p4 = plot!(t, ϵ_hat[:,s],
#         label="Post-fit")

# display(plot(p3,p4,layout=(2,1),size=(1000,600),
#             plot_title="Residuals - Uniform Distribution Process Noise Assumption"))

# part 3 - statistics
# println("Residuals Statistics - Uniform Distribution Process Noise Assumption")

# s=1
# println("Velocity:")
# println("Prediction residuals mean: ", 
#         @sprintf("%.2f",mean(ϵ_bar[:,s])), 
#         "  Post-fit residuals mean: ",
#         @sprintf("%.2f",mean(ϵ_hat[:,s])))
# println("Prediction residuals standard deviation: ",
#         @sprintf("%.2f",std(ϵ_bar[:,s])),
#         "  Post-fit residuals standard deviation: ",
#         @sprintf("%.2f",std(ϵ_hat[:,s])))
# println("")

# s=2
# println("Position:")
# println("Prediction residuals mean: ", 
#         @sprintf("%.2f",mean(ϵ_bar[:,s])), 
#         "  Post-fit residuals mean: ",
#         @sprintf("%.2f",mean(ϵ_hat[:,s])))
# println("Prediction residuals standard deviation: ",
#         @sprintf("%.2f",std(ϵ_bar[:,s])),
#         "  Post-fit residuals standard deviation: ",
#         @sprintf("%.2f",std(ϵ_hat[:,s])))
# println("")
