# TODO(hamzah) Implement the estimator from the project or find a package that does that.

# TODO(hamzah) Tests?

function bayesian_filter_w_dynamic_state(prior_prob)

end


# Questions for David
# 1. How do I compute P(C_t | H_{t-1}, C_{t-1})?
# 2. How do Icompute integrals over the initial conditions? Do I need uncertainties propagated from the initial time?
# 3. A number of places require computing the Stackelberg values?
# 4. I don't understand how we expect the game to keep chaining off of previous game trajectories. Won't we have an
#    exponential number of games at that point?