##
## TODO(documentation)
##

# A control strategy for multiple players.
struct FeedbackGainControlStrategy <: MultiplayerControlStrategy
    num_players::Int                                # number of players
    horizon::Int                                    # horizon
    Ps::AbstractVector{<:AbstractArray{Float64, 3}} # linear feedback gains
    ps::AbstractVector{<:AbstractMatrix{Float64}}   # constant feedback terms
end
FeedbackGainControlStrategy(Ps::AbstractVector{<:AbstractArray{Float64, 3}},
                            ps::AbstractVector{<:AbstractMatrix{Float64}}=[zeros(size(Ps[ii], 1), size(Ps[ii], 3)) for ii in 1:length(Ps)]) = FeedbackGainControlStrategy(length(Ps), size(Ps[1], 3), Ps, ps)

# This function accepts a feedback gain control strategy and applies it to a state at a given time (i.e. index).
function apply_control_strategy(tt::Int, strategy::FeedbackGainControlStrategy, x::AbstractArray{Float64})
    return [-strategy.Ps[ii][:, :, tt] * x - strategy.ps[ii][:, tt] for ii in 1:strategy.num_players]
end


function get_linear_feedback_gains(strategy::FeedbackGainControlStrategy)
    return strategy.Ps
end

function get_linear_feedback_gain(strategy::FeedbackGainControlStrategy, player_idx::Int)
    return strategy.Ps[player_idx]
end

function get_constant_feedback_gains(strategy::FeedbackGainControlStrategy)
    return strategy.ps
end

function get_constant_feedback_gain(strategy::FeedbackGainControlStrategy, player_idx::Int)
    return strategy.ps[player_idx]
end

# Export a commonly used control strategy for convenience and its required method.
export FeedbackGainControlStrategy, apply_control_strategy, get_linear_feedback_gain, get_constant_feedback_gain, get_linear_feedback_gains, get_constant_feedback_gains
