"""
Provides virtual interface for the gray-box environment and simulator.
"""
module GrayBox

using Random
using Distributions

export Simulation,
       Sample,
       Environment,
       EnvironmentSample,
       environment,
       transition!


"""
    GrayBox.Simulation

Abstract base type for a gray-box simulation.
"""
abstract type Simulation end



"""
    GrayBox.Sample

Holds sampled value and log-probability.
"""
mutable struct Sample{T}
    value::T
    logprob::Real
end


"""
    GrayBox.Environment

Alias type for a dictionary of gray-box environment distributions.

     e.g., `Environment(:variable_name => Sampleable)`
"""
const Environment = Dict{Symbol, Sampleable}


"""
    GrayBox.EnvironmentSample

Alias type for a single environment sample.

    e.g., `EnvironmentSample(:variable_name => Sample(value, logprob))`
"""
const EnvironmentSample = Dict{Symbol, Sample}

"""
	GrayBox.State

Alias type for a state.
"""
const State = Union{Vector{Float64},Nothing}


"""
    environment(sim::GrayBox.Simulation)

Return all distributions used in the simulation environment.
"""
function environment(sim::Simulation)::Environment end


"""
    transition!(sim::Union{GrayBox.Simulation, GrayBox.Simulation})::Real

Given an input `sample::EnvironmentSample`, apply transition and return the transition log-probability.
"""
function transition!(sim::Simulation, sample::EnvironmentSample)::Real end


"""
    transition!(sim::GrayBox.Simulation)::Real

Apply a transition step, and return the transition log-probability (used with `ASTSeedAction`).
"""
function transition!(sim::Simulation)::Real end

"""
	state(sim::GrayBox.Simulation)::State

Get current state of simulation as a Vector{Float64}.
"""
function state(sim::Simulation)::State end

"""
	count_actions(sim::Simulation)::Int64

Get total dimension of action space, to be implemented by simulator.
"""
function count_actions(sim::Simulation)::Int64 end

"""
	pack(sim::GrayBox.Simulation, actions::Vector{Float64})::EnvironmentSample

Allows general form of actions, delegates translation from Vector{Float64} to simulator.
"""
function pack(sim::Simulation, actions::Vector{Float64})::EnvironmentSample end

end # module GrayBox
