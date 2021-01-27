# Modified from Shreyas Kowshik's implementation.

function save_policy(policy::Union{CategoricalPolicy, DiagonalGaussianPolicy}, path::String="weights")
    if isdir(path) == false
        mkpath(path)
    end
    save_policy_net(policy, path)
    save_value_net(policy, path)
end


function save_policy_net(policy::CategoricalPolicy, path::String)
    π = policy.π
    @save joinpath(path, "policy_cat.jld") π
end


function save_policy_net(policy::DiagonalGaussianPolicy, path::String)
    μ = policy.μ
    logΣ = policy.logΣ
    @save joinpath(path, "policy_mu.jld") μ
    @save joinpath(path, "policy_sigma.jld") logΣ
end


function save_value_net(policy::Union{CategoricalPolicy, DiagonalGaussianPolicy}, path)
    value_net = policy.value_net
    @save joinpath(path, "value.jld") value_net
end


function load_policy(solver, path::String, ::Type{CategoricalPolicy})
    # TODO: create policy after loading.
    policy = CategoricalPolicy(solver)

    @load joinpath(path, "policy_cat.jld") π
    @load joinpath(path, "value.jld") value_net

    policy.π = π
    policy.value_net = value_net

    return policy
end


function load_policy(solver, path::String, ::Type{DiagonalGaussianPolicy})
    # TODO: create policy after loading.
    policy = DiagonalGaussianPolicy(solver, solver.log_std)

    @load joinpath(path, "policy_mu.jld") μ
    @load joinpath(path, "policy_sigma.jld") logΣ
    @load joinpath(path, "value.jld") value_net

    policy.μ = μ
    policy.logΣ = logΣ
    policy.value_net = value_net

    return policy
end
