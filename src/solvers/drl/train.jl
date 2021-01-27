# Modified from Shreyas Kowshik's implementation.

function train_step!(planner::Union{TRPOPlanner, PPOPlanner},
                     policy::Union{CategoricalPolicy, DiagonalGaussianPolicy},
                     episode_buffer::Buffer, stats_buffer::Buffer)
    env = planner.env
    solver = planner.solver

    clear!(episode_buffer)
    collect_rollouts!(env, solver, policy, episode_buffer, solver.episode_length, stats_buffer)

    if solver isa TRPOSolver
        θ, reconstruct = get_flat_params(get_policy_net(policy))
        old_params = copy(θ)
    end

    num_epochs = solver isa PPOSolver ? solver.ppo_epochs : 1
	mb_losses = Vector{Float64}[]
    for epoch in 1:num_epochs
    	idxs = partition(shuffle(1:size(episode_buffer.exp_dict["states"])[end]), solver.batch_size)

    	for i in idxs
        	mb_states = episode_buffer.exp_dict["states"][:,i]
        	mb_actions = episode_buffer.exp_dict["actions"][:,i]
        	mb_advantages = episode_buffer.exp_dict["advantages"][:,i]
        	mb_returns = episode_buffer.exp_dict["returns"][:,i]
        	mb_log_probs = episode_buffer.exp_dict["log_probs"][:,i]
        	mb_kl_vars = episode_buffer.exp_dict["kl_params"][i]

        	if solver isa TRPOSolver
            	trpo_update!(solver, policy, mb_states, mb_actions, mb_advantages, mb_returns, mb_log_probs, mb_kl_vars, old_params, reconstruct)
        	elseif solver isa PPOSolver
            	kl_div = mean(kl_divergence(policy, mb_kl_vars, mb_states))
            	solver.verbose ? println("KL Sample : $(kl_div)") : nothing

            	ppo_update!(solver, policy, mb_states, mb_actions, mb_advantages, mb_returns, mb_log_probs, mb_kl_vars)
        		push!(mb_losses, [solver.lp, solver.lv, solver.le])
        	end
    	end
    end

	# For plotting
    return mean(hcat(mb_losses...), dims=2)[:]
end


function train!(planner::Union{TRPOPlanner, PPOPlanner})
    solver::Union{TRPOSolver, PPOSolver} = planner.solver

    # Create or load policy
    if solver.resume
        if solver.policy_type == :discrete
            policy = load_policy(solver, "weights", CategoricalPolicy) # TODO: parameterize path
        elseif solver.policy_type == :continuous
            policy = load_policy(solver, "weights", DiagonalGaussianPolicy) # TODO: parameterize path
        end
    else
        policy = get_policy(solver)
    end

    # Define buffers
    episode_buffer::Buffer = initialize_episode_buffer()
    stats_buffer::Buffer = initialize_stats_buffer()
    solver.show_progress ? progress = Progress(solver.num_episodes) : nothing
    losses = Vector{Float64}[]

    for i in 1:solver.num_episodes
        solver.verbose ? println("Episode $i") : nothing
        mb_losses = train_step!(planner, policy, episode_buffer, stats_buffer)
        push!(losses, mb_losses)
        solver.verbose ? println(mean(stats_buffer.exp_dict["rollout_rewards"])) : nothing
        solver.show_progress ? next!(progress) : nothing

        if solver.save && i % solver.save_frequency == 0
            save_policy(policy, "weights") # TODO. Handle where to save policy
        end
    end

    planner.policy = policy

	# Process losses
	m_avg(vs, n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]
	n_smooth = 100
	lp = solver.cₚ*[loss[1] for loss in losses]
	lv = solver.cᵥ*[loss[2] for loss in losses]
	le = solver.cₑ*[loss[3] for loss in losses]
	lt = -lp .+ lv .- le
	lt_smooth = m_avg(lt, n_smooth)

	# Plot losses
	@show mean(lp)
	@show mean(lv)
	@show mean(le)
	@show mean(lt)
	x_shift = collect(1:length(lt_smooth)) .+ n_smooth/2
	p = plot(x_shift, lt_smooth, label="total")
	p = plot!(p, x_shift, m_avg(lp, n_smooth), label="policy")
	p = plot!(p, x_shift, m_avg(lv, n_smooth), label="value")
	p = plot!(p, x_shift, m_avg(le, n_smooth), label="entropy")

	# Save plot
    enc = planner.env.problem.sim.encounter_number
    outdir = "/home/p-rse/ACAS/ACAS_Sim.jl/output"
	fn = "losses_$enc.pdf"
	savefig(p, joinpath(outdir, fn))

    return policy
end
