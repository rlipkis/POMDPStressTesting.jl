"""
Parameters for the AST version of the `MCTS.DPWSolver` solver.
"""
function MCTSPWSolver(; kwargs...)
    return MCTS.DPWSolver(; estimate_value=AST.rollout, # required.
                            enable_state_pw=false, # required.
                            reset_callback=AST.go_to_state, # Custom fork of MCTS.jl
                            tree_in_info=true,
                            show_progress=true,
                            kwargs...)
end



"""
Return optimal action path from MCTS tree (using `info[:tree]` from `(, info) = action_info(...)`).
"""
function get_optimal_path(mdp, tree, snode::Int, actions::Vector{ASTAction}; verbose::Bool=false)
    best_Q = -Inf
    sanode = 0
    for child in tree.children[snode]
        if tree.q[child] > best_Q
            best_Q = tree.q[child]
            sanode = child
        end
    end

    if verbose
        print("State = 0x", string(tree.s_labels[snode].hash, base=16), "\t:\t")
    end
    if sanode != 0
        if verbose
            print("Q = ", tree.q[sanode], "\t:\t")
            println("Action = ", string(tree.a_labels[sanode]))
        end
        push!(actions, tree.a_labels[sanode])

        # Find subsequent maximizing state node
        best_Q_a = -Inf
        snode2 = 0
        for tran in tree.transitions[sanode]
            if tran[2] > best_Q_a
                best_Q_a = tran[2]
                snode2 = tran[1]
            end
        end

        if snode2 != 0
            get_optimal_path(mdp, tree, snode2, actions, verbose=verbose)
        end
    else
        AST.go_to_state(mdp, tree.s_labels[snode])
        if verbose
            if BlackBox.isevent(mdp.sim)
                println("Event.")
            else
                println("End of tree.")
            end
        end
    end

    return actions::Vector{ASTAction}
end
get_optimal_path(mdp, tree, state, actions::Vector{ASTAction}=ASTAction[]; kwargs...) = get_optimal_path(mdp, tree, tree.s_lookup[state], actions; kwargs...)



"""
    AST.search!(planner::DPWPlanner; return_tree::Bool=false, verbose=false)

Search for failure events using the planner of the `MCTSPWSolver`.

This is the main entry function to get a failure trajectories from the planner.
"""
function AST.search!(planner::DPWPlanner; return_tree::Bool=false, verbose=false)
    mdp::ASTMDP = planner.mdp
    Random.seed!(mdp.params.seed) # Determinism

    initstate = AST.initialstate(mdp)
    tree = MCTS.action_info(planner, initstate, tree_in_info=true)[2][:tree] # this runs MCTS
    action_path::Vector = get_optimal_path(mdp, tree, initstate, verbose=verbose)

    return return_tree ? tree : action_path
end
