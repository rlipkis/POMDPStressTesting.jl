import .AST: ASTMDP, ASTMetrics

"""
    print_metrics(planner)
    print_metrics(mdp::ASTMDP)
    print_metrics(metrics::ASTMetrics)

Print failure metrics including:
- iteration of first failure
- number of failures
- failure rate
"""
print_metrics(planner) = print_metrics(planner.mdp.metrics)
print_metrics(mdp::ASTMDP) = print_metrics(mdp.metrics)
function print_metrics(metrics::ASTMetrics)
    E = metrics.event

    if findfirst(E) === nothing
        @info "No failures recorded."
        return 0
    else
        first_failure = findfirst(E)
        num_evals = length(E)
        println("First failure: ", first_failure, " of ", num_evals)
        num_failures = sum(E)
        println("Number of failures: ", num_failures)
        failure_rate = sum(E)/length(E) * 100
        println("Failure rate: ", round(failure_rate, digits=5), "%")
        return failure_rate
    end
end