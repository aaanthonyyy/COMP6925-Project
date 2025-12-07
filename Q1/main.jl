import Pkg

required_packages = ["JuMP", "HiGHS", "Plots"]
installed = [pkg.name for pkg in values(Pkg.dependencies())]

for pkg in required_packages
    if pkg ∉ installed
        Pkg.add(pkg)
    end
end


using JuMP, HiGHS, Plots, Printf

# From Table 1, Workload requirements for numerical results
# W[j, k] = workload for shift j on day k
W = [
    19 16 22 22 22 22 22; # Morning (M)
    19 16 22 22 22 22 22; # Evening (E)
    14 11 16 16 16 16 16  # Night (N)
]

# A[j, k] = additional staff needed for shift j on day k
# Since we are not given the additional staff needed, we use a large value (100) as a placeholder to relax the constraint
A = fill(100, 3, 7)

shifts = 1:3
days = 1:7


function solve_for_n_employees(N)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # x[i,j,k]
    @variable(model, x[1:N, shifts, days], Bin)

    # Minimize total number of assigned shifts
    @objective(model, Min, sum(x))

    # Constraints
    # Eq. (8) - (10): Num consecutive shifts each employee ≤ 2
    max_consecutive_shifts = 2

    @constraint(model, [i = 1:N, k = days], sum(x[i, :, k]) <= max_consecutive_shifts)

    @constraint(
        model,
        [i = 1:N, k = days],
        x[i, 2, k] + x[i, 3, k] + x[i, 1, mod(k, 7)+1] <= max_consecutive_shifts
    )

    @constraint(
        model,
        [i = 1:N, k = days],
        x[i, 3, k] + x[i, 1, mod(k, 7)+1] + x[i, 2, mod(k, 7)+1] <= max_consecutive_shifts
    )

    # Eq. (11): Num shifts worked by each employee ≥ workload
    @constraint(model, [j = shifts, k = days], sum(x[:, j, k]) >= W[j, k])

    # Eq. (12): Num shifts worked by each employee ≤ workload + additional staff
    @constraint(model, [j = shifts, k = days], sum(x[:, j, k]) <= W[j, k] + A[j, k])


    # Eq. (13) Each employee should have 2 consecutive days off
    @constraint(model, [i = 1:N],
        sum(
            (x[i, j, mod(i - 1, 7)+1] + x[i, j, mod(i, 7)+1])
            for j in 1:3
        ) == 0
    )

    # Eq. (14) Each employee should work at least 1 shift per day
    @constraint(model, [i = 1:N, k = 1:5],
        sum(x[i, j, mod(i + k, 7)+1] for j in shifts) >= 1
    )

    optimize!(model)

    return model
end

#= (a) Implement Hosein, Job and Sankar’s [1] model to determine the optimal size of security
personnel while accounting for overtime pay in Julia. Ensure you can replicate Fig. 1 from the
paper.

Recreating results from Hosein et al. for N = 70:90
=#

N_range = 70:90
M = 395     # Total workload
rO = 0.245  # Cost factor for overtime

computed_costs = []
lower_bound_costs = []

lower_bound(N) = N + (max(0, (M - 5 * N)) * rO)

for N in N_range
    model = solve_for_n_employees(N)
    total_shifts = objective_value(model)
    overtime_shifts = max(0, total_shifts - (5 * N))
    cost = N + (overtime_shifts * rO)

    push!(computed_costs, cost)
    push!(lower_bound_costs, lower_bound(N))
end


p = plot(
    N_range, lower_bound_costs,
    label="Lower Bound",
    color=:black,
    linewidth=1.0,
    linestyle=:solid,
    legend=:topleft,
    framestyle=:box,
    xticks=70:2:90,
    ylim=(78, 91),
    xlabel="Number of Employees (N)",
    ylabel="Normalized Total Cost (C/S)",
    title="Variation of Total Cost with Number of Employees",
    fontfamily="Computer Modern",
    guidefontsize=10,
    tickfontsize=9,
    legendfontsize=9,
    size=(600, 450),
    dpi=600
)

plot!(
    N_range, computed_costs,
    label="Computed Solution",
    seriestype=:line,
    color=:navy,
    linewidth=1.0,
    markershape=:circle,
    markercolor=:navy,
    markersize=4,
    markerstrokewidth=0,
    grid=true,
    gridalpha=0.1,
    gridstyle=:dash,
)

# Highlight the optimal solution 
min_cost, idx = findmin(computed_costs)
optimal_N = N_range[idx]

scatter!(
    [optimal_N], [min_cost],
    color=:red,
    markersize=3,
    markerstrokewidth=0,
    label="Optimal Solution (N=$optimal_N)"
)

display(p)


savefig("816008250_q1a_results.png")


#= (b) Check evidence for employee 1 and employee 8 =#
optimal_model = solve_for_n_employees(72)
x = optimal_model[:x]
# both have sat/sun off
println("--- Evidence Check ---")
for k in 1:7
    s1 = "OFF"
    s8 = "OFF"
    for j in 1:3
        if value(x[1, j, k]) > 0.5 s1 = "Shift $j" end
        if value(x[8, j, k]) > 0.5 s8 = "Shift $j" end
    end
    println("Day $k: Emp 1 is $s1 | Emp 8 is $s8")
end
println("-"^70)



#= (c) Assume that the overtime rate is increased (e.g., from baseline of 1.5rn to 2.5rn). 
Solve the problem again and compare the new assignments and figures.
=#

# rn = 0.7/5 * S
rn = 0.7 / 5
rO_old = 1.75 * rn
rO_new = 2.5 * rn

old_computed = []
new_computed = []
old_lb = []
new_lb = []

N_range = 70:90
workload = 395

for N in N_range
    # A. Solve Schedule
    model = solve_for_n_employees(N)

    if termination_status(model) == MOI.OPTIMAL
        total_shifts = objective_value(model)
        overtime = max(0, total_shifts - (5 * N))

        # B. Computed Costs (The Dots)
        push!(old_computed, N + (overtime * rO_old))
        push!(new_computed, N + (overtime * rO_new))
    else
        push!(old_computed, NaN)
        push!(new_computed, NaN)
    end

    # C. Lower Bounds (The Lines)
    min_overtime = max(0, workload - (5 * N))
    push!(old_lb, N + (min_overtime * rO_old))
    push!(new_lb, N + (min_overtime * rO_new))
end

# optimal points for old and new solutions
min_old, idx_old = findmin(old_computed)
min_new, idx_new = findmin(new_computed)

optimal_N_old = N_range[idx_old]
optimal_N_new = N_range[idx_new]


p_c = plot(
    framestyle=:box,
    grid=true,
    gridalpha=0.1,
    gridstyle=:dash,
    xticks=70:2:90,
    legend=:top,
    xlabel="Number of Employees (N)",
    ylabel="Normalized Total Cost (C/S)",
    title="Comparison: Baseline vs. High Overtime Rate",
    fontfamily="Computer Modern",
    size=(700, 500),
    dpi=600
)

# Old Lower Bound
plot!(N_range, old_lb,
    label="Baseline Lower Bound (\$1.75 r_n\$)", color=:gray,
    linewidth=1.0, linestyle=:dash
)

# Old Computed
plot!(N_range, old_computed,
    label="Old Computed (N=$(N_range[idx_old]))",
    color=:gray70, linewidth=1.5, linestyle=:dash,
    marker=:circle, markersize=3, markerstrokewidth=0
)

# New Lower Bound
plot!(N_range, new_lb,
    label="New Lower Bound (\$2.5 r_n\$)", color=:black,
    linewidth=1.0, linestyle=:solid
)

# New Computed
plot!(N_range, new_computed,
    label="New Computed",
    color=:navy, linewidth=1.5,
    marker=:circle, markersize=4, markerstrokewidth=0
)

scatter!(
    [N_range[idx_old]], [min_old],
    color=:gray50,
    markersize=5,
    markerstrokewidth=0,
    label=""
)
scatter!(
    [N_range[idx_new]], [min_new],
    color=:red,
    markersize=6,
    markerstrokewidth=0,
    label="New Optimal (N=$(N_range[idx_new]))"
)

display(p_c)
savefig("816008250_q1c_comparison.png")






function print_detailed_results(N_optimal)
    println("\n--- Generating Tables for Optimal N = $N_optimal ---")

    # Re-solve specific instance to get variables
    model = solve_for_n_employees(N_optimal)

    # 1. TABLE 2 REPLICATION (Aggregate)
    println("\n[TABLE 2 REPLICATION] Aggregate Schedule (Workload)")
    println("-"^70)
    @printf("%-10s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n", "Shift", "Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri")
    println("-"^70)

    shift_names = ["Morning", "Evening", "Night"]

    for j in 1:3
        print(rpad(shift_names[j], 10))
        for k in 1:7
            x = model[:x]
            assigned = sum(value.(x[:, j, k]))
            required = W[j, k]
            s = "$(Int(assigned)) ($required)"
            print(rpad(s, 9))
        end
        println()
    end
    println("-"^70)

    # 2. INDIVIDUAL VALIDATION (Employee #1)
    println("\n[VALIDATION] Schedule for Staff Member #1")
    println("(Proves 2 days off and shift constraints)")
    println("-"^40)
    days_week = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"]

    for k in 1:7
        # Find which shift they are working (if any)
        worked = 0
        s_name = "OFF"
        for j in 1:3
            if value(model[:x][1, j, k]) > 0.5
                worked = 1
                s_name = shift_names[j]
            end
        end
        println("$(days_week[k]): $s_name")
    end
    println("-"^40)
end

println("\n\n")
println("="^70)
println(">>> QUESTION 1(A): BASELINE SCENARIO")
println("    Overtime Rate: 1.75x (Baseline)")
println("    Optimal Staff (N): $optimal_N_old")
println("    Minimum Cost: $(round(optimal_N_old, digits=3))")
println("="^70)
print_detailed_results(optimal_N_old)


println("\n\n")
println("="^70)
println(">>> QUESTION 1(C): HIGH OVERTIME SCENARIO")
println("    Overtime Rate: 2.5x (Increased)")
println("    Optimal Staff (N): $optimal_N_new")
println("    Minimum Cost: $(round(optimal_N_new, digits=3))")
println("="^70)
print_detailed_results(optimal_N_new)

