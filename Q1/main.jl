using JuMP, HiGHS

# From Table 1, Workload requirements for numerical results
# W[j, k] = workload for shift j on day k
W = [
    19 16 22 22 22 22 22; # Morning (M)
    19 16 22 22 22 22 22; # Evening (E)
    14 11 16 16 16 16 16  # Night (N)
]

# A[j, k] = additional staff needed for shift j on day k
A = fill(5, 3, 7)

N = 72 # Number of staff members 

days = 1:7
shifts = 1:3

model = Model(HiGHS.Optimizer)

# x[i,j,k] = 1 if staff i works shift j on day k
@variable(model, x[1:N, shifts, days], Bin)

# Minimize total number of assigned shifts
@objective(model, Min, sum(x))

# Eq. (8) - (10): Num consecutive shifts each employee ≤ 2
max_consecutive_shifts = 2

@constraint(model, [i=1:N, k=days], sum(x[i, :, k]) <= max_consecutive_shifts)

@constraint(
    model, 
    [i=1:N, k=days], 
    x[i,2,k] + x[i,3,k] + x[i,1,mod(k, 7) + 1] <= max_consecutive_shifts
)

@constraint(
    model, 
    [i=1:N, k=days], 
    x[i,3,k] + x[i,1,mod(k, 7) + 1] + x[i,2,mod(k, 7) + 1] <= max_consecutive_shifts
)

# Eq. (11): Num shifts worked by each employee ≥ workload
@constraint(model, [j=shifts, k=days], sum(x[:, j, k]) >= W[j, k])

# Eq. (12): Num shifts worked by each employee ≤ workload + additional staff
@constraint(model, [j=shifts, k=days], sum(x[:, j, k]) <= W[j, k] + A[j, k])


# Eq. (13) Each employee should have 2 consecutive days off
@constraint(model, [i=1:N],
    sum(
        (x[i, j, mod(i-1, 7) + 1] + x[i, j, mod(i, 7) + 1])
        for j in 1:3
    ) == 0
)

# Eq. (14) Each employee should work at least 1 shift per day
@constraint(model, [i=1:N, k=1:5], 
    sum(x[i, j, mod(i+k, 7) + 1] for j in shifts) >= 1
)



optimize!(model)

# 2. Check results
if termination_status(model) == MOI.OPTIMAL
    println("Optimal Schedule Found!")
    println("Total Shifts Assigned: ", objective_value(model))
    
    # Optional: Print how many shifts exceed the minimum workload (Overtime/Excess)
    total_workload = sum(W)
    total_assigned = objective_value(model)
    println("Excess/Overtime Shifts: ", total_assigned - total_workload)
else
    println("Model is infeasible or unbounded.")
end
