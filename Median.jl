using JuMP, Gurobi
tic()
####################################
# CHANGE THE FOLLOWING FOR YOUR PARTICULAR DATA:

# BIG M CONSTANT
M = 50

# PATH TO YOUR DATASETS HERE
trainfile = "Train lpga2009_opt.csv" 
validationfile = "Validation lpga2009_opt.csv" 
testfile = "Test lpga2009_opt.csv"

####################################

train = readcsv(trainfile)
validation = readcsv(validationfile)
test = readcsv(testfile)
print("\nDone reading in data\n")

D = size(train)[2] - 1	
N = size(train)[1]
y_train = train[:, 1]
X_train = train[:,2:(D+1)]
y_validation = validation[:, 1]
X_validation = validation[:,2:(D+1)]
y_test = test[:,1]
X_test = test[:, 2:(D+1)]


# determine centering and scaling factors
mean_X_train = mean(X_train,1);
mean_y_train = mean(y_train);
X_train = X_train .- mean_X_train;
denom_X_train = zeros(D);
for i in 1:D
	denom_X_train[i] = norm(X_train[:,i]);
end

# center and scale the datasets
X_train = X_train ./ denom_X_train';
X_validation = (X_validation .- mean_X_train)./denom_X_train'
X_test = (X_test .- mean_X_train)./denom_X_train'
y_train = y_train .- mean_y_train;
y_validation = y_validation .- mean_y_train
y_test = y_test .- mean_y_train

SST_test = sum((mean(y_train) - y_test).^2)

#######
# Build MIO optimization model

print("Building optimization model\n")

m = Model(solver = GurobiSolver())

# to specify any solver parameters, add them like this:
# m = Model(solver = GurobiSolver(TimeLimit = 10, MIPGap = 0.01))

@defVar(m, Beta[1:D])
# @defVar(m, r[1:N])
@defVar(m, gamma)
@defVar(m, z[1:N], Bin);


# Big M constraints
### Version Dimitris

# @defVar(m, mu_upper[1:N] <= 0);
@defVar(m, mu_lower[1:N] >= 0);
@addConstraint(m,lb1[i=1:N], gamma + mu_lower[i] >= y_train[i] - dot(Beta, vec(X_train[i,:])));
@addConstraint(m,lb2[i=1:N], gamma + mu_lower[i] >= -y_train[i] + dot(Beta, vec(X_train[i,:])));
@addConstraint(m,bigM[i=1:N], 1e6*(1-z[i]) >= mu_lower[i])

### Version Colin
# @addConstraint(m,bigM1[i=1:N], gamma + 1e6*(1-z[i]) >= y_train[i] - dot(Beta, vec(X_train[i,:])))
# @addConstraint(m,bigM2[i=1:N], gamma + 1e6*(1-z[i]) >= -y_train[i] + dot(Beta, vec(X_train[i,:])))


# Median constraint
@addConstraint(m, median, sum{z[i], i=1:N} == round(Int,N/3*2))

# Objective function
setObjective(m, :Min, gamma)
solve(m)
print("\NDone building optimization model\n")

betaVal = getValue(Beta)
println(betaVal)
gammaVal = getValue(gamma)
println(gammaVal)

# Out of sample testing
y_hat_test = X_test*betaVal
RSS_test = sum((y_hat_test - y_test).^2)
R2_test = 1- RSS_test/SST_test

print("\n\n***RESULTS***")
print("\n N:\t", N)
print("\n D:\t", D)
print("\nMIO R2 test\t", R2_test)
print("\n")	
toc()