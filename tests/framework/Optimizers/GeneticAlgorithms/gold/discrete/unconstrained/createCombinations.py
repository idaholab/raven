import itertools
import csv
import itertools
import csv
def createCombinations(nVar, lb, ub, strategy='withReplacement'):
    '''
    This function generates all possible combinations of variables with a specified range.
    The combinations are generated with replacement or without replacement.
        args:
        nVar, int, number of variables to be sampled/perturbed
        lb, list of length = nVar, the lower bound of each variable,
        ub, list of length = nVar, the upper bound of each variable,
        strategy, string, either 'withReplacement', or 'withoutReplacement'
        Returns:
        combinations, an array of shape nCombinations x nvVar
    '''

    # If strategy is 'withReplacement', use itertools.product for combinations with replacement
    if strategy == 'withReplacement':
        combinations = list(itertools.product(*[list(range(lb[i], ub[i] + 1)) for i in range(nVar)]))
    # If strategy is 'withoutReplacement', ensure lb <= combination <= ub
    elif strategy == 'withoutReplacement':
        combinations = []
        min_lb = min(lb)
        max_ub = max(ub)
        for combination in itertools.permutations(range(min_lb, max_ub + 1), nVar):
            # sorted_comb = sorted(combination)
            if len(combination)==len(set(combination)):
                combinations.append(combination)
    else:
        raise ValueError("Invalid strategy. Choose 'withReplacement' or 'withoutReplacement'.")

    # Convert each tuple to list and then to int
    combinations = [list(map(int, comb)) for comb in combinations]

    return combinations

# Test the function
nVar = 3
lb = [1, 1, 1]
ub = [6, 6, 6]
strategy = 'withoutReplacement'
combinations = createCombinations(nVar, lb, ub, strategy)
print(combinations)
# Write the 2D array to a.csv file
# Define variable names
varnames = ['x1', 'x2', 'x3']

# Write the 2D array to a.csv file
with open('samples.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write header row
    writer.writerow(varnames)
    # Write all combinations
    writer.writerows(combinations)