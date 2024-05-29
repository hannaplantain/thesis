
# Simulation code

### Introduction



### Parameters and settings

# MODEL PARAMETERS
N = 10 # length of each genome
K_i = 3 # ruggedness parameter of the NK model for individual fitness
K_g = K_i # ruggedness parameter of the NK model for groups
M = 20 # number of groups
n =  50 # maximum number of individuals per group
mu = 0.1 # when mutation takes place: mutation rate per gene - find good value

I = n*M # total number of individuals (max as starting out with full groups). If changed to less, I must > M as groups cannot be empty
alpha = 1/10 # generation time of groups relative to that of individuals 
t_end = 10 # duration of simulation

# NEUTRALIY
"""
choose "NK" for regular NK, 
"NKp" for probabilistic NK, 
"NKq" for quantised NK
"""
neutrality_i = ["NK"] 
neutrality_g = ["NK"] 
p_i = 0.5 # p of NKp for individual-level fitness
p_g = p_i # p of NKp for group-level fitness
q_i = 4 # q of NKq for individual-level fitness
q_g = q_i # q of NKq for group-level fitness

# NETWORK PROPERTIES
"""
choose "r" for sampling with replacement, 
"nr" for without replacement, 
"block" for blockwise interactions
"""
network_i = ["r"]
network_g = ["r"]

# MUTATION PROBABILITY
"""
choose "yes" for  reproduction with mutation, 
"no" for reproduction without mutation, 
"""
mutation = ["yes"]

# SIMULATION PARAMETERS
my_seed = 10 # random seed
t_end = 10 # end time, in units of individual generation times.

### Packages

#import necessary packages
import numpy as np
import random as rd
import math
from numpy.random import choice 
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from builtins import ValueError
#set style for all plots
plot.style.use("seaborn-v0_8-colorblind")

### Shorthands
B_i = 2**(K_i + 1) # number of hypercube corners for the fitness contributions of each gene
# (= columns in fitness matrix) at the individual level 
B_g = 2**(K_g + 1) # number of hypercube corners for the fitness contributions of each gene
# (= columns in fitness matrix) at the group level

### Global variables
# fitness matrix
fm_i = np.zeros((N, B_i)) # invullen: lege array N x B_j
fm_g = np.zeros((N, B_g)) # invullen: lege arry N x B_g
# epistasis matrices
val = list(range(0, N))

# fitness values
#f_i = # absolute fitness individual level
f_i_comp = np.random.rand(I, N) #randomly generates fitness contributions associated with gene values of all individuals
f_i = np.mean(f_i_comp, axis=1,) #absolute fitness of all individuals (=avg by row of f_i_comp)
f_i = f_i.reshape(-1, 1) 


## Define functions...

### for constructing fitness landscapes

def create_fitness_matrix_i():
    fm_i = np.random.rand(N, B_i)
    match neutrality_i:
        case ["NK"]:
            #
            pass
        case ["NKp"]:
            fm_i = np.where(np.random.rand(*fm_i.shape) < p_i, 0, fm_i) 
        case ["NKq"]:
            fm_i = np.digitize(fm_i, bins=np.linspace(0, 1, q_i+1), right=True) - 1
    return fm_i

def create_fitness_matrix_g():
    fm_g = np.random.rand(N, B_g)
    match neutrality_g:
        case ["NK"]:
            #
            pass
        case ["NKp"]:
            fm_g = np.where(np.random.rand(*fm_g.shape) < p_g, 0, fm_g) 
        case ["NKq"]:
            fm_g = np.digitize(fm_g, bins=np.linspace(0, 1, q_g+1), right=True) - 1
    return fm_g

### for constructing epistasis matrix without repetition
def generate_all_perm_tree(level, nums): 
    """
    makes tree structure with all possible permutations of given 'nums',
    where each number is allowed to move to next position in permutation, excluding repeats
    - level: Current level in the permutation.
    - nums (list): list of numbers to permute.
    returns:
    - dict: nested dictionary representing the permutation tree structure.
        Each key is a number, and corresponding value is the subtree for the next level.
    """
    if len(nums) == 1:
        if level == nums[0]:
            return None
        else:
            return {nums[0]: {}}
    allowed_number = list(nums)
    if level in allowed_number:
        allowed_number.remove(level)
    result = {}
    for number in allowed_number:
        sublevel_number = list(nums)
        if number in sublevel_number:
            sublevel_number.remove(number)
        subtree = generate_all_perm_tree(level + 1, sublevel_number)
        if subtree is not None:
            result[number] = subtree
    if len(result) == 0:
        return None
    return result

def pick_all_moved_perm(all_moved_perm_tree, picked=None):#picks permutation of numbers from previously generated tree, with each number selected only once
    """
    Picks permutation of numbers from previously generated tree, with each number selected only once
    - all_moved_perm_tree (dictionary): The permutation tree generated by generate_all_perm_tree
    - picked: set of numbers already picked.
    Return:
    - list: representing a permutation of numbers
    """
    if picked is None:
        picked = set()
    allowed_num_set = set(all_moved_perm_tree.keys()) - picked
    if not allowed_num_set:
        return []
    number = rd.choice(list(allowed_num_set))
    picked.add(number)
    l = [number]
    sub_tree = all_moved_perm_tree[number]
    if len(sub_tree) > 0:
        l.extend(pick_all_moved_perm(sub_tree, picked))
    return l

def generate_unique_r(tree, num_rows): 
    """
    Generates an array of unique pairs of numbers, with no number repeated in a row
    - tree (dict):the permutation tree generated by generate_all_moved_perm_tree.
    - num_rows: the number of rows to generate.
    Returns:
    - 2d array representing unique pairs of numbers in each row
    """
    result = []
    for _ in range(num_rows):
        row = list(zip(pick_all_moved_perm(tree), pick_all_moved_perm(tree)))
        while any(x[0] == x[1] for x in row):
            row = list(zip(pick_all_moved_perm(tree), pick_all_moved_perm(tree)))
        result.extend(row)
    return np.array(result[:num_rows])

### for choosing preferred epistasis
def create_epistasis_matrix_i(): #epistasis matrix individual level
    em_i = []
    match network_i:
        case ["r"]:
            for row in range(N):
                val = list(range(0, N))
                gene_pair = rd.sample(val[:row] + val[row + 1:], K_i)  
                em_i.append([row] + gene_pair)
            em_i = np.array(em_i)  #first with own gene referenced
            em_i = em_i[:, 1:] #without own gene referenced #remove this row or above
        case ["nr"]:
            tree = generate_all_perm_tree(1, range(1, N+1))
            em_i = generate_unique_r(tree, N)
        #case ["block"]:
           # em = #working on it :)
    return em_i

def create_epistasis_matrix_g():
    em_g = []
    match network_g:
        case ["r"]:
            val = list(range(N))
            for row in range(N):
                gene_pair = rd.sample(val[:row] + val[row + 1:], K_g)  # Select K_g unique genes 
                em_g.append([row] + gene_pair)  # Row number added to the beginning
            em_g = np.array(em_g)  # First with own gene referenced
            em_g = em_g[:, 1:]
        case ["nr"]:
            tree = generate_all_perm_tree(1, range(1, N+1))
            em_g = generate_unique_r(tree, N)
        # case ["block"]:
        #    em = # working on it :)
    return em_g

### for constructing fitness landscapes
def create_fitness_matrix_i():
    fm_i = np.random.rand(N, B_i)
    match neutrality_i:
        case ["NK"]:
            #
            pass
        case ["NKp"]:
            fm_i = np.where(np.random.rand(*fm_i.shape) < p_i, 0, fm_i) 
        case ["NKq"]:
            fm_i = np.digitize(fm_i, bins=np.linspace(0, 1, q_i+1), right=True) - 1
    return fm_i

def create_fitness_matrix_g():
    fm_g = np.random.rand(N, B_g)
    match neutrality_g:
        case ["NK"]:
            #
            pass
        case ["NKp"]:
            fm_g = np.where(np.random.rand(*fm_g.shape) < p_g, 0, fm_g) 
        case ["NKq"]:
            fm_g = np.digitize(fm_g, bins=np.linspace(0, 1, q_g+1), right=True) - 1
    return fm_g

### for calculating coefficients
def calc_a(fitness_matrix): #other functions split by ind/group but not this one. Do or not? think
    """
    calculate and return coefficients based on fitness matrix 
    - B: B_g or B_i: number of binary gene combinations = corners hypercube
    - fm (array): fm_g or fm_i - fitness matrix individuals or groups
    Returns:
    - List containing 'a' coefficients for each row of the input matrix
    """
    a_coefficients = []
    for r in fitness_matrix:
        a = [0.0] * B_g  # initialises list with zeros as floats for each row & X cols
        a[0] = r[0] #set ai0 to Fi0 
        for j in range(1, B_g): 
            sum = 0.0 
            # Calculate next coefs based only on previously calculated coefs
            for l in range(0, j): 
                #check if l equal to bitwise AND of l and j 
                #(ex: 001&101->001 TRUE; 001&100->000 FALSE)
                if l == (l & j): 
                    sum += a[l] 
            a[j] = r[j] - sum 
        a_coefficients.append(a) # append new a's into result array
    return a_coefficients

### for calculating fitness
def gene_fitness(coefficients, epistasis, genome, gene):
    """
    Calculate the fitness component of a specific gene in a genome
    - coefficients (array): coefficient matrix as calculated by calc_a
    - epistasis ("): Epistasis matrix representing interactions between genes
    - genome ("): genome values
    - gene (int): index of specific gene for which fitness component is calculated
    Returns:
    - fitness component of the specified gene in the genome (float)
    """
    result = 0
    for j in range(coefficients.shape[1]):
        contribution = coefficients[gene, j] * (genome[gene] ** (1 & j))
        for k in range(epistasis.shape[1]):
            epi_index = (epistasis[gene, k])
            epi_value = genome[int(epi_index)]
            product_term = epi_value ** ((2**(k+1) & j) / 2**(k+1))
            contribution *= product_term

        result += contribution
    return result

def genome_fitness(coefficients, epistasis, genome):
    """
    Calculate fitness components for all genes within a genome 
    return:
    array containing fitness components for each gene in the genome
    """
    fit_vals = np.zeros(len(genome))
    for gene in range(len(genome)):
        fit_vals[gene] = gene_fitness(coefficients, epistasis, genome, gene)
    return fit_vals

def calculate_fitness(coefficients, epistasis, genomes):
    """
    Calculate the fitness components for all genes in all genomes
    Return:
    3D array containing fitness components for each gene (cols) in each genome (rows)
    """
    if len(genomes.shape) == 2:  #add 3rd dimension for compatibitlyt if 2D array(one group)
        genomes = np.expand_dims(genomes, axis=0)
    fit_val = np.zeros(genomes.shape)
    for group in range(genomes.shape[0]):
        for individual in range(genomes.shape[1]):
            fit_val[group, individual, :] = genome_fitness(coefficients, epistasis, genomes[group, individual, :])
    avg_fit = np.mean(fit_val, axis=2)
    return avg_fit

# CONSTRUCT FITNESS LANDSCAPES 
fm_i = create_fitness_matrix_i()
fm_g = create_fitness_matrix_g()
# Construct epistasis matrices
em_i = create_epistasis_matrix_i()
em_g = create_epistasis_matrix_g()

#coefficients
coef_i = np.array(calc_a(fm_i))
coef_g = np.array(calc_a(fm_g))

#print(coef_g)
fm_i_avg = np.mean(fm_i, axis=0)
#print(fm_i)
#print(fm_i_avg)
population = np.zeros(shape=(M,n,N))

# Add print statements to check array shapes and indices
#print("Coefficient shape:", coef_i.shape)
#print("Epistasis shape:", em_i.shape)
#print("Population shape:", population.shape)
#print(population)
genomes_example = np.random.randint(0, 2, size = (M, n, N))
avg_genomes_example = np.mean(genomes_example, axis = 1)
#print(avg_genomes_example)
#avg_ind_fit = calculate_binary_fitness(genomes_example, fm_i)
#print("individual fitnesses:", avg_ind_fit)
fitness = calculate_fitness(coef_g, em_g, avg_genomes_example)
#print(fitness)
# Call calculate_fitness function
#abs_ind_fitness = calculate_fitness(coef_i, em_i, population)
#print(abs_ind_fitness)

#fitness = np.random.rand(3, 10, 4)
#print(fitness)

#### for updating fitnesses & rates
#Updating fitnesses
#individual
def update_ind_fitness_indrep(event_drawn, population_slice, abs_ind_fitness, coef_i, em_i, offspring_ind_genome, rd_row_index):
    """
    Updates the rates of individual reproduction events after individual reproduction
    -abs_ind_fitness: array containing all current individual absolute fitnesses 
    -coefficients: array with coefficients used for continuous NK based fitness calculation
    -epistasis: array with epistatic interactions used for continuous NK based fitness calculation
    -offspring_ind_genome: genome of only the new individual
    -group_index: index of group slected individual is in
    -ind_index: index of selected individual within group
    -rd_row_index: index of eliminated individual
    Returns: 
    -abs_ind_fitness = array containing the updated absolute fitnesses (changed and unchanged)
    -changed_abs_ind_fitness = array with fitnesses in group in which reproduction happened
    """
    ind_index = (event_drawn- M) % n #within group 
    group_index = ((event_drawn-M)//n) 
    parent_ind_genome = population_slice[ind_index]
    parent_ind_genome = np.array(parent_ind_genome)    
    abs_ind_fitness = np.reshape(abs_ind_fitness, (M, n))   
    population_slice_fit = abs_ind_fitness[group_index, :] 

    if (offspring_ind_genome == parent_ind_genome).all:
        offspring_fitness = population_slice_fit[ind_index] #because offspring fitness = fitness selected individual
        population_slice_fit[rd_row_index] = offspring_fitness
    elif offspring_ind_genome is not parent_ind_genome: #because mutated
        offspring_fitness = genome_fitness(coef_i, em_i, offspring_ind_genome) #mutated so need to recalculate
        population_slice_fit[rd_row_index] = offspring_fitness
        
    unchanged_ind_fit = np.delete(abs_ind_fitness, group_index, axis=0) #fitness of inds in all other groups stays the same
    changed_abs_ind_fitness = np.reshape(population_slice_fit, (1, n))
    abs_ind_fitness = np.append(unchanged_ind_fit, changed_abs_ind_fitness, axis=0)
    #np.concatenate((unchanged_ind_fit, population_slice_fit), axis=0) #fix so correctly insert new fitness (in right spot slices?)
    
    return abs_ind_fitness

def update_ind_fitness_grsplit(assign_mask, abs_ind_fitness, group_index, terminated_group_index):
    """
    Updates the rates of individual reproduction events after a group splitting event 
    -assign_mask: mask created in group splitting function to distribute fitnesses same way as genomes
    -abs_ind_fitness: array containing all current individual absolute fitnesses 
    -group_index: index of group selected for splitting
    Returns:
    -abs_ind_fitness: array containing all current individual absolute fitnesses (changed and unchanged)
    -changed_abs_ind_fitness: array with fitnesses in group in which reproduction happened
    """

    abs_ind_fitness = abs_ind_fitness.reshape((M,n)) #dubbelcheck later
    old_parent_group_fitness = abs_ind_fitness[group_index, :] 

    #parent_group_fitness = old_parent_group_fitness[assign_mask == 0] 
    parent_group_fitness = np.where(assign_mask == 1, np.nan, old_parent_group_fitness)  
    offspring_group_fitness = np.where(assign_mask == 0, np.nan, old_parent_group_fitness) 
    abs_ind_fitness = np.delete(abs_ind_fitness, group_index, axis=0)  #old parent group veriwjderen

    parent_group_fitness_reshape = parent_group_fitness[np.newaxis, :]  #(1, 5)
    offspring_group_fitness_reshape = offspring_group_fitness[np.newaxis, :] 

    abs_ind_fitness = np.concatenate((abs_ind_fitness, parent_group_fitness_reshape, offspring_group_fitness_reshape), axis=0)

    #hier terminated_group_index gebruiken om eliminated group te verwijderen
    abs_ind_fitness = np.delete(abs_ind_fitness, terminated_group_index, axis=0)
    changed_abs_ind_fitness = np.concatenate((parent_group_fitness, offspring_group_fitness), axis=0) #del?
    return abs_ind_fitness

#group 
def update_gr_fitness_indrep(abs_gr_fitness, population_slice, event_drawn, coef_g, em_g):
    """
    -population_slice: 2D array with updated genomes of group reproduction happened in
    -group index: index of group slected individual is in
    -coefficients: array with coefficients used for continuous NK based fitness calculation
    -epistasis: array with epistatic interactions used for continuous NK based fitness calculation
    Returns:
    abs_gr_fitness: 2d array with absolute group fitnesses of all groups 
    """
    group_index = ((event_drawn-M)//n) #note both ind_index and group_index start at 0
    changed_group_genome = np.nanmean(population_slice, axis=0) #new group genome = avg of change group slice
   # print("changed group genome", changed_group_genome)
    changed_abs_gr_fitvals = genome_fitness(coef_g, em_g, changed_group_genome) #calculate abs fitness new group genome
    changed_abs_gr_fitness = np.mean(changed_abs_gr_fitvals, axis=0)
    abs_gr_fitness = abs_gr_fitness.reshape(-1, 1)
    abs_gr_fitness[group_index] = changed_abs_gr_fitness
    #unchanged_abs_gr_fitness = np.delete(abs_gr_fitness, group_index, axis=0)    
    #print("unchanged_abs_gr_fitness", unchanged_abs_gr_fitness)
   # print("unchanged_abs_gr_fitness shape", unchanged_abs_gr_fitness.shape)

   # abs_gr_fitness = np.concatenate((changed_abs_gr_fitness, unchanged_abs_gr_fitness), axis = 0) #recombine new with old

    return abs_gr_fitness

def update_gr_fitness_grsplit(abs_gr_fitness, changed_group, changed_group_index, coef_g, em_g): 
    """
    -changed_group: 2d/3d array containing changed groups
    -changed_group_index: index of changed group or groups, depending on if parent/offspring group gets eliminated or not
    -coefficients: array with coefficients used for continuous NK based fitness calculation
    -epistasis: array with epistatic interactions used for continuous NK based fitness calculation
    Returns: 
    -abs_gr_fitness: 3d array with absolute group fitnesses of all groups 
    -changed_abs_gr_fitness: 2d/3d array with absolute group fitness of the changed groups
    """
    if changed_group.ndim > 2:
        new_group_genomes = np.nanmean(changed_group, axis=1)
        changed_abs_gr_fitness_vals = calculate_fitness(coef_g, em_g, new_group_genomes)
        changed_abs_gr_fitness = np.mean(changed_abs_gr_fitness_vals, axis=0)       
        changed_abs_gr_fitness = changed_abs_gr_fitness[np.newaxis, :] 
        changed_abs_gr_fitness = changed_abs_gr_fitness.reshape(-1,1)      
    else:
        new_group_genomes = np.nanmean(changed_group, axis=0)
        changed_abs_gr_fitness_vals = genome_fitness(coef_g, em_g, new_group_genomes)
        changed_abs_gr_fitness = np.mean(changed_abs_gr_fitness_vals, axis=0)
        changed_abs_gr_fitness = np.array([changed_abs_gr_fitness])     
        changed_abs_gr_fitness = changed_abs_gr_fitness[np.newaxis, :]  
        changed_abs_gr_fitness = changed_abs_gr_fitness.reshape(-1,1)     

    if isinstance(changed_group_index, tuple):
        #Unpack tuple
        index_1, index_2 = changed_group_index
        # Delete both indices from abs_gr_fitness
        unchanged_abs_gr_fitness = np.delete(abs_gr_fitness, (index_1, index_2), axis=0)
        abs_gr_fitness = np.concatenate((changed_abs_gr_fitness, unchanged_abs_gr_fitness), axis= 0)     
    else:
        # Delete the single index from abs_gr_fitness
        unchanged_abs_gr_fitness = np.delete(abs_gr_fitness, changed_group_index-1, axis=0)
        abs_gr_fitness = np.concatenate((changed_abs_gr_fitness, unchanged_abs_gr_fitness), axis= 0)     
    #unchanged_abs_gr_fitness = np.delete(abs_gr_fitness, changed_group_index-1, axis=0) #check if index for changed groups works same next line
    #abs_gr_fitness = np.concatenate((changed_abs_gr_fitness, unchanged_abs_gr_fitness), axis= 0) 
    return abs_gr_fitness 

#Updating rates
def update_rates_indrep(event_drawn, population_slice, abs_ind_fitness, abs_gr_fitness, coef_i, em_i, offspring_ind_genome, rd_row_index, coef_g, em_g):
    abs_ind_fitness = update_ind_fitness_indrep(event_drawn, population_slice, abs_ind_fitness, coef_i, em_i, offspring_ind_genome, rd_row_index)
    abs_ind_fitness = abs_ind_fitness.reshape((M, n))
    changed_f_j = np.nanmean(abs_ind_fitness, axis=1) #average group fitness  = average of ind fitnesses check if taking changed or all
    w_j = abs_ind_fitness / changed_f_j[:, np.newaxis]    
    ind_rates = []
    ind_rates = w_j.flatten()
    ind_rates = w_j.reshape(-1, 1)    
    abs_gr_fitness = update_gr_fitness_indrep(abs_gr_fitness, population_slice, event_drawn, coef_g, em_g)
    F = np.mean(abs_gr_fitness)
    W = abs_gr_fitness / F
    gr_rates = alpha * W
    rates = np.vstack((gr_rates, ind_rates))
    rates = np.nan_to_num(rates, nan=0)
    return rates, abs_ind_fitness, abs_gr_fitness

def update_rates_grsplit(abs_ind_fitness, abs_gr_fitness, event_drawn, changed_group, changed_group_index, assign_mask, terminated_group_index, coef_g, em_g): 
    #group_index = ((event_drawn-M)//n) #note both ind_index and group_index start at 0
    group_index = event_drawn // (N // M)
    abs_ind_fitness = update_ind_fitness_grsplit(assign_mask, abs_ind_fitness, group_index, terminated_group_index)
    changed_f_j = np.nanmean(abs_ind_fitness, axis=1)
    changed_f_j = changed_f_j[:, np.newaxis]
    w_j = abs_ind_fitness / changed_f_j #check 
    ind_rates = []
    ind_rates = w_j.flatten()
    #ind_rates = ind_rates[~np.isnan(ind_rates)]
    ind_rates = ind_rates.reshape(-1, 1)         

    abs_gr_fitness = update_gr_fitness_grsplit(abs_gr_fitness, changed_group, changed_group_index, coef_g, em_g)
    #give groups with one genome fitness 0 (they cannot split)
    nan_counts_per_row = np.sum(np.isnan(abs_ind_fitness), axis=1)
    rows_to_zero = np.where(nan_counts_per_row == abs_ind_fitness.shape[1] - 1)[0]
    abs_gr_fitness[rows_to_zero] = 0

    F = np.mean(abs_gr_fitness)
    W = abs_gr_fitness / F
    gr_rates = alpha * W
    gr_rates = gr_rates.reshape(-1, 1)
    rates = np.vstack((gr_rates, ind_rates))
    rates = np.nan_to_num(rates, nan=0)
    return rates, abs_ind_fitness, abs_gr_fitness

#combine individual and group rates to get new rates array
#def update_rates(assign_mask, rates, abs_ind_fitness, changed_abs_ind_fitness, event_drawn, group_index, coefficients, epistasis, offspring_ind_genome, parent_ind_genome, rd_row_index, ind_index, population_slice, changed_group, changed_group_index): 
    """
    Combines individual and group rates into one array and adds ind/group event, and indexing by event type
    Return:
    2D array (total number of events, 3): event rates, individual/group 0/1, index of ind or group
    """
    
    ind_rates, abs_ind_fitness, changed_abs_ind_fitness = update_ind_rates(assign_mask, ind_rates, abs_ind_fitness, changed_abs_ind_fitness, event_drawn, group_index, coefficients, epistasis, 
              offspring_ind_genome, parent_ind_genome, rd_row_index, ind_index)
    individual = np.zeros_like(ind_rates) #individuals indicated by 0
    ind_index = np.arange(ind_rates.shape[0]).reshape(-1, 1) #index of which individual added
    ind_rates = np.column_stack((ind_rates, individual, ind_index)) 

    group_rates = update_gr_rates(event_drawn, population_slice, group_index, changed_group, changed_group_index, coefficients, epistasis)
    group = np.ones_like(group_rates) #groups indicated by 1
    group_index = np.arange(group_rates.shape[0]).reshape(-1, 1) #index of which group added
    group_rates = np.column_stack((group_rates, group, group_index)) 

    rates = np.vstack((group_rates, ind_rates))

    return rates, ind_index, group_index

### For executing reactions
def ind_reproduction(event_drawn, population, rates):
    """
    -event_drawn: index of selected event (0-total number of events)
    -population: current population state
    Return
    -population_slice: 2D array with updated genomes of group reproduction happened in
    """ 
    #get parent genome
    ind_index = (event_drawn-M) % n #within group 
    print("ind index", ind_index)
    group_index = ((event_drawn-M)//n) #note both ind_index and group_index start at 0
    population_slice = population[group_index,:, :] 
    parent_ind_genome = population_slice[ind_index]
    parent_ind_genome = np.array(parent_ind_genome) 
    rd_row_index = None  
    
    #get offspring genome and add to slice, remove individual if group is full
    offspring_ind_genome = parent_ind_genome 
    offspring_ind_genome = np.array(offspring_ind_genome)  
    match mutation: #no mutation so genome offspring = genome parent
        case ["no"]:
            if np.isnan(population_slice).any(axis=1).any(): #nan rows? group not full- so ind added
                nan_row_indices = np.where(np.isnan(population_slice).any(axis=1))[0]
                rd_nan_row_index = np.random.choice(nan_row_indices)
                population_slice[rd_nan_row_index] = offspring_ind_genome #random nan row replaced by offspring
            elif not np.isnan(population_slice).any(): #no nan rows? group full
                rd_row_index = np.random.randint(0, population_slice.shape[0])  # randomly select ind for replacement                        
                population_slice[rd_row_index] = offspring_ind_genome
            else:
                raise ValueError("group size exceeds max")
        case ["yes"]:
            mutate = rd.random() < mu
            offspring_ind_genome = [1 - x if mutate else x for x in offspring_ind_genome]
            if population_slice.shape[0] < n: #group not full- so ind added
                population_slice = np.append(population_slice, [offspring_ind_genome], axis=0)
            elif population_slice.shape[0] == n:
                    rd_row_index = np.random.randint(0, population_slice.shape[0])  #randomly select ind for elimination
                    population_slice[rd_row_index] = offspring_ind_genome
            else:
                raise ValueError("group size exceeds max")
    #unchanged_population = np.delete(population, group_index, axis=2)   

    return population_slice, offspring_ind_genome, rd_row_index 


def group_splitting(event_drawn, event_index, population, rates): 
    """
    -event_drawn: index of selected event 
    -population: current population state
    Return
    -population_slice (updated number of individuals (only in group), with genomes of length N))
    """ 
    selected_group_index = event_drawn // (n // M)
    parent_group = population[selected_group_index, :, :] #from population take correct slice (=group) based on group_index selected event   

    #zeros (parent group) or ones (offspring group) randomly assigned to each row of mask array to split
    assign_mask = np.random.randint(2, size=parent_group.shape[0])  
    parent_group_new = parent_group[assign_mask == 0]
    offspring_group = parent_group[assign_mask == 1]

    #make sure neither group is empty
    if parent_group_new.shape[0] == 0:     #parent group is empty, randomly select one genome from offspring group for transfer
        while True:
            random_row_index = np.random.randint(0, offspring_group.shape[0])
            selected_genome = offspring_group[random_row_index, np.newaxis, :]
            if not np.isnan(selected_genome).any(): 
                break
                #repeated until non-empty row is selected for transfer to empty parent group
        offspring_group = np.delete(offspring_group, random_row_index, axis=0)
        parent_group_new = selected_genome 
    elif offspring_group.shape[0] == 0:     # offspring group empty, randomly select one genome from parent group for transfer
        while True:
            random_row_index = np.random.randint(0, parent_group.shape[0])
            selected_genome = parent_group[random_row_index, np.newaxis, :]        
            if not np.isnan(selected_genome).any(): 
                break
        parent_group_new = np.delete(parent_group, random_row_index, axis=0)
        offspring_group = selected_genome
    else:
        pass
    
    #add na rows so shape is n,N 
    rows_to_add_p = n - parent_group_new.shape[0] 
    rows_to_add_o = n - offspring_group.shape[0]
    if rows_to_add_p > 0:
        nan_rows_1 = np.full((rows_to_add_p,  N), np.nan)
        parent_group_new = np.concatenate((parent_group_new, nan_rows_1), axis=0)
    if rows_to_add_o > 0:
        nan_rows_2 = np.full((rows_to_add_o, N), np.nan)
        offspring_group = np.concatenate((offspring_group, nan_rows_2), axis=0)
    
    population[selected_group_index] = parent_group_new[np.newaxis, ...]     #replace old parent group with new parent group
    population_newadded = np.concatenate((population, offspring_group[np.newaxis, ...]), axis=0)     #add offspring group to population (now 1 more group than maximum (so 4 instead of 3 example))
    terminated_group_index = np.random.randint(0, population_newadded.shape[0])     #randomly select a group for termination & remove from population
    population = np.delete(population_newadded, terminated_group_index, axis=0) 
    
    #from updated population, correctly extract those groups whose contents changed
    if terminated_group_index == population_newadded.shape[0]-1 or terminated_group_index == population_newadded.shape[0] - 2:
        #then terminated group is parent or offspring group (because last added) and only 1 changed group remains
        if terminated_group_index == population_newadded.shape[0]-1: #(offspring group terminated)
            changed_group = population_newadded[-2, :, :] 
            changed_group_index = population_newadded.shape[0] - 2
        else: #terminated_group_index == population.shape[0] - 2 (parent group terminated)
            changed_group = population_newadded[-1, :, :]
            changed_group_index = population_newadded.shape[0] - 1
    else:
        #parent and offspring groups both included in the changed groups, not terminated
        changed_group = population[-2:, :, :]
        changed_group_index = (population.shape[0] - 2, population.shape[0] - 1)  #index of both second last and last slices
    
    return changed_group, changed_group_index, assign_mask, terminated_group_index

def choose_event(rates): 
    """
    -rates: array (total number of events): event rates
    Return:
    integer: index of selected event
    """
    event_index = np.arange(rates.shape[0])
    total_rate = np.sum(rates)
    probs = rates.flatten() / total_rate #normalized to sum to 1
    event_drawn = choice(event_index, 1, p=probs) #randomly selects event base on individual probabilities of each event
    event_drawn = event_drawn[0] 
    return event_drawn, event_index

    #event_drawn = np.random.choice(event_index, size=1, p=probs)

def execute_reaction(event_drawn, event_index, population):
    if event_drawn < M:  
        return group_splitting(event_drawn, event_index, population)
    elif M < event_drawn < (M + M*n): 
        return ind_reproduction(event_drawn, population, ind_rates)
    else:
        raise ValueError("invalid event indexed")

### for initialisation
def initial_rates(init_abs_ind_fitness, init_abs_gr_fitness):
    init_rates = []
    f_j = np.mean(init_abs_ind_fitness, axis=1) 
    w_j = init_abs_ind_fitness / f_j[:, None]    
    ind_rates = w_j.flatten()
    ind_rates = w_j.reshape(-1, 1)  
    F = np.mean(init_abs_gr_fitness)
    W = init_abs_gr_fitness / F
    gr_rates = alpha * W
    init_rates = np.vstack((gr_rates, ind_rates))
    return init_rates

t=0
# CONSTRUCT FITNESS LANDSCAPES 
fm_i = create_fitness_matrix_i()
fm_g = create_fitness_matrix_g()
# Construct epistasis matrices
em_i = create_epistasis_matrix_i()
em_g = create_epistasis_matrix_g()

print(em_i)
print(em_g)
#coefficients
coef_i = np.array(calc_a(fm_i))
coef_g = np.array(calc_a(fm_g))

print(coef_g)
#initial population
population = np.zeros(shape=(M,n,N))
gr_genomes = np.mean(population, axis=1)
#print(population)
abs_ind_fitness = calculate_fitness(coef_i, em_i, population)
abs_ind_fitness = abs_ind_fitness.reshape(-1,1)
abs_gr_fitness = calculate_fitness(coef_g, em_g, gr_genomes) 
abs_gr_fitness = abs_gr_fitness.reshape(-1,1)
rates = initial_rates(abs_ind_fitness, abs_gr_fitness)

### for running simulation
while t < t_end :
    # Choose the next event based on the current rates
    event_drawn, event_index = choose_event(rates)
    print("event_drawn", event_drawn)
    if event_drawn < M:  
        changed_group, changed_group_index, assign_mask, terminated_group_index = group_splitting(event_drawn, event_index, population, rates)
        rates, abs_ind_fitness, abs_gr_fitness = update_rates_grsplit(abs_ind_fitness, abs_gr_fitness, event_drawn, changed_group, changed_group_index, assign_mask, terminated_group_index, coef_g, em_g)
    elif M-1 < event_drawn < (M + M*n): 
        population_slice, offspring_ind_genome, rd_row_index = ind_reproduction(event_drawn, population, rates)
        rates, abs_ind_fitness, abs_gr_fitness = update_rates_indrep(event_drawn, population_slice, abs_ind_fitness, abs_gr_fitness, coef_i, em_i, offspring_ind_genome, rd_row_index, coef_g, em_g)
    else:
        raise ValueError("invalid event indexed")
    propensity = sum(rates)
    print("propensity", propensity)
    print(population)
    tau = np.random.exponential(scale=1 / propensity)
    # Update time
    t = t + tau
    print("time", t, "tau", tau)
    if t >= t_end:
        t = t_end
        print("End time reached:", t)
        print("End time:", t_end)


### Profiling
#turn file into py file

import json
import os

files = ["simul_code.ipynb"]
created_py_files = []  # List to store the paths of created Python files

for file in files:
    code = json.load(open(file))
    py_file_path = f"{file}.py"  # Construct the path for the Python file
    
    with open(py_file_path, "w+") as py_file:
        for cell in code['cells']:
            if cell['cell_type'] == 'code':
                for line in cell['source']:
                    py_file.write(line)
                py_file.write("\n")
            elif cell['cell_type'] == 'markdown':
                py_file.write("\n")
                for line in cell['source']:
                    if line and line[0] == "#":
                        py_file.write(line)
                py_file.write("\n")
    
    # Add the path of the created Python file to the list
    created_py_files.append(os.path.abspath(py_file_path))

# Print the paths of the created Python files
print("Python files created:")
for py_file in created_py_files:
    print(py_file)
#cprofile
import cProfile
import pstats

cProfile.run('ind_reproduction(event_drawn, population, rates)')
event_drawn = 2
cProfile.run('group_splitting(event_drawn, event_index, population, rates)')

def initial_rates(init_abs_ind_fitness, init_abs_gr_fitness):
    init_rates = []
    f_j = np.mean(init_abs_ind_fitness, axis=1) 
    w_j = init_abs_ind_fitness / f_j[:, None]    
    ind_rates = w_j.flatten()
    ind_rates = w_j.reshape(-1, 1)  
    F = np.mean(init_abs_gr_fitness)
    W = init_abs_gr_fitness / F
    gr_rates = alpha * W
    init_rates = np.vstack((gr_rates, ind_rates))
    return init_rates

def loop():
    # CONSTRUCT FITNESS LANDSCAPES 
    fm_i = create_fitness_matrix_i()
    fm_g = create_fitness_matrix_g()
    # Construct epistasis matrices
    em_i = create_epistasis_matrix_i()
    em_g = create_epistasis_matrix_g()

    print(em_i)
    print(em_g)
    #coefficients
    coef_i = np.array(calc_a(fm_i))
    coef_g = np.array(calc_a(fm_g))

    print(coef_g)
    #initial population
    population = np.zeros(shape=(M,n,N))
    gr_genomes = np.mean(population, axis=1)
    #print(population)
    abs_ind_fitness = calculate_fitness(coef_i, em_i, population)
    abs_ind_fitness = abs_ind_fitness.reshape(-1,1)
    abs_gr_fitness = calculate_fitness(coef_g, em_g, gr_genomes) 
    abs_gr_fitness = abs_gr_fitness.reshape(-1,1)
    rates = initial_rates(abs_ind_fitness, abs_gr_fitness)
    t=0
    while t < t_end :
    # Choose the next event based on the current rates
        event_drawn, event_index = choose_event(rates)
        print("event drawn", event_drawn)
        if event_drawn < M:  
            changed_group, changed_group_index, assign_mask, terminated_group_index = group_splitting(event_drawn, event_index, population, rates)
            rates, abs_ind_fitness, abs_gr_fitness = update_rates_grsplit(abs_ind_fitness, abs_gr_fitness, event_drawn, changed_group, changed_group_index, assign_mask, terminated_group_index, coef_g, em_g)
        elif M-1 < event_drawn < (M + M*n): 
            population_slice, offspring_ind_genome, rd_row_index = ind_reproduction(event_drawn, population, rates)
            rates, abs_ind_fitness, abs_gr_fitness = update_rates_indrep(event_drawn, population_slice, abs_ind_fitness, abs_gr_fitness, coef_i, em_i, offspring_ind_genome, rd_row_index, coef_g, em_g)
        else:
            raise ValueError("invalid event indexed")
        propensity = sum(rates)
        print("propensity", propensity)
        tau = np.random.exponential(scale=1 / propensity)
        # Update time
        t = t + tau
        print("time", t, "tau", tau)
        if t >= t_end:
            t = t_end
            print("End time reached:", t)
            print("End time:", t_end)

def initial_rates(init_abs_ind_fitness, init_abs_gr_fitness):
    init_rates = []
    f_j = np.mean(init_abs_ind_fitness, axis=1) 
    w_j = init_abs_ind_fitness / f_j[:, None]    
    ind_rates = w_j.flatten()
    ind_rates = w_j.reshape(-1, 1)  
    F = np.mean(init_abs_gr_fitness)
    W = init_abs_gr_fitness / F
    gr_rates = alpha * W
    init_rates = np.vstack((gr_rates, ind_rates))
    return init_rates

# CONSTRUCT FITNESS LANDSCAPES 
fm_i = create_fitness_matrix_i()
fm_g = create_fitness_matrix_g()
# Construct epistasis matrices
em_i = create_epistasis_matrix_i()
em_g = create_epistasis_matrix_g()

print(em_i)
print(em_g)
#coefficients
coef_i = np.array(calc_a(fm_i))
coef_g = np.array(calc_a(fm_g))

print(coef_g)
#initial population
population = np.zeros(shape=(M,n,N))
gr_genomes = np.mean(population, axis=1)
#print(population)
abs_ind_fitness = calculate_fitness(coef_i, em_i, population)
abs_ind_fitness = abs_ind_fitness.reshape(-1,1)
abs_gr_fitness = calculate_fitness(coef_g, em_g, gr_genomes) 
abs_gr_fitness = abs_gr_fitness.reshape(-1,1)
rates = initial_rates(abs_ind_fitness, abs_gr_fitness)
t=0


cProfile.run('loop()', 'profile_stats')

#load the profile stats
stats = pstats.Stats('profile_stats')
#stats by cumulative time
stats.sort_stats('cumulative')

# Strip the directories from filenames
stats.strip_dirs()

# Save the stripped stats to a file
stats.dump_stats('profile_stats.stripped')

#print 10 most expensive funcs
stats.sort_stats('cumulative').print_stats(20)