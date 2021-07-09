clear;

addpath(genpath('./'));
addpath(genpath('../active_learning'));
addpath(genpath('../active_search'));

policy_codes
policies = [BATCH_ENS];
policy = policies(1);

verbose           = 1;
batch_size        = 5;
num_queries       = 4;
total_num_queries = num_queries * batch_size;
num_initial       = 1;
data_name         = 'ecfp1'

% Set up problem
[problem, labels, weights, alpha, nns, sims] = load_data(data_name);

model = get_model(@knn_model, weights, alpha);
model = get_model(@model_memory_wrapper, model);

problem.num_queries = num_queries;
problem.batch_size  = batch_size;
problem.verbose     = verbose;
problem.num_initial = num_initial;
label_oracle        = get_label_oracle(@lookup_oracle, labels);

prob_bound = get_probability_bound_wrapper(policy, weights, nns, sims, alpha);
[query_strategy, selector] = get_policy(policy, problem, model, weights, prob_bound);

train_ind       = [100001];
observed_labels = labels(train_ind);

[chosen_ind, chosen_labels] = active_learning( ...
  problem, train_ind, observed_labels, label_oracle, selector, query_strategy ...
);
