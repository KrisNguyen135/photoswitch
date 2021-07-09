clear;

addpath(genpath('./'));
addpath(genpath('../active_learning'));
addpath(genpath('../active_search'));

% Set up problem
[problem, ~, ~, ~, ~, ~] = load_data('toy_problem0');

problem.num_points = 15;
problem.points     = (1:problem.num_points)';
num_groups         = 5;
group_size         = fix(problem.num_points / num_groups);

weights = kron(eye(num_groups), ones(group_size, group_size));
weights(logical(eye(problem.num_points))) = 0;
weights = sparse(weights);
alpha = [0.33, 0.67];

nns = [];
for p = 1:problem.num_points
  start_ = floor((p - 1) / group_size) * group_size + 1;
  end_   = start_ + group_size - 1;
  nns    = [nns; [start_:(p - 1), (p + 1):end_]];
end
nns = nns';

sims = ones(problem.num_points, group_size - 1);
sims = sims'; 

labels = ones(problem.num_points, 1) * 2;
labels(1:group_size) = 1;
labels((end - fix(group_size / 2)):end) = 1;

save('toy_problem.mat', 'labels', 'weights', 'alpha', 'nns', 'sims')
%{
model = get_model(@knn_model, weights, alpha);
model = get_model(@model_memory_wrapper, model);


policy_codes
policies = [BATCH_ENS];
policy   = policies(1);

verbose           = 1;
batch_size        = 2;
num_queries       = 2;
total_num_queries = num_queries * batch_size;
num_initial       = 1;
num_policies      = length(policies);

problem.num_queries = num_queries;
problem.batch_size  = batch_size;
problem.verbose     = verbose;
problem.num_initial = num_initial;
label_oracle        = get_label_oracle(@lookup_oracle, labels);

prob_bound = get_probability_bound_wrapper(policy, weights, nns', sims', alpha);
[query_strategy, selector] = get_policy(policy, problem, model, weights, prob_bound);

train_ind       = [1];
observed_labels = labels(train_ind);
[chosen_ind, chosen_labels] = active_learning( ...
  problem, train_ind, observed_labels, label_oracle, selector, query_strategy...
);
%}
