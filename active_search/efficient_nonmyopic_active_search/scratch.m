clear;

addpath(genpath('./'));
addpath(genpath('../active_learning'));
addpath(genpath('../active_search'));

data_name = 'gpidaph1'

policy_codes
policies = [GREEDY, BATCH_ENS];

verbose           = 0;
batch_size        = 3;
num_queries       = 20;
total_num_queries = num_queries * batch_size;
num_initial       = 1;
num_experiments   = 5;
num_policies      = length(policies);

[problem, labels, weights, alpha, nns, sims] = load_data(data_name);

problem.num_queries = num_queries;
problem.batch_size  = batch_size;
problem.verbose     = verbose;
problem.num_initial = num_initial;
problem.data_name   = data_name;
label_oracle        = get_label_oracle(@lookup_oracle, labels);

model = get_model(@knn_model, weights, alpha);
model = get_model(@model_memory_wrapper, model);

active_search = @active_learning;
num_targets   = nan(total_num_queries, num_experiments, num_policies);

for pp = 1:num_policies
  policy = policies(pp);

  prob_bound = get_probability_bound_wrapper(policy, weights, nns, sims, alpha);
  [query_strategy, selector] = get_policy( ...
    policy, problem, model, weights, prob_bound);

  pos_ind = find(labels == 1);

  for exp = 1:num_experiments
    rng(exp);
    fprintf('\nRunning policy %g experiment %d...', policy, exp);

    tic;
    train_ind       = randsample(pos_ind, num_initial);
    observed_labels = labels(train_ind);

    [chosen_ind, chosen_labels] = active_search( ...
      problem, train_ind, observed_labels, label_oracle, selector, ...
      query_strategy ...
    );

    num_targets(:, exp, pp) = cumsum(chosen_labels == 1);

    fprintf(' Took %.2f seconds.\n', toc);
  end
end

disp(num_targets(end, :, :));
disp(squeeze(mean(num_targets(end, :, :), 2)));
