% function chosen_ind = choose_one_batch()
clear;                % 2148, up to date at iteration 39
iteration      = 40;  % change this accordingly
policy         = 1;  % 1 greedy batch, 2 batch-ens, 33 for negative seq simulation
batch_size     = 50;
num_unfinished = 37;  % up to date at iteration 39
num_queries    = (2000 - num_unfinished) / batch_size;

% the chosen indices will be saved in save_dir/save_name
iter_dir  = sprintf('./data/iterations/iteration%d', iteration);
save_dir  = sprintf('%s/recommended_batch', iter_dir);
save_name = sprintf('policy_%g_chosen_ind', policy);
if ~isdir(save_dir)
  mkdir(save_dir);
end

addpath(genpath('./'));
addpath(genpath('../active_learning'));
addpath(genpath('../active_search'));
addpath(genpath('../efficient_nonmyopic_active_search'));

%%%%%%%%%%%%%%%%%% ignore these parameters for now %%%%%
max_num_samples = 16;
resample = 1;
limit = Inf;  % default: don't limit
sort_upper = 0;  % default: don't sort by upper_bound
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

verbose = true;

%% load data
disp('loading data...');
[problem, labels, weights, alpha, nearest_neighbors, similarities] = ...
  load_data();

%% setup problem
problem.num_queries = num_queries;  % note this is the number of batch queries
problem.batch_size  = batch_size;
problem.verbose     = verbose;  % set to true for debugging/verbose output
problem.do_pruning  = 1;

%% setup model and policy
model = get_model(@knn_model, weights, alpha);
model = get_model(@model_memory_wrapper, model);

%% ignore the pruning details for now
if (policy >= 2 && policy < 3) || (policy >= 4 && policy < 5)
  problem.resample   = resample;
  problem.limit      = limit;
  problem.sort_upper = sort_upper;
end
if ismember(policy, [2 4 31 32 33 34]) || ...
    (policy >= 2 && policy < 3) || (policy >= 4 && policy < 5)
  tight_level = 4;
  probability_bound = get_probability_bound_improved(...
    @knn_probability_bound_improved, ...
    tight_level, weights, nearest_neighbors', similarities', alpha);
else
  probability_bound = get_probability_bound(@knn_probability_bound, ...
    weights, full(max(weights)), alpha);
end

train_ind = (1:length(labels))';
problem.num_initial = length(train_ind) + 87

%%% append returned data
for prev_i = 0:(iteration - 1)
    returned_ind = load(...
        sprintf('./data/iterations/iteration%d/returned_ind_iteration%d', ...
        prev_i, prev_i));
    returned_labels = load(...
        sprintf('./data/iterations/iteration%d/returned_labels_iteration%d', ...
        prev_i, prev_i));

    train_ind = [train_ind; returned_ind];
    labels    = [labels; returned_labels];
end

fprintf('final training data of size %d %d\n', numel(train_ind), numel(labels));

% [query_strategy, selector] = get_policy_w_core(policy, problem, model, ...
%   weights, probability_bound, max_num_samples, numel(train_ind), iteration);
%
% % get list of points to consider for querying this round
% test_ind = selector(problem, train_ind, labels);
% % select location(s) of next observation(s) from the given list
% disp('computing the batch...');
% chosen_ind = query_strategy(problem, train_ind, labels, test_ind);
%
% savepath = fullfile(save_dir, save_name);
% fprintf('saving the results to %s...\n', savepath);
% fid = fopen(savepath, 'w');
% for i = 1:length(chosen_ind)
%   fprintf(fid, '%d\n', chosen_ind(i));
% end
% fclose(fid);
% disp('done');
%
% probs = model(problem, train_ind, labels, chosen_ind);
% probs = probs(:, 1)
% sum(probs)

%%% greedy batch to be used at the end
batch_ind = [];

cores         = load('./data/cores.txt');
core_selector = get_core_selector(numel(train_ind), cores, iteration);

for i = 1:batch_size
    test_ind = core_selector(problem, [train_ind; batch_ind], []);

    probs = model(problem, train_ind, labels, test_ind);
    [~, max_ind] = max(probs(:, 1));

    batch_ind = [batch_ind; test_ind(max_ind)];
end
