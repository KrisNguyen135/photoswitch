% SEQUENTIAL_SIMULATION_BATCH sequentially constructs a batch given a
% query strategy and fictional observations
%
% This is a flexible scaffold for querying in a batch. It can take any
% query strategy and use that strategy repeatedly to construct a series
% of points as though those points were being chosen by the normal
% execution of active learning using the given query substrategy.
%
% Because it can't actually observe the label of these points while the
% batch is constructed, it instead creates some sort of ficiton about
% what the results of its choices were. This fiction is perpetrated by
% the fiction_oracle, which can return any sort of observation based
% on the decisions made.
%
% The strategy also allows for pruning in between steps via the
% fiction_selector, which receives all the ficitonal observations,
% believes them to be true, then makes its decision as normal.
%
% Usage:
%
%    query_ind = sequential_simulation_batch(problem, train_ind, ...
%    observed_labels, test_ind, num_points_remaining, query_substrategy,...
%    fiction_oracle, fiction_selector)
%
% Required Inputs:
%
%              problem: a struct describing the problem, containing fields:
%
%                   points: an (n x d) data matrix for the available points
%              num_classes: the number of classes
%
%            train_ind: a list of indices into problem.points indicating
%                       the thus-far observed points
%      observed_labels: a list of labels corresponding to the observations
%                       in train_ind. In recursive calls to this function,
%                       some of these labels will be fictional
%             test_ind: a list of indices into problem.points indicating
%                       the points eligible for observation
%
% num_points_remaining: an integer indicating the size of the batch to
%                       be selected
%    query_substrategy: A strategy to select the next point that should be
%                       chosen for the batch.
%       fiction_oracle: A label oracle that returns a ficitonal observation
%                       on the point(s) chosen by the query_substrategy
%     fiction_selector: A selector that uses the known observations to
%                       return a list of points that are reasonable to
%                       choose from
%
% Optional Inputs:
%
%       every input to sequential_simulation_batch is mandatory
%
% Output:
%
%   query_ind: The batch of points selected for labeling. A list of length
%              num_points_remaining of indices into problem.points
%
% 9/22/2017

function batch_query_ind = sequential_simulation_batch_iter(problem, train_ind, ...
  observed_labels, ~, batch_size, query_substrategy, ...
  fiction_oracle, fiction_selector)
% save a position for possible (already) pre-pruned points where points ...
% not possible to appear in next batch
%
remaining_budget = problem.num_queries * problem.batch_size ...
  - (numel(train_ind) - problem.num_initial);
remaining_budget = round(remaining_budget);
if batch_size > remaining_budget
  batch_size = remaining_budget;
end
batch_query_ind = nan(batch_size, 1);

% precomputed_ind = [172786; 86828; 67941; 154690; 1; 2; 3; 6; 11; 13; 16; ...
%                    25; 38; 47; 55; 57; 63; 92; 133; 136; 180; 224; 377; ...
%                    389; 395; 467; 823; 858; 974; 1520; 2688];% 4153];
% batch_query_ind(1:numel(precomputed_ind)) = precomputed_ind;
% train_ind = [train_ind; batch_query_ind(1:numel(precomputed_ind))];
% observed_labels = [observed_labels; zeros(numel(precomputed_ind), 1)];

% for i = (numel(precomputed_ind) + 1):batch_size
for i = 1:batch_size
  test_ind = fiction_selector(problem, train_ind, observed_labels);

  tt = tic;
  query_ind = query_substrategy(problem, train_ind, observed_labels, ...
    test_ind);
  time = toc(tt);

  if problem.verbose
      fprintf('%d chosen\n', query_ind);
      fprintf('budget including this batch %d, %d / %d selected from %d points in %.2f sec.\n', ...
        remaining_budget, i, batch_size, numel(test_ind), time);
  end

  batch_query_ind(i) = query_ind;

  % if i == batch_size, continue; end

  fictional_label = fiction_oracle(problem, train_ind, observed_labels, ...
    query_ind);

  train_ind = [train_ind; query_ind];
  observed_labels = [observed_labels; fictional_label];

end
