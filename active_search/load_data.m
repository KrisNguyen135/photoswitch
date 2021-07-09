function [problem, labels, weights, alpha, nearest_neighbors, similarities] = ...
    load_data()

alpha = [0.1, 0.9];
k     = 500;

data_path = './process_molecules/data/morgan2_nearest_neighbors_255794_iter1.mat';
load(data_path);

num_points        = size(similarities, 1);
nearest_neighbors = nearest_neighbors(:, 1:k)';
similarities      = similarities(:, 1:k)';

row_idx = kron((1:num_points)', ones(k, 1));
weights = sparse(row_idx, nearest_neighbors(:), similarities(:), ...
  num_points, num_points);

% load(fullfile(data_dir, 'labeled'));
% labeled_ind = labeled(:, 1);
% raw_labels  = labeled(:, 2);
%
% labels = zeros(num_points, 1);
% labels(labeled_ind) = raw_labels;
load(fullfile('./process_molecules/initial_labeled_data/labels'));
labels(labels ~= 1) = 2;

problem.points            = (1:num_points)';
problem.num_points        = num_points;
problem.num_classes       = 2;
problem.max_num_influence = max(sum(weights > 0, 1));
