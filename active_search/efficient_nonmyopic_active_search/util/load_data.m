function [problem, labels, weights, alpha, nearest_neighbors, similarities] = ...
  load_data(data_name, data_dir)

max_k = 500;
if ~exist('data_dir', 'var')
  data_dir = './data';
end
switch data_name

  case {'toy_problem0', 'toy_problem1'}
    % toy 2D grid problem with deterministic(toy_problem0) or
    % probabilistic(toy_problem1) labels
    data_dir = fullfile(data_dir, 'toy_problem');
    data_path = fullfile(data_dir, data_name);
    load(data_path(1:end-1));
    % prior probabilities of the two classes
    alpha               = [0.1 0.9];

    nn_file  = sprintf('%s_nearest_neighbors.mat', data_name(1:end-1));
    filename = fullfile(data_dir, nn_file);

    if exist(filename, 'file')
      load(filename);
    else
      [nearest_neighbors, distances] = ...
        knnsearch(problem.points, problem.points, ...
        'k', max_k + 1);
      save(filename, 'nearest_neighbors', 'distances');
    end
    k = 50;
    nearest_neighbors = nearest_neighbors(:, 2:(k + 1))';
    distances = distances(:, 2:(k + 1))';
    similarities = 1./distances; %exp(-distances.^2/2);

    % precompute sparse weight matrix
    num_points = problem.num_points;
    row_index = kron((1:num_points)', ones(k, 1));
    weights = sparse(row_index, nearest_neighbors(:), similarities(:), ...
      num_points, num_points);

    labels = 2*ones(problem.num_points, 1);
    rand_label = (data_name(end) == '1');
    if rand_label
      labels(labels_random)        = 1;
    else
      labels(labels_deterministic) = 1;
    end

  case 'citeseer_data'
    data_dir = fullfile(data_dir, 'citeseer');
    data_path = fullfile(data_dir, data_name);
    load(data_path);
    alpha               = [0.05 1];
    num_points = size(x, 1);
    problem.points      = (1:num_points)';
    problem.num_points  = num_points;

    filename = fullfile(data_dir, 'citeseer_data_nearest_neigbors.mat');

    if exist(filename, 'file')
      load(filename);
    else
      [nearest_neighbors, distances] = ...
        knnsearch(x, x, ...
        'k', max_k + 1);

      save(filename, 'nearest_neighbors', 'distances');
    end

    %% there are duplicates in the data
    % e.g. nearest_neighbors(160, 1:2) = [18, 160]
    % that means x(160,:) and x(18,:) are identical
    for i = 1:num_points
      if nearest_neighbors(i, 1) ~= i
        dup_idx = find(nearest_neighbors(i, 2:end) == i);
        nearest_neighbors(i, 1+dup_idx) = nearest_neighbors(i, 1);
        nearest_neighbors(i, 1) = i;
      end
    end

    % limit to only top k
    k = 50;
    nearest_neighbors = nearest_neighbors(:, 2:(k + 1))';
    % distances = distances(:, 2:(k + 1))';
    similarities = ones(size(nearest_neighbors));

    % precompute sparse weight matrix
    row_index = kron((1:num_points)', ones(k, 1));
    weights = sparse(row_index, nearest_neighbors(:), 1, ... %1 / distances(:), ...
      num_points, num_points);

    % create label vector
    labels = 2 * ones(size(x, 1), 1);
    labels(connected_labels == 3) = 1;
    problem.num_classes = 2;
  case 'bmg_data'  % data available upon request
    data_path = fullfile(data_dir, data_name);
    load(data_path);
    % remove labels from features
    x = bmg_data(:, 1:(end - 1));

    % create label vector
    labels = 2 * ones(size(x, 1), 1);
    labels(bmg_data(:, end) <= 1) = 1;

    % remove rows with nans
    ind = (~any(isnan(x), 2));
    x      =      x(ind, :);
    labels = labels(ind);

    num_points = size(x, 1);

    train_portion = 0.1;
    rng('default');
    train_ind = crossvalind('holdout', num_points, 1 - train_portion);

    % can be reproduced above
    ind = [1, 33, 39, 45, 46, 53, 111, 135, 141, 165, 185, 200, 201];

    % limit features to those selected
    x = x(~train_ind, ind);
    %     x = x(:, ind);
    num_points = size(x, 1);

    % remove features with no variance
    x = x(:, std(x) ~= 0);

    % normalize data
    x = bsxfun(@minus, x,     mean(x));
    x = bsxfun(@times, x, 1 ./ std(x));

    problem.points      = x;
    problem.num_classes = 2;
    problem.num_points  = num_points;

    filename = fullfile(data_dir, 'bmg_nearest_neighbors.mat');
    %     filename = fullfile(data_dir, 'bmg_nearest_neigbors.mat');

    if exist(filename, 'file')
      load(filename, 'nearest_neighbors', 'distances');
    else
      [nearest_neighbors, distances] = ...
        knnsearch(problem.points, problem.points, ...
        'k', max_k + 1);

      % deal with a small number of ties in dataset
      for i = 1:num_points
        if (nearest_neighbors(i, 1) ~= i)
          ind = find(nearest_neighbors(i, :) == i);
          nearest_neighbors(i, ind) = nearest_neighbors(i, 1);
          nearest_neighbors(i, 1)   = i;
        end
      end

      save(filename, 'nearest_neighbors', 'distances');
    end

    % limit to only top k
    k = 50;
    nearest_neighbors = nearest_neighbors(:, 2:(k + 1))';
    similarities = ones(size(nearest_neighbors));
    % precompute sparse weight matrix
    row_index = kron((1:num_points)', ones(k, 1));
    weights = sparse(row_index, nearest_neighbors(:), 1, ...
      num_points, num_points);

    alpha = [0.05, 1];

  otherwise  % drug discovery data: https://github.com/rmgarnett/active_virtual_screening.git
    alpha = [0.001, 1];
    num_inactive = 100000;
    if contains(data_name, 'ecfp')
      filename = sprintf('target_%s_ecfp4_nearest_neighbors_%d.mat', ...
        data_name(5:end), num_inactive);
      fingerprint = [data_name(1:4) '4'];
    elseif contains(data_name, 'gpidaph')
      filename = sprintf('target_%s_gpidaph3_nearest_neighbors_%d.mat', ...
        data_name(8:end), num_inactive);
      fingerprint = [data_name(1:7) '3'];
    end

    data_dir = fullfile(data_dir, fingerprint);
    data_path = fullfile(data_dir, filename);
    load(data_path);

    num_points = size(nearest_neighbors, 1);
    num_active = num_points - num_inactive;

    problem.points = (1:num_points)';
    problem.num_classes = 2;
    problem.num_points = num_points;

    % create label vector
    labels = ones(num_points, 1);
    labels(1:num_inactive) = 2;


    % limit to k-nearest neighbors
    k = 100;
    nearest_neighbors = nearest_neighbors(:, 1:k)';
    similarities      = similarities(:, 1:k)';

    % precompute sparse weight matrix
    row_index = kron((1:num_points)', ones(k, 1));
    weights = sparse(row_index, nearest_neighbors(:), similarities(:), ...
      num_points, num_points);
end

problem.max_num_influence = max(sum(weights > 0, 1));  % used for pruning
