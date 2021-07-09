function calculate_nearest_neighbors_include_unlabeled(is_demo, iteration, k)
% demo = 'demo_';  % change this to '' if running on real data
% iteration = 1;
% k         = 100;     % number of nearest neighbors to compute
demo = '';
if is_demo
  demo = 'demo_';
end
rng('default');
parent_dir = ''; %sprintf('%siteration%d', demo, iteration);
data_dir = fullfile(parent_dir, 'data');
if ~isdir(data_dir)
  mkdir(data_dir);
end

% load labeled data
disp('loading labeled data...');
load(fullfile([demo 'initial_labeled_data'], 'labels'));
load(fullfile([demo 'initial_labeled_data'], 'features'));
num_labeled = length(labels);
fprintf('initial #labeled data: %d\n', num_labeled);

%%%%%%% add code to read features of newly labeled data %%%%%%%%%%%%%%%%%%%
for i = 1:(iteration-1)
  data_dir_previous = sprintf('iteration%d/data', i);
  load(fullfile(data_dir_previous, 'new_labels'));
  load(fullfile(data_dir_previous, 'new_features'));
  labels = [labels; new_labels];
  features = [features; ...
    [new_features(:, 1) + num_labeled, new_features(:, 2)]];
  num_labeled = num_labeled + length(new_labels);
  fprintf('#labeled data after iteration %d: %d\n', iteration, num_labeled);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inactive_class = 0;
inactive_ind = find(labels == inactive_class);
active_ind   = find(labels ~= inactive_class);

num_neg = length(inactive_ind);
num_pos = length(active_ind);

% load features of unlabeled data
feature_filepath = fullfile(data_dir, 'features_unlabeled');
disp('loading unlabeled data...');
load(feature_filepath);
num_unlabeled = max(features_unlabeled(:, 1));
fprintf('num unlabeled: %d\n', num_unlabeled);

% note its important to add num_labeled here since the indices of
% features_unlabeled also started from 1
features = [features; ...
  [features_unlabeled(:, 1) + num_labeled, features_unlabeled(:, 2)]];

disp('constructing sparse feature matrix...');
features = sparse(features(:, 2), features(:, 1), 1);

% remove features that are always zero
features = features(any(features, 2), :);

fingerprint = 'morgan2';

nn_filepath = sprintf('%s/%s_nearest_neighbors_%d_iter%d.mat', ...
  data_dir, ...
  fingerprint, ...
  num_unlabeled, ...
  iteration);

tic;

fprintf('computing k nearest neighbors (k=%d) ...\n', k);
[nearest_neighbors, similarities] = jaccard_nn(features, k);
disp('size of nearest_neighbors:');
disp(size(nearest_neighbors));
fprintf('saving nearest_neighbors and similarities to %s ...\n', ...
  nn_filepath);
save(nn_filepath, 'nearest_neighbors', 'similarities');

elapsed = toc;
if (elapsed < 60)
  fprintf('done, took %is.\n', ceil(elapsed));
else
  fprintf('done, took %0.1fm.\n', ceil(elapsed / 6) / 10);
end
