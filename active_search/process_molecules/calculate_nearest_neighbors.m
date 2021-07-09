k = 100;     % number of nearest neighbors to compute
rng('default');

precomputed_directory = 'precomputed';
mkdir(precomputed_directory);

% choose subset of data to use
load('data/labels');

inactive_class = 0;

inactive_ind = find(labels == inactive_class);
active_ind   = find(labels ~= inactive_class);
n = length(labels);
num_neg = length(inactive_ind);
num_pos = length(active_ind);

to_keep = 1:n;

labels = labels(to_keep);

load('data/features');

features = sparse(features(:, 2), features(:, 1), 1);

features = features(:, to_keep);
% remove features that are always zero
features = features(any(features, 2), :);

fingerprint = 'morgan2';

filename = sprintf('%s/%s_%i_nearest_neighbors_%i_%i.mat', ...
  precomputed_directory, ...
  fingerprint, ...
  k, ...
  num_neg, ...
  num_pos);

tic;

[nearest_neighbors, similarities] = jaccard_nn(features, k);

save(filename, 'nearest_neighbors', 'similarities');

elapsed = toc;
if (elapsed < 60)
  fprintf('done, took %is.\n', ceil(elapsed));
else
  fprintf('done, took %0.1fm.\n', ceil(elapsed / 6) / 10);
end
