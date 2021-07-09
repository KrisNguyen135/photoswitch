clear;
k = 500;     % number of nearest neighbors to compute
rng('default');

precomputed_directory = 'precomputed';
mkdir(precomputed_directory);

features = load('data/features_unlabeled');

features = sparse(features(:, 2), features(:, 1), 1);

features = features(any(features, 2), :);

fingerprint = 'morgan2';

filename = sprintf('%s/%s_%i_nearest_neighbors_unlabeled.mat', ...
  precomputed_directory, ...
  fingerprint, ...
  k);

tic;

[nearest_neighbors, similarities] = jaccard_nn(features, k);

save(filename, 'nearest_neighbors', 'similarities');

elapsed = toc;
if (elapsed < 60)
  fprintf('done, took %is.\n', ceil(elapsed));
else
  fprintf('done, took %0.1fm.\n', ceil(elapsed / 6) / 10);
end
