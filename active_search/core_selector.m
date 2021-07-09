function test_ind = core_selector(problem, train_ind, offset, cores, iteration)

% train_member   = ismember(all_ind_w_core, train_ind((offset + 1):end));
% selected_cores = core_ids(train_member);
%
% test_ind    = unlabeled_selector(problem, train_ind, []);
% test_member = ismember(all_ind_w_core, test_ind);
% test_ind    = all_ind_w_core(test_member);
% test_cores  = core_ids(test_member);
%
% core_member = ismember(test_cores, selected_cores);
% test_ind    = test_ind(~core_member);
%
% if isempty(test_ind), test_ind = unlabeled_selector(problem, train_ind, []); end

% test_ind = [];
%
% while isempty(test_ind)
%     train_member   = ismember(all_ind_w_core, train_ind((offset + 1):end));
%     selected_cores = core_ids(train_member);
%     numel(selected_cores)
%
%     test_ind    = unlabeled_selector(problem, train_ind, []);
%     test_member = ismember(all_ind_w_core, test_ind);
%     test_ind    = all_ind_w_core(test_member);
%     test_cores  = core_ids(test_member);
%
%     core_member = ismember(test_cores, selected_cores);
%     test_ind    = test_ind(~core_member);
%
%     offset = offset + 32;
% end

%% rotating requirement
% returned_ind = [];
% for prev_i = 1:(iteration - 1)
%     returned_ind = [returned_ind; load(sprintf( ...
%         './data/iterations/iteration%d/recommended_batch/policy_33_chosen_ind', ...
%         prev_i))];
% end
%
% test_ind = [];
% while isempty(test_ind)
%     selected_cores = unique(cores(train_ind((offset + 1):end)));
%
%     test_ind     = unlabeled_selector(problem, union(train_ind, returned_ind), []);
%     test_cores   = cores(test_ind);
%     invalid_core = ismember(test_cores, selected_cores);
%     test_ind     = test_ind(~invalid_core);
%
%     offset = offset + numel(unique(cores));
% end

%% leave out some cores
remove_cores = [6; 9; 11; 12; 13; 17; 18; 19; 20; 21; 22; 23; 26; 28];

returned_ind = [];
for prev_i = 1:(iteration - 1)
    returned_ind = [returned_ind; load(sprintf( ...
        './data/iterations/iteration%d/recommended_batch/policy_33_chosen_ind', ...
        prev_i))];
end

test_ind = [];
while isempty(test_ind)
    selected_cores = union(unique(cores(train_ind((offset + 1):end))), remove_cores);

    test_ind     = unlabeled_selector(problem, union(train_ind, returned_ind), []);
    test_cores   = cores(test_ind);
    invalid_core = ismember(test_cores, selected_cores);
    test_ind     = test_ind(~invalid_core);

    offset = offset + numel(unique(cores)) - numel(unique(remove_cores));
end
