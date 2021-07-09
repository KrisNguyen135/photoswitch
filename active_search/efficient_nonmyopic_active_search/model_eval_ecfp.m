rng('default');

addpath(genpath('./'));
addpath(genpath('../active_learning'));
addpath(genpath('../active_search'));

datat_dir       = './data';
data_name       = 'ecfp';
which_setting   = 1;

batch_size          = 1;
num_queries         = 20;
total_num_queries   = num_queries * batch_size;
verbose             = 1;

num_initial         = 1;

train_proportion_L  = 0.1;
train_proportion_H  = 0.01;
subset              = false;
theta               = 0.1;
combine_coefs       = [0.25];
n_coefs             = numel(combine_coefs);
ks                  = [5, 10, 15, 20];
n_ks                = numel(ks);
n_experiments       = 20;
n_files             = 20;
paks                = nan(n_coefs + 2, n_files, n_experiments, n_ks);
aucs_roc            = nan(n_coefs + 2, n_files, n_experiments);
aucs_pr             = nan(n_coefs + 2, n_files, n_experiments);
weight_L            = 0.9;

selector    = get_selector(@unlabeled_selector);

for file = 1:n_files
    disp(file);
    
    %% load data
    [problem, labels_H, weights, alpha, nearest_neighbors, similarities] = ...
        load_data(strcat(data_name, int2str(file)));
    
    weights_L       = weight_L * (weights + 1 / weight_L * speye(problem.num_points));

    n_points        = numel(labels_H);
    train_size_L    = fix(n_points * train_proportion_L);
    train_size_H    = fix(n_points * train_proportion_H);

    %% set up problem
    problem.data_name   = data_name;
    label_oracle        = get_label_oracle(@lookup_oracle, labels_H);

    for exp = 1:n_experiments
        % if draw is 1, the corresponding label flips
        flip_probs      = binornd(1, theta, n_points, 1);
        labels_L        = mod(labels_H - 1 + flip_probs, 2) + 1;

        %% set up model
        label_model = get_model(@knn_model, weights, alpha);
        label_model = get_model(@model_memory_wrapper, label_model);
        agree_model = get_model(@knn_model, weights, [1, theta]);
        agree_model = get_model(@model_memory_wrapper, agree_model);

        train_ind_L = randsample(n_points, train_size_L);
        if subset
            train_ind_H         = randsample(train_ind_L, train_size_H);
            train_ind_intersect = train_ind_H;
        else
            train_ind_H         = randsample(n_points, train_size_H);
            train_ind_intersect = intersect(train_ind_L, train_ind_H);
        end

        observed_labels_L           = labels_L(train_ind_L);
        observed_labels_H           = labels_H(train_ind_H);
        observed_labels_intersect   = (labels_L(train_ind_intersect) == ...
            labels_H(train_ind_intersect));
        % convert 0's to 2's
        observed_labels_intersect   = mod(observed_labels_intersect - 1, 2) + 1;

        test_ind    = selector(problem, train_ind_H, observed_labels_H);
        % probs_L     = label_model(problem, train_ind_L, ...
        %     observed_labels_L, test_ind);
        probs_H     = label_model(problem, train_ind_H, ...
            observed_labels_H, test_ind);
        % probs_agree = agree_model(problem, train_ind_intersect, ...
        %     observed_labels_intersect, test_ind);
        
        for c_id = 1:n_coefs
            probs = mf_agree_knn_model(problem, train_ind_L, ...
                observed_labels_L, train_ind_H, observed_labels_H, ...
                train_ind_intersect, observed_labels_intersect, ...
                test_ind, label_model, agree_model, combine_coefs(c_id));

            for k_id = 1:numel(ks)
                paks(c_id, file, exp, k_id) = get_pak(labels_H(test_ind), ...
                    probs, 1, ks(k_id));
            end
        
            [auc_roc, auc_pr]          = get_auc(labels_H(test_ind), probs, 1);
            aucs_roc(c_id, file, exp)  = auc_roc;
            aucs_pr(c_id, file, exp)   = auc_pr;
        end
        
        % probs = mf_knn_model(problem, train_ind_L, observed_labels_L, ...
        %     train_ind_H, observed_labels_H, test_ind, weights, [0.1, 1], 0.9);
        probs = mf_knn_model_v2(problem, train_ind_L, observed_labels_L, ...
            train_ind_H, observed_labels_H, test_ind, weights, [0.1, 1], weights_L);
        
        for k_id = 1:numel(ks)
            paks(n_coefs + 1, file, exp, k_id) = ...
                get_pak(labels_H(test_ind), probs, 1, ks(k_id));
            
            paks(n_coefs + 2, file, exp, k_id) = ...
                get_pak(labels_H(test_ind), probs_H, 1, ks(k_id));
        end
        
        [auc_roc, auc_pr] = get_auc(labels_H(test_ind), probs, 1);
        aucs_roc(n_coefs + 1, file, exp)    = auc_roc;
        aucs_pr(n_coefs + 1, file, exp)     = auc_pr;
        
        [auc_roc, auc_pr] = get_auc(labels_H(test_ind), probs_H, 1);
        aucs_roc(n_coefs + 2, file, exp)    = auc_roc;
        aucs_pr(n_coefs + 2, file, exp)     = auc_pr;
    end
end

results = nan(n_coefs + 2, n_ks + 2);
for k_id = 1:n_ks
    for c_id = 1:n_coefs + 2
        results(c_id, k_id) = mean(paks(c_id, :, :, k_id), 'all');
    end
end

for c_id = 1:n_coefs + 2
    results(c_id, n_ks + 1) = mean(aucs_roc(c_id, :, :), 'all');
    results(c_id, n_ks + 2) = mean(aucs_pr(c_id, :, :), 'all');
end

disp(results');
