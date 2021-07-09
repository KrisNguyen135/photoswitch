function selector = get_core_selector(offset, cores, iteration)

selector = @(problem, train_ind, observed_labels) ...
           core_selector(problem, train_ind, offset, cores, iteration);
