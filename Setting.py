input_dimension = 10
task = 'cifar10'
csv_list = ['complexity.dist_spec_init', 'complexity.fro_dist', 'complexity.fro_over_spec', 'complexity.inverse_margin',
            'complexity.log_prod_of_fro', 'complexity.log_prod_of_fro_over_margin', 'complexity.log_prod_of_spec',
            'complexity.log_prod_of_spec_over_margin', 'complexity.log_spec_init_main', 'complexity.log_spec_orig_main',
            'complexity.log_sum_of_fro', 'complexity.log_sum_of_fro_over_margin', 'complexity.log_sum_of_spec',
            'complexity.log_sum_of_spec_over_margin', 'complexity.pacbayes_flatness', 'complexity.pacbayes_init',
            'complexity.pacbayes_mag_flatness', 'complexity.pacbayes_mag_init', 'complexity.pacbayes_mag_orig',
            'complexity.pacbayes_orig', 'complexity.param_norm', 'complexity.params', 'complexity.path_norm',
            'complexity.path_norm_over_margin', 'hp.model_depth', 'hp.model_width', 'gen.gap']

train_list = ['complexity.fro_dist', 'complexity.log_sum_of_fro', 'complexity.log_sum_of_spec', 'complexity.param_norm',
              'complexity.pacbayes_flatness', 'complexity.pacbayes_init',
              'complexity.pacbayes_mag_flatness', 'complexity.pacbayes_orig', 'complexity.inverse_margin',
              'complexity.log_sum_of_spec_over_margin']

model = 'nn'
