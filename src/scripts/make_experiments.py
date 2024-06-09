
import subprocess
import sys
import os

os.chdir(os.path.dirname(__file__))

TEMPLATE = \
"""
[model]
model = '{model}'

[model_args]
{model_args}

[dataset]
dataset = '{dataset}'
train_test_split = 0.8
min_no_samples = {min_no_samples}

[FedProx]
n_processes = 10
n_clients = -1
n_clients_per_round_fit = 10
n_clients_per_round_eval = -1
drop_stragglers = {drop_stragglers}
learning_rate = {learning_rate}
batch_size = 10
epochs_per_round = 20
steps_per_epoch = -1
n_rounds = {n_rounds}
mu = 1
stragglers = {stragglers}

[quantizer]
{quantizer}

[plumbing]
checkpoint = ''
checkpoint_interval = 10000
plot_detailed_training = false
evaluation_batch_size = 128
rounds_between_evaluations = {rounds_between_evaluations}
seed = {seed}
"""

UNFEDERATED_TEMPLATE = \
"""
[model]
model = '{model}'

[model_args]
{model_args}

[dataset]
dataset = '{dataset}'
train_test_split = 0.8
min_no_samples = {min_no_samples}

[FedProx]
n_processes = 1
n_clients = "unfederated"
n_clients_per_round_fit = 1
n_clients_per_round_eval = -1
drop_stragglers = false
learning_rate = {learning_rate}
batch_size = 10
epochs_per_round = {epochs_per_round}
steps_per_epoch = -1
n_rounds = {n_rounds}
mu = 0
stragglers = 0

[quantizer]
name = "NoCompression"

[plumbing]
checkpoint = ''
checkpoint_interval = {checkpoint_interval}
plot_detailed_training = true
evaluation_batch_size = 128
seed = 123456
"""

FEDPROX_REPRODUCTION_TEMPLATE = \
"""
[model]
model = 'linear_classifier'

[model_args]
n_classes = 10
n_inputs = 60

[dataset]
dataset = '{dataset}'
train_test_split = 0.8
min_no_samples = 1

[FedProx]
n_processes = 10
n_clients = -1
n_clients_per_round_fit = 10
n_clients_per_round_eval = -1
drop_stragglers = {drop_stragglers}
learning_rate = 0.01
batch_size = 10
epochs_per_round = 20
steps_per_epoch = -1
n_rounds = 200
mu = {mu}
stragglers = {stragglers}

[quantizer]
name = "NoCompression"

[plumbing]
checkpoint = ''
checkpoint_interval = 100
plot_detailed_training = false
evaluation_batch_size = 1024
seed = 123456
plot_training_loss = true
"""

def main():
    min_no_samples_sent140 = 30

    experiment_idx = 100
    # Federated experiments
    models_modelArgs =[{'model': 'linear_classifier',
        'model_args': 'n_classes = 10\nn_inputs = 60',
        'dataset': 'synthetic_FedProx_1_1',
        'rounds_between_evaluations': '1',
        'min_no_samples': 1,
        'learning_rate': 0.01,
        'n_rounds': 500},
        {'model': 'cnn',
        'model_args': 'n_classes = 62\nn_inputs = 784',
        'dataset': 'femnist',
        'rounds_between_evaluations': '5',
        'min_no_samples': 1,
        'learning_rate': 0.003,
        'n_rounds': 500,},
        {'model': 'lstm',
        'model_args': 'n_hidden = 256\nn_classes = 2',
        'dataset': 'sent140',
        'rounds_between_evaluations': '5',
        'min_no_samples': 30,
        'learning_rate': 0.3,
        'n_rounds': 1000},  
        ]
    strategies = [{'quantizer': 'name = "QSGD"\nquantization_levels = 1'},
        {'quantizer': 'name = "QSGD"\nquantization_levels = 8'},
        {'quantizer': 'name = "QSGD"\nquantization_levels = 128'},
        {'quantizer': 'name = "FP8"\nimplementation = "search"'},
        {'quantizer': 'name = "NoCompression"'},
    ]
    system_heterogeneities = [
        {'drop_stragglers': 'false', 'stragglers': '0.9'}, # system heterogeneity
        {'drop_stragglers': 'false', 'stragglers': '0'},   # no system heterogeneity
    ]
    plumbing = {'seed': 123456}
    for model_modelArg, experiment_idx in zip(models_modelArgs, [100, 111, 130]):
        # Federated experiments
        print(f"Generating federated experiments for dataset {model_modelArg['dataset']}")
        for strategy in strategies:
            for system_heterogeneity in system_heterogeneities:
                fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
                print("writing to file", os.path.abspath(fname))
                with open(fname, 'w') as f:
                    config_args = {**model_modelArg,
                                    **strategy,
                                    **system_heterogeneity,
                                    **plumbing}
                    config_str = TEMPLATE.format(**config_args)
                    f.write(config_str)
                experiment_idx += 1

    # LFL Experiments
    for model_modelArg, experiment_idx in zip(models_modelArgs, [200, 210, 220]):
        print(f"Generating LFL experiments for dataset {model_modelArg['dataset']}")
        for strategy in [{'quantizer': 'name = "LFL"'},]:
            for system_heterogeneity in system_heterogeneities:
                fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
                print("writing to file", os.path.abspath(fname))
                with open(fname, 'w') as f:
                    config_args = {**model_modelArg,
                                    **strategy,
                                    **system_heterogeneity,
                                    **plumbing}
                    config_str = TEMPLATE.format(**config_args)
                    f.write(config_str)
                experiment_idx += 1

    # Dynamic quantization Experiments
    for model_modelArg, experiment_idx in zip(models_modelArgs, [400, 410, 420]):
        print(f"Generating dynamic quantization experiments for dataset {model_modelArg['dataset']}")
        block_length = 100 if model_modelArg['dataset'] == 'sent140' else 50
        strategy = {
            'quantizer':f'name = "DynamicQuantizer"\nmoving_average = 0.9\nblock_length = {block_length}\nmax_level=128'}
        system_heterogeneity = system_heterogeneities[0]
        fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
        print("writing to file", os.path.abspath(fname))
        with open(fname, 'w') as f:
            config_args = {**model_modelArg,
                            **strategy,
                            **system_heterogeneity,
                            **plumbing}
            config_str = TEMPLATE.format(**config_args)
            f.write(config_str)
        experiment_idx += 1

    # Downlink Experiments
    models_modelArgs_downlink =[{'model': 'linear_classifier',
        'model_args': 'n_classes = 10\nn_inputs = 60',
        'dataset': 'synthetic_FedProx_1_1',
        'rounds_between_evaluations': '1',
        'min_no_samples': 1,
        'learning_rate': 0.01,
        'n_rounds': 1000},
        {'model': 'cnn',
        'model_args': 'n_classes = 62\nn_inputs = 784',
        'dataset': 'femnist',
        'rounds_between_evaluations': '5',
        'min_no_samples': 1,
        'learning_rate': 0.003,
        'n_rounds': 1000,},
        {'model': 'lstm',
        'model_args': 'n_hidden = 256\nn_classes = 2',
        'dataset': 'sent140',
        'rounds_between_evaluations': '5',
        'min_no_samples': 30,
        'learning_rate': 0.3,
        'n_rounds': 2000},  
    ]
    for model_modelArg, experiment_idx in zip(models_modelArgs_downlink, [500, 510, 520]):
        print(f"Generating dynamic quantization experiments for dataset {model_modelArg['dataset']}")
        strategy = {
            'quantizer': 'name = "NoCompression"\ncompress_downlink = true'}
        system_heterogeneity = system_heterogeneities[0]
        fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
        print("writing to file", os.path.abspath(fname))
        with open(fname, 'w') as f:
            config_args = {**model_modelArg,
                            **strategy,
                            **system_heterogeneity,
                            **plumbing}
            config_str = TEMPLATE.format(**config_args)
            f.write(config_str)
        experiment_idx += 1

    # Unfederated experiments
    args =[{'model': 'linear_classifier',
        'model_args': 'n_classes = 10\nn_inputs = 60',
        'dataset': 'synthetic_FedProx_1_1',
        'epochs_per_round': 1,
        'n_rounds': 200,
        'checkpoint_interval': 10000,
        'min_no_samples': 1,
        'learning_rate': 0.01},
        {'model': 'cnn',
        'model_args': 'n_classes = 62\nn_inputs = 784',
        'dataset': 'femnist',
        'epochs_per_round': 1,
        'n_rounds': 30,
        'checkpoint_interval': 100000,
        'min_no_samples': 1,
        'learning_rate': 0.003
        },
        {'model': 'lstm',
        'model_args': 'n_hidden = 256\nn_classes = 2',
        'dataset': 'sent140',
        'epochs_per_round': 1,
        'n_rounds': 30,
        'checkpoint_interval': 10000,
        'min_no_samples': min_no_samples_sent140,
        'learning_rate': 0.3
        },
        ]
    for arg, experiment_idx in zip(args, [110, 121, 140]):
        print(f"Generating unfederated experiments for dataset {arg['dataset']}")
        fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
        print("writing to file", os.path.abspath(fname))
        with open(fname, 'w') as f:
            config_args = {**arg}
            config_str = UNFEDERATED_TEMPLATE.format(**config_args)
            f.write(config_str)

    # FedProx reproduction experiments
    experiment_idx = 122
    # No system heterogeneity
    datasets =[
        {'dataset': 'synthetic_FedProx_iid'},
        # Use 0_0 instead of 1_1 because it has less variance
        # when training.
        {'dataset': 'synthetic_FedProx_0_0'},
    ]
    FedProxs = [
        # FedProx
        {'mu': '1', 'stragglers': '0', 'drop_stragglers': 'false'},
        # FedAvg
        {'mu': '0', 'stragglers': '0', 'drop_stragglers': 'false'},
    ]
    print("Generating reproduction experiments for data heterogeneity")
    for dataset in datasets:
        for FedProx in FedProxs:
                fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
                print("writing to file", os.path.abspath(fname))
                with open(fname, 'w') as f:
                    config_args = {**dataset,
                                   **FedProx}
                    config_str = FEDPROX_REPRODUCTION_TEMPLATE.format(**config_args)
                    f.write(config_str)
                experiment_idx += 1

    # System heterogeneity
    datasets = [
        {'dataset': 'synthetic_FedProx_1_1'},
    ]
    system_heterogeneities = [
        {'stragglers': '0.9'}, # system heterogeneity
        {'stragglers': '0'},   # no system heterogeneity
    ]
    FedProxs = [
        # FedProx
        {'mu': '1', 'drop_stragglers': 'false'},
        # FedAvg
        {'mu': '0', 'drop_stragglers': 'true'},
    ]
    print("Generating reproduction experiments for system heterogeneityx")
    for dataset in datasets:
        for system_heterogeneity in system_heterogeneities:
            for FedProx in FedProxs:
                    fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
                    print("writing to file", os.path.abspath(fname))
                    with open(fname, 'w') as f:
                        config_args = {**system_heterogeneity,
                                    **dataset,
                                    **FedProx}
                        config_str = FEDPROX_REPRODUCTION_TEMPLATE.format(**config_args)
                        f.write(config_str)
                    experiment_idx += 1

    
    # AdaQuantFL Experiments
    for model_modelArg, experiment_idx in zip(models_modelArgs, [1000, 1010, 1020]):
        print(f"Generating AdaQuantFL experiments for dataset {model_modelArg['dataset']}")
        block_length = 100 if model_modelArg['dataset'] == 'sent140' else 50
        strategy = {
            'quantizer':f'name = "AdaQuantFL"\ns0 = 2'}
        system_heterogeneity = {'drop_stragglers': 'false', 'stragglers': '0.9'}
        fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
        print("writing to file", os.path.abspath(fname))
        with open(fname, 'w') as f:
            config_args = {**model_modelArg,
                            **strategy,
                            **system_heterogeneity,
                            **plumbing}
            config_str = TEMPLATE.format(**config_args)
            f.write(config_str)
        experiment_idx += 1

    
    # GZip Experiments
    for model_modelArg, experiment_idx in zip(models_modelArgs, [1100, 1110, 1120]):
        print(f"Generating GZip experiments for dataset {model_modelArg['dataset']}")
        block_length = 100 if model_modelArg['dataset'] == 'sent140' else 50
        strategy = {
            'quantizer':f'name = "GZip"\nquantization_levels = 8'}
        system_heterogeneity = {'drop_stragglers': 'false', 'stragglers': '0.9'}
        fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
        print("writing to file", os.path.abspath(fname))
        with open(fname, 'w') as f:
            config_args = {**model_modelArg,
                            **strategy,
                            **system_heterogeneity,
                            **plumbing}
            config_str = TEMPLATE.format(**config_args)
            f.write(config_str)
        experiment_idx += 1
    
    # Local vs global quantization with equal quantization budget and static quantization-- experiments
    for model_modelArg, experiment_idx, use_local_quantization in zip(models_modelArgs + models_modelArgs,
        [1200, 1220, 1240, 1210, 1230, 1250], [True, True, True, False, False, False]):
            print(f"Generating local quantization experiments for dataset {model_modelArg['dataset']}")
            block_length = 100 if model_modelArg['dataset'] == 'sent140' else 50
            strategy = {
                'quantizer':f'name = "QSGD"\nquantization_levels = 2\nlocalized_quantization = {str(use_local_quantization).lower()}'}
            system_heterogeneity = {'drop_stragglers': 'false', 'stragglers': '0.9'}
            fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
            print("writing to file", os.path.abspath(fname))
            with open(fname, 'w') as f:
                config_args = {**model_modelArg,
                                **strategy,
                                **system_heterogeneity,
                                **plumbing}
                config_str = TEMPLATE.format(**config_args)
                f.write(config_str)
            experiment_idx += 1
    
    # Local vs global quantization with different quantization budgets and static quantization -- experiments
    for model_modelArg, experiment_idx, use_local_quantization in zip(models_modelArgs + models_modelArgs,
        [1300, 1320, 1340, 1310, 1330, 1350], [True, True, True, False, False, False]):
            print(f"Generating local quantization experiments for dataset {model_modelArg['dataset']}")
            block_length = 100 if model_modelArg['dataset'] == 'sent140' else 50
            strategy = {
                'quantizer':f'name = "QSGD"\nquantization_levels = 2\nlocalized_quantization = {str(use_local_quantization).lower()}'}
            system_heterogeneity = {'drop_stragglers': 'false', 'stragglers': '0.9'}
            fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
            print("writing to file", os.path.abspath(fname))
            with open(fname, 'w') as f:
                config_args = {**model_modelArg,
                                **strategy,
                                **system_heterogeneity,
                                **plumbing}
                config_str = TEMPLATE.format(**config_args)
                f.write(config_str)
            experiment_idx += 1
    
    # Local vs global quantization with different quantization budgets and static quantization-- experiments
    for model_modelArg, experiment_idx, use_local_quantization, budget_divider, max_quantization_level in zip(models_modelArgs + models_modelArgs + models_modelArgs,
        [1300, 1320, 1340, 1310, 1330, 1350, 1360, 1370, 1380], [True, True, True, False, False, False, True, True, True], [2, 2, 2, 1, 1, 1, 2, 2, 2], [8,8,8,8,8,8,16,16,16]):
            print(f"Generating local quantization with different quantization budgets and static quantization experiments for dataset {model_modelArg['dataset']}")
            block_length = 100 if model_modelArg['dataset'] == 'sent140' else 50
            strategy = {
                'quantizer':f'name = "QSGD"\nquantization_levels = 8\n'
                f'localized_quantization = {str(use_local_quantization).lower()}\n'
                f'min_quantization_level = 1\n'
                f'max_quantization_level = {max_quantization_level}\n'
                f'quantization_budget_divider = {budget_divider}'}
            system_heterogeneity = {'drop_stragglers': 'false', 'stragglers': '0.9'}
            fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
            print("writing to file", os.path.abspath(fname))
            with open(fname, 'w') as f:
                config_args = {**model_modelArg,
                                **strategy,
                                **system_heterogeneity,
                                **plumbing}
                config_str = TEMPLATE.format(**config_args)
                f.write(config_str)
            experiment_idx += 1
    
    # Local vs global quantization with different quantization budgets and dynamic quantization (AdaQuantFL)-- experiments
    for model_modelArg, experiment_idx, budget_divider, min_q_level in zip(models_modelArgs + [models_modelArgs[1]]*2 + [models_modelArgs[2]]*4,
        [1400, 1410, 1420, 1430, 1440, 1450, 1460, 1470, 1480], [2, 2, 2, 1.5, 1, 1.5, 1, 1.25, 1.25], [1,1,1,1,1,1,1,1,2]):
            print(f"Generating local quantization with different quantization budgets and dynamic quantization (AdaQuantFL) experiments for dataset {model_modelArg['dataset']}")
            block_length = 100 if model_modelArg['dataset'] == 'sent140' else 50
            strategy = {
                'quantizer':f'name = "AdaQuantFL"\ns0 = 2\n'
                f'localized_quantization = true\n'
                f'min_quantization_level = {min_q_level}\n'
                f'max_quantization_level = 8\n'
                f'quantization_budget_divider = {budget_divider}'}
            system_heterogeneity = {'drop_stragglers': 'false', 'stragglers': '0.9'}
            fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
            print("writing to file", os.path.abspath(fname))
            with open(fname, 'w') as f:
                config_args = {**model_modelArg,
                                **strategy,
                                **system_heterogeneity,
                                **plumbing}
                config_str = TEMPLATE.format(**config_args)
                f.write(config_str)
            experiment_idx += 1
    
    # Dynamic quantization Experiments
    for model_modelArg, experiment_idx in zip(models_modelArgs, [1500, 1510, 1520]):
        print(f"Generating dynamic quantization experiments for dataset {model_modelArg['dataset']}")
        block_length = 100 if model_modelArg['dataset'] == 'sent140' else 50
        strategy = {
            'quantizer':f'name = "DynamicQuantizer"\nmoving_average = 0.9\nblock_length = {block_length}\nmax_level=8'}
        system_heterogeneity = {'drop_stragglers': 'false', 'stragglers': '0.9'}
        fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
        print("writing to file", os.path.abspath(fname))
        with open(fname, 'w') as f:
            config_args = {**model_modelArg,
                            **strategy,
                            **system_heterogeneity,
                            **plumbing}
            config_str = TEMPLATE.format(**config_args)
            f.write(config_str)
        experiment_idx += 1


    for model_modelArg, experiment_idx, budget_divider, min_q_level in zip(models_modelArgs*3,
        [1600, 1610, 1620, 1630, 1640, 1650, 1660, 1670, 1680], [1.5, 1.5, 1.5, 1, 1, 1, 2, 2, 2], [1,1,1,1,1,1,1,1,1]):
            print(f"Generating local quantization with different quantization budgets and dynamic quantization (MonoQuant) experiments for dataset {model_modelArg['dataset']}")
            block_length = 100 if model_modelArg['dataset'] == 'sent140' else 50
            strategy = {
                'quantizer':f'name = "DynamicQuantizer"\nmoving_average = 0.9\nblock_length = {block_length}\nmax_level=8\n'
                f'localized_quantization = true\n'
                f'min_quantization_level = {min_q_level}\n'
                f'max_quantization_level = 16\n'
                f'quantization_budget_divider = {budget_divider}'}
            system_heterogeneity = {'drop_stragglers': 'false', 'stragglers': '0.9'}
            fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
            print("writing to file", os.path.abspath(fname))
            with open(fname, 'w') as f:
                config_args = {**model_modelArg,
                                **strategy,
                                **system_heterogeneity,
                                **plumbing}
                config_str = TEMPLATE.format(**config_args)
                f.write(config_str)
            experiment_idx += 1

    # Experiments to find the minimum "nmax_level" required for Sent140 + MonoQuant
    # to reach the target accuracy.
    for model_modelArg, experiment_idx, max_level in zip([models_modelArgs[2]]*2,
        [1700, 1710], [16, 32]):
            print(f"Generating experiments to find the minimum \"nmax_level\" required for Sent140 +"  
                  f"MonoQuant to reach the target accuracy. for max_level {max_level}")
            block_length = 100 if model_modelArg['dataset'] == 'sent140' else 50
            strategy = {
                'quantizer':f'name = "DynamicQuantizer"\nmoving_average = 0.9\nblock_length = {block_length}\nmax_level={max_level}\n'
                f'localized_quantization = false\n'}
            system_heterogeneity = {'drop_stragglers': 'false', 'stragglers': '0.9'}
            fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
            print("writing to file", os.path.abspath(fname))
            with open(fname, 'w') as f:
                config_args = {**model_modelArg,
                                **strategy,
                                **system_heterogeneity,
                                **plumbing}
                config_str = TEMPLATE.format(**config_args)
                f.write(config_str)
            experiment_idx += 1


    for model_modelArg, experiment_idx, budget_divider, min_q_level in zip(models_modelArgs,
        [1800, 1810, 1820], [2, 1.5, 2], [1,1,1]):
            print(f"Generating MonoQuant + SampleQuant experiments for dataset {model_modelArg['dataset']}")
            block_length = 100 if model_modelArg['dataset'] == 'sent140' else 50
            strategy = {
                'quantizer':f'name = "DynamicQuantizer"\nmoving_average = 0.9\nblock_length = {block_length}\nmax_level=8\n'
                f'localized_quantization = true\n'
                f'min_quantization_level = {min_q_level}\n'
                f'max_quantization_level = 16\n'
                f'quantization_budget_divider = {budget_divider}'}
            system_heterogeneity = {'drop_stragglers': 'false', 'stragglers': '0.9'}
            fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
            print("writing to file", os.path.abspath(fname))
            with open(fname, 'w') as f:
                config_args = {**model_modelArg,
                                **strategy,
                                **system_heterogeneity,
                                **plumbing}
                config_str = TEMPLATE.format(**config_args)
                f.write(config_str)
            experiment_idx += 1


    for model_modelArg, experiment_idx, seed, localized_quantization in zip([models_modelArgs[0]]*10,
        [1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [True, False, True, False, True, False, True, False, True, False]):
            print(f"Generating MonoQuant + SampleQuant experiments for dataset {model_modelArg['dataset']} with different seeds")
            block_length = 100 if model_modelArg['dataset'] == 'sent140' else 50
            strategy = {
                'quantizer':f'name = "DynamicQuantizer"\nmoving_average = 0.9\nblock_length = {block_length}\nmax_level=8\n'
                f'localized_quantization = {"true" if localized_quantization else "false"}\n'
                f'constant_variance = true\n'
            }
            system_heterogeneity = {'drop_stragglers': 'false', 'stragglers': '0.9'}
            plumbing = {'seed': str(123456+seed)}
            fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
            print("writing to file", os.path.abspath(fname))
            with open(fname, 'w') as f:
                config_args = {**model_modelArg,
                                **strategy,
                                **system_heterogeneity,
                                **plumbing}
                config_str = TEMPLATE.format(**config_args)
                f.write(config_str)
            experiment_idx += 1

    for model_modelArg, experiment_idx, seed, localized_quantization in zip([models_modelArgs[2]]*10,
        [1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [True, False, True, False, True, False, True, False, True, False]):
            print(f"Generating MonoQuant + SampleQuant experiments for dataset {model_modelArg['dataset']} with different seeds")
            block_length = 100 if model_modelArg['dataset'] == 'sent140' else 50
            strategy = {
                'quantizer':f'name = "DynamicQuantizer"\nmoving_average = 0.9\nblock_length = {block_length}\nmax_level=8\n'
                f'localized_quantization = {"true" if localized_quantization else "false"}\n'
                f'constant_variance = true\n'
            }
            system_heterogeneity = {'drop_stragglers': 'false', 'stragglers': '0.9'}
            plumbing = {'seed': str(123456+seed)}
            fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
            print("writing to file", os.path.abspath(fname))
            with open(fname, 'w') as f:
                config_args = {**model_modelArg,
                                **strategy,
                                **system_heterogeneity,
                                **plumbing}
                config_str = TEMPLATE.format(**config_args)
                f.write(config_str)
            experiment_idx += 1

    for model_modelArg, experiment_idx, seed, localized_quantization in zip([models_modelArgs[1]]*10,
        [1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [True, False, True, False, True, False, True, False, True, False]):
            print(f"Generating MonoQuant + SampleQuant experiments for dataset {model_modelArg['dataset']} with different seeds")
            block_length = 100 if model_modelArg['dataset'] == 'sent140' else 50
            strategy = {
                'quantizer':f'name = "DynamicQuantizer"\nmoving_average = 0.9\nblock_length = {block_length}\nmax_level=8\n'
                f'localized_quantization = {"true" if localized_quantization else "false"}\n'
                f'constant_variance = true\n'
            }
            system_heterogeneity = {'drop_stragglers': 'false', 'stragglers': '0.9'}
            plumbing = {'seed': str(123456+seed)}
            fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
            print("writing to file", os.path.abspath(fname))
            with open(fname, 'w') as f:
                config_args = {**model_modelArg,
                                **strategy,
                                **system_heterogeneity,
                                **plumbing}
                config_str = TEMPLATE.format(**config_args)
                f.write(config_str)
            experiment_idx += 1

if __name__ == "__main__":
    main()