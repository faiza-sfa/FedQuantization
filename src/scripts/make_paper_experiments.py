
import subprocess
import sys
import os

os.chdir(os.path.dirname(__file__))

TEMPLATE = \
"""[model]
model = '{model}'

[model_args]
{model_args}

[dataset]
dataset = '{dataset}'
train_test_split = 0.8
min_no_samples = {min_no_samples}

[FedProx]
n_processes = {n_processes}
n_clients = -1
n_clients_per_round_fit = {n_clients_per_round_fit}
n_clients_per_round_eval = -1
drop_stragglers = false
learning_rate = {learning_rate}
batch_size = 10
epochs_per_round = {epochs_per_round}
steps_per_epoch = -1
n_rounds = {n_rounds}
mu = {mu}
stragglers = 0.9

[quantizer]
{quantizer}

[plumbing]
checkpoint = ''
checkpoint_interval = 10000
plot_detailed_training = false
evaluation_batch_size = 128
rounds_between_evaluations = {rounds_between_evaluations}
seed = {seed}
is_embedded = {is_embedded}
"""

def main():
    models_modelArgs =[
        {'model': 'linear_classifier',
        'model_args': 'n_classes = 10\nn_inputs = 60',
        'dataset': 'synthetic_FedProx_1_1',
        'rounds_between_evaluations': '1',
        'min_no_samples': 1,
        'learning_rate': 0.01,
        'epochs_per_round': 20,
        'mu': 1,
        'n_rounds': 500,
        'is_embedded': 'false',
        'n_processes': 10,
        'n_clients_per_round_fit': 10},
        {'model': 'cnn',
        'model_args': 'n_classes = 62\nn_inputs = 784',
        'dataset': 'femnist',
        'rounds_between_evaluations': '5',
        'min_no_samples': 1,
        'learning_rate': 0.003,
        'epochs_per_round': 20,
        'mu': 1,
        'n_rounds': 500,
        'is_embedded': 'false',
        'n_processes': 10,
        'n_clients_per_round_fit': 10},
        {'model': 'lstm',
        'model_args': 'n_hidden = 256\nn_classes = 2',
        'dataset': 'sent140',
        'rounds_between_evaluations': '5',
        'min_no_samples': 10,
        'learning_rate': 0.3,
        'epochs_per_round': 20,
        'mu': 1,
        'n_rounds': 1000,
        'is_embedded': 'false',
        'n_processes': 10,
        'n_clients_per_round_fit': 10},  
        {'model': 'lstm_shakespeare',
        'model_args': 'n_hidden = 100\nn_classes = 80',
        'dataset': 'shakespeare',
        'rounds_between_evaluations': '2',
        'min_no_samples': 1,
        'learning_rate': 0.8,
        'epochs_per_round': 20,
        'mu': 0.001,
        'n_rounds': 50,
        'is_embedded': 'true',
        'n_processes': 10,
        'n_clients_per_round_fit': 10},  
        {'model': 'cnn_celeba',
        'model_args': 'n_classes = 2',
        'dataset': 'celeba',
        'rounds_between_evaluations': '5',
        'min_no_samples': 1,
        'learning_rate': 0.1,
        'epochs_per_round': 20,
        'mu': 0,
        'n_rounds': 500,
        'is_embedded': 'false',
        'n_processes': 10,
        'n_clients_per_round_fit': 10}, 
        {'model': 'linear_classifier',
        'model_args': 'n_classes = 10\nn_inputs = 784',
        'dataset': 'mnist',
        'rounds_between_evaluations': '1',
        'min_no_samples': 1,
        'learning_rate': 0.03,
        'epochs_per_round': 20,
        'mu': 1,
        'n_rounds': 100,
        'is_embedded': 'false',
        'n_processes': 10,
        'n_clients_per_round_fit': 10}, 
    ]
    quantizers = [
        {'quantizer': 'name = "NoCompression"'},
        {'quantizer': 'name = "QSGD"\nquantization_levels = 1\nlocalized_quantization = false'},
        {'quantizer': 'name = "QSGD"\nquantization_levels = 2\nlocalized_quantization = false'},
        {'quantizer': 'name = "QSGD"\nquantization_levels = 4\nlocalized_quantization = false'},
        {'quantizer': 'name = "QSGD"\nquantization_levels = 8\nlocalized_quantization = false'},
        {'quantizer': 'name = "QSGD"\nquantization_levels = 16\nlocalized_quantization = false'},
        {'quantizer': 'name = "QSGD"\nquantization_levels = 32\nlocalized_quantization = false'},
        {'quantizer': 'name = "QSGD"\nquantization_levels = 64\nlocalized_quantization = false'},
        {'quantizer': 'name = "QSGD"\nquantization_levels = 128\nlocalized_quantization = false'},
        {'quantizer': 'name = "QSGD"\nquantization_levels = 256\nlocalized_quantization = false'},
    ]
    plumbing = {'seed': 123455}

    # Quantization level grid search.

    experiment_idx = 5000
    for model_modelArg in models_modelArgs:
        _quantizers = quantizers
        if model_modelArg['dataset'] == 'shakespeare':
             _quantizers = quantizers + [{'quantizer': 'name = "QSGD"\nquantization_levels = 512\nlocalized_quantization = false'}]
        for quantizer in _quantizers:
            print(f"Generating quantization-search experiment {experiment_idx} for dataset {model_modelArg['dataset']}.")
            fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
            print("writing to file", os.path.abspath(fname))
            with open(fname, 'w') as f:
                config_args = {**model_modelArg,
                                **quantizer,
                                **plumbing}
                config_str = TEMPLATE.format(**config_args)
                f.write(config_str)
            experiment_idx += 1

    q_levels = {'synthetic_FedProx_1_1': 16, 'femnist': 4, 'sent140': 128, 'shakespeare': 512, 'celeba': 16, 'mnist': 16}

    # AdaQuantFL experiments.

    models_modelArgs_Adaquantfl =[
        {'model': 'linear_classifier',
        'model_args': 'n_classes = 10\nn_inputs = 60',
        'rounds_between_evaluations': '1',
        'min_no_samples': 1,
        'learning_rate': 0.01,
        'epochs_per_round': 20,
        'mu': 1,
        'n_rounds': 500,
        'is_embedded': 'false',
        'n_processes': 25,
        'n_clients_per_round_fit': 100,
        'quantizer': 'name = "AdaQuantFL"\ns0 = 2',
        'seed': '123456'
        }, 
        {'model': 'linear_classifier',
        'model_args': 'n_classes = 10\nn_inputs = 60',
        'rounds_between_evaluations': '1',
        'min_no_samples': 1,
        'learning_rate': 0.01,
        'epochs_per_round': 20,
        'mu': 1,
        'n_rounds': 500,
        'is_embedded': 'false',
        'n_processes': 10,
        'n_clients_per_round_fit': 10,
        'seed': '123456',
        'quantizer': '''name = "DynamicQuantizer"
moving_average = 0.9
block_length = 50
max_level=16
min_level=1
localized_quantization = true
constant_variance = true'''
        }, 
    ]
    n_clientss = ['10', '100', '200', '400']

    experiment_idx = 6000
    for m_idx, model_modelArg in enumerate(models_modelArgs_Adaquantfl):
        for p_idx, n_clients in enumerate(n_clientss):
            experiment_idx = 6000 + int(str(p_idx)+str(m_idx))
            dataset = f'synthetic_1_1_{n_clients}'
            print(f"Generating experiment {experiment_idx} for dataset {dataset}.")
            fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
            print("writing to file", os.path.abspath(fname))

            with open(fname, 'w') as f:
                config_args = {**model_modelArg,
                               'dataset': dataset,
                               'n_clients_per_round_fit': 10 if m_idx==1 else n_clients}
                config_str = TEMPLATE.format(**config_args)
                f.write(config_str)
            experiment_idx += 1

    # Baseline experiments.

    plumbings = [{'seed': 123456}, {'seed': 123457}, {'seed': 123458}]
    quantizers = [
        'name = "NoCompression"',
        'name = "QSGD"\nquantization_levels = {q_level}\nlocalized_quantization = false',
        'name = "UVeQFed"\ns_fDesRate = 4\nlocalized_quantization = false',
        'name = "GZip"\nquantization_levels = {q_level}\nlocalized_quantization = false',
        'name = "FP8"\nimplementation = "search"\nlocalized_quantization = false',
    ]

    for p_idx, plumbing in enumerate(plumbings):
        for m_idx, model_modelArg in enumerate(models_modelArgs):
            for q_idx, quantizer in enumerate(quantizers):
                experiment_idx = 7000 + int(str(p_idx)+str(m_idx)+str(q_idx))
                print(f"Generating experiment {experiment_idx} for dataset {model_modelArg['dataset']}.")
                fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
                print("writing to file", os.path.abspath(fname))
                with open(fname, 'w') as f:
                    config_args = {**model_modelArg,
                                    'quantizer': quantizer.format(q_level=q_levels[model_modelArg['dataset']]),
                                    **plumbing}
                    config_str = TEMPLATE.format(**config_args)
                    f.write(config_str)
                experiment_idx += 1

    # Additional UVeQFed baseline (gridsearch)
    plumbings = [{'seed': 123456}]
    quantizers = [
        # 'name = "UVeQFed"\ns_fDesRate = 2\nlocalized_quantization = false',
        'name = "UVeQFed"\ns_fDesRate = 1\nlocalized_quantization = false',
    ]

    for p_idx, plumbing in enumerate(plumbings):
        for m_idx, model_modelArg in enumerate(models_modelArgs):
            for q_idx, quantizer in enumerate(quantizers):
                experiment_idx = 9000 + int(str(p_idx)+str(m_idx)+str(q_idx))
                print(f"Generating experiment {experiment_idx} for dataset {model_modelArg['dataset']}.")
                fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
                print("writing to file", os.path.abspath(fname))
                with open(fname, 'w') as f:
                    config_args = {**model_modelArg,
                                    'quantizer': quantizer.format(q_level=q_levels[model_modelArg['dataset']]),
                                    **plumbing}
                    config_str = TEMPLATE.format(**config_args)
                    f.write(config_str)
                experiment_idx += 1 

    quantizers = [
# Uncompressed
"""
name = "NoCompression"
localized_quantization = false
""",  
# Static quantization
"""
name = "QSGD"
quantization_levels = {q_level}
localized_quantization = false
""",  
# MonoQuant
"""
name = "DynamicQuantizer"
moving_average = 0.9
block_length = {block_length}
max_level={max_level}
min_level={min_level}
localized_quantization = false
""",
# SampleQuant
"""
name = "QSGD"
quantization_levels = {q_level}
localized_quantization = true
constant_variance = true
""", 
# SampleQuant + MonoQuant
"""
name = "DynamicQuantizer"
moving_average = 0.9
block_length = {block_length}
max_level={max_level}
min_level={min_level}
localized_quantization = true
constant_variance = true
""",
    ]
    q_levels = {'synthetic_FedProx_1_1': 16, 'femnist': 4, 'sent140': 128, 'shakespeare': 512, 'celeba': 16, 'mnist': 16}
    block_lengths = {'synthetic_FedProx_1_1': 50, 'femnist': 50, 'sent140': 100, 'shakespeare': 5, 'celeba': 50, 'mnist': 10}
    max_levels = {'synthetic_FedProx_1_1': 16, 'femnist': 4, 'sent140': 128, 'shakespeare': 512, 'celeba': 16, 'mnist': 16}
    min_levels = {'synthetic_FedProx_1_1': 1, 'femnist': 1, 'sent140': 64, 'shakespeare': 256, 'celeba': 8, 'mnist': 2}
    plumbings = [{'seed': 123456}, {'seed': 123457}, {'seed': 123458}]

    for p_idx, plumbing in enumerate(plumbings):
        for m_idx, model_modelArg in enumerate(models_modelArgs):
            for q_idx, quantizer in enumerate(quantizers):
                experiment_idx = 8000 + int(str(p_idx)+str(m_idx)+str(q_idx))
                print(f"Generating experiment {experiment_idx} for dataset {model_modelArg['dataset']}.")
                fname = os.path.join('..', '..', 'experiments',  f'{experiment_idx}.toml')
                print("writing to file", os.path.abspath(fname))
                with open(fname, 'w') as f:
                    if q_idx in {2, 4}:  # SampleQuant
                        import copy
                        model_modelArg_cpy = copy.deepcopy(model_modelArg)
                        model_modelArg_cpy['n_rounds'] = int(model_modelArg_cpy['n_rounds']*1.5)
                        config_args = {**model_modelArg_cpy,
                                        'quantizer': quantizer.format(
                                            q_level=q_levels[model_modelArg['dataset']],
                                            block_length=block_lengths[model_modelArg['dataset']],
                                            max_level=max_levels[model_modelArg['dataset']],
                                            min_level=min_levels[model_modelArg['dataset']],
                                        ),
                                        **plumbing}
                    else:
                        config_args = {**model_modelArg,
                                        'quantizer': quantizer.format(
                                            q_level=q_levels[model_modelArg['dataset']],
                                            block_length=block_lengths[model_modelArg['dataset']],
                                            max_level=max_levels[model_modelArg['dataset']],
                                            min_level=min_levels[model_modelArg['dataset']],
                                        ),
                                        **plumbing}
                    config_str = TEMPLATE.format(**config_args)
                    f.write(config_str)
                experiment_idx += 1

if __name__ == "__main__":
    main()