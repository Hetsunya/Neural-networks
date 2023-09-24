import pandas as pd
from src.neural_network import create_neural_network, train_neural_network_1, train_neural_network_2,\
    DataGenerator, split_data, plot_3d_graph, Neuron

DataGenerator.generate_data('data.csv')

data = pd.read_csv('data.csv')

plot_3d_graph(data['x1'], data['x2'], data['Y'], '3D график до добавления шума')

noise_std = 5.0
DataGenerator.add_noise('data.csv', noise_std)
data_with_noise = pd.read_csv('data.csv')

plot_3d_graph(data_with_noise['x1'], data_with_noise['x2'], data_with_noise['Y'],
              '3D график с добавленным шумом', color='r')


single_neuron_net1 = create_neural_network(1, 2)
single_neuron_net2 = create_neural_network(1, 2)

training_set = []
for index, row in data_with_noise.iterrows():
    inputs = row[['x1', 'x2']].tolist()
    target = row['Y']
    training_set.append((inputs, target))

train_data, test_data = split_data(training_set, train_fraction=0.8)

# при рейте меньше 0.0001 nan
train_neural_network_1(single_neuron_net1, train_data, epochs=100, learning_rate=0.0001, target_error=0.0001)
train_neural_network_2(single_neuron_net2, train_data, epochs=100, learning_rate=0.0001, target_error=0.0001)

print(Neuron.calculate_mse_for_network(single_neuron_net1, test_data))
print(Neuron.calculate_mse_for_network(single_neuron_net2, test_data))

print("\nВеса после обучения для сети с одним нейроном:")
neuron_weights, bias = single_neuron_net1.neurons[0].get_weights()
print(f"Веса нейрона: {neuron_weights}")
print(f"Смещение (bias): {bias}")

print("\nВеса после обучения для сети с одним нейроном:")
neuron_weights, bias = single_neuron_net2.neurons[0].get_weights()
print(f"Веса нейрона: {neuron_weights}")
print(f"Смещение (bias): {bias}")
