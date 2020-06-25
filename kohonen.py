# imports:
import numpy as np
import matplotlib.pyplot as plt


# initialization:
# number of neurons
number_neuron1 = 30
number_neuron2 = 25
# time: increase with every new data
number_iteration = 1000
time_learning_rate = 200
time_constant = int(0.1 * number_iteration)



# W is an array represent the neurons by their weights
# for circle (30)
def initialize():
    W = np.random.normal(0.5, 1, size=(number_neuron1,2))
    return W

# for 5X5 topology (25)
def initialize_mat():
    W = np.random.normal(0.5, 1, size=(number_neuron2,2))
    return W

# finding the winning neuron
def winning_neuron(W, data_point):
    dis = np.sqrt((W[:, 0] - data_point[0]) ** 2 + (W[:, 1] - data_point[1]) ** 2)
    winning_neuron = np.argmin(dis)
    return winning_neuron

def euclidean_dist(W, point):
    dis = np.sqrt((point[0] - W[:,0]) ** 2 + (point[1] - W[:,1]) ** 2)
    return dis

# Updating weights: delta weights = current learning rate * actual distance from data point
def update_location(W, data_point, winning_idx, radius, alpha):
    for node_idx in range(W.shape[0]):
        if abs(winning_idx-node_idx) < radius:
            W[node_idx] += alpha * (data_point - W[node_idx])
    return W

# 5X5 topology
def choose_uniform():
  return np.random.uniform(low=0, high=5.0, size=2)

# circle topology
def generate_circle_point(start_range, end_range):
  ang = np.random.uniform(low = 0.5, high = 3*np.pi, size = 1)
  radius = np.random.uniform(low = start_range, high = end_range, size =1)
  y = radius * np.sin(ang)
  x = radius * np.cos(ang)
  point = np.concatenate([x,y])
  return point

# the kohonen algorithm
def kohonen(question):

    data_container = np.zeros((number_iteration, 2))
    if question == 4:
        W = initialize()
    else:
        W = initialize_mat()

    for time in range(number_iteration):


        if time < time_learning_rate:
            # the learning rate indicates how much we want to adjust our weights.
            new_learning_rate = np.exp(-float(time) / time_constant)
            current_learning_rate = new_learning_rate
            new_radius = float(4) * np.exp(-float(time) / time_constant)
            current_radius = new_radius
        else:
            current_learning_rate = 0.01
            current_radius = 4.0
            # setting up the learning rate and the radius

        # Case C: data chosen randomly and uniformly (5X5 topology)
        if question == 1 :
           data = choose_uniform()
        # Case E: data set is chosen uniformly but in the band between two concentric circles
        if question == 2 :
            data = generate_circle_point(2.0, 4.0)

        data_container[time] = data
        closest_neuron_index = winning_neuron(W, data)
        W = update_location(W, data, closest_neuron_index, current_radius, current_learning_rate)

        # the graph:
        if time % 100 == 0 or time == number_iteration-1:
            if time == 0:
                fig, ax = plt.subplots()
            else:
                plt.clf()
                fig = plt.gcf()
                ax = fig.gca()
            ax.set(xlabel='WEIGHT X',
                    ylabel='WEIGHT Y',
                    title='Kohonen algorithm (Iterations: {}, Radius: {:.3f}, Learning_rate:{:.3f})'
                    .format(time+1, 4, current_learning_rate))
            if question == 4:
                inner_circle = plt.Circle((0, 0), 2, ec="red", fill=False)
                outer_circle = plt.Circle((0, 0), 4, ec="red", fill=False)
                plt.gca().add_patch(outer_circle)
                plt.gca().add_patch(inner_circle)
            for i in range(data_container.shape[0]):
                plt.scatter(data_container[i, 0], data_container[i, 1], color="green", s=30)
            for i in range(W.shape[0]):
                plt.scatter(W[i, 0], W[i, 1], color="blue", s=30)
            plt.plot(W[:, 0], W[:, 1], color='blue', linewidth=1)
            ax.grid()
            plt.pause(1.5)
            plt.xlabel('x')
            plt.ylabel('y')
    plt.show()

kohonen(1)