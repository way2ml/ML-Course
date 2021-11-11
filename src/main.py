import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print(list(training_data)[0])
input()
net = network.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# # print(type(net.weights))
# # print(len(net.weights))
# print(net.weights)
net.save_model()
# net.load_model()
# print(net.weights)

