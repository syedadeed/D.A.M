import pickle

def train(neural_network, train_data, train_label, epoch):
    for i in range(epoch):
        for j in range(len(train_label)):

            #Forward Propogation
            output = train_data[j]
            for k in neural_network:
                output = k.forward_propogation(output)

            #Backward Propogation
            loss = output - train_label[j]
            for m in reversed(neural_network):
                loss = m.backward_propogation(loss)

        with open("parameters.dat", "wb") as file:
            for n in range(0, len(neural_network), 2):
                pickle.dump(neural_network[n].weight, file)
                pickle.dump(neural_network[n].bias, file)
