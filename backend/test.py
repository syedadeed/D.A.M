import numpy as np

def test(neural_network, test_data, test_label):
    correct = 0
    for i in range(len(test_label)):
        output = test_data[i]
        for j in neural_network:
            output = j.forward_propogation(output)

        if np.argmax(output) == np.argmax(test_label[i]):
            correct += 1

    print(f"Accuracy: {(correct / i) * 100:.2f}%")
    print(f"Correct: {correct:,}")
