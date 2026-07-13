import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt(
    "housing.csv",
    delimiter=",",
    skip_header=1,
    usecols=range(9)
)
data = data[~np.isnan(data).any(axis=1)]
X = data[:, :8]
Y = data[:, 8].reshape(-1, 1)


X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
Y = Y / np.max(Y)


sample_number=int(input("enter a sample number"))
X=X[0:sample_number:1]
Y=Y[0:sample_number:1]

input_number = X.shape[1]
hidden_layer=int(input("enter your hidden layer number"))
neurons=[]
for n in range(hidden_layer):
    neuron_number=int(input(f"(enter neuron number for layer {n+1}: "))
    neurons.append(neuron_number)

output_number=1


layers=[input_number] + neurons + [output_number]



weights=[]
biases=[]


for i in range(len(layers)-1):
    w=np.random.randn(layers[i],layers[i+1])
    b=np.zeros((1,layers[i+1]))
    weights.append(w)
    biases.append(b)

m_weights = [np.zeros_like(w) for w in weights]
v_weights = [np.zeros_like(w) for w in weights]

m_biases = [np.zeros_like(b) for b in biases]
v_biases = [np.zeros_like(b) for b in biases]

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

def sigmoid(z):
    return 1/(1+np.exp(-z))

def relu(z):
    return np.maximum(0,z)

def relu_derivative(z):
    return z>0


epoch_number=int(input("Enter epoch number "))
loss_history=[]



lr=float(input("enter lr"))


for epoch in range(epoch_number):
    activations=[X]
    zs=[]
    for i in range (len(weights)):

        z=activations[-1]@weights[i]+ biases[i]
        if i==len(weights)-1:
            a=sigmoid(z)
        else:
            a=relu(z)
        zs.append(z)
        activations.append(a)
    y_pred=activations[-1]
    loss=np.mean((Y-y_pred)**2)
    loss_history.append(loss)
    if epoch % 100 == 0:
        print("epoch:", epoch, "loss:", loss)

        
    grad_weights=[None]*len(weights)
    grad_biases=[None]*len(biases)

    d_loss_y_pred=2*(y_pred -Y)/Y.shape[0]
    d_y_pred_z=y_pred*(1-y_pred)
    d_z=d_loss_y_pred*d_y_pred_z

    for i in reversed(range(len(weights))):

        d_W=activations[i].T@d_z
        d_b=np.sum(d_z,axis=0,keepdims=True)
        grad_weights[i]=d_W
        grad_biases[i]=d_b
        if i !=0:  
            d_a=d_z@weights[i].T
            d_a_z=relu_derivative(zs[i-1])
            d_z =d_a*d_a_z



    t = epoch + 1

    for i in range(len(weights)):
        m_weights[i] = beta1 * m_weights[i] + (1 - beta1) * grad_weights[i]
        v_weights[i] = beta2 * v_weights[i] + (1 - beta2) * (grad_weights[i] ** 2)

        m_biases[i] = beta1 * m_biases[i] + (1 - beta1) * grad_biases[i]
        v_biases[i] = beta2 * v_biases[i] + (1 - beta2) * (grad_biases[i] ** 2)

        m_w_hat = m_weights[i] / (1 - beta1 ** t)
        v_w_hat = v_weights[i] / (1 - beta2 ** t)

        m_b_hat = m_biases[i] / (1 - beta1 ** t)
        v_b_hat = v_biases[i] / (1 - beta2 ** t)

        weights[i] = weights[i] - lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        biases[i] = biases[i] - lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)


plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Graph")
plt.show()