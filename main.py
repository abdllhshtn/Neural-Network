import numpy as np

X=np.array(
    [
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ]
)
Y= np.array(
    [
        [0],
        [1],
        [1],
        [0]
    ]
)

W1= np.random.randn(2,2)
b1= np.zeros((1,2))
W2=np.random.randn(2,1)
b2=np.zeros((1,1))

def sigmoid(z):
    return 1/(1+np.exp(-z))
for epoch in range (10000):

    z1= X@W1+b1
    a1= sigmoid(z1)
    z2=a1@W2+b2
    y_pred= sigmoid(z2)

    loss= np.mean((Y-y_pred)**2)

    d_loss_y_pred= 2*(y_pred-Y)/Y.shape[0]

    d_y_pred_z2= y_pred*(1-y_pred)

    d_z2=d_y_pred_z2 * d_loss_y_pred

    d_W2=a1.T@d_z2

    d_b2=np.sum(d_z2,axis=0, keepdims=True)

    d_a1=d_z2@W2.T
    d_a1_z1 = a1 * (1 - a1)
    d_z1=d_a1*d_a1_z1

    d_W1=X.T@d_z1
    d_b1=np.sum(d_z1,axis=0,keepdims=True)

    lr=0.1

    W1=W1-lr*d_W1
    b1=b1-lr*d_b1
    W2=W2-lr*d_W2
    b2=b2-lr*d_b2
    if epoch%1000==0:
        print(loss)

print("Final predictions:")
print(y_pred)


