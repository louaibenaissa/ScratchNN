import numpy as np 
class NeuralNet : 
    """
    Neural network class, input as row vectors for the features.
    """
    def __init__(self,dims):
        """
        Initialize neural network with specified dimensions.
        dims: list containing number of neurons in each layer (including input and output)
        """
        self.dims = dims
        self.W = [None] 
        self.b = [None]
        # W size is length of neural network - the input layer (dims-1)
        for i in range(1,len(dims)): 
            self.W.append(np.random.randn(dims[i], dims[i-1]) * np.sqrt(2.0 / dims[i-1]))
            self.b.append(np.random.randn(dims[i], 1))


    def forward_prop(self,X) : 
        """
        Forward propagation through the network
        X: input data (samples x features)
        """
        self.A = [np.array(X).T]
        for i in range(1,len(self.dims)):
            Z = np.dot(self.W[i],self.A[i-1]) + self.b[i]
            self.A.append(1/(1+np.exp(-Z)))


    def back_prop(self,y) : 
        """
        Backward propagation to compute gradients
        y: true labels (one-hot encoded)
        """
        m = y.shape[1] if len(y.shape) > 1 else len(y)  # Number of samples
        y = np.array(y).reshape(-1, m) if len(y.shape) == 1 else np.array(y).T
   

        self.dW = [None]*len(self.dims)
        self.db = [None]*len(self.dims)
        self.dZ = [None]*len(self.dims)

        self.dZ[-1] = self.A[-1] - y

        for i in range(len(self.W)-1,0,-1) :
            
            self.dW[i] = 1/m *np.dot(self.dZ[i],self.A[i-1].T)
            self.db[i] = 1/m *np.sum(self.dZ[i],axis=1,keepdims=True)
            if i>1 : 
                self.dZ[i-1] = np.dot(self.W[i].T, self.dZ[i]) * (self.A[i-1] * (1 - self.A[i-1]))
            
    def update_params(self, learning_rate=0.001):
        """
        Update weights and biases using computed gradients
        """
        for i in range(1, len(self.dims)):
            self.W[i] = self.W[i] - learning_rate * self.dW[i]
            self.b[i] = self.b[i] - learning_rate * self.db[i]


    def train(self, X, y, nb_iters, learning_rate=0.001, batch_size=None, verbose=False):
        """
        Train the neural network
        X: input data
        y: labels (one-hot encoded)
        nb_iters: number of iterations
        learning_rate: learning rate for gradient descent
        batch_size: size of mini-batches (None for full batch)
        verbose: whether to print progress
        """
        m = X.shape[0]
        
        # Handle batch_size
        if batch_size is None:
            batch_size = m
            
        losses = []
        for i in range(nb_iters):
            # Mini-batch processing
            for j in range(0, m, batch_size):
                end = min(j + batch_size, m)
                X_batch = X[j:end]
                y_batch = y[j:end]
                
                # Forward pass
                self.forward_prop(X_batch)
                
                # Backward pass
                self.back_prop(y_batch)
                
                # Update parameters
                self.update_params(learning_rate)
            
            # Compute full dataset loss for tracking (can be expensive for large datasets)
            if verbose:
                self.forward_prop(X)
                loss = -np.mean(y * np.log(self.A[-1].T + 1e-8) + (1-y) * np.log(1 - self.A[-1].T + 1e-8))
                losses.append(loss)
                
                if i % 10 == 0:
                    print(f"Iteration {i}, Loss: {loss:.6f}")
        
        return losses
    

    def predict(self, X):
        """
        Make predictions with trained network
        X: input data
        Returns: predictions (samples x classes)
        """
        self.forward_prop(X)
        return self.A[-1].T  # Return in shape (samples x classes)
    

    def evaluate(self, X, y):
        """
        Evaluate the model accuracy
        X: input data
        y: true labels (not one-hot encoded)
        """
        predictions = self.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_classes == y)
        return accuracy
    

