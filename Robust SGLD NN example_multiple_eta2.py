#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import pickle
import joblib
import contextlib
from tqdm import tqdm
from joblib import Parallel, delayed


# In[2]:


# Parallel Job Progress Bar

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield tqdm_object

    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# In[3]:


# define U and its gradient wrt theta

def U_NN(theta, x, **kwargs):
    w, b = theta[:-1], theta[-1]
    z, y = x[:,:-1], x[:,-1]
    ip_NN = np.dot(z, w) + b
    return np.abs(y - 1 / (1 + np.exp(-ip_NN)))**2

def grad_U_NN(theta, x, **kwargs):
    w, b = theta[:-1], theta[-1]
    z, y = x[:,:-1], x[:,-1]
    ip_NN = np.dot(z, w) + b
    sigmoid_op = 1 / (1 + np.exp(-ip_NN))
    error = y - sigmoid_op
    
    der_w = -2 * error[:, np.newaxis] * z * sigmoid_op[:, np.newaxis] * (1 - sigmoid_op[:, np.newaxis])
    der_b = -2 * error * sigmoid_op * (1 - sigmoid_op)
    
    grad = np.zeros((x.shape[0], theta.shape[0]))
    grad[:, :-1] = der_w
    grad[:, -1] = der_b
    return grad


# In[4]:


# Discretisation of Compact Support

def set_prod(*arr_lst):
    n_arrs = len(arr_lst)
    datatype = np.result_type(*arr_lst)
    arr = np.empty([len(a) for a in arr_lst] + [n_arrs], dtype=datatype)
    for i, a in enumerate(np.ix_(*arr_lst)):
        arr[..., i] = a
    return arr.reshape(-1, n_arrs)



ell = 3
j = 1

# uniform distribution
xi_range = [-3, 3]

# discretised space
dyadic_set = np.arange(-2**(ell-1), 2**(ell-1), 1/2**j) # dyadic
dyadic_set = dyadic_set[np.logical_and(dyadic_set >= xi_range[0], dyadic_set <= xi_range[1])]
dyadic_set

# m = dimension of data
m = 4

discrete_space = set_prod(*[dyadic_set]*m)

print(discrete_space)


# In[5]:


# (Robust) SGLD
def iota(alpha):
    return np.log(np.cosh(alpha))

def grad_iota(alpha):
    return np.tanh(alpha)

# returns vector; evaluated at all points of discretised space
def F_delta(theta_bar, x, delta, p, U, iota, discrete_space, **kwargs):
    theta, alpha = theta_bar[:-1], theta_bar[-1]
    return np.exp(
        (U(theta, discrete_space, **kwargs)-iota(alpha)*np.linalg.norm(x-discrete_space, axis=1)**p)/delta
    )

def grad_V_delta(theta_bar, x, delta, p, eta1, eta2, U, iota, grad_U, grad_iota, discrete_space, **kwargs):
    theta, alpha = theta_bar[:-1], theta_bar[-1]
    F = F_delta(theta_bar, x, delta, p, U, iota, discrete_space, **kwargs)
    grad_V = np.zeros(theta_bar.shape[0])
    
    grad_V[:-1] = eta1*theta + np.dot(F, grad_U(theta, discrete_space, **kwargs))/np.sum(F)
    grad_V[-1] = (
        eta2*iota(alpha)*grad_iota(alpha)-
        grad_iota(alpha)*np.dot(F, np.linalg.norm(x-discrete_space, axis=1)**p)/np.sum(F)
    )
    
    return grad_V


# In[6]:


def sgld(n_iter, step, beta, theta_0, x_data, grad_U, **kwargs): 
    # initialise algorithm
    theta = theta_0
    
    # Store theta for each iteration
    theta_history = np.zeros((n_iter, theta_0.shape[0]))
    
    # generate random normal samples
    Z = np.random.normal(size=(n_iter, theta_0.shape[0]))
    
    # generate data
    X_idx = np.random.choice(x_data.shape[0], size=n_iter)
    
    for n in range(n_iter):
        theta = (theta +
            np.sqrt(2 * step / beta) * Z[n] - 
            step * grad_U(theta, x_data[X_idx[n]][None, :], **kwargs).ravel()
        )
        
        # Store the current theta
        theta_history[n] = theta

    return theta_history


def robust_sgld(n_iter, step, beta, 
                theta_bar_0, x_data, delta, p, eta1, eta2, U, iota, grad_U, grad_iota, discrete_space, **kwargs):
    
    # initialise algorithm
    theta_bar = theta_bar_0
    
    # Store theta_bar for each iteration
    theta_bar_history = np.zeros((n_iter, theta_bar_0.shape[0]))
    
    # generate random normal samples
    Z = np.random.normal(size=(n_iter, theta_bar_0.shape[0]))
    
    # generate data
    X_idx = np.random.choice(x_data.shape[0], size=n_iter)
    
    for n in range(n_iter):
        theta_bar = (theta_bar +
            np.sqrt(2*step/beta)*Z[n] - 
            step*grad_V_delta(theta_bar, x_data[X_idx[n]], delta, p, eta1, eta2, U, iota, grad_U, grad_iota, discrete_space, **kwargs)
        )
        
        # Store the current theta_bar
        theta_bar_history[n] = theta_bar

    return theta_bar_history


# In[7]:


#data generation

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
q = 0.3
n_train_samples = 10000
n_test_samples = 5000

# Define theta_op
w_op = np.array([-0.5, 0.5, 0.1])
b_op = -0.2
theta_op = np.append(w_op, b_op)  # theta_op = (0.5, 0.5, 0.5, 0.2)

# Function to compute sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate training samples
Z_train = []
Y_train = []

for _ in range(n_train_samples):
    # Generate samples from uniform distribution in [-1, 1]
    z_clean = np.random.uniform(-1, 1, size=3)
    
    # Draw y from normal distribution using clean z
    mean_y_clean = sigmoid(np.dot(w_op, z_clean) + b_op)
    y_clean = mean_y_clean + 0.1*np.random.binomial(n=1, p=0.5)
    
    if np.random.rand() < (1 - q):
        # Use the clean z
        z = z_clean
        y = y_clean
    else:
        # Replace z with a sample from the contaminated distribution
        z = np.random.uniform(2, 2.5, size=3)
        y = y_clean + np.random.binomial(n=1, p=0.5)
    
    # Store the samples
    Z_train.append(z)
    Y_train.append(y)  # Take the first element

Z_train = np.array(Z_train)
Y_train = np.array(Y_train)

# Reshape Y_train to be a column vector (1500, 1)
Y_train_reshaped = Y_train.reshape(-1, 1)  # Shape (1500, 1)

# Concatenate Z_train and Y_train_reshaped
X_train = np.concatenate((Z_train, Y_train_reshaped), axis=1)

# Generate test samples
Z_test = []
Y_test = []

for _ in range(n_test_samples):
    # Generate samples from uniform distribution in [-1, 1]
    z = np.random.uniform(-1, 1, size=3)
    
    # Draw y from normal distribution
    mean_y = sigmoid(np.dot(w_op, z) + b_op)
    y = mean_y + 0.1*np.random.binomial(n=1, p=0.5)
    
    # Store the samples
    Z_test.append(z)
    Y_test.append(y)  # Take the first element

Z_test = np.array(Z_test)
Y_test = np.array(Y_test)

# Reshape Y_test to be a column vector (7000, 1)
Y_test_reshaped = Y_test.reshape(-1, 1)

# Concatenate Z_test and Y_test_reshaped
X_test = np.concatenate((Z_test, Y_test_reshaped), axis=1)


# In[8]:


# set algorithm parameters
n_iter = 25000
step = 0.01
beta = 10**9
theta_0 = np.array([-2,-2,-2,-2])
theta_bar_0 = np.array([-2,-2,-2,-2,0])
delta = 0.1
p = 2 

eta1 = 10**-3
eta2_values = [0.01,0.1,0.5,1,1.5,2]#[0.01, 0.05, 0.075, 0.1, 0.5, 1] # bigger eta2 means larger "radius"


# In[9]:


# Number of runs
n_runs = 100

# Initialize a dictionary to store theta_bar_histories for each eta2
theta_bar_histories_dict = {}

# Start the timer
start_time = time.time()

# Loop over each eta2 value
for eta2 in eta2_values:
    # Run robust SGLD in parallel for each eta2 and store results
    theta_bar_histories = np.array(Parallel(n_jobs=-1)(
        delayed(robust_sgld)(
            n_iter=n_iter,
            step=step,
            beta=beta,
            theta_bar_0=theta_bar_0,
            x_data=X_train,
            delta=delta,
            p=p,
            eta1=eta1,
            eta2=eta2,  # Use current eta2 value
            U=U_NN,
            iota=iota,
            grad_U=grad_U_NN,
            grad_iota=grad_iota,
            discrete_space=discrete_space
        ) for _ in tqdm(range(n_runs), desc=f"Processing eta2={eta2}")
    ))

    # Store the results in the dictionary with eta2 as the key
    theta_bar_histories_dict[eta2] = theta_bar_histories
    
# End the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Time taken to run robust SGLD: {elapsed_time:.2f} seconds")


# In[10]:


# Dictionary to store the mean and std for the last iteration of each eta2
last_iteration_stats = {}

# Loop through each eta2 to compute the last iteration stats
for eta2, histories in theta_bar_histories_dict.items():
    # Calculate mean and std for the last iteration
    theta_bar_mean_last = np.mean(histories[:, -1, :], axis=0)
    theta_bar_std_last = np.std(histories[:, -1, :], axis=0)
    
    # Store in the dictionary
    last_iteration_stats[eta2] = {
        'mean': theta_bar_mean_last,
        'std': theta_bar_std_last
    }

# Return or print the dictionary as needed
print(last_iteration_stats)


# In[26]:


# Plotting the trajectory of theta_bar for each eta2 value
plt.figure(figsize=(12, 8))

# Loop through each eta2 to plot the trajectories
for eta2, histories in theta_bar_histories_dict.items():
    # Calculate mean and std for robust SGLD for each eta2
    theta_bar_mean = np.mean(histories, axis=0)
    theta_bar_std = np.std(histories, axis=0)
    
    # Plot each parameter trajectory excluding the last column (assuming it's bias)
    for i in range(theta_bar_mean.shape[1] - 1):  # Exclude the last column
        plt.plot(theta_bar_mean[:, i], label=f'Theta {i} (eta2={eta2})')  # Plot each weight
        plt.fill_between(range(n_iter), 
                         theta_bar_mean[:, i] - theta_bar_std[:, i], 
                         theta_bar_mean[:, i] + theta_bar_std[:, i], 
                         alpha=0.2)
        
# Finalize the plot
plt.title('Trajectory of Theta during Robust SGLD with Std Dev')
plt.xlabel('Iteration')
plt.ylabel('Theta values')
plt.legend()
plt.grid()
plt.show()


# In[41]:


# Determine the number of plots per row
plots_per_row = 3

# Calculate number of rows needed
num_eta2 = len(theta_bar_histories_dict)
num_rows = (num_eta2 + plots_per_row - 1) // plots_per_row

# Initialize the big figure
fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(18, 6 * num_rows))
axes = axes.flatten()
labels = [r'$w_0$', r'$w_1$', r'$w_2$', r'$b_0$']
reference_values = [-0.5, 0.5, 0.1, -0.2]

# Loop through each eta2 to plot the trajectories
for idx, (eta2, histories) in enumerate(theta_bar_histories_dict.items()):
    theta_bar_mean = np.mean(histories, axis=0)
    theta_bar_std = np.std(histories, axis=0)
    
    # Plot each parameter trajectory excluding the last column (assuming it's bias)
    for i in range(theta_bar_mean.shape[1] - 1):  # Exclude the last column
        axes[idx].plot(theta_bar_mean[:, i], label=labels[i])
        axes[idx].fill_between(range(n_iter), 
                               theta_bar_mean[:, i] - theta_bar_std[:, i], 
                               theta_bar_mean[:, i] + theta_bar_std[:, i], 
                               alpha=0.2)
        #axes[idx].axhline(y=reference_values[i], color='r', linestyle='--')
        
    # Finalize each subplot
    axes[idx].set_title(r'Trajectory of $\theta$ for robust SGLD with $\eta_2={}$'.format(eta2))
    axes[idx].set_xlabel('Iteration')
    axes[idx].set_ylabel(r'$\theta$ values')
    axes[idx].legend(loc='lower right')
    axes[idx].grid()

# Hide any unused subplots
for idx in range(num_eta2, len(axes)):
    fig.delaxes(axes[idx])

# Finalize the layout
plt.tight_layout()
plt.savefig('robust_sgld_parameter_traj.png')
plt.show()


# In[25]:


# Number of runs
#n_runs = 1

# Initialize arrays to accumulate the theta histories
theta_histories = np.zeros((n_runs, n_iter, theta_0.shape[0]))

# Start the timer
start_time = time.time()

# Loop for SGLD
for run in range(n_runs):
    theta_histories[run] = sgld(n_iter=n_iter, step=step, beta=beta, theta_0=theta_0, x_data=X_train, grad_U=grad_U_NN)


# End the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Time taken to run SGLD: {elapsed_time:.2f} seconds")


# Calculate mean and std for SGLD
theta_mean = np.mean(theta_histories, axis=0)
theta_std = np.std(theta_histories, axis=0)

print(theta_mean[-1])
print(theta_std[-1])


# In[42]:


# Plotting the trajectory of theta with std for SGLD
plt.figure(figsize=(12, 8))
for i in range(theta_mean.shape[1]):
    plt.plot(theta_mean[:, i], label=labels[i])
    plt.fill_between(range(n_iter), 
                     theta_mean[:, i] - theta_std[:, i], 
                     theta_mean[:, i] + theta_std[:, i], 
                     alpha=0.2)
plt.title(r'Trajectory of $\theta$ for SGLD')
plt.xlabel('Iteration')
plt.ylabel(r'$\theta$ values')
plt.legend()
plt.grid()
plt.savefig('sgld_parameter_traj.png')
plt.show()


# In[44]:


# Calculate loss for SGLD and SGLD_robust using theta_mean, theta_mean_bar and X_test
w_mean = theta_mean[:, :-1]  # Mean weights
b_mean = theta_mean[:, -1]    # Mean bias
z_test = X_test[:, :-1]
y_test = X_test[:, -1]

# Calculate loss using mean theta values
loss_sgld = np.zeros(n_iter)

for n in range(n_iter):
    ip_NN = np.dot(z_test, w_mean[n]) + b_mean[n]
    loss_sgld[n] = np.mean(np.abs(y_test - 1 / (1 + np.exp(-ip_NN)))**2)

# Compute reference loss using theta_0 with X_test
w_ref = theta_op[:-1]
b_ref = theta_op[-1]

# Calculate reference loss using X_test
reference_loss = np.zeros(n_iter)

for n in range(n_iter):
    ip_NN = np.dot(z_test, w_ref) + b_ref
    reference_loss[n] = np.mean(np.abs(y_test - 1 / (1 + np.exp(-ip_NN)))**2)



# Initialize the main plot
plt.figure(figsize=(12, 8))

# Plot mean loss for SGLD and SGLD_robust
plt.plot(reference_loss, label='Reference Loss', linestyle='--')
plt.plot(loss_sgld, label='SGLD')

# Superimpose parameter trajectories for each eta2 value
for eta2, histories in theta_bar_histories_dict.items():
    # Calculate the mean trajectory for this eta2
    theta_bar_mean_eta2 = np.mean(histories, axis=0)
    
    # Extract mean weights and bias for this eta2
    w_mean_robust_eta2 = theta_bar_mean_eta2[:, :-2]
    b_mean_robust_eta2 = theta_bar_mean_eta2[:, -2]
    
    # Calculate the loss trajectory for this eta2
    loss_trajectory_eta2 = np.zeros(n_iter)
    for n in range(n_iter):
        ip_NN = np.dot(z_test, w_mean_robust_eta2[n]) + b_mean_robust_eta2[n]
        loss_trajectory_eta2[n] = np.mean(np.abs(y_test - 1 / (1 + np.exp(-ip_NN)))**2)
    
    # Plot the loss trajectory for this eta2
    plt.plot(loss_trajectory_eta2, label=r'Robust SGLD $\eta_2={}$'.format(eta2))

# Finalize the main plot
plt.title('Mean Squared Loss Trajectory for SGLD and robust SGLD with Reference Loss using test dataset')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.savefig('sgld_robust_sgld_loss_traj1.png')
# Add a subplot for the specific eta2=2
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(reference_loss, label='Reference Loss', linestyle='--')
ax.plot(loss_sgld, label='Mean Squared Loss SGLD')

# Calculate and plot the loss trajectory for eta2=2
eta2 = 2
histories = theta_bar_histories_dict[eta2]
theta_bar_mean_eta2 = np.mean(histories, axis=0)
w_mean_robust_eta2 = theta_bar_mean_eta2[:, :-2]
b_mean_robust_eta2 = theta_bar_mean_eta2[:, -2]

loss_trajectory_eta2 = np.zeros(n_iter)
for n in range(n_iter):
    ip_NN = np.dot(z_test, w_mean_robust_eta2[n]) + b_mean_robust_eta2[n]
    loss_trajectory_eta2[n] = np.mean(np.abs(y_test - 1 / (1 + np.exp(-ip_NN)))**2)

ax.plot(loss_trajectory_eta2, color='gray', label=r'Robust SGLD $\eta2={}$'.format(eta2))

# Finalize the subplot
ax.set_title(r'Mean Squared Loss Trajectory for robust SGLD with $\eta_2={}$'.format(eta2))
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.legend()
ax.grid()

plt.savefig('sgld_robust_sgld_loss_traj2.png')
plt.show()


# In[15]:


# Last value for reference
last_loss_reference = reference_loss[-1]

# Last value for SGLD loss
last_loss_sgld = loss_sgld[-1]

# Dictionary to store last loss for each eta2 in SGLD_robust
last_loss_robust = {}

# Calculate the last loss value for each eta2
for eta2, histories in theta_bar_histories_dict.items():
    theta_bar_mean_eta2 = np.mean(histories, axis=0)
    w_mean_robust_eta2 = theta_bar_mean_eta2[:, :-2]
    b_mean_robust_eta2 = theta_bar_mean_eta2[:, -2]
    
    # Calculate the loss trajectory for this eta2
    loss_trajectory_eta2 = np.zeros(n_iter)
    for n in range(n_iter):
        ip_NN = np.dot(z_test, w_mean_robust_eta2[n]) + b_mean_robust_eta2[n]
        loss_trajectory_eta2[n] = np.mean(np.abs(y_test - 1 / (1 + np.exp(-ip_NN)))**2)
    
    # Store the last loss value for this eta2
    last_loss_robust[eta2] = loss_trajectory_eta2[-1]

# Print or return the last loss values
print(f"Last reference Loss: {last_loss_reference}")
print(f"Last SGLD Loss: {last_loss_sgld}")
print("Last SGLD_robust Loss for each eta2:", last_loss_robust)


# In[ ]:




