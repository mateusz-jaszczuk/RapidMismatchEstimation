import numpy as np
from jax import vmap, jit, lax, value_and_grad
import jax.numpy as jnp
import jax.random as random
import jax.scipy.stats as stats
from functools import partial 
import optax
import torch
import torch.nn as nn
import warnings
warnings.simplefilter("ignore", category=FutureWarning)


class RapidMismatchEstimation():
    def __init__(self, vi_max_iterations=2500, auto_warmup = True):
        ## Initialize parameters for the model
        print('### Rapdid Mismatch Estimation ###\n')
        self.current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.NeuralNetwork = RME_NeuralNetwork().to(self.current_device)
        self.VariationalInference = RME_VariationalInference(max_iterations=vi_max_iterations)
        self.fastFK = computeFK()
        if auto_warmup:
            self.load_and_warm_up()

    def load_and_warm_up(self,nn_params='/home/mateusz/mateusz_ws/src/rapid_mismatch_estimation/scripts/model_weights.pth'):
        self.NeuralNetwork.rme_load(nn_params,current_device=self.current_device)
        print('###        Warm-up run         ###')
        self.VariationalInference.warm_up_compiler()
        J_warm_up = jnp.zeros((50, 6, 7))
        tau_warm_up = jnp.zeros((50, 7))
        q_warmp_up = jnp.zeros((7,50))
        _ = self.convert_NN_inputs(J_warm_up, tau_warm_up)
        _,_ = self.computeJacobian_R0e(q_warmp_up)
        tau_warm_up = jnp.zeros((200, 7))
        q_warmp_up = jnp.zeros((7,200))
        _ = self.estimate_mismatch(tau_warm_up,(q_warmp_up.T))
        print("Warm-up for vmap completed.")

    def convert_NN_inputs(self,Jacobians, true_torques):
        # Converts inputs to pseudo wrenches
        VI_data_shape = Jacobians.shape[0]
        NN_desired_shape = 20
        idx = np.linspace(0,VI_data_shape-1,NN_desired_shape).astype(int)
        Jacobians_nn = Jacobians[idx,:,:]
        true_torques_nn = true_torques[idx,:]
        def one_step(J, torque):
            pseudo_inverse = jnp.linalg.inv(J @ J.T + jnp.eye(6) * 1e-2) @ J
            return pseudo_inverse @ torque
        W = vmap(one_step, in_axes=(0, 0))(Jacobians_nn, true_torques_nn)
        return np.array(W)
    
    @partial(jit, static_argnums=(0,))
    def computeJacobian_R0e(self,q):
        Jac_Rt = vmap(self.fastFK.compute_Jacobian_R0e, in_axes=1)(q)
        return Jac_Rt[0].transpose((0,2,1)), Jac_Rt[1]
    
    def convert_VI_inputs(self, torque_list, q_list, vi_sequence_length):
        # Convert input lists to proper jnp.arrays
        idx = np.linspace(0,q_list.shape[0],vi_sequence_length).astype(int)
        q = (jnp.array(q_list).T)[:,idx]
        torque = (jnp.array(torque_list))[idx,:]
        return q, torque

    def estimate_mismatch(self, tau_ext_list, q_list, vi_sequence_length = 50):
        # Prepare inputs for estimation
        q, ext_torque = self.convert_VI_inputs(tau_ext_list, q_list, vi_sequence_length)
        Jacobians, R = self.computeJacobian_R0e(q)
        nn_inputs = torch.tensor(self.convert_NN_inputs(Jacobians.transpose(0,2,1), ext_torque), dtype=torch.float32).reshape(1, 20, 6)
      
        # Define VI iitial parameters with NN
        mu_init = jnp.array(self.NeuralNetwork.nn_estimation(nn_inputs,current_device=self.current_device).cpu())
        log_sigma_init = jnp.array([1.0, 0.05,  0.05, 0.1])
        initial_parameters = jnp.concatenate([mu_init, log_sigma_init])

        # Run Variational Inference Estimation
        estimated_params = self.VariationalInference.estimate_posterior(initial_parameters, 
                                                                        true_torques = ext_torque, 
                                                                        Jacobians = Jacobians, R = R)
        return np.array(estimated_params)
    

class RME_VariationalInference:
    def __init__(self, max_iterations):
    ## Initialize parameters for the model
        self.fastFK = computeFK()
        self.max_iterations = max_iterations
        self.std_prior = jnp.array([0.5, 0.02, 0.02, 0.05])     
        self.opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=0.025))
        print('###   Variational Inference    ###')
        print('Model initialized with max',self.max_iterations,'iterations.\n')
   
    def objective_function(self, params, rng, true_torques, Jacobians, R, mean_prior, likelihood_noise):
        mu, log_sigma = params[:4], params[4:]
        # Reparametrization trick
        epsilon = random.normal(rng, shape=(4,))
        theta_sample = mu + jnp.exp(log_sigma) * epsilon
        # Compute ELBO
        log_prior = jnp.sum(stats.norm.logpdf(theta_sample, loc = mean_prior, scale = self.std_prior))
        predicted_torques = self.fastFK.compute_ext_torque(mass = theta_sample[0],
                                                           r_com = jnp.array(theta_sample[1:]),
                                                           Jacobians = Jacobians, R = R)
        log_likelihood = jnp.sum(stats.norm.logpdf(true_torques, loc = predicted_torques, scale = likelihood_noise))
        log_q = jnp.sum(stats.norm.logpdf(theta_sample, loc=mu, scale=jnp.exp(log_sigma)))
        elbo = log_likelihood + log_prior - log_q
        return -elbo
    
    @partial(jit, static_argnums=(0,))
    def step(self, params, opt_state, rng, true_torques, Jacobians, R, prior, likelihood_noise):
        loss, grads = value_and_grad(self.objective_function)(params, rng, true_torques, Jacobians, R, prior, likelihood_noise)
        updates, opt_state = self.opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    def warm_up_compiler(self):
        # Warm-up inputs
        warmup_params = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.05, 0.05, 0.05])
        warmup_torques = jnp.zeros((50, 7))
        warmup_Jacobians = jnp.zeros((50, 7, 6))
        warmup_R = jnp.zeros((50, 3, 3))
        prior = warmup_params[:4]
        # Run warm-up pass on estimate_posterior
        _ = self.estimate_posterior(warmup_params, warmup_torques, warmup_Jacobians, warmup_R)
        # Run warm-up pass on step()
        opt_state = self.opt.init(warmup_params)
        rng = random.PRNGKey(0)
        rng, subkey = random.split(rng)
        _ = self.step(warmup_params, opt_state, subkey, warmup_torques, warmup_Jacobians, warmup_R,prior, jnp.array([0.1]*7))
        print('Warm-up run for JIT compiler completed.')

    def estimate_posterior(self,params, true_torques, Jacobians, R):
        opt_state = self.opt.init(params)
        mean_prior = params[:4]
        likelihood_noise = 0.25 * (jnp.max(true_torques,axis=0) - jnp.min(true_torques,axis=0))
        likelihood_noise = jnp.maximum(likelihood_noise, jnp.array([0.1]*7))
        # Conditions for early stopping
        check_conditions = 700
        tol = 4e-2
        stop_verification = 2
        # Training loop 
        rng = random.PRNGKey(1)
        prev_params = params  
        no_improve_count = 0  
        
        for iteration in range(self.max_iterations):
            rng, subkey = random.split(rng)
            params, opt_state, _ = self.step(params, opt_state, 
                                             subkey, true_torques, 
                                             Jacobians, R, mean_prior, 
                                             likelihood_noise)
            # Start checking improvement after a number of iterations
            if (iteration > check_conditions) and iteration % 10 == 0:  
                # Compute L2-norm difference between consecutive parameter updates
                param_change = jnp.linalg.norm(params - prev_params)
                prev_params = params  # Update previous parameters
                # Check stopping condition
                if param_change < tol:
                    no_improve_count += 1
                else:
                    no_improve_count = 0  # Reset if improvement is made
                if no_improve_count >= stop_verification:
                    break
        return params
    

class RME_NeuralNetwork(nn.Module):
    def __init__(self, seq_len=20, feature_dim=6, hidden_dim=64, num_mlp_layers = 3, mlp_dim=256, dropout_rate = 0.1, num_heads=8, output_dim=4):
      super().__init__()
      # Define Model
      self.Convolution = nn.Conv1d(in_channels=feature_dim, out_channels=hidden_dim, kernel_size=5, padding=2)
      self.PositionalEmbedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
      self.SelfAttention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
      self.num_mlp_layers = num_mlp_layers
      self.MLP_Layer = nn.Sequential(
          nn.Linear(hidden_dim, mlp_dim),
          nn.ReLU(),
          nn.Linear(mlp_dim, hidden_dim),
          nn.Dropout(dropout_rate))
      self.ModelHead = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
      # Convolution Layer
      x = x.permute(0, 2, 1) # Swap axis for Convolution
      x = self.Convolution(x)
      # Positional Encoding
      x = x.permute(0, 2, 1) # Swap axis back
      x = x + self.PositionalEmbedding
      # Multi-Head Self-Attention
      attention_output, _ = self.SelfAttention(x, x, x)
      x = torch.mean(attention_output, dim=1)
      # Multi-Layer Perceptron
      for i in range(self.num_mlp_layers):
        x = self.MLP_Layer(x)
      # Model Head
      x = self.ModelHead(x).squeeze()
      return x

    def rme_load(self, parameters_file,current_device):
      # Load model from the saved parameters
      self.load_state_dict(torch.load(parameters_file,map_location=current_device,weights_only=True))
      self.to(current_device)
      # Print total number of model parameters
      n_parameters = sum(p.numel() for p in self.parameters())
      n_parameters = format(n_parameters, ",")
      print('###       Neural Network       ###')
      print('Model initialized with ',n_parameters,'parameters.')
      print("Model parameters loaded from", parameters_file, "\n")

    def nn_estimation(self, input,current_device):
      # Function to get model pedictions
      normalization_paramters = torch.load("/home/mateusz/mateusz_ws/src/rapid_mismatch_estimation/scripts/normalization_paramters.pth",weights_only=False)
      inputs = input.clone().detach().to(current_device)
      inputs = (inputs - normalization_paramters[0,:,:,:]) / normalization_paramters[1,:,:,:]
      # Get model predictions
      self.eval()
      with torch.no_grad():
          predictions = self(inputs)
      return predictions
    

class computeFK:
    def __init__(self):
        # DH parameters for Franka Emika Arm
        self.a = jnp.array([0, 0, 0.0825, 0.0825, 0, 0.088, 0])
        self.d = jnp.array([0.333, 0, 0.316, 0, 0.384, 0, 0.107])
        self.alpha = jnp.array([-jnp.pi/2, jnp.pi/2, jnp.pi/2, jnp.pi/2, -jnp.pi/2, jnp.pi/2, 0])
        self.theta_offset = jnp.array([0, 0, 0, jnp.pi, 0, -jnp.pi, -jnp.pi/4])
        self.g = -9.81

    def compute_A(self, n, q):
        # Compute single A transformation matrix
        theta = q[n] + self.theta_offset[n]
        A = jnp.array([
            [jnp.cos(theta), -jnp.sin(theta) * jnp.cos(self.alpha[n]),  jnp.sin(theta) * jnp.sin(self.alpha[n]), self.a[n] * jnp.cos(theta)],
            [jnp.sin(theta),  jnp.cos(theta) * jnp.cos(self.alpha[n]), -jnp.cos(theta) * jnp.sin(self.alpha[n]), self.a[n] * jnp.sin(theta)],
            [0, jnp.sin(self.alpha[n]), jnp.cos(self.alpha[n]), self.d[n]],
            [0, 0, 0, 1]])
        return A
    
    def calcFK(self, q):
        # Compute the rotation matrix betwen EF and WF
        q = jnp.array(q)
        A_matrices = vmap(self.compute_A, (0, None))(jnp.arange(7), q)
        Transformations = lax.associative_scan(jnp.matmul, A_matrices)
        T0e = Transformations[-1] 
        # Return all transformations and rotation between WF and EF frames
        return Transformations, T0e[:3,:3]

    def compute_Jac_R0e(self, q):
        # Compute Jacobian of the robot
        q = jnp.asarray(q)  
        Transformations, R0e = self.compute_FK(q)
        # Get origins and rotation axis for Jacobians computation
        origins = jnp.vstack([jnp.zeros((1, 3)), Transformations[:, :3, 3]])
        rotation_axis = jnp.vstack([jnp.array([[0, 0, 1]]), Transformations[:, :3, 2]])
        # Compute individual columns of Jacobian
        def right_jacobian(idx):
            Jv = jnp.cross(rotation_axis[idx], origins[-1]  - origins[idx])
            Jw = rotation_axis[idx]
            return jnp.hstack((Jv, Jw))
        # Compute all columns of Jacobian with vmap
        J = vmap(right_jacobian)(jnp.arange(7)).T
        return J, R0e

    # Compile with JIT for efficiency
    compute_FK = jit(calcFK, static_argnums=(0,))
    compute_Jacobian_R0e = jit(compute_Jac_R0e, static_argnums=(0,))

    def compute_ext_torque(self, mass, r_com, Jacobians, R):
        # Run Robot Inverse Dynamics to compute external torque
        r_com = jnp.einsum("tij,j->ti", R, r_com) # Convert r_com to WF
        F = jnp.tile(jnp.array([0.0, 0.0, self.g * mass]), (Jacobians.shape[0], 1))
        W_ext = jnp.hstack((F, jnp.cross(r_com, F)))
        external_torque = jnp.einsum("tji,ti->tj", Jacobians, W_ext)
        return external_torque