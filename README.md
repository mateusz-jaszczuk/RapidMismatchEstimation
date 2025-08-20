# Rapid Mismatch Estimation
Welcome to [Rapid Mismatch Estimation](https://mateusz-jaszczuk.github.io/rme/) GitHub Page! Below you can find attached project code.

## Robot Compatibility
The following code was designed for [Franka Emika Research](https://www.franka.de/) manipulator. If you wish to use it with different robots, adjust the DH-parameters in Forward Kinematics code, and re-train the Neural Network.


## Using the Code
Main framework's function is 
<pre><code>RapidMismatchEstimation().estimate_mismatch(tau_ext, q)</code></pre>
which estimates load applied to the end-effector in about 200 ms. As inputs, user provides time histories of joint positions and external torques applied at each joint over 200 ms time window.

### Activation Condition and Data Collection
Above, we provide function we use as the framework activation condition. We suggest that activation condition and then data collection are both done within main control loop to avoid system delays. Data collection should **only begin after triggering activation condition,** which should be follwed by passing collected data to the estimation framework.

### ROS Package
For optimal performance, we created a ROS Package rapid_mismatch_estimation, allowing to run the model in parrallel to the main control loop. After data collection, user should publish joint position and external torque measurement hirsotiries over 200 ms time-window to the following topic
<pre><code>'/rme_data/feedback_data'</code></pre>
Upon finishing the estimation, package will publish estimation result as a list (mass, r_x, r_y, r_z) to the following topic
<pre><code>'/rme_result/estimated_mismatch'</code></pre>

## Contact
In case of any questions, please contact Mateusz Jaszczuk at [jaszczuk@seas.upenn.edu](mailto:jaszczuk@seas.upenn.edu).

## Citation
If you find this work useful, please consider citing:
<pre><code>@inproceedings{jaszczuk2025rme,
      author = {Jaszczuk, Mateusz and Figueroa, Nadia},
      title = {Rapid Mismatch Estimation via Neural Network Informed Variational Inference},
      booktitle = {9th Conference on Robot Learning (CoRL)},
      year = {2025}
}</code></pre>
