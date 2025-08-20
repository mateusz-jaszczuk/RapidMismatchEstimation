def activation_condition(torque_to_check,F_ext_list):
    long_term_increase = np.abs((torque_to_check[-1] - torque_to_check[0])) > 0.7
    stabilization_checks = [sum(torque_to_check[-70:   -50])/20, 
                            sum(torque_to_check[-150: -130])/20, 
                            sum(torque_to_check[-230: -210])/20]
    end_stabilization = sum(torque_to_check[-10:])/10
    stabilization = all(abs(end_stabilization - checks) < 0.2 for checks in stabilization_checks)

    # Check forces
    F_ext = np.array(F_ext_list)
    #F_ext_mean_init = np.mean(F_ext[:20,:],axis=0)
    F_ext_mean = np.mean(F_ext[-50:,:],axis=0) #- F_ext_mean_init 
    z_principal_force = (np.abs(F_ext_mean[2]) > F_ext_mean[0]) and (np.abs(F_ext_mean[2]) > F_ext_mean[0])
    side_forces = (F_ext_mean[0] < 2.5) and (F_ext_mean[1] < 2.5)
    force_check = z_principal_force and side_forces