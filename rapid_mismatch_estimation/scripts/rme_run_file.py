#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
from RapidMismatchEstimation import RapidMismatchEstimation
import os
import time

class subscriber_rme:
    def __init__(self):
        self.analysis_data_msg = rospy.Subscriber('/rme_data/feedback_data', Float32MultiArray,  self.analysis_data_callback, queue_size=10)
        self.q_input = list(np.zeros(700))
        self.tau_ext_input = list(np.zeros(700))
        self.new_data_received = False
    def analysis_data_callback(self, msg):
        received_data = np.array(msg.data).reshape((400, 7))
        self.q_input = received_data[:200,:]
        self.tau_ext_input = received_data[200:,:]
        self.new_data_received = True
    def reset_analysis(self):
        self.new_data_received = False
    def return_message(self):
        return self.q_input, self.tau_ext_input

if __name__ == "__main__":
    # Initialize ROS
    rospy.init_node("rme_analysis")
    Subscriber_RME = subscriber_rme()
    pub_rme = rospy.Publisher("/rme_result/estimated_mismatch", Float32MultiArray, queue_size = 10)
    rme = RapidMismatchEstimation()
    while not rospy.is_shutdown():
        if Subscriber_RME.new_data_received:
            q, tau_ext = Subscriber_RME.return_message()
            estimated_params = rme.estimate_mismatch(tau_ext_list = tau_ext, q_list = q)[:4]
            estimated_params = estimated_params.tolist()
            rme_result_msg = Float32MultiArray()
            rme_result_msg.data = estimated_params
            pub_rme.publish(rme_result_msg)
            Subscriber_RME.reset_analysis()
        else:
            rospy.sleep(0.01)