<meta name="google-site-verification" content="FJJsOvNIncYaSRIiRjRr1NVSGaEFbwiMKFWWsgKC_KQ" />

# US-CNN: Detection of overlapping ultrasonic echoes with deep neural networks

This code package implements the method described in the paper "Detection of overlapping ultrasonic echoes with deep neural networks
". Code package includes simulation of overlapping ultrasonic echoes and detection of echoes using CNNs.

Link to paper: https://www.sciencedirect.com/science/article/abs/pii/S0041624X21002183

run:

main for running an experiment.   In it, the experiment main arguments are set:
    - assumption: defines for the simulated data distribution if it is general or specific
    - test_data_type: what data to train on

    Other parameters can be set in kwargs dictionary.


configuration:

     - config: the experiment parameters configuration
     - path_config: path configuration

