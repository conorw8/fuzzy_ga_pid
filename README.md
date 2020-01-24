# fuzzy_ga_pid

In this project, a UGV is perturbed with three performance conditions, healthy, left fault, and right fault all of which are subject to various noise functions to simulate varying intensities of the faults. The goal is to assign a proportional control law to accommodate the loss in performance associated with each condition. This is done by first training a LSTM neural network to identify the severity of the fault based on various performance data such as position, velocity, orientation, etc. This serverity prediction (presumably mean residual variance), will then be plugged into a fuzzy logic controller (FLC). The FLC will output a set of PID control parameters x = [Kp, Ki, Kd] such that the specififed PID controller will be proportional to the severity of the predicted fault. The fuzzy membership functions will be optimized for this task by implementing a genetic algorithm.

# To do:

- simulate fault data
- develop regression LSTM network
- develop FLC
- develop Unity simulation
