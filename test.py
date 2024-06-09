
# Function RunServer():
# 2:     Initialize global model parameters W_0
# 3:     for each round t = 1, 2, ... do
# 4:         St ← select a subset of clients
# 5:         for each client c_k in St do
# 6:             Q_level ← AdaptiveQuantizationLevel(c_k)
# 7:             Send(global_model, Q_level to c_k)
# 8:         end for
# 9:         AggregateQuantizedModelUpdates()
# 10:         UpdateGlobalModel()
# 11:     end for
# 12: EndFunction

# 13: Function RunClient(c_k, global_model, Q_level):
# 14:     local_model ← TrainLocalModel(global_model, Q_level)
# 15:     quantized_gradients ← CGQ(local_model, Q_level)
# 16:     clipped_gradients ← SCS(quantized_gradients)
# 17:     Send(ClippedGradients to server)
# 18: EndFunction

# // Adaptive Quantization Level for the forward pass
# 19: Function AdaptiveQuantizationLevel(c_k):
# 20:     data_variance ← CalculateDataVariance(c_k)
# 21:     return DetermineQuantizationLevel(data_variance)
# 22: EndFunction

# // Compensative Gradient Quantization
# 23: Function CGQ(local_model, Q_level):
# 24:     gradients ← ComputeGradients(local_model)
# 25:     compensation_factor ← CalculateCompensationFactor(gradients)
# 26:     return QuantizeGradients(gradients, Q_level, compensation_factor)
# 27: EndFunction

# // Symmetric Clipping and Scaling
# 28: Function SCS(quantized_gradients):
# 29:     clipping_threshold ← CalculateClippingThreshold(quantized_gradients)
# 30:     clipped_gradients ← ClipGradients(quantized_gradients, clipping_threshold)
# 31:     scaling_factor ← CalculateScalingFactor(clipping_threshold)
# 32:     return ScaleGradients(clipped_gradients, scaling_factor)
# 33: EndFunction