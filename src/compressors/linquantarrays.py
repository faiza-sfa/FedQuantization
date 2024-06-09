import numpy as np
from src.compressors.compressor import Compressor
print("Initializing Julia integration.")
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.eval("using Random; Random.seed!(0)")
Main.include("src/compressors/linquantarrays.jl")
print("Julia module loaded successfully.")

class LinQuantArrays(Compressor):
    def __init__(self):
        pass

    def encode(self, array: np.ndarray, dtype: np.dtype = np.uint8) -> bytes:
        """Quantizes the numpy array using the specified unsigned integer dtype and returns the encoded bytes."""
        assert len(array.shape) == 1  # Ensure it's a 1D array
        assert array.dtype == np.float32 or array.dtype == np.float64, "Array must be of type float32 or float64"
        
        # Mapping numpy dtype to Julia UInt type
        julia_dtype = {np.uint8: 'UInt8', np.uint16: 'UInt16', np.uint32: 'UInt32', np.uint64: 'UInt64'}[dtype]
        
        # Calling the Julia function to perform quantization
        result = Main.LinQuantization(julia_dtype, array)
        return bytes(result.A)  # Access the UInt array and convert to bytes

    def decode(self, data: bytes, dtype: np.dtype, shape: tuple) -> np.ndarray:
        """Reconstructs the numpy array from the quantized data."""
        assert dtype in [np.uint8, np.uint16, np.uint32, np.uint64], "dtype must be an unsigned integer type"
        
        # Convert bytes back to numpy array of the appropriate dtype
        quant_array = np.frombuffer(data, dtype=dtype).reshape(shape)
        
        # Mapping numpy dtype to Julia UInt type
        julia_dtype = {np.uint8: 'UInt8', np.uint16: 'UInt16', np.uint32: 'UInt32', np.uint64: 'UInt64'}[dtype]
        
        # Create a fake LinQuantArray structure expected by Julia for dequantization
        Main.eval(f"""
        struct LinQuantArray{T,N} <: AbstractArray{{Unsigned,N}}
            A::Array{{T,N}}
            min::Float64
            max::Float64
        end
        """)
        
        # Create the LinQuantArray object in Julia with dummy min and max since we don't store them
        # Note: Replace `dummy_min` and `dummy_max` with actual values if needed
        dummy_min, dummy_max = 0.0, 1.0
        Main.eval(f"q_array = LinQuantArray{{{julia_dtype}, 1}}(reshape({quant_array.tolist()}, {shape}), {dummy_min}, {dummy_max})")
        
        # Call the dequantization function
        result = Main.Array(np.float32, Main.q_array)
        return result

# Example of using LinQuantArrays
if __name__ == "__main__":
    compressor = LinQuantArrays()
    float_array = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    encoded = compressor.encode(float_array, dtype=np.uint8)
    decoded = compressor.decode(encoded, dtype=np.uint8, shape=(3,))
    print("Decoded array:", decoded)
