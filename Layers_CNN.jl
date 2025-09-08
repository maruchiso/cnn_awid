module Layers_CNN

using ..AutoDiff
import ..Layers_NN: forward, Layer
using Random, LinearAlgebra

export Conv1D, MaxPool1D


struct Conv1D <: Layer
    weights::Node
    bias::Node
    activation::Function
    stride::Int
    padding::Int
end

function Conv1D(in_channels::Int, out_channels::Int, kernel_width::Int; activation=identity, stride=1, padding=0, dtype=Float32)
    limit = sqrt(dtype(6.0) / dtype(kernel_width * in_channels))
    W_val = randn(dtype, kernel_width, in_channels, out_channels) .* limit
    b_val = zeros(dtype, out_channels)
    Conv1D(Node(W_val; is_trainable=true), Node(b_val; is_trainable=true), activation, stride, padding)
end

function forward(layer::Conv1D, x_node::Node)
    x = value(x_node)
    W = value(layer.weights)
    b = value(layer.bias)
    
    in_width, in_channels, batch_size = size(x)
    kernel_w, _, out_channels = size(W)
    
    out_width = floor(Int, (in_width + 2 * layer.padding - kernel_w) / layer.stride) + 1
    
    # convolution operation
    out_raw = zeros(Float32, out_width, out_channels, batch_size)
    for n in 1:batch_size, c_out in 1:out_channels, w_out in 1:out_width
        w_start = (w_out - 1) * layer.stride + 1
        receptive_field = x[w_start:(w_start + kernel_w - 1), :, n]
        out_raw[w_out, c_out, n] = sum(receptive_field .* W[:, :, c_out]) + b[c_out]
    end
    
    # Node for convolution output
    local conv_output_node
    function _conv_backward()
        grad_out_raw = grad(conv_output_node)
        
        grad_x = zeros(Float32, size(x))
        grad_W = zeros(Float32, size(W))
        grad_b = zeros(Float32, size(b))

        for n in 1:batch_size, c_out in 1:out_channels, w_out in 1:out_width
            w_start = (w_out - 1) * layer.stride + 1
            receptive_field = x[w_start:(w_start + kernel_w - 1), :, n]
            
            grad_slice = grad_out_raw[w_out, c_out, n]
            grad_b[c_out] += grad_slice
            grad_W[:, :, c_out] .+= receptive_field .* grad_slice
            grad_x[w_start:(w_start + kernel_w - 1), :, n] .+= W[:, :, c_out] .* grad_slice
        end
        
        accumulate_gradient!(x_node, grad_x)
        accumulate_gradient!(layer.weights, grad_W)
        accumulate_gradient!(layer.bias, grad_b)
    end
    
    conv_output_node = Node(out_raw, [x_node, layer.weights, layer.bias], _conv_backward)
    
    # Activation function on the convolution output
    final_output_node = layer.activation(conv_output_node)
    
    return final_output_node
end

get_params(layer::Conv1D) = [layer.weights, layer.bias]


struct MaxPool1D <: Layer
    pool_size::Int
    stride::Int
end
MaxPool1D(pool_size::Int) = MaxPool1D(pool_size, pool_size)

function forward(layer::MaxPool1D, x_node::Node)
    x = value(x_node)
    in_width, channels, batch_size = size(x)
    
    out_width = floor(Int, (in_width - layer.pool_size) / layer.stride) + 1
    
    out_val = zeros(Float32, out_width, channels, batch_size)
    indices = Array{CartesianIndex{3}, 3}(undef, out_width, channels, batch_size)

    for n in 1:batch_size, c in 1:channels, w_out in 1:out_width
        w_start = (w_out - 1) * layer.stride + 1
        window = x[w_start:(w_start + layer.pool_size - 1), c, n]
        max_val, rel_idx = findmax(window)
        out_val[w_out, c, n] = max_val
        indices[w_out, c, n] = CartesianIndex(w_start + rel_idx - 1, c, n)
    end
    
    local output_node
    function _backward()
        grad_out = grad(output_node)
        grad_x = zeros(Float32, size(x))
        for i in eachindex(grad_out)
            grad_x[indices[i]] += grad_out[i]
        end
        accumulate_gradient!(x_node, grad_x)
    end
    
    output_node = Node(out_val, [x_node], _backward)
    return output_node
end

get_params(layer::MaxPool1D) = []

end