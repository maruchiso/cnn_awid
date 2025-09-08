module Layers_NN

using ..AutoDiff
using Random, LinearAlgebra

abstract type Layer end
export Layer, Dense, Model, get_params, Embedding, Flatten, Permute, forward


struct Dense <: Layer
    W::Node
    b::Node
    activation::Function
    
    function Dense(input_size::Int, output_size::Int, activation_func::Function=identity; dtype::Type{<:Real}=Float32)
        # Glorot/Xavier initialization
        limit = sqrt(dtype(6.0) / (dtype(input_size) + dtype(output_size)))
        W_val = rand(dtype, input_size, output_size) .* dtype(2.0) .* limit .- limit
        b_val = zeros(dtype, 1, output_size)
        W_node = Node(W_val; is_trainable=true)
        b_node = Node(b_val; is_trainable=true)
        new(W_node, b_node, activation_func)
    end
end

function forward(layer::Dense, x::Node)
    linear_combination = matmul(x, layer.W) + layer.b
    return layer.activation(linear_combination)
end

function get_params(layer::Dense)
    return [layer.W, layer.b]
end

struct Model
    layers::Vector{<:Layer}
    
    function Model(model_layers::Layer...)
        layers_vec = collect(Layer, model_layers)
        new(layers_vec)
    end
end

function forward(model::Model, x::Node)
    current_output = x
    for layer_item in model.layers
        current_output = forward(layer_item, current_output)
    end
    return current_output
end

(model::Model)(x::Node) = forward(model, x)

function get_params(model::Model)
    all_params = Node[]
    for l in model.layers
        if hasmethod(get_params, (typeof(l),))
            append!(all_params, get_params(l))
        end
    end
    return unique(all_params)
end

struct Embedding <: Layer
    weights::Node
    function Embedding(vocab_size::Int, embed_dim::Int; dtype=Float32)
        limit = sqrt(dtype(1.0) / dtype(embed_dim))
        W_val = rand(dtype, vocab_size, embed_dim) .* dtype(2.0) .* limit .- limit
        new(Node(W_val; is_trainable=true))
    end
end

function forward(layer::Embedding, x::Node)
    indices = Int.(value(x))
    batch_size, seq_len = size(indices)
    embed_dim = size(value(layer.weights), 2)
    
    output_val = zeros(Float32, batch_size, seq_len, embed_dim)
    for b in 1:batch_size, t in 1:seq_len
        idx = indices[b, t]
        if idx > 0
            output_val[b, t, :] = value(layer.weights)[idx, :]
        end
    end

    local output_node
    function _backward()
        grad_out = grad(output_node)
        grad_w = zeros(Float32, size(value(layer.weights)))
        for b in 1:batch_size, t in 1:seq_len
            idx = indices[b, t]
            if idx > 0
                grad_w[idx, :] .+= grad_out[b, t, :]
            end
        end
        accumulate_gradient!(layer.weights, grad_w)
    end
    
    output_node = Node(output_val, [layer.weights], _backward)
    return output_node
end

get_params(layer::Embedding) = [layer.weights]


struct Flatten <: Layer end
function forward(layer::Flatten, x::Node)
    val = value(x)
    batch_size = size(val)[end]
    flattened_val = reshape(val, :, batch_size)'

    local output_node
    function _backward()
        grad_out = grad(output_node)
        reshaped_grad = reshape(grad_out', size(val))
        accumulate_gradient!(x, reshaped_grad)
    end
    output_node = Node(flattened_val, [x], _backward)
    return output_node
end
get_params(layer::Flatten) = []

struct Permute <: Layer
    dims::Tuple
end
function forward(layer::Permute, x::Node)
    val = value(x)
    permuted_val = permutedims(val, layer.dims)
    inv_dims = invperm(collect(layer.dims))

    local output_node
    function _backward()
        grad_out = grad(output_node)
        unpermuted_grad = permutedims(grad_out, inv_dims)
        accumulate_gradient!(x, unpermuted_grad)
    end
    output_node = Node(permuted_val, [x], _backward)
    return output_node
end
get_params(layer::Permute) = []

end
