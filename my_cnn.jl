include("./autodiff/autodiff.jl")
include("./Layers_NN.jl")
include("./Layers_CNN.jl")
include("./loss_functions.jl")
include("./optimizer.jl")
include("./data.jl")

using .AutoDiff, .Layers_NN, .Layers_CNN, .LossFunctions, .Optimizers, .Data
using Printf, Random, Statistics

const LEARNING_RATE = 0.001f0
const EPOCHS = 5
const BATCH_SIZE = 64
const EMBEDDING_DIM = 50
const KERNEL_WIDTH = 3
const OUT_CHANNELS = 8
const POOL_SIZE = 8
const PARAM_DTYPE = Float32

# Load data
X_train, y_train, X_test, y_test, vocab = get_imdb_data()
VOCAB_SIZE = length(vocab)

# Calculate the input feature size for the Dense layer
DENSE_IN_FEATURES = 12 * OUT_CHANNELS

# Model definition
model = Model(
    Embedding(VOCAB_SIZE, EMBEDDING_DIM),
    Permute((2, 3, 1)),
    Conv1D(EMBEDDING_DIM, OUT_CHANNELS, KERNEL_WIDTH, activation=relu),
    MaxPool1D(POOL_SIZE),
    Flatten(),
    Dense(DENSE_IN_FEATURES, 1, sigmoid),
)

params = get_params(model)
typed_params = Vector{Node{PARAM_DTYPE}}(params)
optimizer = Adam(typed_params; lr=LEARNING_RATE)


num_samples = size(X_train, 1)
num_batches = ceil(Int, num_samples / BATCH_SIZE)

for epoch in 1:EPOCHS
    shuffled_indices = shuffle(1:num_samples)

    total_loss = 0.0
    start_time = time()

    for i in 1:num_batches
        start_idx = (i-1) * BATCH_SIZE + 1
        end_idx = min(i * BATCH_SIZE, num_samples)
        batch_indices = shuffled_indices[start_idx:end_idx]

        x_batch = X_train[batch_indices, :]
        y_batch = y_train[batch_indices, :]

        # Forward pass
        y_pred = model(Node(x_batch))

        # Calculate loss
        loss = binary_cross_entropy(y_pred, y_batch)
        total_loss += value(loss)

        # Backward pass and optimization
        zero_grad!(params)
        backward!(loss)
        update!(optimizer)

        if i % 50 == 0
            @printf("Epoch %d, Batch %d/%d, Avg loss: %.4f\n", epoch, i, num_batches, total_loss / i)
        end
    end

    # Predictions on the test set
    y_test_pred_vals = value(model(Node(X_test)))
    preds = y_test_pred_vals .> 0.5

    # Calculate accuracy
    test_accuracy = mean(preds .== y_test) * 100

    elapsed_time = time() - start_time
    @printf("\nEpoch %d ended in %.2fs. Avg training loss: %.4f, Accuracy on test set: %.2f%%\n\n", epoch, elapsed_time, total_loss / num_batches, test_accuracy)
end

