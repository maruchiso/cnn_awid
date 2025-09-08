include("./data.jl")
using .Data

using Flux
using Printf, Random, Statistics
using Flux: Losses, Optimisers

const LEARNING_RATE = 0.001f0
const EPOCHS = 5
const BATCH_SIZE = 64
const EMBEDDING_DIM = 50
const KERNEL_WIDTH = 3
const OUT_CHANNELS = 8
const POOL_SIZE = 8

X_train, y_train, X_test, y_test, vocab = get_imdb_data()
VOCAB_SIZE = length(vocab)

y_train = Float32.(reshape(y_train, 1, :))
y_test = Float32.(reshape(y_test, 1, :))

const DENSE_IN_FEATURES = 12 * OUT_CHANNELS

model = Chain(
    Embedding(VOCAB_SIZE, EMBEDDING_DIM),
    x -> permutedims(x, (2, 1, 3)),
    Conv((KERNEL_WIDTH,), EMBEDDING_DIM => OUT_CHANNELS, relu),
    MaxPool((POOL_SIZE,)),
    Flux.flatten,
    Dense(DENSE_IN_FEATURES, 1, sigmoid),
)

loss_function(m, x, y) = Losses.binarycrossentropy(m(x), y)
optimizer = Optimisers.setup(Optimisers.Adam(LEARNING_RATE), model)

num_samples = size(X_train, 1)
num_batches = ceil(Int, num_samples / BATCH_SIZE)

for epoch in 1:EPOCHS
    shuffled_indices = shuffle(1:num_samples)
    total_loss = 0.0
    start_time = time()

    for i in 1:num_batches
        start_idx = (i - 1) * BATCH_SIZE + 1
        end_idx = min(i * BATCH_SIZE, num_samples)
        batch_indices = shuffled_indices[start_idx:end_idx]

        x_batch = X_train[batch_indices, :]'
        y_batch = y_train[:, batch_indices]

        loss, grads = Flux.withgradient(model) do m
            loss_function(m, x_batch, y_batch)
        end
        
        Optimisers.update!(optimizer, model, grads[1])

        total_loss += loss

        if i % 50 == 0
            @printf("Epoch %d, Batch %d/%d, Avg loss: %.4f\n", epoch, i, num_batches, total_loss / i)
        end
    end

    y_test_pred_vals = model(X_test')
    preds = y_test_pred_vals .> 0.5f0
    test_accuracy = mean(preds .== y_test) * 100

    elapsed_time = time() - start_time
    @printf("\nEpoch %d ended in %.2fs. Avg training loss: %.4f, Accuracy on test set: %.2f%%\n\n", epoch, elapsed_time, total_loss / num_batches, test_accuracy)
end