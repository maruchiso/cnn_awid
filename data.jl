module Data
using MLDatasets, Random, LinearAlgebra, JLD2, WordTokenizers, StatsBase
using JLD2: load, @load, @save

export get_mnist_loader

function onehot(label::Int, num_classes::Int=10)
    output = zeros(Float32, 1, num_classes)
    output[label + 1] = 1.0f0
    return output
end

function get_mnist_loader(; batch_size::Int=64, split::Symbol=:train)
    mnist_data = MLDatasets.MNIST(split=split)
    flat_x = Float32.(reshape(mnist_data.features, 28^2, :))
    onehot_y = vcat([onehot(y) for y in mnist_data.targets]...)'
    num_samples = size(flat_x, 2)
    indices = shuffle(1:num_samples)

    batches = []
    for i in 1:batch_size:num_samples
        batch_indices = indices[i:min(i + batch_size - 1, num_samples)]
        
        batch_x = permutedims(flat_x[:, batch_indices])
        batch_y = permutedims(onehot_y[:, batch_indices])
        
        push!(batches, (batch_x, batch_y))
    end
    
    return batches
end

export get_imdb_data

const VOCAB_SIZE = 2000
const SENTENCE_LENGTH = 100

function get_imdb_data(;force_preprocess=false)
    data_dir = "data"
    processed_file = joinpath(data_dir, "imdb_processed.jld2")
    raw_file = joinpath(data_dir, "imdb_dataset.jld2")

    if !force_preprocess && isfile(processed_file)
        println("Loading preprocessed data from $processed_file...")
        @load processed_file X_train y_train X_test y_test vocab
    else
        println("Preprocessing IMDb data from local file... (this may take a while the first time)")
        
        if !isfile(raw_file)
            error("Raw data file not found at $raw_file. Make sure imdb_dataset.jld2 is in the 'data' subfolder.")
        end
        reviews_all = load(raw_file, "reviews")
        labels_all = load(raw_file, "labels")
        
        # shuffle data
        indices = shuffle(1:length(labels_all))
        reviews_all = reviews_all[indices]
        labels_all = labels_all[indices]

        # data split 80/20
        split_ratio = 0.8
        split_idx = floor(Int, split_ratio * length(labels_all))
        
        train_reviews = reviews_all[1:split_idx]
        train_labels = labels_all[1:split_idx]
        
        test_reviews = reviews_all[split_idx+1:end]
        test_labels = labels_all[split_idx+1:end]
        
        # Dictionary creation onlt from training data
        tokenized_reviews = tokenize.(lowercase.(train_reviews))
        word_counts = countmap(vcat(tokenized_reviews...))
        
        sorted_words = sort(collect(pairs(word_counts)), by=x->x[2], rev=true)
        vocab_list = [word for (word, count) in sorted_words[1:min(end, VOCAB_SIZE - 2)]]
        
        vocab = ["<PAD>", "<UNK>"]
        append!(vocab, vocab_list)
        word_to_ix = Dict(word => i for (i, word) in enumerate(vocab))
        
        function process_reviews(reviews, sentiments, word_map)
            tokenized = tokenize.(lowercase.(reviews))
            
            X = ones(Int, length(reviews), SENTENCE_LENGTH)
            for (i, doc) in enumerate(tokenized)
                for (j, token) in enumerate(doc)
                    if j > SENTENCE_LENGTH break end
                    X[i, j] = get(word_map, token, 2) # 2 to indeks <UNK>
                end
            end
            
            Y = reshape(Float32.(sentiments), :, 1)
            return X, Y
        end

        X_train, y_train = process_reviews(train_reviews, train_labels, word_to_ix)
        X_test, y_test = process_reviews(test_reviews, test_labels, word_to_ix)
        
        println("Saving preprocessed data to $processed_file...")
        if !isdir(data_dir)
            mkdir(data_dir)
        end
        @save processed_file X_train y_train X_test y_test vocab
    end

    return X_train, y_train, X_test, y_test, vocab
end

end
