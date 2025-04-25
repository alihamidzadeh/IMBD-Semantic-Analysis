// Parallel version of IMDB sentiment analysis using OpenMP
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <set>
#include <map>
#include <regex>
#include <cmath>
#include <unordered_map>
#include <random>
#include <omp.h>

using namespace std;

// Structure to hold review text and sentiment label
struct Review {
    string text;
    string sentiment;
};

// define stopwords globally to reuse
set<string> get_stopwords() {
    return {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself", "she", "her", "hers", "herself",
        "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing",
        "a", "an", "the", "and", "but", "if", "or", "because", "as",
        "until", "while", "of", "at", "by", "for", "with", "about", "against",
        "between", "into", "through", "during", "before", "after", "above", "below",
        "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
        "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "any", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "can", "will", "just", "don", "should", "now"
    };
}

string clean_text(const string& text) {
    string cleaned = text;

    // Convert to lowercase
    transform(cleaned.begin(), cleaned.end(), cleaned.begin(), ::tolower);

    // Remove HTML tags like <br />
    cleaned = regex_replace(cleaned, regex("<[^>]+>"), " ");

    // Remove numbers
    cleaned = regex_replace(cleaned, regex("[0-9]+"), " ");

    // Remove punctuation
    cleaned.erase(remove_if(cleaned.begin(), cleaned.end(),
        [](unsigned char c) {
            return ispunct(c);
        }), cleaned.end());

    // Remove extra spaces
    cleaned = regex_replace(cleaned, regex("\\s+"), " ");

    return cleaned;
}

// Parallel CSV loader
vector<Review> load_csv(const string& filename) {
    vector<Review> reviews;
    ifstream file(filename);
    string line;
    vector<string> lines;

    // First, read all lines
    getline(file, line); // Skip header
    while (getline(file, line)) {
        lines.push_back(line);
    }

    // Parallel processing of lines
    reviews.resize(lines.size());
    #pragma omp parallel for
    for (size_t i = 0; i < lines.size(); ++i) {
        size_t comma_pos = lines[i].rfind(',');
        if (comma_pos != string::npos) {
            string review_text = lines[i].substr(0, comma_pos);
            string sentiment = lines[i].substr(comma_pos + 1);

            // Remove quotes if present
            if (!review_text.empty() && review_text.front() == '"')
                review_text = review_text.substr(1, review_text.size() - 2);
            if (!sentiment.empty() && sentiment.front() == '"')
                sentiment = sentiment.substr(1, sentiment.size() - 2);

            reviews[i].text = clean_text(review_text);
            reviews[i].sentiment = sentiment;
        }
    }

    return reviews;
}

// Function to tokenize a string by space
vector<string> tokenize(const string& text) {
    stringstream ss(text);
    string word;
    vector<string> tokens;

    while (ss >> word) {
        tokens.push_back(word);
    }

    return tokens;
}

// Parallel function to build vocabulary
set<string> build_vocabulary(const vector<Review>& reviews) {
    set<string> vocab;
    #pragma omp parallel
    {
        set<string> local_vocab;
        #pragma omp for nowait
        for (size_t i = 0; i < reviews.size(); ++i) {
            vector<string> tokens = tokenize(reviews[i].text);
            for (const auto& word : tokens) {
                if (word.length() >= 3 &&
                    !regex_match(word, regex("([a-z])\\1{2,}")) &&
                    get_stopwords().find(word) == get_stopwords().end()) {
                    local_vocab.insert(word);
                }
            }
        }
        #pragma omp critical
        vocab.insert(local_vocab.begin(), local_vocab.end());
    }
    return vocab;
}

// Parallel TF computation
map<string, double> compute_tf(const vector<string>& tokens) {
    map<string, double> tf;
    int total_words = tokens.size();

    // Count words in parallel
    #pragma omp parallel
    {
        map<string, double> local_tf;
        #pragma omp for nowait
        for (size_t i = 0; i < tokens.size(); ++i) {
            local_tf[tokens[i]] += 1.0;
        }
        #pragma omp critical
        {
            for (const auto& [word, count] : local_tf) {
                tf[word] += count;
            }
        }
    }

    // Normalize
    for (auto& [word, count] : tf) {
        count /= total_words;
    }

    return tf;
}

// Parallel IDF computation
map<string, double> compute_idf(const vector<vector<string>>& all_tokens, const set<string>& vocabulary) {
    map<string, double> idf;
    int N = all_tokens.size();

    #pragma omp parallel
    {
        map<string, double> local_idf;
        vector<string> vocab_vec(vocabulary.begin(), vocabulary.end());
        
        #pragma omp for nowait
        for (size_t i = 0; i < vocab_vec.size(); ++i) {
            const string& word = vocab_vec[i];
            int doc_count = 0;
            for (const auto& tokens : all_tokens) {
                if (find(tokens.begin(), tokens.end(), word) != tokens.end()) {
                    doc_count++;
                }
            }
            if (doc_count > 0) {
                local_idf[word] = log((double)N / doc_count);
            }
        }
        #pragma omp critical
        idf.insert(local_idf.begin(), local_idf.end());
    }

    return idf;
}

// Parallel TF-IDF computation
map<string, double> compute_tfidf(const map<string, double>& tf, const map<string, double>& idf) {
    map<string, double> tfidf;
    vector<pair<string, double>> tf_vec(tf.begin(), tf.end());

    #pragma omp parallel
    {
        map<string, double> local_tfidf;
        #pragma omp for nowait
        for (size_t i = 0; i < tf_vec.size(); ++i) {
            const auto& [word, tf_value] = tf_vec[i];
            auto idf_it = idf.find(word);
            if (idf_it != idf.end()) {
                local_tfidf[word] = tf_value * idf_it->second;
            }
        }
        #pragma omp critical
        tfidf.insert(local_tfidf.begin(), local_tfidf.end());
    }

    return tfidf;
}

// Sigmoid function
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// Predict probability for a single example
double predict_prob(const unordered_map<string, double>& features, const map<string, double>& weights) {
    double z = 0.0;
    vector<pair<string, double>> features_vec(features.begin(), features.end());
    
    #pragma omp parallel for reduction(+:z)
    for (size_t i = 0; i < features_vec.size(); ++i) {
        const auto& [word, value] = features_vec[i];
        auto it = weights.find(word);
        if (it != weights.end()) {
            z += value * it->second;
        }
    }
    return sigmoid(z);
}

// Parallel training of logistic regression
map<string, double> train_logistic_regression(
    const vector<unordered_map<string, double>>& X,
    const vector<int>& y,
    int epochs = 10,
    double lr = 0.1
) {
    map<string, double> weights;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        #pragma omp parallel
        {
            map<string, double> local_weights;
            #pragma omp for nowait
            for (size_t i = 0; i < X.size(); ++i) {
                double pred = predict_prob(X[i], weights);
                double error = y[i] - pred;

                for (const auto& [word, value] : X[i]) {
                    local_weights[word] += lr * error * value;
                }
            }
            #pragma omp critical
            {
                for (const auto& [word, value] : local_weights) {
                    weights[word] += value;
                }
            }
        }
        cout << "Epoch " << epoch + 1 << " done\n";
    }

    return weights;
}

// Predict label: 0 or 1
int predict_label(const unordered_map<string, double>& features, const map<string, double>& weights) {
    return predict_prob(features, weights) >= 0.5 ? 1 : 0;
}

// Parallel accuracy calculation
double calculate_accuracy(
    const vector<unordered_map<string, double>>& X,
    const vector<int>& y,
    const map<string, double>& weights
) {
    int correct = 0;
    #pragma omp parallel for reduction(+:correct)
    for (size_t i = 0; i < X.size(); ++i) {
        int predicted = predict_label(X[i], weights);
        if (predicted == y[i]) {
            correct++;
        }
    }
    return (double)correct / X.size() * 100.0;
}

int main() {
    // Set number of threads
    omp_set_num_threads(omp_get_max_threads());
    cout << "Using " << omp_get_max_threads() << " threads\n";

    // Start timing
    auto start_time = chrono::high_resolution_clock::now();

    // string filename = "Minimal_IMDB_Dataset.csv";
    string filename = "Original_IMDB_Dataset.csv";
    vector<Review> reviews = load_csv(filename);

    cout << "Number of reviews: " << reviews.size() << endl;

    // Step 1: Tokenize and filter all reviews
    vector<vector<string>> all_tokens(reviews.size());
    #pragma omp parallel for
    for (size_t i = 0; i < reviews.size(); ++i) {
        all_tokens[i] = tokenize(reviews[i].text);
    }

    // Step 2: Build vocabulary
    set<string> vocabulary = build_vocabulary(reviews);
    cout << "Vocabulary size: " << vocabulary.size() << endl;

    // Step 3: Compute IDF
    map<string, double> idf = compute_idf(all_tokens, vocabulary);

    // Step 4: Convert reviews to TF-IDF features
    vector<unordered_map<string, double>> X(reviews.size());
    vector<int> y(reviews.size());
    #pragma omp parallel for
    for (size_t i = 0; i < reviews.size(); ++i) {
        map<string, double> tf = compute_tf(all_tokens[i]);
        map<string, double> tfidf = compute_tfidf(tf, idf);
        X[i] = unordered_map<string, double>(tfidf.begin(), tfidf.end());
        y[i] = reviews[i].sentiment == "positive" ? 1 : 0;
    }

    // Step 5: Train the model
    cout << "\nTraining the model..." << endl;
    map<string, double> weights = train_logistic_regression(X, y);

    // Calculate and display accuracy
    double accuracy = calculate_accuracy(X, y, weights);
    cout << "\nTraining Accuracy: " << accuracy << "%" << endl;

    // End timing
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    
    cout << "\nTotal processing time: " << duration.count() << " milliseconds" << endl;
    cout << "Total processing time: " << duration.count() / 1000.0 << " seconds" << endl;

    return 0;
} 