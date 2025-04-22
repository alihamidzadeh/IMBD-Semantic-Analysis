// cls; g++ .\1.cpp -o .\1.exe; .\1.exe

//1
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

//2
#include <set>
#include <map>
#include <regex>

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



// Modified CSV loader
vector<Review> load_csv(const string& filename) {
    vector<Review> reviews;
    ifstream file(filename);
    string line;

    getline(file, line); // Skip header if exists

    while (getline(file, line)) {
        size_t comma_pos = line.rfind(','); // Find last comma
        if (comma_pos != string::npos) {
            string review_text = line.substr(0, comma_pos);
            string sentiment = line.substr(comma_pos + 1);

            // Remove quotes if present
            if (!review_text.empty() && review_text.front() == '"')
                review_text = review_text.substr(1, review_text.size() - 2);
            if (!sentiment.empty() && sentiment.front() == '"')
                sentiment = sentiment.substr(1, sentiment.size() - 2);

            Review review;
            review.text = clean_text(review_text);
            review.sentiment = sentiment;
            reviews.push_back(review);
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

// Function to build vocabulary from all reviews
set<string> build_vocabulary(const vector<Review>& reviews) {
    set<string> vocab;

    for (const auto& review : reviews) {
        vector<string> tokens = tokenize(review.text);
        for (const auto& word : tokens) {
            // Filter by length, repetition, and stopword check
            if (
                word.length() >= 3 &&
                !regex_match(word, regex("([a-z])\\1{2,}")) &&
                get_stopwords().find(word) == get_stopwords().end()
            ) {
                vocab.insert(word);
            }
        }
    }

    return vocab;
}


//3

map<string, double> compute_tf(const vector<string>& tokens) {
    map<string, double> tf;
    int total_words = tokens.size();

    for (const string& word : tokens) {
        tf[word] += 1.0;
    }

    // Normalize: divide by total word count
    for (auto& pair : tf) {
        pair.second /= total_words;
    }

    return tf;
}

#include <cmath>

map<string, double> compute_idf(const vector<vector<string>>& all_tokens, const set<string>& vocabulary) {
    map<string, double> idf;
    int N = all_tokens.size();

    for (const auto& word : vocabulary) {
        int doc_count = 0;
        for (const auto& tokens : all_tokens) {
            if (find(tokens.begin(), tokens.end(), word) != tokens.end()) {
                doc_count++;
            }
        }
        if (doc_count > 0) {
            idf[word] = log((double)N / doc_count);
        }
    }

    return idf;
}

map<string, double> compute_tfidf(const map<string, double>& tf, const map<string, double>& idf) {
    map<string, double> tfidf;

    for (const auto& [word, tf_value] : tf) {
        auto it = idf.find(word);
        if (it != idf.end()) {
            tfidf[word] = tf_value * it->second;
        }
    }

    return tfidf;
}

#include <unordered_map>
#include <random>

// Sigmoid function
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// Predict probability for a single example
double predict_prob(const unordered_map<string, double>& features, const map<string, double>& weights) {
    double z = 0.0;
    for (const auto& [word, value] : features) {
        auto it = weights.find(word);
        if (it != weights.end()) {
            z += value * it->second;
        }
    }
    return sigmoid(z);
}

// Train logistic regression with SGD
map<string, double> train_logistic_regression(
    const vector<unordered_map<string, double>>& X, // list of tf-idf maps per review
    const vector<int>& y, // labels: 0 or 1
    int epochs = 10,
    double lr = 0.1
) {
    map<string, double> weights;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < X.size(); ++i) {
            double pred = predict_prob(X[i], weights);
            double error = y[i] - pred;

            for (const auto& [word, value] : X[i]) {
                weights[word] += lr * error * value;
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


int main() {
    // string filename = "IMDB_Dataset.csv";
    string filename = "Minimal_IMDB_Dataset.csv";
    vector<Review> reviews = load_csv(filename);

    cout << "Number of reviews: " << reviews.size() << endl;

    set<string> stopwords = get_stopwords();

    // Step 1: Tokenize and filter all reviews
    vector<vector<string>> all_tokens;
    for (const auto& review : reviews) {
        vector<string> raw_tokens = tokenize(review.text);
        vector<string> filtered_tokens;
        for (const auto& word : raw_tokens) {
            if (
                word.length() >= 3 &&
                !regex_match(word, regex("([a-z])\\1{2,}")) &&
                stopwords.find(word) == stopwords.end()
            ) {
                filtered_tokens.push_back(word);
            }
        }
        all_tokens.push_back(filtered_tokens);
    }

    // Step 2: Build vocabulary (from all filtered tokens)
    set<string> vocabulary;
    for (const auto& tokens : all_tokens) {
        vocabulary.insert(tokens.begin(), tokens.end());
    }

    // Step 3: Compute IDF
    map<string, double> idf = compute_idf(all_tokens, vocabulary);

    // Step 4: Compute and show TF-IDF for first review
    // for (int i = 0; i < 1 && i < all_tokens.size(); ++i) {
    //     map<string, double> tf = compute_tf(all_tokens[i]);
    //     map<string, double> tfidf = compute_tfidf(tf, idf);

    //     cout << "\nTF-IDF for review #" << i + 1 << ":\n";
    //     int count = 0;
    //     for (const auto& [word, score] : tfidf) {
    //         // if (count++ >= 10) break;
    //         cout << word << ": " << score << "\n";
    //     }
    // }

    // Step 5: Build TF-IDF vector (X) and label vector (y)
    vector<unordered_map<string, double>> X;
    vector<int> y;

    for (size_t i = 0; i < reviews.size(); ++i) {
        const auto& tokens = all_tokens[i];
        map<string, double> tf = compute_tf(tokens);
        map<string, double> tfidf = compute_tfidf(tf, idf);

        // Convert tfidf map (std::map) to unordered_map (lighter and faster lookup)
        unordered_map<string, double> tfidf_vec(tfidf.begin(), tfidf.end());
        X.push_back(tfidf_vec);

        // Convert sentiment to label: positive → 1, negative → 0
        y.push_back(reviews[i].sentiment == "positive" ? 1 : 0);
    }

    map<string, double> weights = train_logistic_regression(X, y);

    cout << "---------------";

    int predicted = predict_label(X[0], weights);
    cout << "\nPredicted sentiment for review #1: " << (predicted == 1 ? "positive" : "negative") << endl;
    cout << "Actual sentiment: " << reviews[0].sentiment << endl;

    return 0;
}