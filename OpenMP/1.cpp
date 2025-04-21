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

// // Function to clean text: lowercase and remove punctuation
// string clean_text(const string& text) {
//     string cleaned = text;
//     transform(cleaned.begin(), cleaned.end(), cleaned.begin(), ::tolower);
//     cleaned.erase(remove_if(cleaned.begin(), cleaned.end(),
//         [](unsigned char c) { return ispunct(c); }), cleaned.end());
//     return cleaned;
// }

// // Function to load CSV file
// vector<Review> load_csv(const string& filename) {
//     vector<Review> reviews;
//     ifstream file(filename);
//     string line;

//     while (getline(file, line)) {
//         stringstream ss(line);
//         string review_text, sentiment;

//         if (getline(ss, review_text, ',') && getline(ss, sentiment)) {
//             Review review;
//             review.text = clean_text(review_text);
//             review.sentiment = sentiment;
//             reviews.push_back(review);
//         }
//     }
//     return reviews;
// }

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

    // Define a simple set of English stopwords
    set<string> stopwords = {
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

    for (const auto& review : reviews) {
        vector<string> tokens = tokenize(review.text);
        for (const auto& word : tokens) {
            // Filter by length, repetition, and stopword check
            if (
                word.length() >= 3 &&
                !regex_match(word, regex("([a-z])\\1{2,}")) &&
                stopwords.find(word) == stopwords.end()
            ) {
                vocab.insert(word);
            }
        }
    }

    return vocab;
}

int main() {
    // string filename = "IMDB_Dataset.csv";
    string filename = "Minimal_IMDB_Dataset.csv";
    vector<Review> reviews = load_csv(filename);

    cout << "Number of reviews: " << reviews.size() << endl;

    // Tokenization + Vocabulary
    set<string> vocabulary = build_vocabulary(reviews);
    cout << "Vocabulary size: " << vocabulary.size() << endl;

    // Optional: print first 10 words from vocabulary
    int count = 0;
    for (const auto& word : vocabulary) {
        if (count++ >= 10) break;
        cout << word << " ";
    }
    cout << "\n";

    return 0;
}