#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

// Structure to hold review text and sentiment label
struct Review {
    std::string text;
    std::string sentiment;
};

// Function to clean text: lowercase and remove punctuation
std::string clean_text(const std::string& text) {
    std::string cleaned = text;
    std::transform(cleaned.begin(), cleaned.end(), cleaned.begin(), ::tolower);
    cleaned.erase(std::remove_if(cleaned.begin(), cleaned.end(),
        [](unsigned char c) { return std::ispunct(c); }), cleaned.end());
    return cleaned;
}

// Function to load CSV file
std::vector<Review> load_csv(const std::string& filename) {
    std::vector<Review> reviews;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string review_text, sentiment;

        if (std::getline(ss, review_text, ',') && std::getline(ss, sentiment)) {
            Review review;
            review.text = clean_text(review_text);
            review.sentiment = sentiment;
            reviews.push_back(review);
        }
    }
    return reviews;
}

int main() {
    std::string filename = "IMDB_Dataset.csv";
    std::vector<Review> reviews = load_csv(filename);

    std::cout << "Number of reviews: " << reviews.size() << std::endl;

    for (size_t i = 1; i <= 5 && i < reviews.size(); ++i) {
        std::cout << "Review: " << reviews[i].text << std::endl;
        std::cout << "Sentiment: " << reviews[i].sentiment << std::endl;
        std::cout << "-----------------------------" << std::endl;
    }

    return 0;
}
