#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;
// Structure to hold review text and sentiment label
struct Review {
    string text;
    string sentiment;
};

// Function to clean text: lowercase and remove punctuation
string clean_text(const string& text) {
    string cleaned = text;
    transform(cleaned.begin(), cleaned.end(), cleaned.begin(), ::tolower);
    cleaned.erase(remove_if(cleaned.begin(), cleaned.end(),
        [](unsigned char c) { return ispunct(c); }), cleaned.end());
    return cleaned;
}

// Function to load CSV file
vector<Review> load_csv(const string& filename) {
    vector<Review> reviews;
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        string review_text, sentiment;

        if (getline(ss, review_text, ',') && getline(ss, sentiment)) {
            Review review;
            review.text = clean_text(review_text);
            review.sentiment = sentiment;
            reviews.push_back(review);
        }
    }
    return reviews;
}

int main() {
    string filename = "IMDB_Dataset.csv";
    vector<Review> reviews = load_csv(filename);

    cout << "Number of reviews: " << reviews.size() << endl;

    for (size_t i = 1; i <= 5 && i < reviews.size(); ++i) {
        cout << "Review: " << reviews[i].text << endl;
        cout << "Sentiment: " << reviews[i].sentiment << endl;
        cout << "-----------------------------" << endl;
    }

    return 0;
}
