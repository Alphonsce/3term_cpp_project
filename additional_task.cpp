#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

std::vector<std::string> generate_ngrams(std::string w, size_t n) {
    std::vector<std::string> ngrams;

    for (int i = 0; i <= w.length() - n; ++i)
    {
        ngrams.push_back(w.substr(i, n));
    }

    return ngrams; 
}

int main() {
    std::string w = "heloloworld";
    size_t n = 2;
    std::vector<std::string> ngrams = generate_ngrams(w, n);
    std::vector<std::string> printed;

    for (int i = 0; i < ngrams.size(); i++) {

        int occurrences = 0;
        std::string::size_type pos = 0;
        while ((pos = w.find(ngrams[i], pos)) != std::string::npos) {
            occurrences += 1;
            pos += ngrams[i].length();
        }
        if (std::find(printed.begin(), printed.end(), ngrams[i]) == printed.end()) {
            if (std::find(ngrams[i].begin(), ngrams[i].end(), ' ') == ngrams[i].end())
                std::cout << ngrams[i] << ": " << occurrences << std::endl;
            printed.push_back(ngrams[i]);
        }
    }
    return 0;
}