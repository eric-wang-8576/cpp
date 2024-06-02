struct TrieNode {
    std::array<std::unique_ptr<TrieNode>, 26> children;
    bool isEnd;
    TrieNode() : isEnd(false) {}
};

class Trie {
private:
    std::unique_ptr<TrieNode> root;
    
public:
    Trie() : root(std::make_unique<TrieNode>()) {}
    
    void insert(string word) {
        TrieNode* curr = root.get();
        for (char c : word) {
            int idx = c - 'a';
            if (curr->children[idx] == nullptr) {
                curr->children[idx] = std::make_unique<TrieNode>();
            }
            curr = curr->children[idx].get();
        }
        curr->isEnd = true;
    }
    
    bool search(string word) {
        TrieNode* curr = root.get();
        for (char c : word) {
            int idx = c - 'a';
            if (curr->children[idx] == nullptr) {
                return false;
            }
            curr = curr->children[idx].get();
        }
        return curr->isEnd;
    }
    
    bool startsWith(string prefix) {
        TrieNode* curr = root.get();
        for (char c : prefix) {
            int idx = c - 'a';
            if (curr->children[idx] == nullptr) {
                return false;
            }
            curr = curr->children[idx].get();
        }
        return true;
    }
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */
