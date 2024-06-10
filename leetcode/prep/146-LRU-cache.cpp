struct Node {
    Node* left;
    Node* right;
    int key;
    int val;
};

class LRUCache {
private:
    Node* leftS;
    Node* rightS;
    const int maxKeys;
    int numKeys;

    // Maps the key to the node in the linked list of nodes
    // A key is present iff it is in the linked list
    std::unordered_map<int, Node*> keyToNode;

    // Only operate on linked list
    void popNode(Node* nodePtr) {
        Node* leftN = nodePtr->left;
        Node* rightN = nodePtr->right;
        leftN->right = rightN;
        rightN->left = leftN;
    }

    // Only operate on linked list
    void addToFront(Node* nodePtr) {
        Node* rightN = leftS->right;

        leftS->right = nodePtr;
        nodePtr->left = leftS;

        rightN->left = nodePtr;
        nodePtr->right = rightN;
    }
    
public:
    LRUCache(int capacity) : maxKeys(capacity) {
        numKeys = 0;
        leftS = new Node();
        rightS = new Node();
        leftS->right = rightS;
        rightS->left = leftS;
    }
    
    // Move element to most recent location
    int get(int key) {
        if (keyToNode.find(key) == keyToNode.end()) {
            return -1;
        }

        Node* nodePtr = keyToNode[key];
        popNode(nodePtr);
        addToFront(nodePtr);
        return nodePtr->val;
    }
    
    // Add element to the start, evict if necessary 
    void put(int key, int value) {
        // If it's already there, just update the value
        if (keyToNode.find(key) != keyToNode.end()) {
            Node* nodePtr = keyToNode[key];
            nodePtr->val = value;
            popNode(nodePtr);
            addToFront(nodePtr);
        } else {
            numKeys++;
            if (numKeys > maxKeys) {
                // Evict
                numKeys--;
                Node* lastN = rightS->left;
                popNode(lastN);
                keyToNode.erase(lastN->key);
            }
            Node* newN = new Node();
            newN->key = key;
            newN->val = value;
            keyToNode[key] = newN;
            addToFront(newN);
        }
    }


    void dump() {
        Node* curr = leftS->right;
        std::cout << "[ ";
        while (curr != rightS) {
            std::cout << "key: " << curr->key << " value: " << curr->val << "\t";
            curr = curr->right;
        }
        std::cout << " ]" << std::endl;
    }
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */
