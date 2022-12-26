struct Node {
    int key;
    int value;
    Node* prev;
    Node* next;
    Node(int key, int value) : key(key), value(value) {}
};

class LRUCache {
public:
    LRUCache(int capacity): cacheCapacity(capacity) {
        numNodes = 0;
        head = new Node(0, 0);
        tail = new Node(0, 0);
        head->next = tail;
        tail->prev = head;
    }
    
    int get(int key) {
        const auto it = keyToNode.find(key);
        if (it == keyToNode.cend()) {
            return -1;
        } else {
            Node* node = it->second;
            moveToFront(node);
            return node->value;
        }
    }
    
    void put(int key, int value) {
        const auto it = keyToNode.find(key);
        if (it == keyToNode.cend()) { // not in the cache
            
            Node* newNode = new Node(key, value);
        
            if (keyToNode.size() == cacheCapacity) { // evict
                keyToNode.erase(popNode(tail->prev));
            }
            
            keyToNode.insert({key, newNode});
            addToFront(newNode);
        } else { // already in the cache
            Node* node = it->second;
            node->value = value;
            moveToFront(node);
        }
        
    }
private:
    const int cacheCapacity;
    int numNodes;
    unordered_map<int, Node*> keyToNode;
    Node* head;
    Node* tail;
    
    void addToFront(Node* node) {
        Node* orig_right = head->next;
        head->next = node;
        node->prev = head;
        node->next = orig_right;
        orig_right->prev = node;
    }
    
    int popNode(Node* node) {
        Node* orig_left = node->prev;
        Node* orig_right = node->next;
        orig_left->next = orig_right;
        orig_right->prev = orig_left;
        return node->key;
    }
    
    void moveToFront(Node* node) {
        popNode(node);
        addToFront(node);
    }
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */