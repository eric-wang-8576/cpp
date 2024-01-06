class Strategy {
public:
    Game* game;
    int numRed;
    int numBlack;

    Strategy(Game* g) : 
        game(g), 
        numRed(26), 
        numBlack(26) {}

    void play();
    void random();
};
