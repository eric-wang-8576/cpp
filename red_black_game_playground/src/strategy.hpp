class Strategy {
public:
    Game* game;
    int numRed;
    int numBlack;

    Strategy(Game* g) : 
        game(g), 
        numRed(32), 
        numBlack(32) {}

    void play();
    void random();
};
