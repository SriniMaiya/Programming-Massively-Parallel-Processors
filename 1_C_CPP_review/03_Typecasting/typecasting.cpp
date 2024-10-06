#include <iostream>

class Entity
{
public:
    virtual void PrintName() {}
};

class Player : public Entity
{
public:
    void PrintName() {}
};

class Enemy : public Entity
{
public:
    void PrintName() {}
};

int main()
{
    // Dynamic casting.
    Player *player = new Player();
    Entity *actualEnemy = new Enemy();

    Entity *actualPlayer = player;

    Player *p0 = dynamic_cast<Player *>(actualPlayer);
    Player *p1 = dynamic_cast<Player *>(actualEnemy);

    if (p0)
    {
        std::cout << "Succeded in typecasting Entity type (Actually player) to Player dynamically" << std::endl;
    }
    if (!p1)
    {
        std::cout << "Failed in typecasting Entity type (Actually Enemy!!) to Player dynamically" << std::endl;
    }
}