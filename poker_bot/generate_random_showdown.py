import random

values = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "j", "q", "k", "a"]
suits = ["c", "d", "h", "s"]

cards = []
for value in values:
    for suit in suits:
        cards.append(value + suit)
random.shuffle(cards)

num_boards = 4
num_people = 8
people = ["Wang", "Josh", "Jeff", "Brian", "Zach", "Chen", "Jason", "Allen"]
people.sort()

print("PLO")
print(num_boards)
for board in range(num_boards):
    to_print = ""
    for offset in range(0, 5):
        ind = board*5 + offset
        to_print += cards[ind] + " "
    print(to_print)
print(num_people)
for person in range(len(people)):
    to_print = people[person] + " " + str(round(random.uniform(50, 150), 2)) + " "
    total_offset = num_boards*5 + person*4
    for offset in range(0, 4):
        ind = total_offset + offset
        to_print += cards[ind] + " "
    print(to_print)