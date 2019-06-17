import numpy as np


def menu():
    new_case = [''] * 7

    # Introduction
    print('\n\033[1m'+'WELCOME TO THE PLAYLIST GENERATOR. '+'\033[0m'+'\nPlease, answer to the following questions:')

    # Age feature
    new_case[0] = int(input('*\033[1mIntroduce you Age:\033[0m '))
    var1 = input('*Are you Male or Female?\n   0 -Male\n   1 -Female\n   \033[1mIntroduce the number of your option:\033[0m ')
    if var1 == '0':
        new_case[1] = 'Male'
    else:
        new_case[1] = 'Female'

    # Selected music genres
    print('*Choose your three favourite genres from the following list:')
    print('   0 -Blues\n   1 -Dance\n   2 -Hard Rock\n   3 -Intrumental-Classical\n   4 -Jazz\n   5 -Latin\n   6 -Pop'
          '\n   7 -Reggae\n   8 -Reggaeton\n   9 -Rock\n   10 -Vocal')
    var2 = []
    var2.append(int(input('   \033[1mIntroduce the numbers of your first option:\033[0m ')))
    var2.append(int(input('   \033[1mIntroduce the numbers of your second option:\033[0m ')))
    var2.append(int(input('   \033[1mIntroduce the numbers of your third option:\033[0m ')))
    var2 = np.sort(var2)
    string_list = ['Blues','Dance', 'Hard Rock', 'Intrumental-Classical', 'Jazz', 'Latin', 'Pop', 'Reggae',
                   'Reggaeton', 'Rock', 'Vocal']
    new_case[2] = string_list[var2[0]]+','+string_list[var2[1]]+','+string_list[var2[2]]

    # Select concentration
    print('*Choose how much concentration do you need while listening to the playlist:')
    print('   0 -Not at all\n   1 -Some concentration\n   2 -Yes, my full concentration')
    var3 = int(input('   \033[1mIntroduce the number of your option:\033[0m '))
    string_list = ['not at all', 'some concentration', 'yes, my full concentration']
    new_case[3] = string_list[var3]

    # Select hapiness
    print('*Choose your level of hapiness:')
    print('   0 -Really unhappy\n   1 -Not so happy\n   2 -Neutral\n   3 -Happy\n   4 -Very happy')
    var4 = int(input('   \033[1mIntroduce the number of your option:\033[0m '))
    string_list = ['really unhappy', 'not so happy', 'neutral', 'happy', 'very happy']
    new_case[4] = string_list[var4]

    # Select energy
    print('*Choose your level of energy:')
    print('   0 -Very calm\n   1 -Calm\n   2 -Neutral\n   3 -With some energy\n   4 -With a lot of energy')
    var5 = int(input('   \033[1mIntroduce the number of your option:\033[0m '))
    string_list = ['very calm', 'calm', 'neutral', 'with some energy', 'with a lot of energy']
    new_case[5] = string_list[var5]

    # Select future hapiness
    print('*Choose the level of hapiness do you want to be after listening to the song:')
    print('   0 -Be less happy\n   1 -Keep the same feeling\n   2 -Be more happy')
    var6 = int(input('   \033[1mIntroduce the number of your option:\033[0m '))
    string_list = ['I get less happy', 'I keep the same', 'I get happier']
    new_case[6] = string_list[var6]

    # Select future energy
    print('*Choose the level of energy do you want to be after listening to the song:')
    print('   0 -Be more relaxed\n   1 -Keep the same feeling\n   2 -Feel more energy')
    var7 = int(input('   \033[1mIntroduce the number of your option:\033[0m '))
    string_list = ['I get more relaxed', 'I keep the same', 'I feel with more energy']
    new_case[7] = string_list[var7]

    return new_case