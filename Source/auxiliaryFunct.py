def menu(new_case):
    print('\n\033[1m'+'Welcome to our Case Based Reasoning. '+'\033[0m'+'\nPlease, answer to the following questions:')
    new_case[0] = int(input('Age: '))
    new_case[1] = str(input('Male or Female: '))
    print('Choose your three favourite genres from the following list in order separated by a comma (without '
          'blancks). For example: Blues,Instrumental-Classical,Rock')
    print('"Blues", "Dance", "Hard Rock", "Intrumental-Classical", "Jazz", "Latin", "Pop", "Reggae", "Reggaeton", "Rock", "Vocal".')
    new_case[2] = str(input('Introduce your option: '))
    new_case[3] = str(input('How much concentration do you need now ("not at all", "some concentration or "yes, '
                            'my full concentration". Introduce your option: '))
    new_case[4] = str(input('What is your level of hapiness ("really unhappy", "not so happy", "neutral", "happy", "very happy". '
                            'Introduce your option: '))
    new_case[5] = str(input('What is your level of energy ("very calm", "calm", "neutral", "with some energy", "with a lot of energy". '
                            'Introduce your option: '))
    new_case[6] = str(input('What is the level of hapiness do you want to be after listening to the song ("I get less '
                            'happy", "I keep the same", "I get happier". Introduce your option: '))
    new_case[7] = str(input('What is the level of energy do you want to be after listening to the song ("I get more '
                            'relaxed", "I keep the same", "I feel with more energy". Introduce your option: '))
    return new_case