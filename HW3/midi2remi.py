import numpy as np
import miditoolkit
import copy
import pickle

from utils import Item, read_items, quantize_items, chord_extract, \
    group_items, Event, item2event, word_to_event, write_midi


if __name__ == '__main__':
    path = './TA_demo/000.mid'
    midi = miditoolkit.midi.parser.MidiFile(path)
    print(midi)

    # notes
    print('Notes')
    print(*midi.instruments[0].notes[0:5], sep='\n')
    # temples
    print('\nTempos')
    print(*midi.tempo_changes[0:5], sep='\n')

    note_items, tempo_items = read_items(path)

    print('Note items')
    print(*note_items[:5], sep='\n')
    print('\nTempo items')
    print(*tempo_items[:5], sep='\n')

    note_items = quantize_items(note_items)

    print('Quantized note items')
    print(*note_items[:5], sep='\n')

    # chord_items = chord_extract(path, note_items[-1].end)

    # print('Chord items')
    # print(*chord_items[:5], sep='\n')

    # if using chord items
    # items = chord_items + tempo_items + note_items

    # if not using chord items
    items = tempo_items + note_items

    max_time = note_items[-1].end
    groups = group_items(items, max_time)

    for g in groups:
        print(*g, sep='\n')
        print()

    events = item2event(groups)

    print('Events')
    print(*events[:10], sep='\n')

    # read your dictionary
    event2word, word2event = pickle.load(open('./TA_demo/basic_event_dictionary.pkl', 'rb'))

    words = []
    for event in events:
        e = '{}_{}'.format(event.name, event.value)
        if e in event2word:
            words.append(event2word[e])
        else:
            # OOV
            if event.name == 'Note Velocity':
                # replace with max velocity based on our training data
                words.append(event2word['Note Velocity_21'])
            else:
                # something is wrong
                # you should handle it for your own purpose
                print('something is wrong! {}'.format(e))

    print('Tokens')
    print(words[:10])
