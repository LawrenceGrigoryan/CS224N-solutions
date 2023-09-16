# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.
import utils


if __name__ == '__main__':
    pred = 'London'
    dev_data = open('../birth_dev.tsv', 'r').read()
    lines = dev_data.strip().split('\n')
    birth_places = [line.split('\t')[1] for line in lines]
    correct = len(list(filter(lambda x: x == pred, birth_places)))
    print(f'Accuracy when predicting only "London": {correct/len(birth_places)}')
