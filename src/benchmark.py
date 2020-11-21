import src.utils as utils
from tqdm.auto import tqdm


def single_benchmark(model, target_file, html=False):
    tokenized_file = utils.my_tokenize(target_file)

    result = HTML(target_file)
    if len(tokenized_file) == 0:
        return (0, 0)

    proper_predictions = 0
    total_predictions = 0
    prediction = model.get_prediction(tokenized_file[0], 1)
    for token in tokenized_file[1:]:

        if token in prediction:
            proper_predictions += 1
            result.add_correct(token)

        else:
            if prediction[0] == '<UNK>':
                result.add_wrong('UNK')
            else:
                result.add_wrong(token)

        total_predictions += 1

        prediction = model.get_prediction(token, 1)

    if html:
        result.render(proper_predictions, total_predictions)
        result.save(target_file)
    return proper_predictions, total_predictions


class HTML:
    output = '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Title</title></head><body>'

    def __init__(self, name):
        self.output += name.split('\\')[-1] + '<br>'

    def add_correct(self, token):
        if token == '\n':
            self.output += f'<span style="background-color: #00FF00">&nbsp</span><br>'
        else:
            self.output += f'<span style="background-color: #00FF00">{token} </span>'

    def add_wrong(self, token):
        if token == '\n':
            self.output += f'<span style="background-color: #FF0000">&nbsp</span><br>'
        else:
            self.output += f'<span style="background-color: #FF0000">{token} </span>'

    def render(self, proper, all):
        self.output += f'<br>{proper}/{all} = {proper / all}<br>'
        self.output += '</body></html>'
        return self.output

    def save(self, name):
        name = '.\\results\\' + name.split('\\')[-1] + '.html'

        with open(name, 'w') as f:
            f.write(self.output)
