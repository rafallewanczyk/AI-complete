import numpy as np
import src.utils as utils
from src.ngram import Ngram
from tqdm.auto import tqdm


def single_benchmark(model, target_file, html=False):
    tokenized_file = utils.my_tokenize(target_file)

    result = HTML(target_file)
    if len(tokenized_file) == 0:
        return 0, 0, 0, 0

    proper_predictions = 0
    top_3 = 0
    top_5 = 0
    ranked = 0
    total_predictions = 0
    prediction = model.get_prediction(tokenized_file[0], 10, 1)
    for i, token in enumerate(tokenized_file[1:], 1):

        if '<UNK>' in prediction:
            trigram = Ngram(3, tokenized_file[:i])
            bigram = Ngram(2, tokenized_file[:i])
            grams = trigram.predict(tokenized_file[:i]) + bigram.predict(tokenized_file[:i])
            grams = [p[0] for p in grams]
            grams = list(dict.fromkeys(grams))[:5]
            prediction[prediction.index('<UNK>')] = grams
            prediction = np.hstack(prediction).tolist()

        if token in prediction[:3]:
            top_3 += 1

        if token in prediction[:5]:
            top_5 += 1
            result.add_correct(token)

        if token in prediction[:10]:
            ranked += 1/(prediction.index(token)+1)


        # elif prediction[0] == '<UNK>':
        #     trigram = Ngram(3, tokenized_file[:i])
        #     bigram = Ngram(2, tokenized_file[:i])
        #     predictions = trigram.predict(tokenized_file[:i]) + bigram.predict(tokenized_file[:i])
        #     # predictions = bigram.predict(tokenized_file[:i])
        #     predictions = [p[0] for p in predictions]
        #     predictions = list(dict.fromkeys(predictions))[:5]
        #     # predictions = predictions[:5]
        #     if token in predictions:
        #         result.add_correct(token)
        #         proper_predictions += 1
        #     else:
        #         result.add_wrong(token)

        else:
            # if prediction[0] == '<UNK>':
            #     result.add_wrong('UNK')
            # else:
            result.add_wrong(token)

        total_predictions += 1

        prediction = model.get_prediction(token, 10, 1)

    if html:
        result.render(top_5, total_predictions)
        result.save(target_file, f'top_5: {"%.2f" % (top_5/ total_predictions)}')
    return top_3, top_5, ranked, total_predictions


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

    def save(self, name, accuracy):
        name = '.\\results\\' + accuracy + '_' + name.split('\\')[-1] + '.html'

        with open(name, 'w') as f:
            f.write(self.output)
