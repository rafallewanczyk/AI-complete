import requests


class Generator:

    def generate(input):
        r = requests.post('http://127.0.0.1:8000', json={"file_name": "name", "code": input})
        predictions = r.text.split('#')

        if '\n' in predictions:
            predictions[predictions.index('\n')] = '&lt;ENTER&gt;'

        if '<UNK>' in predictions:
            predictions[predictions.index('<UNK>')] = '&lt;UNK&gt;'
        merged = '<br>'
        merged = merged.join(predictions)
        return merged
