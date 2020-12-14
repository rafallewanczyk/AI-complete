"""
Very simple HTTP server in python (Updated for Python 3.7)
Usage:
    ./dummy-web-server.py -h
    ./dummy-web-server.py -l localhost -p 8000
Send a GET request:
    curl http://localhost:8000
Send a HEAD request:
    curl -I http://localhost:8000
Send a POST request:
    curl -d "foo=bar&bin=baz" http://localhost:8000
"""
import argparse
import tensorflow as tf
import json
import numpy as np
import pickle as pk
from message import Message
from http.server import HTTPServer, BaseHTTPRequestHandler
from src.model import Model
from src.ngram import Ngram
import src.utils as utils


class S(BaseHTTPRequestHandler):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    vocab_size = 20001  # CONST
    embedding_dim = 32  # CONST
    rnn_units = 512
    batch_size = 128  # CONST
    win_size = 5
    model = Model(vocab_size, embedding_dim, rnn_units, batch_size, win_size, '..\\checkpoints\\4\\model.h5', None)
    model.prepare_predictions('..\\drivers\\vocabulary.voc', '..\\checkpoints\\4\\model.h5')


    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def _html(self, message):
        """This just generates an HTML document that includes `message`
        in the body. Override, or re-write this do do more interesting stuff.
        """
        content = f"<html><body><h1>{message}</h1></body></html>"
        return content.encode("utf8")  # NOTE: must return a bytes object!

    def do_GET(self):
        self._set_headers()
        self.wfile.write(self._html("nothin to see here"))

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        # Doesn't do anything with posted data
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        msg = (Message(**json.loads(post_data)))
        print("received:\n", msg.code)

        #get RNN prediction
        prediction = self.model.get_prediction(msg.code)

        #get Ngram prediction
        tokenized_file = utils.tokenize_string(msg.code)
        if '<UNK>' in prediction:
            trigram = Ngram(3, tokenized_file)
            bigram = Ngram(2, tokenized_file)
            grams = trigram.predict(tokenized_file) + bigram.predict(tokenized_file)
            grams = [p[0] for p in grams]
            grams = list(dict.fromkeys(grams))[:5]
            prediction[prediction.index('<UNK>')] = grams
            prediction = np.hstack(prediction).tolist()
        print(prediction)

        self._set_headers()
        self.wfile.write(bytes('#'.join(prediction[:5]), encoding='utf-8'))


def run(server_class=HTTPServer, handler_class=S, addr="localhost", port=8000):
   # # model.train('.\\vocabulary', '..\\data\\django\\django\\apps', 4)
    # model.prepare_predictions('..\\src\\vocab10k', '..\\checkpoints\\10k.h5')
    # S.start_model()
    server_address = (addr, port)
    httpd = server_class(server_address, handler_class)

    print(f"Starting httpd server on {addr}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    import os
    print(os.getcwd())
    parser = argparse.ArgumentParser(description="Run a simple HTTP server")
    parser.add_argument(
        "-l",
        "--listen",
        default="localhost",
        help="Specify the IP address on which the server listens",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Specify the port on which the server listens",
    )
    args = parser.parse_args()
    run(addr=args.listen, port=args.port)
