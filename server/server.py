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
import json
from message import Message
from http.server import HTTPServer, BaseHTTPRequestHandler
from src.model import Model


class S(BaseHTTPRequestHandler):

    vocab_size = 10001
    embedding_dim = 128
    rnn_units = 512
    batch_size = 128
    win_size = 10
    model = Model(vocab_size, embedding_dim, rnn_units, batch_size, win_size)
    model.prepare_predictions('..\\src\\vocab10k', '..\\checkpoints\\10k.h5')


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
        prediction = self.model.get_prediction(msg.code, 100 )
        print(prediction)
        # print("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n", str(self.path), str(self.headers),
        #       post_data.decode('utf-8'))
        self._set_headers()
        self.wfile.write(self._html(''.join(prediction[0])))


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
