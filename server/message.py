import json
class Message:

    def __init__(self, file_name, code):
        self.file_name = file_name
        self.code = code
    def __repr__(self):
        return (self.file_name, self.code).__str__()


msg = Message('tensorflow.txt', 'from os imoprt io\nfrom io import X')
jsonobj = json.dumps(msg.__dict__)
new_msg = (Message(**json.loads(jsonobj)))
print(repr(new_msg))

