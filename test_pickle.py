import pickle

class Test(object):

    def __init__(self):
        self.val = 20

    def save(self):
        with open('test_pickle', 'wb') as f:
            pickle.dump(self, f)

test = Test()
test.save()

with open('test_pickle', 'rb') as f:
    loaded = pickle.load(f)

print(loaded.val)
