class Man:
    def __init__(self, name):
        self.name = name
        print('Initialized')
        
    def hello(self):
        print('hello ' + self.name)
    
    def goodbye(self):
        print('see you ' + self.name)
        
m = Man("David")

m.hello()
m.goodbye()