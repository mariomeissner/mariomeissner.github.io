import argparse # To read the word to parse from stdin

# An error we will throw when a parse error occurs
class ParseError(Exception):
    pass

# The meat and potatoes
class Parser():

    def __init__(self, string):
        self.string = string
        self.position = 0

    def read(self, token):

        # First shift out any whitespace
        while self.string[self.position] == " ":
            self.position += 1


        if self.string[self.position] == token[0]: 
            self.position += 1
        else: return False

        for char in token[1:]:
            if self.string[self.position] == char:
                self.position += 1
            else:
                raise ParseError("Attempted to read {} but found char {}".format(token, char))
        return True

    def program(self): 

        if not self.read("{"): return False
        if not self.statement_list(): raise ParseError()
        if not self.read("}"): raise ParseError()
        return True

    def statement_list(self):

        if self.statement(): 
            if not self.statement_list(): raise ParseError()
            return True
        else: return True

    def statement(self):

        if self.id():

            if not self.read("="): raise ParseError()
            if not self.expr(): raise ParseError()
            if not self.read(";"): raise ParseError()
            return True

        elif self.read("if"): 
            
            if not self.read("("): raise ParseError()
            if not self.expr(): raise ParseError()
            if not self.read(")") or not self.read("then"): raise ParseError()
            if not self.statement(): raise ParseError()
            return True

        else: return False

    def expr(self): 

        if self.num(): return True
        elif self.id():
            if not self.expr_tail(): raise ParseError()
            return True
        else: return False

    def expr_tail(self): 

        if self.read("+") or self.read("-"):
            if not self.expr(): raise ParseError()
            return True
        else: return True

    # For simplicity, let's only allow id values a, b or c:
    def id(self):
        if self.read("a") or self.read("b") or self.read("c"): return True
        else: return False

    # For simplicity, only accept single digit numbers
    def num(self):
        for num_str in [str(num) for num in range(10)]:
            if self.read(num_str): return True
        return False

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Parses a string and tells you if it is part of the language or not")
    argparser.add_argument('string', metavar='str', type=str, nargs=1, help="The string to parse")
    args = argparser.parse_args()
    word = args.string[0]
    print("Parsing: {}".format(word))

    parser = Parser(word)
    try:
        if parser.program(): print("The given string is part of the language!")
        else: print("This word does not start with '{', which is required.")
    except ParseError:
        print("ParseError was thrown, word is not part of the language.")
    