
class Position():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return str(list((self.x, self.y)))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    def __ne__(self, other):
        return self.x != other.x or self.y != other.y

    def __add__(self, other):
        return Position(self.x + other.x, self.x + other.y)
    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __sub__(self, other):
        return Position(self.x - other.x, self.y - other.y)
    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        return self

    def __mul__(self, other):
        return Position(self.x * other, self.y * other)
    def __imul__(self, other):
        self.x *= other
        self.y *= other
        return self

    def __div__(self, other):
        return Position(self.x / other, self.y / other)
    def __truediv__(self, other):
        return Position(self.x / other, self.y / other)
    def __idiv__(self, other):
        self.x /= other
        self.y /= other
        return self