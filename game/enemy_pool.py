
class EnemyPool():
    def __init__(self):
        self.__pool = []

    def add(self, enemy):
        self.__pool.append(enemy)

    def find_by_position(self, position):
        """
        This method might return None when the coresspoids was not found
        """
        for enemy in self.__pool:
            if enemy.position == position:
                return enemy
        return None
